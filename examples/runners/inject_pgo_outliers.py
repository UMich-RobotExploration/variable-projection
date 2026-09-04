#!/usr/bin/env python3
"""Inject outlier loop-closures into each PGO dataset.

Two outlier models are supported:

  --mode identity (default): each outlier is an identity transform (zero
    translation, identity rotation) between a randomly sampled pose pair.
    Matches the convention used in the riSAM paper. Has a known failure mode
    for Implicit formulations because identity outliers collapse trajectories
    toward origin.

  --mode random: each outlier has a uniformly random rotation and a
    translation drawn uniformly from the trajectory's bounding box. Matches
    the Yang et al. 2020 (GNC) main-experiment convention. No origin
    attractor, so outliers don't all pull in the same direction.

The fraction is computed as a percent of existing loop closures (non-
sequential pose-pose edges, i.e. pairs (s1, s2) with same prefix character
and index(s2) != index(s1) + 1).

Output goes to examples/data/pgo_outliers/<name>_o<pct>/<name>_o<pct>.pyfg
for pct ∈ {10, 20, 30}. Counts are reported per dataset.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent.parent
PGO_DIR = REPO / "examples" / "data" / "pgo"
OUT_ROOT = REPO / "examples" / "data" / "pgo_outliers"

# Identity transform components per edge type:
#   EDGE_SE2:  dx dy dth  (3 numbers)
#   EDGE_SE3:QUAT: dx dy dz qx qy qz qw  (7 numbers)
IDENTITY_SE2 = "0.000000 0.000000 0.000000"
IDENTITY_SE3 = "0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000"


def random_se2(rng: np.random.Generator, bbox_min: np.ndarray,
                bbox_max: np.ndarray) -> str:
    tx = rng.uniform(bbox_min[0], bbox_max[0])
    ty = rng.uniform(bbox_min[1], bbox_max[1])
    th = rng.uniform(-np.pi, np.pi)
    return f"{tx:.6f} {ty:.6f} {th:.6f}"


def random_se3(rng: np.random.Generator, bbox_min: np.ndarray,
                bbox_max: np.ndarray) -> str:
    tx = rng.uniform(bbox_min[0], bbox_max[0])
    ty = rng.uniform(bbox_min[1], bbox_max[1])
    tz = rng.uniform(bbox_min[2], bbox_max[2])
    # Uniform random rotation: sample 4D Gaussian and normalize. Each direction
    # in S^3 corresponds to a rotation in SO(3) (up to sign, which is a
    # redundancy that washes out for sampling).
    q = rng.normal(0.0, 1.0, 4)
    q /= np.linalg.norm(q)
    qx, qy, qz, qw = q.tolist()
    return f"{tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"


def parse_pose_symbol(s: str) -> tuple[str, int] | None:
    """`A123` → ('A', 123). Returns None if it doesn't look like one."""
    m = re.match(r"^([A-Za-z]+)(\d+)$", s)
    return (m.group(1), int(m.group(2))) if m else None


def is_sequential(s1: str, s2: str) -> bool:
    p1, p2 = parse_pose_symbol(s1), parse_pose_symbol(s2)
    if p1 is None or p2 is None:
        return False
    return p1[0] == p2[0] and p2[1] == p1[1] + 1


def inject(pyfg_in: Path, pyfg_out: Path, fraction: float, seed: int,
            mode: str) -> dict:
    """Append `fraction * n_loop_closures` outliers to the .pyfg.

    `mode` is "identity" (zero translation, identity rotation) or "random"
    (uniform random SE(d) transform with translation in the trajectory bbox).
    """
    pose_symbols: list[str] = []
    seen_pose_symbols: set[str] = set()
    positions: list[list[float]] = []
    edges: list[tuple[str, str]] = []
    edge_kind: str | None = None   # "SE2" or "SE3"
    info_str: str | None = None    # last seen edge's info-matrix tail
    lines: list[str] = []

    with pyfg_in.open() as f:
        for raw in f:
            lines.append(raw)
            tok = raw.split()
            if not tok:
                continue
            if tok[0] == "VERTEX_SE2":
                sym = tok[2]
                if sym not in seen_pose_symbols:
                    seen_pose_symbols.add(sym)
                    pose_symbols.append(sym)
                    # VERTEX_SE2 <ts> <sym> <tx> <ty> <theta>
                    positions.append([float(tok[3]), float(tok[4])])
            elif tok[0] == "VERTEX_SE3:QUAT":
                sym = tok[2]
                if sym not in seen_pose_symbols:
                    seen_pose_symbols.add(sym)
                    pose_symbols.append(sym)
                    # VERTEX_SE3:QUAT <ts> <sym> <tx> <ty> <tz> <q...>
                    positions.append(
                        [float(tok[3]), float(tok[4]), float(tok[5])])
            elif tok[0] == "EDGE_SE2":
                edges.append((tok[2], tok[3]))
                if edge_kind is None:
                    edge_kind = "SE2"
                    # EDGE_SE2 <ts> <i> <j> <dx> <dy> <dth> <6 info numbers>
                    info_str = " ".join(tok[7:])
            elif tok[0] == "EDGE_SE3:QUAT":
                edges.append((tok[2], tok[3]))
                if edge_kind is None:
                    edge_kind = "SE3"
                    # <ts> <i> <j> <dx dy dz> <qx qy qz qw> <21 info numbers>
                    info_str = " ".join(tok[11:])

    if edge_kind is None or info_str is None:
        raise RuntimeError(f"{pyfg_in}: no pose-pose edges found")

    loop_closures = [(a, b) for (a, b) in edges if not is_sequential(a, b)]
    n_outliers = int(round(fraction * len(loop_closures)))

    existing = set()
    for a, b in edges:
        existing.add((a, b))
        existing.add((b, a))

    rng = np.random.default_rng(seed)
    n_poses = len(pose_symbols)
    pos_arr = np.asarray(positions, dtype=float)
    # Translation sampling box: the actual trajectory bbox in `random` mode.
    # Pad by 10% so corners are reachable, and guard against degenerate axes.
    bbox_min = pos_arr.min(axis=0)
    bbox_max = pos_arr.max(axis=0)
    span = bbox_max - bbox_min
    span[span < 1e-6] = 1.0
    bbox_min -= 0.05 * span
    bbox_max += 0.05 * span

    outlier_lines: list[str] = []
    attempts = 0
    max_attempts = max(10 * n_outliers, 100)
    while len(outlier_lines) < n_outliers and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(n_poses, size=2, replace=False)
        s1, s2 = pose_symbols[int(i)], pose_symbols[int(j)]
        if (s1, s2) in existing:
            continue
        existing.add((s1, s2))
        existing.add((s2, s1))
        if edge_kind == "SE2":
            tx = IDENTITY_SE2 if mode == "identity" else random_se2(rng, bbox_min, bbox_max)
            outlier_lines.append(
                f"EDGE_SE2 0.000000 {s1} {s2} {tx} {info_str}\n")
        else:
            tx = IDENTITY_SE3 if mode == "identity" else random_se3(rng, bbox_min, bbox_max)
            outlier_lines.append(
                f"EDGE_SE3:QUAT 0.000000 {s1} {s2} {tx} {info_str}\n")

    pyfg_out.parent.mkdir(parents=True, exist_ok=True)
    with pyfg_out.open("w") as out:
        out.writelines(lines)
        out.writelines(outlier_lines)

    return {
        "poses": n_poses,
        "loop_closures": len(loop_closures),
        "requested_outliers": n_outliers,
        "actual_outliers": len(outlier_lines),
        "edge_kind": edge_kind,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--percentages", nargs="+", type=int, default=[10, 20, 30])
    ap.add_argument("--mode", choices=("identity", "random"), default="random",
                     help="outlier transform model (default: random SE(d))")
    ap.add_argument("--force", action="store_true",
                     help="wipe examples/data/pgo_outliers/ before writing")
    args = ap.parse_args()

    if args.force and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"mode={args.mode}")
    print(f"{'dataset':<18}{'edge':>4}{'poses':>9}{'LCs':>7}"
           + "".join(f"{f'+o{p}':>8}" for p in args.percentages))
    for dataset_dir in sorted(PGO_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        pyfg_files = list(dataset_dir.glob("*.pyfg"))
        if not pyfg_files:
            continue
        pyfg_in = pyfg_files[0]
        name = pyfg_in.stem
        row_counts: list[int] = []
        seed_base = abs(hash(name)) % (2**31)
        for pct in args.percentages:
            out_name = f"{name}_o{pct}"
            out_dir = OUT_ROOT / out_name
            out_pyfg = out_dir / f"{out_name}.pyfg"
            stats = inject(pyfg_in, out_pyfg, pct / 100.0,
                            seed_base + pct, args.mode)
            row_counts.append(stats["actual_outliers"])
        print(f"{name:<18}{stats['edge_kind']:>4}{stats['poses']:>9}"
               f"{stats['loop_closures']:>7}"
               + "".join(f"{c:>8}" for c in row_counts))
    print(f"\nwrote {OUT_ROOT.relative_to(REPO)}/<name>_o{{10,20,30}}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
