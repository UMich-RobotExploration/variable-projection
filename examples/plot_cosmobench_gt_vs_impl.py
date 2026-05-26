"""Overlay plot: GT (dashed) vs Implicit (solid) per robot, on one cosmobench
dataset. Uses the per-robot-aligned solution so trajectory shape is visible
without the multi-robot gauge offset.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROBOT_COLORS = {
    "A": "#1f77b4", "B": "#d62728", "C": "#2ca02c",
    "D": "#9467bd", "E": "#ff7f0e",
}


def parse_pyfg_gt(pyfg: Path):
    names, xyz = [], []
    for ln in pyfg.read_text().splitlines():
        parts = ln.split()
        if parts and parts[0] == "VERTEX_SE3:QUAT":
            names.append(parts[2])
            xyz.append([float(parts[3]), float(parts[4]), float(parts[5])])
    return names, np.asarray(xyz, dtype=float)


def parse_tum_xyz(p: Path):
    rows = [list(map(float, ln.split())) for ln in p.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    return np.asarray(rows, dtype=float)[:, 1:4]


def umeyama_no_scale(src, dst):
    A = src - src.mean(0)
    B = dst - dst.mean(0)
    U, _S, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = Vt.T @ D @ U.T
    t = dst.mean(0) - R @ src.mean(0)
    return R, t


def split_by_robot(names: List[str], pts: np.ndarray):
    out: Dict[str, list] = defaultdict(list)
    for n, p in zip(names, pts):
        out[n[0]].append(p)
    return {r: np.asarray(v) for r, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pyfg", type=Path, required=True)
    ap.add_argument("--tum",  type=Path, required=True,
                     help="Implicit final TUM (from irls_robust)")
    ap.add_argument("-o",    type=Path, required=True)
    args = ap.parse_args()

    names, gt = parse_pyfg_gt(args.pyfg)
    robots = np.array([n[0] for n in names])
    est = parse_tum_xyz(args.tum)
    if est.shape[0] != len(names):
        raise SystemExit(f"row count mismatch: pyfg {len(names)} vs tum {est.shape[0]}")

    # per-robot rigid align to GT
    aligned = np.empty_like(est)
    for r in sorted(set(robots)):
        m = robots == r
        R, t = umeyama_no_scale(est[m], gt[m])
        aligned[m] = est[m] @ R.T + t

    gt_by_r = split_by_robot(names, gt)
    est_by_r = split_by_robot(names, aligned)

    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    for r in sorted(gt_by_r.keys()):
        c = ROBOT_COLORS.get(r, "k")
        g = gt_by_r[r]
        e = est_by_r[r]
        ax.plot(g[:, 0], g[:, 1], g[:, 2], "--", color=c, linewidth=1.3, alpha=0.85,
                label=f"robot {r} — GT")
        ax.plot(e[:, 0], e[:, 1], e[:, 2], "-",  color=c, linewidth=1.3,
                label=f"robot {r} — Implicit")
        # start/end markers from GT (so they sit on the ground-truth track)
        ax.scatter(*g[0],  color=c, marker="o", s=30, edgecolor="black", linewidth=0.5)
        ax.scatter(*g[-1], color=c, marker="s", s=30, edgecolor="black", linewidth=0.5)

    all_xyz = np.concatenate([gt, aligned], axis=0)
    mid = (all_xyz.max(0) + all_xyz.min(0)) / 2.0
    half = max((all_xyz.max(0) - all_xyz.min(0)).max() / 2.0, 1e-3) * 1.05
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"{args.pyfg.stem} — Implicit (solid) vs GT (dashed) per robot, "
                 f"per-robot Umeyama aligned", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    args.o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.o, dpi=150, bbox_inches="tight")
    print(f"wrote {args.o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
