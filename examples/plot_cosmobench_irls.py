"""Plot ground-truth + IRLS-optimized trajectories for a cosmobench dataset.

Reads a PyFG file (for ground-truth pose chains + per-robot grouping) and the
three IRLS-output TUM files produced by `irls_robust` with formulations
`explicit`, `expvp`, `impl`. Renders a 2x2 figure:
  (1) ground truth   (2) Explicit
  (3) Explicit VarProj   (4) Implicit
Each optimized trajectory is rigidly aligned to GT via Umeyama (no scale) so
the geometric shape and the magnitude of residual outlier-pull are
visually comparable.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROBOT_COLORS = {
    "A": "#1f77b4",
    "B": "#d62728",
    "C": "#2ca02c",
    "D": "#9467bd",
    "E": "#ff7f0e",
}


def parse_pyfg_gt(pyfg: Path) -> Tuple[List[str], Dict[str, List[Tuple[float, float, float]]]]:
    """Return (vertex_names_in_pyfg_order, gt_xyz_per_robot)."""
    names: List[str] = []
    gt_by_robot: Dict[str, List[Tuple[float, float, float]]] = {}
    for line in pyfg.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] != "VERTEX_SE3:QUAT":
            continue
        name = parts[2]
        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
        names.append(name)
        robot = name[0]
        gt_by_robot.setdefault(robot, []).append((x, y, z))
    return names, {r: pts for r, pts in gt_by_robot.items()}


def parse_tum(tum: Path) -> np.ndarray:
    rows = [list(map(float, ln.split())) for ln in tum.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    return np.asarray(rows, dtype=float)


def umeyama_no_scale(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid SE(3) alignment: find R, t minimizing ||R src + t - dst||^2."""
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    A = src - src_mean
    B = dst - dst_mean
    H = A.T @ B
    U, _S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = Vt.T @ D @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def apply_rigid(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return pts @ R.T + t


def split_by_robot(names: List[str], pts: np.ndarray) -> Dict[str, np.ndarray]:
    by_robot: Dict[str, List[np.ndarray]] = {}
    for n, p in zip(names, pts):
        by_robot.setdefault(n[0], []).append(p)
    return {r: np.asarray(v) for r, v in by_robot.items()}


def plot_panel(ax, traj_by_robot: Dict[str, np.ndarray], title: str,
               limits: np.ndarray) -> None:
    for robot, pts in sorted(traj_by_robot.items()):
        color = ROBOT_COLORS.get(robot, "k")
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color=color, linewidth=1.0,
                label=f"robot {robot}")
        ax.scatter(*pts[0], color=color, marker="o", s=24, edgecolor="black",
                    linewidth=0.4, zorder=5)
        ax.scatter(*pts[-1], color=color, marker="s", s=24, edgecolor="black",
                    linewidth=0.4, zorder=5)
    mid = (limits.max(axis=0) + limits.min(axis=0)) / 2
    half = max((limits.max(axis=0) - limits.min(axis=0)).max() / 2.0, 1e-3) * 1.05
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pyfg", type=Path, required=True,
                     help="Ground-truth pyfg file (vertex order must match TUM line order).")
    ap.add_argument("--tum-explicit", type=Path, required=True)
    ap.add_argument("--tum-expvp", type=Path, required=True)
    ap.add_argument("--tum-impl", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--no-align", action="store_true",
                     help="skip Umeyama alignment of optimized trajectories to GT")
    ap.add_argument("--per-robot-align", action="store_true",
                     help="Umeyama-align each robot's trajectory to GT independently "
                          "(shows intrinsic shape error, hides inter-robot gauge error)")
    args = ap.parse_args()

    names, gt_by_robot = parse_pyfg_gt(args.pyfg)
    gt_xyz = np.asarray([gt_by_robot[n[0]][0] for n in names])  # placeholder
    # Reconstruct GT in vertex-line order so it aligns row-for-row with TUM.
    gt_xyz_list = []
    counter: Dict[str, int] = {}
    for n in names:
        r = n[0]
        i = counter.get(r, 0)
        gt_xyz_list.append(gt_by_robot[r][i])
        counter[r] = i + 1
    gt_xyz = np.asarray(gt_xyz_list)

    methods = [
        ("Explicit", args.tum_explicit),
        ("Explicit VarProj", args.tum_expvp),
        ("Implicit", args.tum_impl),
    ]
    robots = np.array([n[0] for n in names])
    method_xyz: Dict[str, np.ndarray] = {}
    for label, tum in methods:
        rows = parse_tum(tum)
        if rows.shape[0] != len(names):
            raise ValueError(
                f"{tum.name}: {rows.shape[0]} rows but pyfg has {len(names)} vertices"
            )
        xyz = rows[:, 1:4]
        if args.per_robot_align:
            aligned = np.empty_like(xyz)
            for r in sorted(set(robots)):
                mask = robots == r
                R, t = umeyama_no_scale(xyz[mask], gt_xyz[mask])
                aligned[mask] = apply_rigid(xyz[mask], R, t)
            xyz = aligned
        elif not args.no_align:
            R, t = umeyama_no_scale(xyz, gt_xyz)
            xyz = apply_rigid(xyz, R, t)
        method_xyz[label] = xyz

    # Per-method per-robot trajectory dicts and an overall bbox for consistent
    # axis limits across all 4 panels.
    gt_by_robot_arr = split_by_robot(names, gt_xyz)
    method_by_robot = {label: split_by_robot(names, xyz)
                        for label, xyz in method_xyz.items()}

    all_pts = [gt_xyz] + list(method_xyz.values())
    limits = np.concatenate(all_pts, axis=0)

    fig = plt.figure(figsize=(13.5, 11))
    titles_and_data = [
        ("Ground truth", gt_by_robot_arr),
        ("Explicit (IRLS, GM)", method_by_robot["Explicit"]),
        ("Explicit VarProj (IRLS, GM)", method_by_robot["Explicit VarProj"]),
        ("Implicit (IRLS, GM)", method_by_robot["Implicit"]),
    ]
    for i, (title, data) in enumerate(titles_and_data, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        plot_panel(ax, data, title, limits)

    suptitle = f"{args.pyfg.stem} — IRLS (GM kernel, odom init, no priors)"
    fig.suptitle(suptitle, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
