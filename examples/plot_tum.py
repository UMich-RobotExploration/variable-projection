#!/usr/bin/env python3
"""Plot a TUM trajectory file.

TUM format: each line is `timestamp tx ty tz qx qy qz qw`. Comment lines
(starting with `#`) and blank lines are ignored.

Usage:
    python3 plot_tum.py <trajectory.tum> [<trajectory.tum> ...] [-o out.pdf]

If multiple trajectories are passed, they are drawn on the same axes (with
distinct colours and a legend). The plotter auto-detects 2D trajectories
(z constant) and switches to a flat xy projection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

import numpy as np


def load_tum(path: Path) -> np.ndarray:
    """Return an (N, 8) float array: ts, tx, ty, tz, qx, qy, qz, qw."""
    rows = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) != 8:
                raise ValueError(f"{path}: bad TUM line (expected 8 fields, got {len(parts)}): {s!r}")
            rows.append([float(x) for x in parts])
    if not rows:
        raise ValueError(f"{path}: no usable rows")
    return np.asarray(rows, dtype=float)


def is_planar(xyz: np.ndarray, eps: float = 1e-6) -> bool:
    return float(np.ptp(xyz[:, 2])) < eps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path,
                     help="one or more TUM trajectory files")
    ap.add_argument("-o", "--output", type=Path, default=None,
                     help="save figure to this path instead of showing")
    ap.add_argument("--force-3d", action="store_true",
                     help="render as 3D even if all trajectories are planar")
    ap.add_argument("--no-markers", action="store_true",
                     help="hide start/end markers")
    ap.add_argument("--subplots", action="store_true",
                     help="put each trajectory in its own panel instead of overlaying them")
    args = ap.parse_args()

    if args.output is None:
        matplotlib.use("TkAgg" if sys.platform != "linux" else matplotlib.get_backend())
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajs: list[tuple[Path, np.ndarray]] = []
    for p in args.paths:
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1
        trajs.append((p, load_tum(p)))

    all_xyz = np.concatenate([t[:, 1:4] for _, t in trajs], axis=0)
    planar = is_planar(all_xyz) and not args.force_3d

    cmap = plt.get_cmap("tab10")

    def draw(ax, data, color, label, planar):
        if planar:
            ax.plot(data[:, 1], data[:, 2], "-", color=color, linewidth=1.4,
                     label=label)
            if not args.no_markers:
                ax.plot(data[0, 1], data[0, 2], "o", color=color, markersize=8,
                         markeredgecolor="black", markeredgewidth=0.6)
                ax.plot(data[-1, 1], data[-1, 2], "s", color=color, markersize=8,
                         markeredgecolor="black", markeredgewidth=0.6)
        else:
            ax.plot(data[:, 1], data[:, 2], data[:, 3], "-", color=color,
                     linewidth=1.4, label=label)
            if not args.no_markers:
                ax.scatter(data[0, 1],  data[0, 2],  data[0, 3],
                            color=color, marker="o", s=50,
                            edgecolor="black", linewidth=0.6)
                ax.scatter(data[-1, 1], data[-1, 2], data[-1, 3],
                            color=color, marker="s", s=50,
                            edgecolor="black", linewidth=0.6)

    def style(ax, planar, xyz_for_box=None):
        if planar:
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            if xyz_for_box is not None:
                spans = xyz_for_box.max(axis=0) - xyz_for_box.min(axis=0)
                mid = (xyz_for_box.max(axis=0) + xyz_for_box.min(axis=0)) / 2.0
                half = max(spans.max() / 2.0, 1e-6)
                ax.set_xlim(mid[0] - half, mid[0] + half)
                ax.set_ylim(mid[1] - half, mid[1] + half)
                ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.grid(True, alpha=0.3)

    n = len(trajs)
    if args.subplots and n > 1:
        # Side-by-side panels (1 row if ≤ 3, else 2 rows).
        cols = n if n <= 3 else (n + 1) // 2
        rows = 1 if n <= 3 else 2
        fig = plt.figure(figsize=(5.5 * cols, 5.0 * rows))
        for i, (p, data) in enumerate(trajs):
            kwargs = {"projection": "3d"} if not planar else {}
            ax = fig.add_subplot(rows, cols, i + 1, **kwargs)
            draw(ax, data, cmap(i % 10), p.name, planar)
            style(ax, planar, all_xyz if not planar else None)
            ax.set_title(p.name, fontsize=11)
    else:
        fig = plt.figure(figsize=(7.5, 6.5))
        kwargs = {"projection": "3d"} if not planar else {}
        ax = fig.add_subplot(111, **kwargs)
        for i, (p, data) in enumerate(trajs):
            draw(ax, data, cmap(i % 10), p.name, planar)
        style(ax, planar, all_xyz if not planar else None)
        title = "1 trajectory" if n == 1 else f"{n} trajectories"
        ax.set_title(f"{title}  ({'2D' if planar else '3D'})")
        if n > 1:
            ax.legend(loc="best", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    if args.output is None:
        plt.show()
    else:
        fig.savefig(args.output, bbox_inches="tight",
                     dpi=160 if args.output.suffix.lower() == ".png" else None)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
