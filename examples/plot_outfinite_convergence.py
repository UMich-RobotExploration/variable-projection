"""Convergence plot for the outfinite (RA-SLAM) dataset.

Two panels (objective vs iteration, objective vs wall time). Each formulation
gets 5 thin lines (one per init) plus a bold line at the iteration-wise / time-
binned median across the 5 inits.

Reads:  examples/data/raslam/outfinite/cached_results/results_rank5_*.json
Writes: examples/data/raslam/outfinite/outfinite_convergence.png
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("examples/data/raslam/outfinite/cached_results")
OUT_PNG  = Path("examples/data/raslam/outfinite/outfinite_convergence.png")

# formulation int code -> (display label, color)
FORM_META = {
    0: ("Explicit",          "tab:red"),
    1: ("Explicit + VarPro", "tab:orange"),
    2: ("Implicit (Ours)",   "tab:blue"),
}
ORDER = [2, 1, 0]  # draw Explicit last so it doesn't visually dominate


def load_runs() -> Dict[int, List[Tuple[np.ndarray, np.ndarray]]]:
    """code -> list of (iters, costs, times) per init."""
    runs: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    for p in sorted(DATA_DIR.glob("results_rank5_*.json")):
        d = json.loads(p.read_text())[0]
        costs = np.asarray(d["costs"], dtype=float)
        times = np.asarray(d["times"], dtype=float)
        iters = np.arange(len(costs))
        runs[d["formulation"]].append((iters, costs, times))
    return runs


def median_by_iter(runs):
    max_n = max(len(c) for _, c, _ in runs)
    M = np.full((len(runs), max_n), np.nan)
    for i, (_, c, _) in enumerate(runs):
        M[i, : len(c)] = c
        # After a run finishes, hold its final value so the median doesn't
        # collapse when only a subset of runs are still active.
        M[i, len(c):] = c[-1]
    return np.arange(max_n), np.nanmedian(M, axis=0)


def median_by_time(runs, n_grid: int = 400):
    t_min = max(t[0] for _, _, t in runs)  # 0 for everyone, kept for safety
    t_max = max(t[-1] for _, _, t in runs)
    grid = np.linspace(max(t_min, 1e-3), t_max, n_grid)
    M = np.empty((len(runs), n_grid))
    for i, (_, c, t) in enumerate(runs):
        # piecewise-constant interpolation: at grid time g, take the cost
        # at the last iteration whose timestamp <= g (or the final cost if g
        # exceeds the run's last timestamp).
        idx = np.searchsorted(t, grid, side="right") - 1
        idx = np.clip(idx, 0, len(c) - 1)
        M[i] = c[idx]
    return grid, np.median(M, axis=0)


def main() -> int:
    runs = load_runs()
    fig, (ax_iter, ax_time) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    for code in ORDER:
        label, color = FORM_META[code]
        for _, c, t in runs[code]:
            ax_iter.plot(np.arange(len(c)), c, color=color, alpha=0.25, linewidth=1.0)
            ax_time.plot(t, c, color=color, alpha=0.25, linewidth=1.0)
        it_x, it_y = median_by_iter(runs[code])
        ax_iter.plot(it_x, it_y, color=color, linewidth=2.2, label=label)
        tg_x, tg_y = median_by_time(runs[code])
        ax_time.plot(tg_x, tg_y, color=color, linewidth=2.2, label=label)

    for ax, xl in [(ax_iter, "Iteration"), (ax_time, "Wall time (s)")]:
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("Objective")
        ax.grid(True, which="both", axis="both", alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    fig.suptitle("outfinite (RA-SLAM, rank 5) — convergence "
                 "(thin lines: 5 random inits; bold: median)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
