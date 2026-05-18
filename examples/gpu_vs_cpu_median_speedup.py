#!/usr/bin/env python3
"""Time-weighted GPU/CPU speedup per formulation, aggregated across the
standard datasets — same style as gpu_vs_cpu_speedups.png but with one bar
per (category, formulation) instead of one bar per dataset.

For each standard dataset we compute the median time-to-convergence in
each (backend, formulation) cell (matching the convention used by
sweep_plots_lines.py and standard_speedup_plots.py: first time the cost
drops within (1+TOL)*target_cost where target_cost is the best Implicit
final cost on that dataset; absolute-tolerance fallback for near-zero
SNL targets).

Per-category bars are aggregated by **summing the per-dataset median
times** in each category and then taking the ratio of the sums:

    category_speedup =  Σ_d  median(CPU t_conv)_d
                      / Σ_d  median(GPU t_conv)_d

This is deployment-realistic: it answers "if I ran every dataset in the
category back-to-back, how much faster would the GPU finish?" Long-running
datasets dominate the ratio in proportion to their actual compute time,
which is the right behaviour when comparing throughput.

Output: examples/data/analysis/gpu_vs_cpu_median_speedup.pdf (+ .png)
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

# Reuse the loader + speedup-extraction logic from standard_speedup_plots.py
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from standard_speedup_plots import (
    collect_speedups, FORMULATIONS, FORM_LABELS, FORM_COLORS,
)


REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "examples" / "data" / "analysis"

# Only the three categories from the reference figure.
CATEGORIES = ["pgo", "raslam", "sfm"]
CATEGORY_LABEL = {"pgo": "PGO", "raslam": "RA-SLAM", "sfm": "SfM"}

# Bar order + colours match the reference legend: Original (blue),
# Orig.+VP (orange), Ours (green).
BAR_ORDER = ["Explicit", "ExplicitVarPro", "Implicit"]


plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.0,
})


def _log_tick_fmt(v, _pos):
    if v >= 1:
        return f"{v:g}" + r"$\times$"
    return rf"$1/{1.0 / v:g}\times$"


def main():
    _, backend_speedups = collect_speedups()

    # Time-weighted aggregation: ratio of sums, not median of ratios.
    # collect_speedups returns lists of (speedup, t_cpu, t_gpu) tuples.
    speedups = {}
    counts = {}
    for cat in CATEGORIES:
        for form in BAR_ORDER:
            entries = backend_speedups.get((cat, form), [])
            counts[(cat, form)] = len(entries)
            if not entries:
                speedups[(cat, form)] = None
                continue
            total_cpu = sum(t_cpu for _, t_cpu, _ in entries)
            total_gpu = sum(t_gpu for _, _, t_gpu in entries)
            speedups[(cat, form)] = (total_cpu / total_gpu) if total_gpu > 0 else None

    # Console summary so it's easy to drop into a report.
    print("Time-weighted (Σ CPU times) / (Σ GPU times) per category:\n")
    print(f"{'category':<10} {'n datasets':>10} | "
          f"{'Original':>10} {'Orig.+VP':>10} {'Ours':>10}")
    for cat in CATEGORIES:
        n = max(counts[(cat, f)] for f in BAR_ORDER)
        cells = []
        for f in BAR_ORDER:
            m = speedups[(cat, f)]
            cells.append(f"{m:.2f}x" if m is not None else "  —  ")
        print(f"{CATEGORY_LABEL[cat]:<10} {n:>10} | "
              f"{cells[0]:>10} {cells[1]:>10} {cells[2]:>10}")

    # ---- Plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11.0, 5.4), dpi=200)
    x = np.arange(len(CATEGORIES), dtype=float)
    width = 0.27

    for slot, form in enumerate(BAR_ORDER):
        offset = -width + slot * width
        ys = []
        for cat in CATEGORIES:
            m = speedups[(cat, form)]
            ys.append(m if m is not None else np.nan)
        bars = ax.bar(
            x + offset, ys, width=width,
            color=FORM_COLORS[form],
            edgecolor="0.25", linewidth=0.8,
            label=f"CPU / GPU ({FORM_LABELS[form]})",
        )
        # Annotate each bar with its median value.
        for rect, y in zip(bars, ys):
            if not np.isfinite(y):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                y * 1.04,
                f"{y:.2f}" + r"$\times$",
                ha="center", va="bottom",
                fontsize=11, color="0.15",
            )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_yscale("log")
    # Reasonable tick ladder that covers both speedup and slowdown side.
    ticks = [1/3, 1/2, 1, 1.5, 2, 3]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(_log_tick_fmt))
    # Pad the limits a little above the biggest value.
    all_vals = [m for m in speedups.values() if m is not None and np.isfinite(m)]
    ymax = max(all_vals + [1.0]) * 1.35
    ymin = min(all_vals + [1.0]) / 1.4
    ax.set_ylim(ymin, ymax)

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABEL[c] for c in CATEGORIES])
    ax.set_ylabel("Runtime Improvement Factor (CPU / GPU)")
    ax.set_title("CPU-to-GPU Speedup")
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(ncol=3, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 1.0), columnspacing=1.8, handletextpad=0.6)

    out_pdf = ANALYSIS / "gpu_vs_cpu_median_speedup.pdf"
    out_png = ANALYSIS / "gpu_vs_cpu_median_speedup.png"
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"\nwrote {out_pdf.relative_to(REPO)}")
    print(f"wrote {out_png.relative_to(REPO)}")


if __name__ == "__main__":
    main()
