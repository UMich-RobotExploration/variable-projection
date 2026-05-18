#!/usr/bin/env python3
"""Two-panel bar chart: total solve time vs. the shared precompute portion.

For each grid3d sweep scenario, plots six bars per group — Explicit, Original
+ V.P., and Ours, each on CPU and GPU — with a dark overlay marking the
algorithm-agnostic precompute (CR / Cholesky / B). The precompute is only
overlaid on the V.P. and Ours bars, since the plain Explicit formulation
doesn't form the Schur structure.

Inputs (produced by earlier sweep stages):
  examples/data/analysis/preprocess_times.json
  examples/data/grid3d_sweep/<axis>/<scenario>/cached_results/*.json

Outputs:
  examples/data/analysis/grid3d_sweep_preprocess_breakdown.{pdf,png}
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size":       13,
    "axes.labelsize":  14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
})


REPO = Path(__file__).resolve().parent.parent
SWEEP_ROOT = REPO / "examples" / "data" / "grid3d_sweep"
ANALYSIS_DIR = REPO / "examples" / "data" / "analysis"

FORMULATION_NAME = {0: "Exp", 1: "ExpVP", 2: "Impl"}

# (formulation, backend, face, edge, hatch, legend label, has_precompute)
# CPU bars first then GPU; within each backend: Explicit, Original+V.P., Ours.
SPECS = [
    ("Exp",   "CPU", "#aac8e4", "#1f77b4", "",   "Original (CPU)",         False),
    ("ExpVP", "CPU", "#ffd699", "#ff7f0e", "",   "Original + V.P. (CPU)",  True),
    ("Impl",  "CPU", "#a8d5a2", "#2ca02c", "",   "Ours (CPU)",             True),
    ("Exp",   "GPU", "#aac8e4", "#1f77b4", "//", "Original (GPU)",         False),
    ("ExpVP", "GPU", "#ffd699", "#ff7f0e", "//", "Original + V.P. (GPU)",  True),
    ("Impl",  "GPU", "#a8d5a2", "#2ca02c", "//", "Ours (GPU)",             True),
]


def collect_times() -> tuple[dict, dict]:
    """Walk cached_results/, group times by (axis, scenario) → (form, backend).
    Returns (times, meta)."""
    times: dict = defaultdict(lambda: defaultdict(list))
    meta: dict = {}
    for axis_dir in sorted(SWEEP_ROOT.iterdir()):
        if not axis_dir.is_dir():
            continue
        for scn in sorted(axis_dir.iterdir()):
            cache = scn / "cached_results"
            meta_p = scn / "meta.json"
            if not (cache.exists() and meta_p.exists()):
                continue
            meta[(axis_dir.name, scn.name)] = json.loads(meta_p.read_text())
            for f in cache.iterdir():
                if not f.name.startswith(("results_rank", "gpu_results_rank")):
                    continue
                be = "GPU" if f.name.startswith("gpu_results_rank") else "CPU"
                try:
                    arr = json.loads(f.read_text())
                except json.JSONDecodeError:
                    continue
                for r in arr:
                    t = r.get("times") or []
                    form = FORMULATION_NAME.get(r.get("formulation"), "?")
                    if form in ("Exp", "ExpVP", "Impl") and t:
                        times[(axis_dir.name, scn.name)][(form, be)].append(t[-1])
    return times, meta


def median_or_nan(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def render_panel(ax, keys, prep_times, times, xtick_labels, xlabel):
    x = np.arange(len(keys))
    n = len(SPECS)
    w = 0.88 / n
    offsets = (np.arange(n) - (n - 1) / 2.0) * w

    # The cached `times` arrays record solver-only elapsed time (precompute
    # happens before the iteration loop and isn't included). For V.P. / Ours
    # we stack the precompute (dark) at the bottom so the bar height is the
    # true wall-clock total: solver + precompute. Explicit has no precompute.
    preps_per_scn = np.array([prep_times[f"{k[0]}/{k[1]}"] for k in keys])
    prep_label_used = False
    for off, (form, be, face, edge, hatch, lbl, has_prep) in zip(offsets, SPECS):
        solver = np.array(
            [median_or_nan(times[k].get((form, be), [])) for k in keys])
        if has_prep:
            ax.bar(
                x + off, preps_per_scn, width=w * 0.95,
                color="#222222", edgecolor="black", linewidth=0.4, zorder=3,
                label=("Precompute (CR / Chol / B)" if not prep_label_used else None),
            )
            prep_label_used = True
            ax.bar(
                x + off, solver, bottom=preps_per_scn, width=w * 0.95,
                color=face, edgecolor=edge, linewidth=1.1,
                hatch=hatch, label=lbl, zorder=2,
            )
        else:
            ax.bar(
                x + off, solver, width=w * 0.95,
                color=face, edgecolor=edge, linewidth=1.1,
                hatch=hatch, label=lbl, zorder=2,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (s)")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def main() -> int:
    prep_times = json.loads((ANALYSIS_DIR / "preprocess_times.json").read_text())
    times, meta = collect_times()

    size_keys  = sorted(
        [k for k in times if k[0] == "size"],
        key=lambda k: meta[k]["total_vars"])
    ratio_keys = sorted(
        [k for k in times if k[0] == "ratio" and k[1] != "ratio_1_100"],
        key=lambda k: meta[k]["landmarks"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    render_panel(
        axes[0], size_keys, prep_times, times,
        [k[1].replace("size_", "") for k in size_keys],
        "Size Sweep",
    )
    render_panel(
        axes[1], ratio_keys, prep_times, times,
        [f"1:{int(round(meta[k]['landmarks']/meta[k]['poses']))}" for k in ratio_keys],
        "Constrained : Unconstrained Ratio Sweep",
    )

    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, hi)
        ax.legend(loc="upper left", ncol=2, framealpha=0.95)

    fig.tight_layout()
    out_pdf = ANALYSIS_DIR / "grid3d_sweep_preprocess_breakdown.pdf"
    out_png = ANALYSIS_DIR / "grid3d_sweep_preprocess_breakdown.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=140)
    print(f"wrote {out_pdf.relative_to(REPO)}")
    print(f"wrote {out_png.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
