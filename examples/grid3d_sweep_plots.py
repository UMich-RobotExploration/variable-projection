#!/usr/bin/env python3
"""Line-plot summaries of the grid3d sweep.

Produces four two-panel figures (size sweep + ratio sweep) under
examples/data/analysis/:

  grid3d_sweep_all_runs.{pdf,png}      — solve time per formulation/backend
  grid3d_sweep_iterations.{pdf,png}    — median iterations to converge
  grid3d_sweep_impl_speedup.{pdf,png}  — speedup of Ours vs each baseline
  grid3d_sweep_gpu_speedup.{pdf,png}   — GPU/CPU speedup per formulation

Both axes are numeric (log-scaled): size by total variable count, ratio by
landmarks/poses. The ratio panel intentionally drops the `ratio_1_100`
scenario.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size":       13,
    "axes.labelsize":  14,
    "axes.titlesize":  14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})


REPO = Path(__file__).resolve().parent.parent
SWEEP_ROOT = REPO / "examples" / "data" / "grid3d_sweep"
ANALYSIS_DIR = REPO / "examples" / "data" / "analysis"

FORMS = ["Explicit", "ExpVP", "Impl"]
BACKENDS = ["CPU", "GPU"]
FORMULATION_NAME = {0: "Explicit", 1: "ExpVP", 2: "Impl"}

COLORS = {"Explicit": "#1f77b4", "ExpVP": "#ff7f0e", "Impl": "#2ca02c"}
LABEL  = {"Explicit": "Original", "ExpVP": "Original + V.P.", "Impl": "Ours"}
LS = {"CPU": "--", "GPU": "-"}
MK = {"CPU": "o",  "GPU": "s"}

# Skip this scenario on the ratio panel — caller asked for it gone.
RATIO_SKIP = {"ratio_1_100"}


def collect():
    """Walk cached_results/, return per-scenario time and iter samples + meta."""
    times: dict = defaultdict(lambda: defaultdict(list))
    iters: dict = defaultdict(lambda: defaultdict(list))
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
                backend = "GPU" if f.name.startswith("gpu_results_rank") else "CPU"
                try:
                    arr = json.loads(f.read_text())
                except json.JSONDecodeError:
                    continue
                for r in arr:
                    t = r.get("times") or []
                    c = r.get("costs") or []
                    form = FORMULATION_NAME.get(r.get("formulation"), "?")
                    if not t or form == "?":
                        continue
                    key = (axis_dir.name, scn.name)
                    times[key][(form, backend)].append(t[-1])
                    iters[key][(form, backend)].append(len(c))
    return times, iters, meta


def med(metric, key, form, backend):
    vs = metric[key].get((form, backend), [])
    return statistics.median(vs) if vs else float("nan")


def sync_ylim(axes):
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)


def panel_axes(times, meta):
    """Return per-panel (keys, xs, xtick_labels, xlabel, log_x) tuples.

    Both axes are numeric: size uses the actual total-variable count so the
    sparse high end (10k / 15k / 22k) doesn't get the same spacing as the
    dense low end (2k / 3k / 5k); ratio uses landmarks/poses. Both are
    log-scaled.
    """
    size_keys  = sorted(
        [k for k in times if k[0] == "size"],
        key=lambda k: meta[k]["total_vars"])
    ratio_keys = sorted(
        [k for k in times if k[0] == "ratio" and k[1] not in RATIO_SKIP],
        key=lambda k: meta[k]["landmarks"])

    xs_size = [meta[k]["total_vars"] for k in size_keys]
    xs_ratio = [meta[k]["landmarks"] / max(meta[k]["poses"], 1) for k in ratio_keys]
    size_labels  = [k[1].replace("size_", "") for k in size_keys]
    ratio_labels = [f"1:{int(round(x))}" for x in xs_ratio]
    return [
        (size_keys,  xs_size,  size_labels,  "Size Sweep",                              True, 0),
        (ratio_keys, xs_ratio, ratio_labels, "Constrained : Unconstrained Ratio Sweep", True, 0),
    ]


def _format_xaxis(ax, xs, xtick_labels, rotate=0):
    """Set explicit tick locations + labels and disable matplotlib's automatic
    minor tick labels.

    The ratio panel uses real numeric x values (1, 2, 3, 5, 10, 20, 50) on a
    *linear* scale, which packs the low end into a 4-unit window. A 45°
    rotation plus a small font size keeps the labels readable.
    """
    import matplotlib.ticker as mt
    ax.set_xticks(xs)
    if rotate:
        ha = "center" if rotate == 90 else "right"
        ax.set_xticklabels(xtick_labels, rotation=rotate, ha=ha,
                            rotation_mode="anchor", fontsize=10)
    else:
        ax.set_xticklabels(xtick_labels)
    ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.xaxis.set_minor_formatter(mt.NullFormatter())


def plot_all_runs(times, meta, out_stem):
    panels = panel_axes(times, meta)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, (keys, xs, xtick_labels, xlabel, log_x, rot) in zip(axes, panels):
        for form in FORMS:
            for be in BACKENDS:
                sx, sy = [], []
                for x, k in zip(xs, keys):
                    for v in times[k].get((form, be), []):
                        sx.append(x); sy.append(v)
                ax.scatter(sx, sy, color=COLORS[form], marker=MK[be], s=18,
                            alpha=0.18, edgecolors="none", zorder=1)
                ys = [med(times, k, form, be) for k in keys]
                ax.plot(xs, ys, color=COLORS[form], linestyle=LS[be], marker=MK[be],
                         markersize=7, linewidth=2.0, zorder=3,
                         label=f"{LABEL[form]} ({be})")
        if log_x:
            ax.set_xscale("log")
        _format_xaxis(ax, xs, xtick_labels, rotate=rot)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Solve time (s)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    sync_ylim(axes)
    for ax in axes:
        ax.legend(ncol=2, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(ANALYSIS_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_iterations(iters, meta, out_stem):
    panels = panel_axes(iters, meta)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, (keys, xs, xtick_labels, xlabel, log_x, rot) in zip(axes, panels):
        for form in FORMS:
            for be in BACKENDS:
                ys = [med(iters, k, form, be) for k in keys]
                ax.plot(xs, ys, color=COLORS[form], linestyle=LS[be], marker=MK[be],
                         markersize=7, linewidth=2.0, label=f"{LABEL[form]} ({be})")
        if log_x:
            ax.set_xscale("log")
        _format_xaxis(ax, xs, xtick_labels, rotate=rot)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Iterations to converge")
        ax.set_ylim(0, 50)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(ANALYSIS_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_impl_speedup(times, meta, out_stem):
    panels = panel_axes(times, meta)
    comps = [("Explicit", "vs Original",        "#9467bd"),
             ("ExpVP",    "vs Original + V.P.", "#17becf")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, (keys, xs, xtick_labels, xlabel, log_x, rot) in zip(axes, panels):
        for baseline, lab, color in comps:
            for be in BACKENDS:
                ys = [med(times, k, baseline, be) / med(times, k, "Impl", be)
                       for k in keys]
                ax.plot(xs, ys, color=color, linestyle=LS[be], marker=MK[be],
                         markersize=7, linewidth=2.0, label=f"{lab} ({be})")
        ax.axhline(1.0, color="grey", linewidth=1, linestyle=":")
        if log_x:
            ax.set_xscale("log")
        _format_xaxis(ax, xs, xtick_labels, rotate=rot)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Speedup of Ours  (baseline time / Ours time)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    sync_ylim(axes)
    for ax in axes:
        ax.legend(ncol=2, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(ANALYSIS_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=140)
    plt.close(fig)


def plot_gpu_speedup(times, meta, out_stem):
    panels = panel_axes(times, meta)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, (keys, xs, xtick_labels, xlabel, log_x, rot) in zip(axes, panels):
        for form in FORMS:
            ys = [med(times, k, form, "CPU") / med(times, k, form, "GPU") for k in keys]
            ax.plot(xs, ys, color=COLORS[form], marker="o",
                     markersize=7, linewidth=2.0, label=LABEL[form])
        ax.axhline(1.0, color="grey", linewidth=1, linestyle=":")
        if log_x:
            ax.set_xscale("log")
        _format_xaxis(ax, xs, xtick_labels, rotate=rot)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("GPU speedup  (CPU time / GPU time)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    sync_ylim(axes)
    for ax in axes:
        ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(ANALYSIS_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=140)
    plt.close(fig)


def main() -> int:
    times, iters, meta = collect()
    plot_all_runs(times, meta, "grid3d_sweep_all_runs")
    plot_iterations(iters, meta, "grid3d_sweep_iterations")
    plot_impl_speedup(times, meta, "grid3d_sweep_impl_speedup")
    plot_gpu_speedup(times, meta, "grid3d_sweep_gpu_speedup")
    for stem in ("grid3d_sweep_all_runs",
                 "grid3d_sweep_iterations",
                 "grid3d_sweep_impl_speedup",
                 "grid3d_sweep_gpu_speedup"):
        print(f"wrote examples/data/analysis/{stem}.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
