#!/usr/bin/env python3
"""Box-and-whisker plots of the SfM sweep results.

For each scenario we compute the *time* and *iteration index* at which each
solver run first reaches within CONVERGENCE_TOL (=1%) of the median Implicit
final cost for that scenario. The 5 random initialisations form the box-plot
distribution at each sweep point.

Outputs (under examples/data/analysis/):
  sweep_size_time.{png,pdf}    - time to 1% of Implicit min,  size sweep
  sweep_size_iters.{png,pdf}   - iterations to 1% of Implicit min, size sweep
  sweep_ratio_time.{png,pdf}   - time to 1% of Implicit min,  ratio sweep
  sweep_ratio_iters.{png,pdf}  - iterations to 1% of Implicit min, ratio sweep

CPU and GPU runs share a plot per sweep, distinguished by box face shade.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


# Publication-style typography: STIX serif math, embeddable fonts, IEEE-ish
# default sizes. Touched once at import.
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 26,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.6,
})


REPO = Path(__file__).resolve().parent.parent
INPUT_JSON = REPO / "examples" / "data" / "analysis" / "sweep_results.json"
OUTPUT_DIR = REPO / "examples" / "data" / "analysis"

CONVERGENCE_TOL = 0.01

FORMULATIONS = ["Implicit", "Explicit", "ExplicitVarPro"]
FORMULATION_LABELS = {
    "Implicit": "Ours",
    "Explicit": "Original",
    "ExplicitVarPro": "Orig.+VP",
}
FORMULATION_COLORS = {
    "Implicit": "#2ca02c",          # green
    "Explicit": "#1f77b4",          # blue
    "ExplicitVarPro": "#ff7f0e",    # orange
}
BACKENDS = ["Cpu", "Gpu"]
BACKEND_ALPHA = {"Cpu": 0.35, "Gpu": 0.9}


# ---------------------------------------------------------------------------
# Loading + per-run convergence-point extraction
# ---------------------------------------------------------------------------

def normalize_backend(raw: str | None) -> str | None:
    if not isinstance(raw, str):
        return None
    s = raw.lower()
    if s.startswith("cpu"):
        return "Cpu"
    if s.startswith("gpu"):
        return "Gpu"
    return None


def first_within_tol(times, costs, target):
    """Return (time, iteration_index) when cost first drops within
    (1+CONVERGENCE_TOL) * target. Skip non-finite cost entries. None if
    the run never reaches the threshold."""
    if target is None or not math.isfinite(target) or target <= 0.0:
        return None
    threshold = (1.0 + CONVERGENCE_TOL) * target
    for i, c in enumerate(costs):
        if not isinstance(c, (int, float)) or not math.isfinite(c):
            continue
        if c <= threshold:
            t = times[i] if i < len(times) else None
            return (float(t) if isinstance(t, (int, float)) else None, int(i))
    return None


def scenario_sort_key(scn: dict) -> tuple:
    axis = scn["axis"]
    name = scn["scenario"]
    if axis == "size":
        return (scn.get("total_vars", 0),)
    # ratio: sort by landmark/pose ratio ascending (1:1 first, 1:100 last)
    return (scn.get("pose_to_landmark_ratio", 1.0),)


def short_label(scn: dict) -> str:
    """Tex-style scenario label for x-tick: e.g. '$22\\mathrm{k}$' or '$1\\!:\\!100$'."""
    name = scn["scenario"]
    if scn["axis"] == "size":
        m = re.match(r"size_(\d+)([a-zA-Z]*)$", name)
        if m:
            num, suf = m.group(1), m.group(2)
            if suf:
                return rf"${num}\mathrm{{{suf}}}$"
            return rf"${num}$"
        return name
    m = re.match(r"ratio_(\d+)_(\d+)$", name)
    if m:
        return rf"${m.group(1)}\!:\!{m.group(2)}$"
    return name


def aggregate(scenarios: list[dict]):
    """Return:
      ordered: {axis: [scenario_dict, ...] sorted}
      per_run: {axis: {scenario_name: {(backend, formulation): [(time, iters), ...]}}}
      targets: {axis: {scenario_name: target_cost or None}}
    """
    by_axis: dict[str, list[dict]] = defaultdict(list)
    for scn in scenarios:
        by_axis[scn["axis"]].append(scn)
    ordered = {a: sorted(scns, key=scenario_sort_key) for a, scns in by_axis.items()}

    per_run: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    targets: dict = defaultdict(dict)

    for axis, scns in ordered.items():
        for scn in scns:
            name = scn["scenario"]
            implicit_finals = [
                r["final_cost"] for r in scn["runs"]
                if r.get("formulation") == "Implicit"
                and isinstance(r.get("final_cost"), (int, float))
                and math.isfinite(r["final_cost"])
            ]
            # Use the best (minimum) Implicit final cost as the gold standard,
            # not the median: this ensures every Implicit run that actually
            # reached the minimum is counted, and missing boxes for other
            # (backend, formulation) combos correctly signal "never reached
            # the best minimum".
            target = float(np.min(implicit_finals)) if implicit_finals else None
            targets[axis][name] = target

            for r in scn["runs"]:
                backend = normalize_backend(r.get("backend"))
                form = r.get("formulation")
                if backend is None or form not in FORMULATIONS:
                    continue
                hit = first_within_tol(r.get("times", []), r.get("costs", []), target)
                if hit is None:
                    continue
                per_run[axis][name][(backend, form)].append(hit)

    return ordered, per_run, targets


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_metric(
    ax: plt.Axes,
    metric: str,
    scenarios: list[dict],
    per_scn: dict,
    ylim: tuple[float, float] | None,
) -> list[tuple[int, float, str]]:
    """Render the violin cluster for a single metric on the given axis.

    Returns the list of (scenario_index, x_offset, color) tuples for combos
    that produced zero converged inits — to be marked by the caller above the
    axis frame.
    """
    metric_idx = 0 if metric == "time" else 1

    names = [s["scenario"] for s in scenarios]
    x_positions = np.arange(len(scenarios), dtype=float)

    group_width = 0.84
    n_boxes = len(FORMULATIONS) * len(BACKENDS)
    box_width = group_width / n_boxes

    missing: list[tuple[int, float, str]] = []
    for f_idx, form in enumerate(FORMULATIONS):
        for b_idx, backend in enumerate(BACKENDS):
            slot = f_idx * len(BACKENDS) + b_idx
            offset = -group_width / 2 + (slot + 0.5) * box_width

            data, positions = [], []
            for i, name in enumerate(names):
                pairs = per_scn[name].get((backend, form), [])
                vals = [p[metric_idx] for p in pairs if p[metric_idx] is not None]
                if vals:
                    data.append(vals)
                    positions.append(x_positions[i] + offset)
                else:
                    missing.append((i, offset, FORMULATION_COLORS[form]))
            if not data:
                continue
            face = FORMULATION_COLORS[form]
            alpha = BACKEND_ALPHA[backend]
            vp = ax.violinplot(
                data,
                positions=positions,
                widths=box_width * 0.95,
                showmeans=False, showmedians=False, showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(face)
                body.set_edgecolor(face)
                body.set_alpha(alpha)
                body.set_linewidth(0.6)
            for series, pos in zip(data, positions):
                ax.scatter(
                    np.full(len(series), pos), series,
                    s=10, color=face, alpha=min(1.0, alpha + 0.2),
                    edgecolor="none", zorder=4,
                )
                med = float(np.median(series))
                ax.plot(
                    [pos - box_width * 0.40, pos + box_width * 0.40],
                    [med, med], color="black", linewidth=1.2, zorder=5,
                )

    if metric == "time":
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks(x_positions)
    ax.set_xlim(-0.6, len(scenarios) - 0.4)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return missing


def plot_combined_figure(
    ordered: dict,
    per_run: dict,
    y_ranges: dict,
    out_base: Path,
) -> None:
    """Single figure spanning both sweeps:
        [Time | Size]   [Time | Ratio]
        [Iters | Size]  [Iters | Ratio]
    Each column has its own scenario labels; the two rows share y-extents
    (set by compute_global_y_ranges) so columns are directly comparable.
    One legend serves all four panels.
    """
    # Column order: size on the left, ratio on the right (natural reading order
    # for "as the problem grows, then as the constrained/unconstrained mix
    # shifts").
    AXIS_ORDER = ["size", "ratio"]
    AXIS_TITLE = {"size": "Size Sweep", "ratio": "Ratio Sweep"}
    AXIS_XLABEL = {
        "size": "Number of Variables",
        "ratio": "Ratio of Constrained to Unconstrained Variables",
    }
    axes_present = [a for a in AXIS_ORDER if ordered.get(a)]

    fig, axarr = plt.subplots(
        2, len(axes_present),
        figsize=(34.0, 14.0), dpi=300, sharey="row",
        gridspec_kw=dict(hspace=0.22, wspace=0.05),
    )
    if len(axes_present) == 1:
        axarr = axarr.reshape(2, 1)

    for col_idx, axis in enumerate(axes_present):
        scenarios = ordered[axis]
        per_scn = per_run[axis]
        labels = [short_label(s) for s in scenarios]
        x_positions = np.arange(len(scenarios), dtype=float)

        ax_time = axarr[0, col_idx]
        ax_iters = axarr[1, col_idx]

        missing_time = _draw_metric(ax_time, "time", scenarios, per_scn, y_ranges.get("time"))
        missing_iters = _draw_metric(ax_iters, "iters", scenarios, per_scn, y_ranges.get("iters"))

        for ax, missing in ((ax_time, missing_time), (ax_iters, missing_iters)):
            if not missing:
                continue
            y_top = ax.get_ylim()[1]
            for i, off, color in missing:
                ax.plot(
                    x_positions[i] + off, y_top,
                    marker="x", markersize=8, markeredgewidth=1.6,
                    color=color, clip_on=False, zorder=4,
                )

        # Per-column titles, x-labels, x-tick labels.
        ax_time.set_title(AXIS_TITLE[axis])
        ax_time.tick_params(labelbottom=False)
        ax_iters.set_xticks(x_positions)
        ax_iters.set_xticklabels(labels)
        ax_iters.set_xlabel(AXIS_XLABEL[axis])

    # Y-axis labels only on the leftmost column.
    axarr[0, 0].set_ylabel("Time until convergence (s)")
    axarr[1, 0].set_ylabel("Iterations until convergence")

    # Single legend above the figure.
    handles = []
    for form in FORMULATIONS:
        for backend in BACKENDS:
            handles.append(Patch(
                facecolor=FORMULATION_COLORS[form],
                alpha=BACKEND_ALPHA[backend],
                edgecolor=FORMULATION_COLORS[form],
                label=f"{backend.upper()} — {FORMULATION_LABELS[form]}",
            ))
    handles.append(plt.Line2D(
        [0], [0], marker="x", color="0.4", linestyle="none",
        markersize=8, markeredgewidth=1.6, label="Not Converged",
    ))
    fig.legend(
        handles=handles, ncol=4, frameon=False,
        loc="lower center", bbox_to_anchor=(0.5, -0.05),
        columnspacing=2.2, handletextpad=0.7,
    )

    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def compute_global_y_ranges(per_run: dict) -> dict:
    """Compute shared y-axis bounds for time (log) and iters (linear) across
    every scenario in every axis, so the size-sweep and ratio-sweep figures
    use the same y-extent and can be compared side-by-side."""
    all_t, all_i = [], []
    for axis, scn_map in per_run.items():
        for name, combos in scn_map.items():
            for (_, _), pairs in combos.items():
                for (t, n) in pairs:
                    if t is not None and math.isfinite(t) and t > 0:
                        all_t.append(t)
                    if n is not None and math.isfinite(n):
                        all_i.append(n)
    out = {}
    if all_t:
        log_pad = 0.08
        out["time"] = (
            10 ** (math.log10(min(all_t)) - log_pad),
            10 ** (math.log10(max(all_t)) + log_pad),
        )
    if all_i:
        i_max = max(all_i)
        out["iters"] = (0.0, i_max * 1.10)
    return out


def main() -> int:
    if not INPUT_JSON.exists():
        print(f"{INPUT_JSON} not found. Run `python examples/sfm_sweep.py aggregate` first.")
        return 1
    blob = json.loads(INPUT_JSON.read_text())
    scenarios = blob.get("scenarios", [])
    if not scenarios:
        print("No scenarios in sweep_results.json")
        return 1

    ordered, per_run, targets = aggregate(scenarios)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_ranges = compute_global_y_ranges(per_run)

    for axis, scns in ordered.items():
        if not scns:
            continue
        print(f"--- {axis} sweep ---")
        for s in scns:
            t = targets[axis][s["scenario"]]
            print(f"  {s['scenario']}: Implicit target cost ≈ {t!r}")

    out = OUTPUT_DIR / "sweep"
    plot_combined_figure(ordered, per_run, y_ranges, out)
    print(f"wrote {out.with_suffix('.pdf').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
