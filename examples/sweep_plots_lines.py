#!/usr/bin/env python3
"""Alternative plot for sweep_results.json: median + IQR lines, plus a
separate speedup figure.

Outputs (under examples/data/analysis/):
  sweep_lines.pdf    -- median + IQR ribbon for time-to-convergence (log)
                        and iterations-to-convergence (linear), per
                        (formulation, backend). 6 series per panel.
  sweep_speedup.pdf  -- speedup of Ours (Implicit) over each baseline,
                        defined as
                            speedup = median(baseline t_conv) /
                                      median(Ours      t_conv)
                        where t_conv is wall-clock time to reach within 1%
                        of the best Implicit final cost for that scenario.
                        Same backend on both sides of the ratio (CPU/CPU
                        and GPU/GPU). 4 series per panel: 2 baselines × 2
                        backends. Dotted reference at 1×.

Convention matches sweep_plots.py:
  Implicit       = Ours  (green)
  Explicit       = Original  (blue)
  ExplicitVarPro = Orig.+VP  (orange)
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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np


# Publication-style typography (matches sweep_plots.py).
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
    "lines.linewidth": 2.0,
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
BACKEND_LS = {"Cpu": "--", "Gpu": "-"}
BACKEND_MARKER = {"Cpu": "o", "Gpu": "s"}
BACKEND_ALPHA = {"Cpu": 0.55, "Gpu": 1.0}


# ---------------------------------------------------------------------------
# Loading + per-run convergence-point extraction (identical to sweep_plots.py)
# ---------------------------------------------------------------------------

def normalize_backend(raw):
    if not isinstance(raw, str):
        return None
    s = raw.lower()
    if s.startswith("cpu"):
        return "Cpu"
    if s.startswith("gpu"):
        return "Gpu"
    return None


def first_within_tol(times, costs, target):
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


def scenario_sort_key(scn):
    axis = scn["axis"]
    if axis == "size":
        return (scn.get("total_vars", 0),)
    return (scn.get("pose_to_landmark_ratio", 1.0),)


def short_label(scn):
    name = scn["scenario"]
    if scn["axis"] == "size":
        m = re.match(r"size_(\d+)([a-zA-Z]*)$", name)
        if m:
            num, suf = m.group(1), m.group(2)
            return rf"${num}\mathrm{{{suf}}}$" if suf else rf"${num}$"
        return name
    m = re.match(r"ratio_(\d+)_(\d+)$", name)
    if m:
        return rf"${m.group(1)}\!:\!{m.group(2)}$"
    return name


# ---------------------------------------------------------------------------
# Aggregation: per (axis, scenario, backend, formulation) → list of (time,iters)
# for runs that reached the convergence threshold, plus total init count for
# success-rate computation.
# ---------------------------------------------------------------------------

def aggregate(scenarios):
    by_axis = defaultdict(list)
    for scn in scenarios:
        by_axis[scn["axis"]].append(scn)
    ordered = {a: sorted(scns, key=scenario_sort_key) for a, scns in by_axis.items()}

    # converged[(axis, scn_name, backend, form)] = [(t, n), ...]
    converged = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # total_inits[(axis, scn_name, backend, form)] = int
    total_inits = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    targets = defaultdict(dict)

    for axis, scns in ordered.items():
        for scn in scns:
            name = scn["scenario"]
            implicit_finals = [
                r["final_cost"] for r in scn["runs"]
                if r.get("formulation") == "Implicit"
                and isinstance(r.get("final_cost"), (int, float))
                and math.isfinite(r["final_cost"])
            ]
            target = float(np.min(implicit_finals)) if implicit_finals else None
            targets[axis][name] = target

            for r in scn["runs"]:
                backend = normalize_backend(r.get("backend"))
                form = r.get("formulation")
                if backend is None or form not in FORMULATIONS:
                    continue
                total_inits[axis][name][(backend, form)] += 1
                hit = first_within_tol(r.get("times", []), r.get("costs", []), target)
                if hit is not None:
                    converged[axis][name][(backend, form)].append(hit)

    return ordered, converged, total_inits, targets


def quantiles_for_metric(pairs, metric_idx):
    """Return (median, q25, q75, n) for the time (idx 0) or iters (idx 1)
    component of a list of (time, iters) tuples. Returns Nones if empty."""
    vals = [p[metric_idx] for p in pairs if p[metric_idx] is not None]
    if not vals:
        return None, None, None, 0
    arr = np.asarray(vals, dtype=float)
    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    return float(med), float(q25), float(q75), len(vals)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _series_xy_band(scenarios, per_scn, total_per_scn, backend, form, metric_idx):
    """For one (backend, formulation) series, walk scenarios in order and
    return arrays (xs, median, q25, q75, success_rate). NaN-fills missing
    cells so the line still plots through them as a break."""
    xs, med, lo, hi, success = [], [], [], [], []
    for i, scn in enumerate(scenarios):
        name = scn["scenario"]
        pairs = per_scn[name].get((backend, form), [])
        m, q25, q75, n_conv = quantiles_for_metric(pairs, metric_idx)
        n_total = total_per_scn[name].get((backend, form), 0)
        xs.append(i)
        if m is None:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        else:
            med.append(m); lo.append(q25); hi.append(q75)
        success.append((n_conv / n_total) if n_total > 0 else 0.0)
    return (np.asarray(xs), np.asarray(med), np.asarray(lo),
            np.asarray(hi), np.asarray(success))


def _draw_metric_lines(ax, scenarios, per_scn, total_per_scn, metric_idx,
                       log_y):
    x_positions = np.arange(len(scenarios), dtype=float)
    for form in FORMULATIONS:
        for backend in BACKENDS:
            xs, med, lo, hi, _ = _series_xy_band(
                scenarios, per_scn, total_per_scn, backend, form, metric_idx)
            color = FORMULATION_COLORS[form]
            alpha = BACKEND_ALPHA[backend]
            ls = BACKEND_LS[backend]
            mk = BACKEND_MARKER[backend]
            ax.fill_between(x_positions, lo, hi,
                             color=color, alpha=alpha * 0.18,
                             linewidth=0, zorder=1)
            ax.plot(x_positions, med, color=color, alpha=alpha,
                     linestyle=ls, linewidth=1.8, marker=mk,
                     markersize=8, markeredgecolor="white",
                     markeredgewidth=0.6, zorder=3)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlim(-0.4, len(scenarios) - 0.6)
    ax.set_xticks(x_positions)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _format_speedup_yaxis(ax, ticks):
    """Pin the (log) y-axis of a speedup plot to a fixed set of human-
    readable ticks like 1×, 2×, 5×, 10×. The provided `ticks` is the
    multiplicative ladder to label."""
    ymin, ymax = min(ticks), max(ticks)
    ax.set_ylim(ymin / 1.08, ymax * 1.08)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())

    def fmt(v, _pos):
        if v >= 1:
            # 1×, 2×, 5×, 10× — drop trailing ".0"
            return (f"{v:g}" + r"$\times$")
        # Fractional: show 1/N× so the reader can read it as "N× slower".
        return rf"$1/{1.0 / v:g}\times$"

    ax.yaxis.set_major_formatter(FuncFormatter(fmt))


def _draw_speedup_formulation(ax, scenarios, per_scn, total_per_scn):
    """One ratio line per (baseline, backend): median(baseline time) /
    median(Ours time), where Ours = Implicit. Log-y so Nx and 1/Nx are
    symmetric around the 1× reference. Missing-data points (either side
    has no converged runs) are dropped from the line."""
    x_positions = np.arange(len(scenarios), dtype=float)
    baselines = [f for f in FORMULATIONS if f != "Implicit"]
    for baseline in baselines:
        for backend in BACKENDS:
            ratios = []
            for scn in scenarios:
                name = scn["scenario"]
                ours = per_scn[name].get((backend, "Implicit"), [])
                base = per_scn[name].get((backend, baseline), [])
                m_ours, *_ = quantiles_for_metric(ours, 0)
                m_base, *_ = quantiles_for_metric(base, 0)
                if m_ours and m_ours > 0 and m_base and m_base > 0:
                    ratios.append(m_base / m_ours)
                else:
                    ratios.append(np.nan)
            color = FORMULATION_COLORS[baseline]
            alpha = BACKEND_ALPHA[backend]
            ls = BACKEND_LS[backend]
            mk = BACKEND_MARKER[backend]
            ax.plot(x_positions, ratios, color=color, alpha=alpha,
                     linestyle=ls, linewidth=1.8, marker=mk,
                     markersize=8, markeredgecolor="white",
                     markeredgewidth=0.6, zorder=3)
    # Reference line at 1× (no speedup).
    ax.axhline(1.0, color="0.4", linewidth=1.0, linestyle=":", zorder=2)
    ax.set_yscale("log")
    _format_speedup_yaxis(ax, ticks=[1, 2, 3, 5, 8])
    ax.set_xlim(-0.4, len(scenarios) - 0.6)
    ax.set_xticks(x_positions)
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_speedup_backend(ax, scenarios, per_scn, total_per_scn):
    """One ratio line per formulation: median(CPU time) / median(GPU time).
    Same formulation on both sides of the ratio, so the result isolates the
    hardware acceleration effect. Log-y, dotted 1× reference."""
    x_positions = np.arange(len(scenarios), dtype=float)
    for form in FORMULATIONS:
        ratios = []
        for scn in scenarios:
            name = scn["scenario"]
            cpu = per_scn[name].get(("Cpu", form), [])
            gpu = per_scn[name].get(("Gpu", form), [])
            m_cpu, *_ = quantiles_for_metric(cpu, 0)
            m_gpu, *_ = quantiles_for_metric(gpu, 0)
            if m_cpu and m_cpu > 0 and m_gpu and m_gpu > 0:
                ratios.append(m_cpu / m_gpu)
            else:
                ratios.append(np.nan)
        color = FORMULATION_COLORS[form]
        ax.plot(x_positions, ratios, color=color, alpha=1.0,
                 linestyle="-", linewidth=1.8, marker="D",
                 markersize=8, markeredgecolor="white",
                 markeredgewidth=0.6, zorder=3)
    ax.axhline(1.0, color="0.4", linewidth=1.0, linestyle=":", zorder=2)
    ax.set_yscale("log")
    # Backend ratios on PGO+SfM tend to be in [~1/3, ~1.5]: GPU sometimes
    # wins, sometimes loses, depending on formulation.
    _format_speedup_yaxis(ax, ticks=[1/3, 1/2, 1, 1.5])
    ax.set_xlim(-0.4, len(scenarios) - 0.6)
    ax.set_xticks(x_positions)
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


AXIS_ORDER = ["size", "ratio"]
AXIS_TITLE = {"size": "Size Sweep", "ratio": "Ratio Sweep"}
AXIS_XLABEL = {
    "size": "Number of Variables",
    "ratio": "Ratio of Constrained to Unconstrained Variables",
}


def plot_main(ordered, converged, totals, out_path):
    """Two-row figure: time-to-convergence (log y) and
    iterations-to-convergence (linear y), per (formulation, backend).
    6 series per panel, IQR ribbon, median line + scenario markers."""
    axes_present = [a for a in AXIS_ORDER if ordered.get(a)]

    fig, axarr = plt.subplots(
        2, len(axes_present),
        figsize=(34.0, 14.0), dpi=300, sharey="row",
        gridspec_kw=dict(hspace=0.20, wspace=0.05),
    )
    if len(axes_present) == 1:
        axarr = axarr.reshape(2, 1)

    for col_idx, axis in enumerate(axes_present):
        scenarios = ordered[axis]
        per_scn = converged[axis]
        total_per_scn = totals[axis]
        labels = [short_label(s) for s in scenarios]
        x_positions = np.arange(len(scenarios), dtype=float)

        ax_time = axarr[0, col_idx]
        ax_iters = axarr[1, col_idx]

        _draw_metric_lines(ax_time, scenarios, per_scn, total_per_scn,
                            metric_idx=0, log_y=True)
        _draw_metric_lines(ax_iters, scenarios, per_scn, total_per_scn,
                            metric_idx=1, log_y=False)

        ax_time.set_title(AXIS_TITLE[axis])
        ax_time.tick_params(labelbottom=False)
        ax_iters.set_xticks(x_positions)
        ax_iters.set_xticklabels(labels)
        ax_iters.set_xlabel(AXIS_XLABEL[axis])

    axarr[0, 0].set_ylabel("Time until convergence (s)")
    axarr[1, 0].set_ylabel("Iterations until convergence")

    # Legend describes the 6 series per panel.
    handles = []
    for form in FORMULATIONS:
        for backend in BACKENDS:
            handles.append(plt.Line2D(
                [0], [0],
                color=FORMULATION_COLORS[form],
                alpha=BACKEND_ALPHA[backend],
                linestyle=BACKEND_LS[backend],
                marker=BACKEND_MARKER[backend], markersize=8,
                markeredgecolor="white", markeredgewidth=0.6,
                linewidth=1.8,
                label=f"{backend.upper()} — {FORMULATION_LABELS[form]}",
            ))
    fig.legend(handles=handles, ncol=3, frameon=False,
                loc="lower center", bbox_to_anchor=(0.5, -0.03),
                columnspacing=2.2, handletextpad=0.7)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_speedup(ordered, converged, totals, out_path):
    """Dedicated speedup figure with two rows.

    Row 1 — formulation comparison (backend fixed):
        speedup = median(baseline t_conv) / median(Ours t_conv)
        CPU/CPU and GPU/GPU, so the ratio isolates the formulation effect.
        4 series: 2 baselines × 2 backends.

    Row 2 — backend comparison (formulation fixed):
        speedup = median(CPU t_conv) / median(GPU t_conv)
        Same formulation on both sides, so the ratio isolates the
        GPU-acceleration effect. 3 series, one per formulation (incl. Ours).

    Reference dotted line at 1× on both rows.
    """
    axes_present = [a for a in AXIS_ORDER if ordered.get(a)]

    fig, axarr = plt.subplots(
        2, len(axes_present),
        figsize=(34.0, 13.0), dpi=300, sharey="row",
        gridspec_kw=dict(hspace=0.30, wspace=0.05),
    )
    if len(axes_present) == 1:
        axarr = axarr.reshape(2, 1)

    for col_idx, axis in enumerate(axes_present):
        scenarios = ordered[axis]
        per_scn = converged[axis]
        total_per_scn = totals[axis]
        labels = [short_label(s) for s in scenarios]
        x_positions = np.arange(len(scenarios), dtype=float)

        ax_form = axarr[0, col_idx]
        ax_back = axarr[1, col_idx]

        _draw_speedup_formulation(ax_form, scenarios, per_scn, total_per_scn)
        _draw_speedup_backend    (ax_back, scenarios, per_scn, total_per_scn)

        ax_form.set_title(AXIS_TITLE[axis])
        ax_form.tick_params(labelbottom=False)
        ax_back.set_xticks(x_positions)
        ax_back.set_xticklabels(labels)
        ax_back.set_xlabel(AXIS_XLABEL[axis])

    axarr[0, 0].set_ylabel(r"Formulation speedup ($\times$)")
    axarr[1, 0].set_ylabel(r"GPU vs CPU speedup ($\times$)")

    # Row 1 legend: 4 ratio series (2 baselines × 2 backends) + 1× ref.
    handles_form = []
    for baseline in [f for f in FORMULATIONS if f != "Implicit"]:
        for backend in BACKENDS:
            handles_form.append(plt.Line2D(
                [0], [0],
                color=FORMULATION_COLORS[baseline],
                alpha=BACKEND_ALPHA[backend],
                linestyle=BACKEND_LS[backend],
                marker=BACKEND_MARKER[backend], markersize=8,
                markeredgecolor="white", markeredgewidth=0.6,
                linewidth=1.8,
                label=f"{backend.upper()}: Ours vs {FORMULATION_LABELS[baseline]}",
            ))
    handles_form.append(plt.Line2D(
        [0], [0], color="0.4", linewidth=1.0, linestyle=":",
        label=r"$1\times$",
    ))
    # Place row 1 legend in the gap between row 1 and row 2.
    leg_form = fig.legend(handles=handles_form, ncol=5, frameon=False,
                           loc="center", bbox_to_anchor=(0.5, 0.5),
                           columnspacing=2.2, handletextpad=0.7)

    # Row 2 legend: 3 series — one per formulation (incl. Ours).
    handles_back = []
    for form in FORMULATIONS:
        handles_back.append(plt.Line2D(
            [0], [0],
            color=FORMULATION_COLORS[form], alpha=1.0,
            linestyle="-", linewidth=1.8, marker="D",
            markersize=8, markeredgecolor="white", markeredgewidth=0.6,
            label=f"GPU vs CPU — {FORMULATION_LABELS[form]}",
        ))
    handles_back.append(plt.Line2D(
        [0], [0], color="0.4", linewidth=1.0, linestyle=":",
        label=r"$1\times$",
    ))
    fig.legend(handles=handles_back, ncol=4, frameon=False,
                loc="lower center", bbox_to_anchor=(0.5, -0.06),
                columnspacing=2.2, handletextpad=0.7)
    # Re-add the row-1 legend (fig.legend on a second call replaces the first
    # unless we add it back manually as an artist).
    fig.add_artist(leg_form)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if not INPUT_JSON.exists():
        print(f"{INPUT_JSON} not found. Run `python examples/sfm_sweep.py aggregate` first.")
        return 1
    blob = json.loads(INPUT_JSON.read_text())
    scenarios = blob.get("scenarios", [])
    if not scenarios:
        print("No scenarios in sweep_results.json")
        return 1

    ordered, converged, totals, targets = aggregate(scenarios)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_main = OUTPUT_DIR / "sweep_lines.pdf"
    plot_main(ordered, converged, totals, out_main)
    print(f"wrote {out_main.relative_to(REPO)}")

    out_speedup = OUTPUT_DIR / "sweep_speedup.pdf"
    plot_speedup(ordered, converged, totals, out_speedup)
    print(f"wrote {out_speedup.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
