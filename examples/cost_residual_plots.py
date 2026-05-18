#!/usr/bin/env python3
"""Cost-residual convergence plots.

For each (scenario, formulation, backend, init) we plot the trajectory of
    f(x_t) - f*
where f* is the *best* cost achieved across all formulations, backends and
inits on that scenario. Y-axis is log scale (with a small floor to keep
log(0) at bay on perfectly-converged tails).

Two views per scenario:
  - vs iteration index
  - vs wall-clock time (seconds since the solver's t=0)

Outputs (under examples/data/analysis/):
  cost_residual_sweep_size.pdf      rows = size scenarios
  cost_residual_sweep_ratio.pdf     rows = ratio scenarios
  cost_residual_pgo.pdf             rows = PGO datasets
  cost_residual_raslam.pdf          rows = RA-SLAM datasets
  cost_residual_sfm.pdf             rows = SfM datasets
  cost_residual_snl.pdf             rows = SNL datasets

Convention matches sweep_plots.py:
  Implicit       = Ours       (green)
  Explicit       = Original   (blue)
  ExplicitVarPro = Orig.+VP   (orange)
  CPU            = dashed line, lower alpha
  GPU            = solid line
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
import numpy as np


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data"
ANALYSIS = DATA / "analysis"


# Formulation enum used in the per-dataset JSONs (int) and the sweep JSON
# (already stringified).
FORM_INT_TO_STR = {0: "Explicit", 1: "ExplicitVarPro", 2: "Implicit"}
FORMULATIONS = ["Implicit", "Explicit", "ExplicitVarPro"]
FORM_LABELS = {
    "Implicit": "Ours",
    "Explicit": "Original",
    "ExplicitVarPro": "Orig.+VP",
}
FORM_COLORS = {
    "Implicit": "#2ca02c",        # green
    "Explicit": "#1f77b4",        # blue
    "ExplicitVarPro": "#ff7f0e",  # orange
}
BACKENDS = ["Cpu", "Gpu"]
BACKEND_LS = {"Cpu": "--", "Gpu": "-"}
BACKEND_ALPHA = {"Cpu": 0.35, "Gpu": 0.65}

# Floor on (cost − cost_min). Below this we clip — keeps log plots usable
# on scenarios where some runs hit machine precision (e.g., SNL on tiny
# datasets converges to ~1e-14 and zero residuals would break log y).
FLOOR = 1e-10


plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.0,
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _normalize_backend(raw):
    if not isinstance(raw, str):
        return None
    s = raw.lower()
    if s.startswith("cpu"): return "Cpu"
    if s.startswith("gpu"): return "Gpu"
    return None


def _formulation_name(raw):
    if isinstance(raw, str):
        return raw
    try:
        return FORM_INT_TO_STR[int(raw)]
    except (KeyError, ValueError, TypeError):
        return None


def _coerce_runs(raw_runs):
    """Coerce a list of raw run dicts into the (formulation, backend,
    costs, times) form used by the plotter. Drops entries we can't classify."""
    out = []
    for r in raw_runs:
        form = _formulation_name(r.get("formulation"))
        backend = _normalize_backend(r.get("backend"))
        costs = r.get("costs") or []
        times = r.get("times") or []
        if form is None or backend is None or not costs:
            continue
        out.append({
            "formulation": form,
            "backend": backend,
            "costs": costs,
            "times": times,
        })
    return out


def load_standard_dataset(ds_dir: Path):
    """Read results.json + gpu_results.json from a single dataset directory."""
    raw = []
    for fname in ("results.json", "gpu_results.json"):
        p = ds_dir / fname
        if not p.exists():
            continue
        try:
            arr = json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"  warning: {p} unreadable, skipping")
            continue
        if isinstance(arr, list):
            raw.extend(arr)
    return _coerce_runs(raw)


def load_sweep(path=None):
    """Return {axis: [(scenario_name, runs), ...]} from a sweep_results.json
    file. Defaults to the SfM-sweep aggregate; pass a different path to plot
    e.g. the Grid3D sweep."""
    if path is None:
        path = ANALYSIS / "sweep_results.json"
    sweep_path = Path(path)
    if not sweep_path.exists():
        return {}
    blob = json.loads(sweep_path.read_text())
    by_axis = defaultdict(list)
    for s in blob.get("scenarios", []):
        runs = _coerce_runs(s.get("runs", []))
        if runs:
            by_axis[s["axis"]].append((s["scenario"], runs))
    return dict(by_axis)


# ---------------------------------------------------------------------------
# cost-min and trajectory utilities
# ---------------------------------------------------------------------------

def cost_min(runs):
    """Best finite cost achieved across all runs in `runs`."""
    best = math.inf
    for r in runs:
        for c in r["costs"]:
            if isinstance(c, (int, float)) and math.isfinite(c) and c < best:
                best = c
    return best if math.isfinite(best) else None


def _residual_trajectory(run, c_min):
    """Return (xs_iters, xs_time, ys_residual) with NaN where the cost was
    non-finite or below the floor."""
    costs = run["costs"]
    times = run["times"]
    n = min(len(costs), len(times)) if times else len(costs)
    if n == 0:
        return None
    xs_iters = np.arange(n, dtype=float)
    xs_time = (
        np.asarray(times[:n], dtype=float)
        if times else np.full(n, np.nan)
    )
    ys = np.full(n, np.nan)
    for i, c in enumerate(costs[:n]):
        if isinstance(c, (int, float)) and math.isfinite(c):
            ys[i] = max(c - c_min, FLOOR)
    return xs_iters, xs_time, ys


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_panel(ax, runs, c_min, x_kind):
    """Draw every individual trajectory as a thin low-alpha line."""
    plotted = False
    for r in runs:
        traj = _residual_trajectory(r, c_min)
        if traj is None:
            continue
        xs_iters, xs_time, ys = traj
        xs = xs_iters if x_kind == "iters" else xs_time
        # Drop trailing NaNs so the line ends cleanly.
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not mask.any():
            continue
        plotted = True
        ax.plot(
            xs[mask], ys[mask],
            color=FORM_COLORS[r["formulation"]],
            linestyle=BACKEND_LS[r["backend"]],
            alpha=BACKEND_ALPHA[r["backend"]],
            linewidth=0.9,
            zorder=2,
        )
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(FLOOR * 0.5, 1e-12))
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return plotted


def plot_category(title, datasets, out_path):
    """Two-column figure: vs-iters (left) and vs-time (right), one row per
    dataset/scenario in `datasets` (list of (name, runs))."""
    n = len(datasets)
    if n == 0:
        print(f"  skipping {out_path.name}: no data")
        return

    # Row height tuned so the y-axis is readable even with many datasets.
    fig_h = max(2.4, 1.8 * n + 1.2)
    fig, axarr = plt.subplots(
        n, 2,
        figsize=(13.0, fig_h),
        dpi=180,
        sharey="row",
        gridspec_kw=dict(hspace=0.55, wspace=0.04),
    )
    if n == 1:
        axarr = axarr.reshape(1, 2)

    for i, (name, runs) in enumerate(datasets):
        c_min = cost_min(runs)
        ax_iter = axarr[i, 0]
        ax_time = axarr[i, 1]
        if c_min is None:
            ax_iter.text(0.5, 0.5, "no data", ha="center", va="center")
            ax_iter.set_axis_off()
            ax_time.set_axis_off()
            continue
        _draw_panel(ax_iter, runs, c_min, "iters")
        _draw_panel(ax_time, runs, c_min, "time")
        ax_iter.set_ylabel(f"{name}\n$f - f^*$", labelpad=4)
        if i == n - 1:
            ax_iter.set_xlabel("iterations")
            ax_time.set_xlabel("time (s)")
        # f* annotation in the corner so the reader knows the reference.
        ax_iter.text(
            0.99, 0.98, f"$f^*$ = {c_min:.4g}",
            transform=ax_iter.transAxes,
            ha="right", va="top",
            fontsize=8, color="0.35",
        )

    fig.suptitle(title, fontsize=16, y=0.995)

    # Legend at the bottom.
    handles = []
    for form in FORMULATIONS:
        for backend in BACKENDS:
            handles.append(plt.Line2D(
                [0], [0],
                color=FORM_COLORS[form],
                linestyle=BACKEND_LS[backend],
                linewidth=1.8,
                label=f"{backend.upper()} — {FORM_LABELS[form]}",
            ))
    fig.legend(
        handles=handles, ncol=3, frameon=False,
        loc="lower center", bbox_to_anchor=(0.5, -0.01 if n > 4 else -0.03),
        columnspacing=2.0, handletextpad=0.6,
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _sweep_sort_key(item):
    name, _ = item
    m = re.match(r"size_(\d+)k?", name)
    if m: return (int(m.group(1)),)
    m = re.match(r"ratio_(\d+)_(\d+)", name)
    if m: return (int(m.group(2)),)
    return (name,)


def _standard_dataset_order(name):
    # Heuristic: keep alphabetical, but put smaller "tiny/small/test" first
    # so the eye lands on the small problems before the large ones.
    weight = 0
    low = name.lower()
    if "tiny" in low or "test" in low or "small" in low:
        weight = -1
    return (weight, name.lower())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Cost − cost_min vs iters/time, per scenario.")
    ap.add_argument("--sweep-input", default=None,
                    help="aggregated sweep JSON (default: "
                         "examples/data/analysis/sweep_results.json)")
    ap.add_argument("--sweep-prefix", default="cost_residual_sweep",
                    help="output PDF basename for sweep axes — gets "
                         "_<axis>.pdf appended. Default: %(default)s.")
    ap.add_argument("--sweep-title", default="Sweep",
                    help="title prefix for sweep figures (e.g. \"Grid3D sweep\")")
    ap.add_argument("--no-standard", action="store_true",
                    help="skip the per-category standard-dataset plots "
                         "(useful when you only want the sweep output)")
    args = ap.parse_args(argv)

    ANALYSIS.mkdir(parents=True, exist_ok=True)

    # Standard datasets, grouped by category.
    if not args.no_standard:
        for cat in ["pgo", "raslam", "sfm", "snl"]:
            cat_dir = DATA / cat
            if not cat_dir.exists():
                continue
            ds_list = []
            for ds_dir in sorted(cat_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                runs = load_standard_dataset(ds_dir)
                if runs:
                    ds_list.append((ds_dir.name, runs))
            ds_list.sort(key=lambda item: _standard_dataset_order(item[0]))
            out = ANALYSIS / f"cost_residual_{cat}.pdf"
            plot_category(cat.upper(), ds_list, out)

    # Sweep, split by axis.
    sweep = load_sweep(args.sweep_input)
    for axis in ("size", "ratio"):
        scns = sweep.get(axis, [])
        scns.sort(key=_sweep_sort_key)
        out = ANALYSIS / f"{args.sweep_prefix}_{axis}.pdf"
        plot_category(f"{args.sweep_title} — {axis}", scns, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
