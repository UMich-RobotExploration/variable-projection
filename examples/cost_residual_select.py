#!/usr/bin/env python3
"""Cost-residual convergence plots for a hand-picked set of datasets,
with GTSAM added as a fourth method alongside our three formulations.

Datasets (one PDF each):
  - Intel             (PGO)
  - Single-Drone      (RA-SLAM)
  - SNL MIT           (SNL)
  - MipNeRF Garden    (SfM)

For each, we plot f(x_t) − f* on a log y-axis vs both iteration index and
wall-clock time. f* is the *best* cost achieved across every method,
formulation, backend, and init on that dataset (so GTSAM is in the
running for the optimum).

Result-file conventions:
  ours:   examples/data/<cat>/<name>/results.json     (CpuRTR / TNT)
          examples/data/<cat>/<name>/gpu_results.json (GpuRTR)
  gtsam:  /home/nikolas/varProj-gtsam/data/<cat>/<name>/results.json
          (single file, formulation == "gtsam", no backend field)

Outputs (under examples/data/analysis/):
  cost_residual_select_intel.pdf
  cost_residual_select_single_drone.pdf
  cost_residual_select_snl_mit.pdf
  cost_residual_select_mipnerf_garden.pdf
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent
OURS_DATA = REPO / "examples" / "data"
GTSAM_DATA = Path("/home/nikolas/varProj-gtsam/data")
ANALYSIS = OURS_DATA / "analysis"


# Display config.
FORM_INT_TO_STR = {0: "Explicit", 1: "ExplicitVarPro", 2: "Implicit"}

FORM_LABELS = {
    "Implicit": "Ours",
    "Explicit": "Original",
    "ExplicitVarPro": "Orig. + VP",
    "gtsam": "GTSAM",
}
FORM_COLORS = {
    "Implicit":        "#2ca02c",  # green
    "Explicit":        "#1f77b4",  # blue
    "ExplicitVarPro":  "#ff7f0e",  # orange
    "gtsam":           "#d62728",  # red
}

# Plot order (drawn back-to-front, so later items end up on top).
METHODS = ["Explicit", "ExplicitVarPro", "Implicit", "gtsam"]

# f - f* is clipped from below by this floor so log y stays well-defined
# on perfectly-converged tails (synthetic SNL hits ~1e-14).
FLOOR = 1e-10


plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.2,
})


# ---------------------------------------------------------------------------
# Datasets to plot
# ---------------------------------------------------------------------------

# (display_name, category_dir, ours_dirname, gtsam_dirname, out_slug)
DATASETS = [
    ("Intel",             "pgo",    "intel",          "intel",          "intel"),
    ("Single-Drone",      "raslam", "single_drone",   "single_drone",   "single_drone"),
    ("SNL — MIT",         "snl",    "MIT_snl",        "MIT_snl",        "snl_mit"),
    ("MipNeRF Garden",    "sfm",    "MipNerf-garden", "MipNerf-garden", "mipnerf_garden"),
]


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


def _load_json(path):
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"  warning: {path} unreadable")
        return []


def load_runs(category, ours_dirname, gtsam_dirname):
    """Return a list of run dicts {formulation, backend (or None), costs,
    times} from both our and GTSAM JSON files."""
    runs = []

    # Ours: CpuRTR (results.json) + GpuRTR (gpu_results.json).
    for fname in ("results.json", "gpu_results.json"):
        for r in _load_json(OURS_DATA / category / ours_dirname / fname):
            form = _formulation_name(r.get("formulation"))
            backend = _normalize_backend(r.get("backend"))
            if form is None or backend is None or not r.get("costs"):
                continue
            runs.append({
                "method": form,           # "Implicit" / "Explicit" / "ExplicitVarPro"
                "backend": backend,
                "costs": r["costs"],
                "times": r.get("times") or [],
            })

    # GTSAM: single results.json, no backend, formulation == "gtsam".
    # The GTSAM driver reports per-iteration *delta* times (each entry is
    # the time spent in that iteration, not the elapsed time since t=0).
    # Convert to cumulative wall-clock so it lines up with our convention.
    for r in _load_json(GTSAM_DATA / category / gtsam_dirname / "results.json"):
        if not r.get("costs"):
            continue
        raw_times = r.get("times") or []
        cum_times = list(np.cumsum(raw_times)) if raw_times else []
        runs.append({
            "method": "gtsam",
            "backend": None,
            "costs": r["costs"],
            "times": cum_times,
        })

    return runs


# ---------------------------------------------------------------------------
# cost-min and trajectories
# ---------------------------------------------------------------------------

def _aggregate_method(runs, method):
    """Aggregate trajectories for a single method across (backends, inits).

    Returns:
        iters_x : np.ndarray of shape (T,)   — iteration indices 0..T-1
        time_x  : np.ndarray of shape (T,)   — median wall-clock time at
                                                each iteration index across
                                                the contributing runs
        med_y   : np.ndarray of shape (T,)   — median cost
        q25_y   : np.ndarray of shape (T,)
        q75_y   : np.ndarray of shape (T,)

    Returns None if no run for this method has any data.
    """
    method_runs = [r for r in runs if r["method"] == method]
    if not method_runs:
        return None

    T = max(len(r["costs"]) for r in method_runs)
    iters_x = np.arange(T, dtype=float)
    time_x  = np.full(T, np.nan)
    med_y   = np.full(T, np.nan)
    q25_y   = np.full(T, np.nan)
    q75_y   = np.full(T, np.nan)

    for i in range(T):
        ys_at_i = []
        ts_at_i = []
        for r in method_runs:
            if i >= len(r["costs"]):
                continue
            c = r["costs"][i]
            if isinstance(c, (int, float)) and math.isfinite(c):
                ys_at_i.append(max(c, FLOOR))
            t = r["times"][i] if i < len(r["times"]) else None
            if isinstance(t, (int, float)) and math.isfinite(t):
                ts_at_i.append(t)
        if ys_at_i:
            arr = np.asarray(ys_at_i, dtype=float)
            med_y[i]  = float(np.median(arr))
            q25_y[i]  = float(np.percentile(arr, 25))
            q75_y[i]  = float(np.percentile(arr, 75))
        if ts_at_i:
            time_x[i] = float(np.median(ts_at_i))
    return iters_x, time_x, med_y, q25_y, q75_y


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw(ax, agg_by_method, x_kind):
    for method in METHODS:
        a = agg_by_method.get(method)
        if a is None:
            continue
        iters_x, time_x, med_y, q25_y, q75_y = a
        xs = iters_x if x_kind == "iters" else time_x
        mask = np.isfinite(xs) & np.isfinite(med_y)
        if not mask.any():
            continue
        color = FORM_COLORS[method]
        # IQR band
        band_mask = mask & np.isfinite(q25_y) & np.isfinite(q75_y)
        if band_mask.any():
            ax.fill_between(
                xs[band_mask], q25_y[band_mask], q75_y[band_mask],
                color=color, alpha=0.18, linewidth=0, zorder=1,
            )
        # Median line
        ax.plot(
            xs[mask], med_y[mask],
            color=color, linewidth=2.0, zorder=3,
            label=FORM_LABELS[method],
        )
    ax.set_yscale("log")
    # Let matplotlib pick the y-range from the actual data; no artificial
    # floor extension into empty log decades.
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _legend_handles():
    return [
        plt.Line2D(
            [0], [0],
            color=FORM_COLORS[m], linewidth=2.0,
            label=FORM_LABELS[m],
        )
        for m in METHODS
    ]


def plot_dataset(display_name, runs, out_path):
    if not runs:
        print(f"  no data for {display_name}; skipping")
        return

    agg = {m: _aggregate_method(runs, m) for m in METHODS}

    fig, (ax_i, ax_t) = plt.subplots(1, 2, figsize=(14.0, 5.0), dpi=200,
                                       sharey=True,
                                       gridspec_kw=dict(wspace=0.04))
    _draw(ax_i, agg, "iters")
    _draw(ax_t, agg, "time")

    ax_i.set_ylabel("Cost")
    ax_i.set_xlabel("iterations")
    ax_t.set_xlabel("time (s)")
    fig.suptitle(display_name, fontsize=15, y=1.00)

    fig.legend(handles=_legend_handles(), ncol=4, frameon=False,
                loc="lower center", bbox_to_anchor=(0.5, -0.06),
                columnspacing=2.4, handletextpad=0.7)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")


# ---------------------------------------------------------------------------

def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for display_name, cat, ours_name, gtsam_name, slug in DATASETS:
        runs = load_runs(cat, ours_name, gtsam_name)
        # Quick coverage check on the console.
        per_method = {}
        for r in runs:
            key = (r["method"], r.get("backend") or "—")
            per_method[key] = per_method.get(key, 0) + 1
        print(f"\n{display_name}:")
        for k in sorted(per_method):
            print(f"  {k[0]:<16} {k[1]:<4}: {per_method[k]} runs")

        out = ANALYSIS / f"cost_residual_select_{slug}.pdf"
        plot_dataset(display_name, runs, out)


if __name__ == "__main__":
    main()
