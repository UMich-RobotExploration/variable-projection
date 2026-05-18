#!/usr/bin/env python3
"""Aggregate (not per-dataset) speedup plot for the standard datasets.

For each standard dataset we compute per-(formulation, backend) time-to-
convergence using the same definition as sweep_plots_lines.py:
  - target_cost = best Implicit final cost across all inits on that dataset
  - t_conv      = first time the cost drops within (1+TOL) * target_cost
                   (median over inits, separately per formulation+backend)
Then per dataset we compute speedup ratios:
  formulation speedup:  median(baseline_t_conv) / median(Ours_t_conv)
                         (same backend; baseline ∈ {Explicit, ExplicitVarPro})
  backend speedup:      median(CPU_t_conv)      / median(GPU_t_conv)
                         (same formulation)

We then **aggregate across datasets** rather than plotting per-dataset
points: box plots show the distribution of per-dataset speedups within each
problem family (PGO / RA-SLAM / SfM / SNL / ALL combined), with the
median annotated. Two output panels:

  Row 1 — Formulation speedup (backend held fixed): 4 box positions per
          family (CPU/GPU × Original/Orig.+VP).
  Row 2 — Backend speedup (formulation held fixed): 3 box positions per
          family (Ours / Original / Orig.+VP).

Output: examples/data/analysis/standard_speedup.pdf
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data"
ANALYSIS = DATA / "analysis"

CONVERGENCE_TOL = 0.01

FORM_INT_TO_STR = {0: "Explicit", 1: "ExplicitVarPro", 2: "Implicit"}
FORMULATIONS = ["Implicit", "Explicit", "ExplicitVarPro"]
FORM_LABELS = {
    "Implicit": "Ours",
    "Explicit": "Original",
    "ExplicitVarPro": "Orig.+VP",
}
FORM_COLORS = {
    "Implicit": "#2ca02c",
    "Explicit": "#1f77b4",
    "ExplicitVarPro": "#ff7f0e",
}
BACKENDS = ["Cpu", "Gpu"]
BACKEND_LS = {"Cpu": "--", "Gpu": "-"}
BACKEND_ALPHA = {"Cpu": 0.55, "Gpu": 1.0}

CATEGORIES = ["pgo", "raslam", "sfm", "snl"]
CATEGORY_LABEL = {
    "pgo": "PGO", "raslam": "RA-SLAM", "sfm": "SfM", "snl": "SNL", "all": "ALL",
}


plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.4,
})


# ---------------------------------------------------------------------------
# Loading + convergence-point extraction
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


def _load_runs(ds_dir):
    raw = []
    for fname in ("results.json", "gpu_results.json"):
        p = ds_dir / fname
        if not p.exists():
            continue
        try:
            raw.extend(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    out = []
    for r in raw:
        form = _formulation_name(r.get("formulation"))
        backend = _normalize_backend(r.get("backend"))
        costs = r.get("costs") or []
        times = r.get("times") or []
        if form is None or backend is None or not costs:
            continue
        out.append({
            "formulation": form, "backend": backend,
            "costs": costs, "times": times,
        })
    return out


# Some synthetic / clean datasets (notably SNL) converge to ~0 cost, so a
# pure relative tolerance (1+TOL)*target collapses to the floor. Pair it
# with an absolute-tolerance fallback so the convergence criterion is
# meaningful even when target ≈ 0.
CONVERGENCE_ABS_TOL = 1e-6


def _first_within_tol(times, costs, target):
    """Time at which `costs` first drops below max((1+TOL)*target, abs_tol)."""
    if target is None or not math.isfinite(target):
        return None
    threshold = max((1.0 + CONVERGENCE_TOL) * target, CONVERGENCE_ABS_TOL)
    for i, c in enumerate(costs):
        if not isinstance(c, (int, float)) or not math.isfinite(c):
            continue
        if c <= threshold:
            t = times[i] if i < len(times) else None
            return float(t) if isinstance(t, (int, float)) else None
    return None


def _per_dataset_median_times(runs):
    """Returns {(backend, formulation): median time-to-convergence}
    plus the target cost used."""
    # Target = min Implicit final cost (matches sweep convention).
    implicit_finals = []
    for r in runs:
        if r["formulation"] != "Implicit":
            continue
        finite = [c for c in r["costs"]
                  if isinstance(c, (int, float)) and math.isfinite(c)]
        if finite:
            implicit_finals.append(min(finite))
    if not implicit_finals:
        return {}, None
    target = min(implicit_finals)
    # Don't bail out at target ≈ 0 anymore — the absolute-tolerance branch
    # in _first_within_tol handles it.

    times_by_combo = defaultdict(list)
    for r in runs:
        t = _first_within_tol(r["times"], r["costs"], target)
        if t is None:
            continue
        times_by_combo[(r["backend"], r["formulation"])].append(t)
    medians = {k: float(np.median(v)) for k, v in times_by_combo.items() if v}
    return medians, target


def collect_speedups():
    """For every standard dataset, compute formulation and backend speedups.
    Returns:
      form_speedups[(category, backend, baseline)]  -> list of (s, t_base, t_ours)
      backend_speedups[(category, formulation)]      -> list of (s, t_cpu, t_gpu)

    Each entry carries both the ratio and the underlying times so callers
    can either median over ratios (dataset-equal weighting) or sum the
    times (time-weighted / deployment-realistic).
    """
    form_speedups = defaultdict(list)
    backend_speedups = defaultdict(list)

    for cat in CATEGORIES:
        cat_dir = DATA / cat
        if not cat_dir.exists():
            continue
        for ds_dir in sorted(cat_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            runs = _load_runs(ds_dir)
            if not runs:
                continue
            medians, target = _per_dataset_median_times(runs)
            if not medians:
                continue

            # Formulation speedups: Ours vs each baseline, per backend.
            for backend in BACKENDS:
                t_ours = medians.get((backend, "Implicit"))
                if t_ours is None or t_ours <= 0.0:
                    continue
                for baseline in ("Explicit", "ExplicitVarPro"):
                    t_base = medians.get((backend, baseline))
                    if t_base is None or t_base <= 0.0:
                        continue
                    s = t_base / t_ours
                    form_speedups[(cat, backend, baseline)].append((s, t_base, t_ours))

            # Backend speedup: CPU/GPU per formulation.
            for form in FORMULATIONS:
                t_cpu = medians.get(("Cpu", form))
                t_gpu = medians.get(("Gpu", form))
                if not (t_cpu and t_cpu > 0 and t_gpu and t_gpu > 0):
                    continue
                backend_speedups[(cat, form)].append((t_cpu / t_gpu, t_cpu, t_gpu))

    return form_speedups, backend_speedups


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _log_tick_formatter(v, _pos):
    if v >= 1:
        return f"{v:g}" + r"$\times$"
    return rf"$1/{1.0 / v:g}\times$"


def _format_speedup_yaxis(ax, ticks):
    ymin, ymax = min(ticks), max(ticks)
    ax.set_ylim(ymin / 1.12, ymax * 1.12)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(_log_tick_formatter))


def _draw_box(ax, positions, datasets, color, label_below=None):
    """`datasets` is a list, each entry a list of speedup tuples (s, ...)."""
    # Note: matplotlib expects iterables of floats. Filter empties so the
    # box loop is happy.
    data = [[t[0] for t in d] if d else [np.nan] for d in datasets]
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="0.3", linewidth=1.0),
        capprops=dict(color="0.3", linewidth=1.0),
        boxprops=dict(facecolor=color, alpha=0.45, edgecolor="0.3", linewidth=1.0),
    )
    # Annotate the median value above each box.
    for d, pos in zip(datasets, positions):
        if not d:
            continue
        med = float(np.median([t[0] for t in d]))
        ax.text(
            pos, med,
            f"{med:.2g}" + r"$\times$",
            ha="center", va="center",
            fontsize=10, color="black",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85),
            zorder=4,
        )


# ---------------------------------------------------------------------------
# Two-row figure
# ---------------------------------------------------------------------------

def _aggregate_all(speedups_by_cat):
    """Merge per-category lists into a single 'ALL' bucket keyed by everything
    *except* the category dimension."""
    out = defaultdict(list)
    for key, vals in speedups_by_cat.items():
        new_key = key[1:]  # drop the category dim
        out[new_key].extend(vals)
    return out


def plot(form_speedups, backend_speedups, out_path):
    cats_with_data = [c for c in CATEGORIES + ["all"]]
    family_xs = np.arange(len(cats_with_data), dtype=float)

    fig, axarr = plt.subplots(
        2, 1, figsize=(16.0, 11.0), dpi=200,
        gridspec_kw=dict(hspace=0.45),
    )

    # ------ Row 1: Formulation speedup ----------------------------------
    ax_f = axarr[0]
    # 4 boxes per family: (backend, baseline) ordered as
    #   CPU-Original, CPU-Orig.+VP, GPU-Original, GPU-Orig.+VP
    box_combos = [
        ("Cpu", "Explicit"),
        ("Cpu", "ExplicitVarPro"),
        ("Gpu", "Explicit"),
        ("Gpu", "ExplicitVarPro"),
    ]
    group_width = 0.85
    n_box = len(box_combos)
    box_w = group_width / n_box

    all_form = _aggregate_all(form_speedups)

    for slot, (backend, baseline) in enumerate(box_combos):
        offset = -group_width / 2 + (slot + 0.5) * box_w
        positions = family_xs + offset
        per_family = []
        for cat in cats_with_data:
            if cat == "all":
                vals = all_form.get((backend, baseline), [])
            else:
                vals = form_speedups.get((cat, backend, baseline), [])
            per_family.append(vals)
        color = FORM_COLORS[baseline]
        _draw_box(ax_f, positions, per_family, color)

    ax_f.axhline(1.0, color="0.4", linewidth=1.0, linestyle=":", zorder=2)
    ax_f.set_yscale("log")
    _format_speedup_yaxis(ax_f, ticks=[1/2, 1, 2, 5, 10, 20])
    ax_f.set_xticks(family_xs)
    ax_f.set_xticklabels([CATEGORY_LABEL[c] for c in cats_with_data])
    ax_f.set_ylabel(r"Formulation speedup ($\times$)")
    ax_f.set_title("Median time-to-convergence speedup of Ours vs baseline "
                    "(per dataset, then aggregated across category)")
    ax_f.grid(True, axis="y", which="major", linestyle="--", alpha=0.4, zorder=0)
    ax_f.spines["top"].set_visible(False)
    ax_f.spines["right"].set_visible(False)

    # Formulation legend.
    handles_f = []
    for baseline in ("Explicit", "ExplicitVarPro"):
        for backend in BACKENDS:
            handles_f.append(plt.Rectangle(
                (0, 0), 1, 1,
                facecolor=FORM_COLORS[baseline],
                alpha=0.45 if backend == "Cpu" else 0.85,
                edgecolor="0.3",
                label=f"{backend.upper()}: Ours vs {FORM_LABELS[baseline]}",
            ))
    handles_f.append(plt.Line2D([0], [0], color="0.4", linestyle=":",
                                  linewidth=1.0, label=r"$1\times$"))
    ax_f.legend(handles=handles_f, ncol=5, loc="upper center",
                 bbox_to_anchor=(0.5, -0.16), frameon=False,
                 columnspacing=1.8, handletextpad=0.6)

    # ------ Row 2: Backend (GPU vs CPU) speedup ------------------------
    ax_b = axarr[1]
    backend_box_forms = ["Implicit", "Explicit", "ExplicitVarPro"]
    n_box = len(backend_box_forms)
    box_w = group_width / n_box

    all_back = _aggregate_all(backend_speedups)

    for slot, form in enumerate(backend_box_forms):
        offset = -group_width / 2 + (slot + 0.5) * box_w
        positions = family_xs + offset
        per_family = []
        for cat in cats_with_data:
            if cat == "all":
                vals = all_back.get((form,), [])
            else:
                vals = backend_speedups.get((cat, form), [])
            per_family.append(vals)
        color = FORM_COLORS[form]
        _draw_box(ax_b, positions, per_family, color)

    ax_b.axhline(1.0, color="0.4", linewidth=1.0, linestyle=":", zorder=2)
    ax_b.set_yscale("log")
    _format_speedup_yaxis(ax_b, ticks=[1/3, 1/2, 1, 1.5, 2])
    ax_b.set_xticks(family_xs)
    ax_b.set_xticklabels([CATEGORY_LABEL[c] for c in cats_with_data])
    ax_b.set_ylabel(r"GPU vs CPU speedup ($\times$)")
    ax_b.set_title("Median GPU vs CPU speedup, per formulation "
                    "(per dataset, then aggregated across category)")
    ax_b.grid(True, axis="y", which="major", linestyle="--", alpha=0.4, zorder=0)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    handles_b = [
        plt.Rectangle((0, 0), 1, 1,
                       facecolor=FORM_COLORS[f], alpha=0.45,
                       edgecolor="0.3",
                       label=f"GPU vs CPU — {FORM_LABELS[f]}")
        for f in backend_box_forms
    ]
    handles_b.append(plt.Line2D([0], [0], color="0.4", linestyle=":",
                                  linewidth=1.0, label=r"$1\times$"))
    ax_b.legend(handles=handles_b, ncol=4, loc="upper center",
                 bbox_to_anchor=(0.5, -0.16), frameon=False,
                 columnspacing=1.8, handletextpad=0.6)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# Console table (for the report)
# ---------------------------------------------------------------------------

def print_tables(form_speedups, backend_speedups):
    all_form = _aggregate_all(form_speedups)
    all_back = _aggregate_all(backend_speedups)

    print("\n=== Formulation speedup: median(baseline t_conv) / median(Ours t_conv) ===")
    print(f"{'category':<10} | n datasets | Ours vs Original (CPU/GPU) | Ours vs Orig.+VP (CPU/GPU)")
    for cat in CATEGORIES + ["all"]:
        if cat == "all":
            buckets = all_form
            n = max(len(buckets.get(k, [])) for k in
                    [("Cpu", "Explicit"), ("Gpu", "Explicit")])
        else:
            buckets = {(b, base): form_speedups.get((cat, b, base), [])
                        for b in BACKENDS for base in ("Explicit", "ExplicitVarPro")}
            n = max(len(v) for v in buckets.values()) if buckets else 0
        def med(b, base):
            v = buckets.get((b, base), [])
            return f"{np.median([t[0] for t in v]):.2f}x" if v else "  —  "
        print(f"{CATEGORY_LABEL[cat]:<10} | {n:>10} | "
              f"{med('Cpu','Explicit'):>6} / {med('Gpu','Explicit'):>6}        | "
              f"{med('Cpu','ExplicitVarPro'):>6} / {med('Gpu','ExplicitVarPro'):>6}")

    print("\n=== GPU vs CPU speedup (per formulation) ===")
    print(f"{'category':<10} | Ours    Original   Orig.+VP")
    for cat in CATEGORIES + ["all"]:
        if cat == "all":
            buckets = all_back
        else:
            buckets = {(f,): backend_speedups.get((cat, f), []) for f in FORMULATIONS}
        def med(f):
            v = buckets.get((f,), [])
            return f"{np.median([t[0] for t in v]):.2f}x" if v else "  —  "
        print(f"{CATEGORY_LABEL[cat]:<10} | "
              f"{med('Implicit'):>6}   {med('Explicit'):>6}   {med('ExplicitVarPro'):>6}")


# ---------------------------------------------------------------------------

def main() -> int:
    form_speedups, backend_speedups = collect_speedups()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    plot(form_speedups, backend_speedups, ANALYSIS / "standard_speedup.pdf")
    print_tables(form_speedups, backend_speedups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
