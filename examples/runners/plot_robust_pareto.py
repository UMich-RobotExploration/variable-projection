#!/usr/bin/env python3
"""Accuracy-vs-runtime Pareto plot for the robust (GM + GNC) experiments.

Reads the per-dataset JSONs written by the two robust sweeps,

  examples/data/analysis/cosmobench_irls_gnc_5init/    (irls_robust.cpp: the
                                                       VarPro formulations)
  examples/data/analysis/cosmobench_gtsam_sesync_5init/ (~/varProj-gtsam
                                                       SESync_GNC_example, the
                                                       residual-matched baseline)

and plots one mark per (dataset, method) as ATE against solver wall time, with
CosmoBench and Nebula pooled into a single panel.  Lower left is better.

Each sweep runs five inits per dataset (seed 0 noiseless odometry, seeds 1-4
perturbing every odometry edge by 0.5 deg / 0.02 m before chaining), and all
methods -- including the SESync baseline, via its --init-tum -- see the same
init for a given seed.  Like the table (make_robust_table.py), the figure
reports a *single trial*: --seed 0 by default, the noiseless odometry init.
--seed all instead draws each dataset's median over inits, and --show-seeds
then scatters the individual inits faintly behind them.  The single-init
sweeps are still plottable with --irls-dir / --sesync-dir.

A run is "converged" if its ATE is within --ate-tol (default 5%) of the best
ATE any method reached on that dataset, or its GM robust cost is within
--cost-tol (default 1%) of the best -- the same rule that puts \\mredx in the
table.  Runs outside it are still plotted (the accuracy axis is the
point of the figure) but hollow, and under --normalize best a dashed line
marks the cutoff at 1 + ate_tol.

Pooling datasets that span 600-3500 poses means raw seconds and raw metres are
not comparable across the cloud, so both axes are normalized per dataset
(--normalize):

  best    each dataset's own fastest time and lowest ATE become the reference,
          so a mark reads "k times slower / k worse than the best method on
          that dataset".  Default; privileges no method.
  median  divide by the per-dataset median over the methods instead.
  ours    divide by our numbers, so a mark reads as a speedup / accuracy factor
          against us.
  none    raw seconds and metres.

Note that on the log accuracy axis the near-optimal band is a 0.04-decade
sliver at 1.0 sitting next to divergences at 90x, so the spread of the runs
that all reached the same optimum is not resolvable; quote the medians from the
summary table alongside the figure.

Styling matches the convergence figures: tab10 with Original blue / Orig. + VP
orange / Ours green / GTSAM red, whitegrid chrome, frameless legend below.

The four TUHH datasets are dropped by default: with the per-robot priors
stripped nothing ties the robots to a common gauge, so every method scores
104-140 m global ATE there while per-robot aligned ATE stays under 1.1 m.  That
is a property of the data, not of the solvers.  --include-tuhh puts them back.

Usage:
  .venv/bin/python examples/runners/plot_robust_pareto.py
  .venv/bin/python examples/runners/plot_robust_pareto.py --normalize none
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = REPO / "examples" / "data" / "analysis"
IRLS_DIR = ANALYSIS_DIR / "cosmobench_irls_gnc_5init"
SESYNC_DIR = ANALYSIS_DIR / "cosmobench_gtsam_sesync_5init"
# The earlier single-init sweeps, still plottable via --irls-dir / --sesync-dir.
IRLS_DIR_1INIT = ANALYSIS_DIR / "cosmobench_irls_gnc_full"
SESYNC_DIR_1INIT = ANALYSIS_DIR / "cosmobench_gtsam_sesync"

NEBULA = {"finals", "kentucky_underground", "tunnel", "urban"}

# method key -> (legend label, colour, marker).  Hues match the convergence
# figures.  Marker shape is the secondary encoding, so identity survives
# greyscale printing and the red/green pair.
METHODS = [
    ("explicit", "Original", "#1f77b4", "s"),
    ("expvp", "Orig. + VP", "#ff7f0e", "^"),
    ("impl", "Ours", "#2ca02c", "o"),
    ("gtsam", "GTSAM", "#d62728", "P"),
]

GRID = "#e6e6e6"
SPINE = "#000000"
INK = "#1a1a1a"

AXES = {
    "best": ("Time / fastest (×)", "ATE / best (×)"),
    "median": ("Time / median (×)", "ATE / median (×)"),
    "ours": ("Time / Ours (×)", "ATE / Ours (×)"),
    "none": ("Time (s)", "ATE RMSE (m)"),
}


def summarize(runs: dict, seed: str = "all") -> dict:
    """Collapse one method's per-seed runs to the mark the figure draws.

    With a single `seed` the mark is that run -- the same two numbers the table
    quotes. With seed="all", `wall_s`/`ate` are the medians across the ok
    inits, and `seeds` keeps every individual run so --show-seeds can scatter
    them. A method counts as ok if at least one selected init finished.
    """
    if seed != "all":
        runs = {k: v for k, v in runs.items() if str(k) == seed}
    ok = [r for r in runs.values() if r.get("status") == "ok"
          and r.get("wall_s") and r.get("ate_global_rmse_m")]
    if not ok:
        statuses = sorted({str(r.get("status")) for r in runs.values()})
        return {"wall_s": None, "ate": None, "cost": None,
                "status": statuses[0] if statuses else "missing", "seeds": []}
    return {
        "wall_s": statistics.median(r["wall_s"] for r in ok),
        "ate": statistics.median(r["ate_global_rmse_m"] for r in ok),
        "cost": statistics.median(r["robust_cost"] for r in ok
                                  if r.get("robust_cost") is not None)
                if any(r.get("robust_cost") is not None for r in ok) else None,
        "status": "ok",
        "n_ok": len(ok),
        "n_runs": len(runs),
        "seeds": [{"wall_s": r["wall_s"], "ate": r["ate_global_rmse_m"],
                   "seed": r.get("seed")} for r in ok],
    }


def method_runs(entry: dict) -> dict:
    """Per-seed runs of one method, for either record shape.

    Multi-init records nest them under `seeds`/`runs`; the single-init sweeps
    stored one flat stats dict, which is just a one-element run set.
    """
    if not isinstance(entry, dict):
        return {}
    for key in ("seeds", "runs"):
        if isinstance(entry.get(key), dict):
            return entry[key]
    return {"0": entry}


def load(include_tuhh: bool, irls_dir: Path, sesync_dir: Path,
         seed: str = "all") -> dict:
    """dataset -> {family, n_poses, ..., m: {method: {wall_s, ate, status}}}."""
    rows: dict[str, dict] = {}
    for path in sorted(irls_dir.glob("*.json")):
        if path.stem == "summary":
            continue
        d = json.loads(path.read_text())
        name = d["dataset"]
        if not include_tuhh and name.startswith("tuhh"):
            continue
        rows[name] = {
            "family": "Nebula" if name in NEBULA else "CosmoBench",
            "n_poses": d["n_poses"], "n_edges": d["n_edges"],
            "n_outliers": d["n_outliers_truth"],
            "m": {k: summarize(method_runs(v), seed)
                  for k, v in d["methods"].items()},
        }
    for path in sorted(sesync_dir.glob("*.json")):
        if path.stem == "summary":
            continue
        d = json.loads(path.read_text())
        if d["dataset"] not in rows:
            continue
        runs = d["runs"] if isinstance(d.get("runs"), dict) else {"0": d.get("result", {})}
        rows[d["dataset"]]["m"]["gtsam"] = summarize(runs, seed)
    return rows


def apply_norm(rows: dict, mode: str, ate_tol: float = 0.05,
               cost_tol: float = 0.01) -> None:
    """Stamp each ok run with the plotted coordinates (v["x"], v["y"]) and
    whether it converged (v["conv"]): within `ate_tol` of the dataset's best
    ATE, or within `cost_tol` of its best robust cost."""
    keys = [k for k, _, _, _ in METHODS]
    for r in rows.values():
        ok = {k: r["m"][k] for k in keys
              if r["m"].get(k) and r["m"][k]["status"] == "ok"
              and r["m"][k]["wall_s"] and r["m"][k]["ate"]}
        best_ate = min(v["ate"] for v in ok.values())
        costs = [v["cost"] for v in ok.values() if v.get("cost") is not None]
        best_cost = min(costs) if costs else None
        for v in ok.values():
            v["conv"] = v["ate"] <= best_ate * (1.0 + ate_tol) or (
                best_cost is not None and v.get("cost") is not None
                and v["cost"] <= best_cost * (1.0 + cost_tol))
        # Time reference from converged runs only: a fast run that lands in
        # the wrong basin is not a runtime anyone can claim.
        conv = [v for v in ok.values() if v["conv"]]
        if mode == "best":
            tx = min(v["wall_s"] for v in conv)
            ty = best_ate
        elif mode == "median":
            tx = statistics.median(v["wall_s"] for v in conv)
            ty = statistics.median(v["ate"] for v in ok.values())
        elif mode == "ours":
            ref = ok.get("impl")
            tx, ty = (ref["wall_s"], ref["ate"]) if ref else (None, None)
        else:
            tx = ty = 1.0
        for v in ok.values():
            v["x"] = v["wall_s"] / tx if tx else None
            v["y"] = v["ate"] / ty if ty else None
            # Individual inits share the dataset's reference, so a seed mark is
            # comparable with the median mark it sits next to.
            for s in v.get("seeds", []):
                s["x"] = s["wall_s"] / tx if tx else None
                s["y"] = s["ate"] / ty if ty else None


def rel(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise (--out may point away)."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def points(rows: dict, family: str | None, method: str, per_seed: bool = False,
           conv: bool | None = None):
    """Plotted (x, y) of one method; `conv` keeps only runs inside (True) or
    outside (False) the ATE tolerance, None keeps both."""
    out = []
    for r in rows.values():
        if family is not None and r["family"] != family:
            continue
        v = r["m"].get(method)
        if not (v and v["status"] == "ok"):
            continue
        if conv is not None and v.get("conv") != conv:
            continue
        if per_seed:
            out += [(s["x"], s["y"]) for s in v.get("seeds", [])
                    if s.get("x") and s.get("y")]
        elif v.get("x") and v.get("y"):
            out.append((v["x"], v["y"]))
    return out


def draw_panel(ax, rows, norm, show_seeds: bool = False,
               ate_tol: float = 0.05):
    for key, label, colour, marker in METHODS:
        if show_seeds:
            # Every init as a faint mark, with the per-dataset median on top:
            # the cloud shows init sensitivity, the solid mark is what the
            # table quotes.
            seed_pts = points(rows, None, key, per_seed=True)
            if seed_pts:
                xs, ys = zip(*seed_pts)
                ax.scatter(xs, ys, s=18, marker=marker, c=colour, alpha=0.22,
                           linewidths=0.0, zorder=2)
        pts = points(rows, None, key, conv=True)
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=46, marker=marker, c=colour, alpha=0.6,
                       linewidths=0.6, edgecolors="white", zorder=3)
        # Not converged (outside both tolerances): the \mredx cells of the table.
        pts = points(rows, None, key, conv=False)
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=46, marker=marker, facecolors="none",
                       edgecolors=colour, alpha=0.8, linewidths=1.1, zorder=3)

    if norm == "best":
        ax.axhline(1.0 + ate_tol, color="#666666", ls="--", lw=0.9, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(AXES[norm][0])
    ax.set_ylabel(AXES[norm][1])


def render(rows, out_stem: Path, formats, title: str | None, norm: str,
           show_seeds: bool = False, ate_tol: float = 0.05):
    fig, ax = plt.subplots(figsize=(7.2, 4.75))
    fig.subplots_adjust(left=0.125, right=0.98, top=0.965, bottom=0.185)
    draw_panel(ax, rows, norm, show_seeds, ate_tol)

    vals = [(v["x"], v["y"]) for r in rows.values() for v in r["m"].values()
            if v and v["status"] == "ok" and v.get("x") and v.get("y")]
    if show_seeds:
        vals += [(s["x"], s["y"]) for r in rows.values() for v in r["m"].values()
                 if v and v["status"] == "ok"
                 for s in v.get("seeds", []) if s.get("x") and s.get("y")]
    xs, ys = zip(*vals)
    ax.set_xlim(min(xs) * 0.62, max(xs) * 1.7)
    ax.set_ylim(min(ys) * 0.7, max(ys) * 1.9)

    handles = [Line2D([], [], marker=mk, ls="none", markersize=7.5,
                      markerfacecolor=c, markeredgecolor="white",
                      markeredgewidth=1.0, label=lb)
               for _, lb, c, mk in METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=len(METHODS),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.4, columnspacing=2.2)

    if title:
        fig.suptitle(title, fontsize=12.5, color=INK, y=0.99)

    for ext in formats:
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=220, facecolor="white", pad_inches=0.06)
        print(f"wrote {rel(path)}")
    plt.close(fig)


def write_table(rows, out_stem: Path):
    """The medians the figure no longer draws, for the paper text.

    Each dataset contributes its median over inits, so `median wall` and
    `median ATE` mean the same thing they did in the single-init tables. The
    extra columns describe the inits themselves: `median ATE range` is the
    median across datasets of (max - min) ATE over that dataset's inits (how
    much the init mattered), and `ATE < 3 m` counts individual (dataset, init)
    runs, not datasets.
    """
    lines_csv = ["group,method,n_datasets,median_wall_s,median_ate_m,"
                 "median_x_norm,median_y_norm,median_ate_range_m,"
                 "n_runs_ate_under_3m,n_runs,n_within_ate_tol"]
    lines_md = ["| group | method | n | median wall (s) | median ATE (m) | "
                "median x (norm) | median y (norm) | median ATE range (m) | "
                "ATE < 3 m | within ATE tol |",
                "|---|---|---|---|---|---|---|---|---|---|"]
    for family in (None, "CosmoBench", "Nebula"):
        for key, label, _, _ in METHODS:
            pts = points(rows, family, key)
            if not pts:
                continue
            nx, ny = zip(*pts)
            raw = []
            ranges = []
            n_under, n_runs, n_conv = 0, 0, 0
            for r in rows.values():
                if family is not None and r["family"] != family:
                    continue
                v = r["m"].get(key)
                if not (v and v["status"] == "ok" and v.get("x")):
                    continue
                raw.append((v["wall_s"], v["ate"]))
                n_conv += bool(v.get("conv"))
                ates = [s["ate"] for s in v.get("seeds", [])]
                if ates:
                    ranges.append(max(ates) - min(ates))
                    n_runs += len(ates)
                    n_under += sum(1 for a in ates if a < 3.0)
            rw, ra = zip(*raw)
            group = family or "All"
            rng = statistics.median(ranges) if ranges else float("nan")
            lines_csv.append(
                f"{group},{label},{len(pts)},{statistics.median(rw):.3f},"
                f"{statistics.median(ra):.4f},{statistics.median(nx):.3f},"
                f"{statistics.median(ny):.3f},{rng:.4f},{n_under},{n_runs},{n_conv}")
            lines_md.append(
                f"| {group} | {label} | {len(pts)} | {statistics.median(rw):.2f} "
                f"| {statistics.median(ra):.2f} | {statistics.median(nx):.2f} "
                f"| {statistics.median(ny):.2f} | {rng:.2f} "
                f"| {n_under}/{n_runs} | {n_conv}/{len(pts)} |")
    out_stem.with_suffix(".csv").write_text("\n".join(lines_csv) + "\n")
    out_stem.with_suffix(".md").write_text("\n".join(lines_md) + "\n")
    print(f"wrote {rel(out_stem.with_suffix('.csv'))}")
    print(f"wrote {rel(out_stem.with_suffix('.md'))}")
    print("\n".join(lines_md))


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": ["DejaVu Sans"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.9,
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.labelcolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.minor.size": 0, "ytick.minor.size": 0,
        "legend.fontsize": 10,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ANALYSIS_DIR / "robust_pareto",
                    help="output path stem (extensions are appended)")
    ap.add_argument("--include-tuhh", action="store_true",
                    help="keep the four gauge-degenerate TUHH datasets")
    ap.add_argument("--normalize", choices=["best", "median", "ours", "none"],
                    default="best",
                    help="per-dataset normalization of both axes "
                         "(default: best, i.e. each dataset's own optimum)")
    ap.add_argument("--title", default=None,
                    help="optional suptitle (off by default; LaTeX captions it)")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    ap.add_argument("--irls-dir", type=Path, default=IRLS_DIR,
                    help="VarPro sweep output (default: the 5-init sweep; pass "
                         f"{IRLS_DIR_1INIT.name} for the single-init results)")
    ap.add_argument("--sesync-dir", type=Path, default=SESYNC_DIR,
                    help="SESync baseline output (default: the 5-init sweep; "
                         f"pass {SESYNC_DIR_1INIT.name} for the single-init one)")
    ap.add_argument("--seed", default="0",
                    help="init seed to plot, matching the table (default 0: "
                         "noiseless odometry); 'all' plots the per-dataset "
                         "median over inits")
    ap.add_argument("--ate-tol", type=float, default=0.05,
                    help="relative ATE tolerance vs the per-dataset best; runs "
                         "outside it (and outside --cost-tol) are drawn "
                         "hollow (default 0.05)")
    ap.add_argument("--cost-tol", type=float, default=0.01,
                    help="relative robust-cost tolerance vs the per-dataset "
                         "best; a run within it also counts as converged "
                         "(default 0.01)")
    ap.add_argument("--show-seeds", action="store_true",
                    help="also scatter every individual init as a faint mark "
                         "behind the per-dataset medians")
    args = ap.parse_args()

    if not args.irls_dir.is_dir() or not args.sesync_dir.is_dir():
        missing = [str(d) for d in (args.irls_dir, args.sesync_dir)
                   if not d.is_dir()]
        print(f"missing sweep output: {', '.join(missing)}")
        return 2
    rows = load(args.include_tuhh, args.irls_dir, args.sesync_dir, args.seed)
    apply_norm(rows, args.normalize, args.ate_tol, args.cost_tol)
    apply_style()
    render(rows, args.out, args.formats, args.title, args.normalize,
           show_seeds=args.show_seeds, ate_tol=args.ate_tol)
    write_table(rows, args.out.with_name(args.out.name + "_summary"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
