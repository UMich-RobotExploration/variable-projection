#!/usr/bin/env python3
"""Standard (non-robust) benchmark: performance profile, or a cost-vs-time
Pareto scatter.

Companion to plot_robust_pareto.py.  The standard sweep optimizes an unweighted
least-squares objective and records cost/time traces, so the accuracy axis is
cost, not ATE.

Reads the traces the paper_experiments sweep leaves in the tree:

  examples/data/**/results.json       CPU  (--device cpu)
  examples/data/**/gpu_results.json   GPU  (--device gpu)

Each file holds one record per (formulation, init): the full cost/time trace
plus precompute_s.  The 51 standard datasets carry the full 4 formulations x 5
inits grid; sphere_sweep/ is a separate scaling study and outfinite only ever
ran one formulation on CPU, so both are skipped, as are the toy problems in
SKIP_DATASETS.

--form profile (default): a Dolan-More performance profile.  One problem is one
(dataset, init), so the starting point is always held fixed.  A method *solves*
a problem when its final cost lands within --tau of the best cost any method
reached from that init -- without that test a method could win on time by
stopping early at a worse point.  The curve plots the fraction of problems a
method solves within k times the fastest solver's time, so the value at k = 1
is how often it is fastest and the right-hand plateau is how often it succeeds
at all.  A curve that never reaches 1.0 failed the remainder.

--form pareto: one mark per (dataset, method), time against final cost, both
divided by the best value any method reached on that (dataset, init) and then
reduced over the five inits by median.  Legible, but on this benchmark ~88% of
the marks land on cost ratio 1.0, which is why the profile is the default.

Time is precompute + solve by default (--time solve drops precompute).  Note
the recorded precompute for Implicit and Dense includes a host CHOLMOD
factorization the GPU path never uses, so the GPU figure charges both of them
for work they do not do; that is conservative for Implicit and roughly neutral
for Dense, whose precompute is dominated by forming the Schur complement.

Datasets whose optimum is ~0 (cost ~1e-14) are dropped: a cost ratio and a 1%
band are both meaningless there.  --min-cost changes the cutoff.

GTSAM is a CPU-only baseline and is read from
examples/data/analysis/standard_gtsam/<dataset>.json, each holding
  {"dataset": str, "runs": [{"init": str, "wall_s": float,
                             "final_cost": float}, ...]}
If that directory is absent the series is simply omitted.

Usage:
  .venv/bin/python examples/runners/plot_standard_pareto.py --device gpu
  .venv/bin/python examples/runners/plot_standard_pareto.py --device cpu
  .venv/bin/python examples/runners/plot_standard_pareto.py --form pareto
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
DATA_DIR = REPO / "examples" / "data"
GTSAM_DIR = DATA_DIR / "analysis" / "standard_gtsam"

# Formulation index in the JSON -> our method key.
FORM_KEY = {0: "explicit", 1: "expvp", 2: "impl", 3: "dense"}

# method key -> (legend label, colour, marker).  Hues match the convergence
# figures; Dense takes tab10's next slot.
STYLE = {
    "explicit": ("Original", "#1f77b4", "s"),
    "expvp": ("Orig. + VP", "#ff7f0e", "^"),
    "impl": ("Ours", "#2ca02c", "o"),
    "gtsam": ("GTSAM", "#d62728", "P"),
    "dense": ("Dense", "#9467bd", "D"),
}
# GTSAM is a CPU baseline; Dense is the GPU memory-scaling baseline.
DEVICE_METHODS = {
    "cpu": ["explicit", "expvp", "impl", "gtsam"],
    "gpu": ["explicit", "expvp", "impl", "dense"],
}
RESULT_FILE = {"cpu": "results.json", "gpu": "gpu_results.json"}

# sphere_sweep is the p-scaling study, not part of the standard benchmark.
SKIP_FAMILIES = {"sphere_sweep"}

# Toy problems: tinyGrid3D solves in under a millisecond on CPU so its time
# ratio is not resolvable there, and smallGrid3D / factor_graph_small are too
# small to say anything about scaling.  Excluded on both devices so the CPU and
# GPU figures cover the same datasets.  (The _snl variants of the grids are
# dropped separately, for a ~0 optimum.)
SKIP_DATASETS = {"tinyGrid3D", "smallGrid3D", "factor_graph_small"}

GRID = "#e6e6e6"
SPINE = "#000000"
INK = "#1a1a1a"

AXES = ("Time / fastest (×)", "Cost / best (×)")


def load(device: str, use_precompute: bool) -> dict:
    """dataset -> {family, runs: {init: {method: {"t": s, "c": cost}}}}."""
    rows: dict[str, dict] = {}
    for path in sorted(DATA_DIR.rglob(RESULT_FILE[device])):
        family = path.relative_to(DATA_DIR).parts[0]
        if family in SKIP_FAMILIES:
            continue
        recs = json.loads(path.read_text())
        # The standard grid is 4 formulations x 5 inits; anything else is a
        # partial run (outfinite) or a different experiment.
        if len(recs) != 20:
            continue
        runs: dict[str, dict] = {}
        for r in recs:
            if r.get("skip_reason") or not r["times"] or not r["costs"]:
                continue
            init = Path(r["init_file"]).stem
            t = r["times"][-1] + (r["precompute_s"] if use_precompute else 0.0)
            runs.setdefault(init, {})[FORM_KEY[r["formulation"]]] = {
                "t": t, "c": r["costs"][-1]}
        if runs and path.parent.name not in SKIP_DATASETS:
            rows[path.parent.name] = {"family": family, "runs": runs}
    return rows


def merge_gtsam(rows: dict) -> int:
    """Fold the CPU GTSAM baseline in, if it has been generated. Returns n."""
    if not GTSAM_DIR.is_dir():
        return 0
    n = 0
    for path in sorted(GTSAM_DIR.glob("*.json")):
        d = json.loads(path.read_text())
        row = rows.get(d["dataset"])
        if row is None:
            continue
        for run in d.get("runs", []):
            if run["init"] in row["runs"]:
                row["runs"][run["init"]]["gtsam"] = {
                    "t": run["wall_s"], "c": run["final_cost"]}
                n += 1
    return n


def normalize(rows: dict, methods: list[str], min_cost: float):
    """dataset -> {method: (median time ratio, median cost ratio)}.

    Ratios are formed within a single (dataset, init) so the comparison always
    holds the starting point fixed, then reduced over the inits by median.
    """
    out, dropped, no_time = {}, [], []
    for name, row in rows.items():
        per_method: dict[str, list] = {m: [] for m in methods}
        best_overall = min((v["c"] for r in row["runs"].values()
                            for v in r.values()), default=None)
        if best_overall is None or best_overall < min_cost:
            dropped.append(name)
            continue
        timed = False
        for run in row["runs"].values():
            present = {m: v for m, v in run.items() if m in methods}
            if not present:
                continue
            tref = min(v["t"] for v in present.values())
            cref = min(v["c"] for v in present.values())
            if tref <= 0 or cref <= 0:
                # Trace times are recorded to the millisecond; a solve that
                # rounds to 0.000 s has no resolvable ratio.
                continue
            timed = True
            for m, v in present.items():
                per_method[m].append((v["t"] / tref, v["c"] / cref))
        pts = {}
        for m, vals in per_method.items():
            if vals:
                pts[m] = (statistics.median(v[0] for v in vals),
                          statistics.median(v[1] for v in vals))
        if pts:
            out[name] = pts
        elif not timed:
            no_time.append(name)
    return out, dropped, no_time


def profile_ratios(rows: dict, methods: list[str], min_cost: float,
                   tau: float):
    """Dolan-More performance ratios, one problem per (dataset, init).

    A method "solves" a problem when its final cost is within tau of the best
    cost any method reached from that same init; otherwise its ratio is
    infinite and it never contributes to the profile.  Without that test a
    method could win on time simply by stopping early at a worse point.
    """
    per = {m: [] for m in methods}
    n, dropped, no_time = 0, [], []
    for name, row in rows.items():
        best_overall = min((v["c"] for r in row["runs"].values()
                            for v in r.values()), default=None)
        if best_overall is None or best_overall < min_cost:
            dropped.append(name)
            continue
        timed = False
        for run in row["runs"].values():
            present = {m: v for m, v in run.items() if m in methods}
            if not present:
                continue
            cbest = min(v["c"] for v in present.values())
            if cbest <= 0:
                continue
            solved = {m: v for m, v in present.items()
                      if v["c"] <= cbest * (1.0 + tau)}
            tref = min((v["t"] for v in solved.values()), default=0.0)
            if tref <= 0:
                continue
            timed, n = True, n + 1
            for m in methods:
                per[m].append(solved[m]["t"] / tref if m in solved
                              else float("inf"))
        if not timed:
            no_time.append(name)
    return per, n, dropped, no_time


def render_profile(per: dict, n: int, methods: list[str], out_stem: Path,
                   formats, title: str | None, tau: float):
    fig, ax = plt.subplots(figsize=(7.2, 4.75))
    fig.subplots_adjust(left=0.115, right=0.98, top=0.965, bottom=0.185)

    finite = [r for rs in per.values() for r in rs if r != float("inf")]
    kmax = max(finite) * 1.35
    for m in methods:
        rs = sorted(r for r in per[m] if r != float("inf"))
        if not rs:
            continue
        label, colour, marker = STYLE[m]
        # Step up by 1/n at each solved problem, then hold flat to the right
        # edge; a curve that stops below 1.0 failed the rest.
        xs = [1.0] + [r for r in rs for _ in (0, 1)] + [kmax]
        ys = [0.0, 0.0] + [i / n for i in range(1, len(rs)) for _ in (0, 1)] \
             + [len(rs) / n, len(rs) / n]
        ax.plot(xs, ys, color=colour, lw=2.0, solid_joinstyle="round",
                zorder=4 if m == "impl" else 3)
        # Sparse markers carry identity when the hues are hard to separate.
        idx = [max(0, round(f * (len(rs) - 1))) for f in
               (0.08, 0.28, 0.5, 0.72, 0.92)]
        ax.plot([rs[i] for i in idx], [(i + 1) / n for i in idx], ls="none",
                marker=marker, ms=6.5, color=colour, markeredgecolor="white",
                markeredgewidth=0.8, zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(1.0, kmax)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Time / fastest (×)")
    ax.set_ylabel(f"Fraction solved within {tau:.0%} of best cost")

    handles = [Line2D([], [], color=STYLE[m][1], lw=2.0, marker=STYLE[m][2],
                      ms=6.5, markeredgecolor="white", markeredgewidth=0.8,
                      label=STYLE[m][0]) for m in methods if per[m]]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.5, columnspacing=2.2)
    if title:
        fig.suptitle(title, fontsize=12.5, color=INK, y=0.99)

    for ext in formats:
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=220, facecolor="white", pad_inches=0.06)
        print(f"wrote {rel(path)}")
    plt.close(fig)


def write_profile_table(per: dict, n: int, methods: list[str], out_stem: Path,
                        tau: float):
    lines_csv = ["method,n_problems,frac_solved,frac_fastest,within_2x,"
                 "within_10x,median_ratio_when_solved"]
    lines_md = ["| method | solved | fastest | within 2× | within 10× | "
                "median ratio |", "|---|---|---|---|---|---|"]
    for m in methods:
        rs = per[m]
        if not rs:
            continue
        ok = [r for r in rs if r != float("inf")]
        frac = lambda k: sum(1 for r in rs if r <= k) / n
        med = statistics.median(ok) if ok else float("nan")
        lines_csv.append(f"{STYLE[m][0]},{n},{len(ok) / n:.4f},{frac(1.001):.4f},"
                         f"{frac(2):.4f},{frac(10):.4f},{med:.3f}")
        lines_md.append(
            f"| {STYLE[m][0]} | {len(ok) / n:.0%} | {frac(1.001):.0%} | "
            f"{frac(2):.0%} | {frac(10):.0%} | {med:.2f}× |")
    out_stem.with_suffix(".csv").write_text("\n".join(lines_csv) + "\n")
    out_stem.with_suffix(".md").write_text("\n".join(lines_md) + "\n")
    print(f"wrote {rel(out_stem.with_suffix('.csv'))}")
    print(f"wrote {rel(out_stem.with_suffix('.md'))}")
    print("\n".join(lines_md))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def render(pts: dict, methods: list[str], out_stem: Path, formats,
           title: str | None):
    fig, ax = plt.subplots(figsize=(7.2, 4.75))
    fig.subplots_adjust(left=0.125, right=0.98, top=0.965, bottom=0.185)

    drawn = []
    order = [m for m in methods if m != "impl"] + [m for m in methods
                                                   if m == "impl"]
    for z, m in enumerate(order):
        xy = [p[m] for p in pts.values() if m in p]
        if not xy:
            continue
        _, colour, marker = STYLE[m]
        xs, ys = zip(*xy)
        ax.scatter(xs, ys, s=46, marker=marker, c=colour, alpha=0.6,
                   linewidths=0.6, edgecolors="white", zorder=3 + z)
        drawn.append(m)
    drawn = [m for m in methods if m in drawn]

    allxy = [v for p in pts.values() for v in p.values()]
    xs, ys = zip(*allxy)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(xs) * 0.62, max(xs) * 1.7)
    ax.set_ylim(min(ys) * 0.7, max(ys) * 1.9)
    ax.set_xlabel(AXES[0])
    ax.set_ylabel(AXES[1])

    handles = [Line2D([], [], marker=STYLE[m][2], ls="none", markersize=7.5,
                      markerfacecolor=STYLE[m][1], markeredgecolor="white",
                      markeredgewidth=1.0, label=STYLE[m][0])
               for m in drawn]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.4, columnspacing=2.2)
    if title:
        fig.suptitle(title, fontsize=12.5, color=INK, y=0.99)

    for ext in formats:
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=220, facecolor="white", pad_inches=0.06)
        print(f"wrote {rel(path)}")
    plt.close(fig)


def write_table(pts: dict, rows: dict, methods: list[str], out_stem: Path):
    lines_csv = ["group,method,n_datasets,median_time_ratio,median_cost_ratio,"
                 "wins_time,wins_cost"]
    lines_md = ["| group | method | n | median time / fastest | "
                "median cost / best | fastest | best cost |",
                "|---|---|---|---|---|---|---|"]
    families = sorted({rows[n]["family"] for n in pts})
    for group in [None] + families:
        for m in methods:
            sel = [p[m] for n, p in pts.items() if m in p
                   and (group is None or rows[n]["family"] == group)]
            if not sel:
                continue
            xs, ys = zip(*sel)
            wt = sum(1 for x in xs if x < 1.001)
            wc = sum(1 for y in ys if y < 1.001)
            name = group or "All"
            lines_csv.append(f"{name},{STYLE[m][0]},{len(sel)},"
                             f"{statistics.median(xs):.3f},"
                             f"{statistics.median(ys):.4f},{wt},{wc}")
            lines_md.append(
                f"| {name} | {STYLE[m][0]} | {len(sel)} | "
                f"{statistics.median(xs):.2f} | {statistics.median(ys):.3f} | "
                f"{wt}/{len(sel)} | {wc}/{len(sel)} |")
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
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--out", type=Path, default=None,
                    help="output stem (default: analysis/standard_<form>_<device>)")
    ap.add_argument("--time", choices=["total", "solve"], default="total",
                    help="total = precompute + solve (default)")
    ap.add_argument("--min-cost", type=float, default=1e-6,
                    help="drop datasets whose optimum is below this; a cost "
                         "ratio is meaningless when the optimum is ~0")
    ap.add_argument("--form", choices=["profile", "pareto"], default="profile",
                    help="profile: Dolan-More performance profile (default); "
                         "pareto: the cost-vs-time scatter")
    ap.add_argument("--tau", type=float, default=0.01,
                    help="cost tolerance for counting a problem solved")
    ap.add_argument("--title", default=None)
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = ap.parse_args()

    stem = "standard_profile" if args.form == "profile" else "standard_pareto"
    out = args.out or (DATA_DIR / "analysis" / f"{stem}_{args.device}")
    methods = list(DEVICE_METHODS[args.device])

    rows = load(args.device, use_precompute=args.time == "total")
    if not rows:
        print(f"no {RESULT_FILE[args.device]} traces under {DATA_DIR}")
        return 2
    if "gtsam" in methods:
        n = merge_gtsam(rows)
        if n == 0:
            methods.remove("gtsam")
            print(f"note: no GTSAM baseline under {rel(GTSAM_DIR)} - "
                  "plotting the VarPro formulations only")

    apply_style()
    if args.form == "profile":
        per, n, dropped, no_time = profile_ratios(rows, methods, args.min_cost,
                                                  args.tau)
        print(f"{args.device.upper()}: {n} problems (dataset x init), "
              f"{len(dropped)} datasets dropped for a ~0 optimum "
              f"({', '.join(dropped)})")
        if no_time:
            print(f"note: dropped for sub-millisecond runtimes: "
                  f"{', '.join(no_time)}")
        render_profile(per, n, methods, out, args.formats, args.title, args.tau)
        write_profile_table(per, n, methods,
                            out.with_name(out.name + "_summary"), args.tau)
        return 0

    pts, dropped, no_time = normalize(rows, methods, args.min_cost)
    print(f"{args.device.upper()}: {len(pts)} datasets plotted, "
          f"{len(dropped)} dropped for a ~0 optimum ({', '.join(dropped)})")
    if no_time:
        print(f"note: dropped for sub-millisecond runtimes: "
              f"{', '.join(no_time)}")
    for m in methods:
        missing = [n for n in pts if m not in pts[n]]
        if missing:
            print(f"note: {STYLE[m][0]} missing on {', '.join(missing)}")

    render(pts, methods, out, args.formats, args.title)
    write_table(pts, rows, methods, out.with_name(out.name + "_summary"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
