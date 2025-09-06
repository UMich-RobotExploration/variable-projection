import os
import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# --- Paths ---
BASE_DIRS = [
    "/home/nikolas/variable-projection/examples/data",      # Explicit / Implicit / VarPro
    "/home/nikolas/variable-projection/examples/data_nik",  # GTSAM
]

EXP_SUBDIRS = [
   # "/raslam/factor_graph_small/results.json",
    "/raslam/single_drone/results.json",
    "/raslam/plaza2/results.json",
    #"/sfm/bal-392/results.json",
    # "/sfm/TUM-desk/results.json",
    # "/sfm/MipNerf-garden/results.json",
    # "/sfm/IMC-gate/results.json",
    # "/sfm/IMC-temple/results.json",
    # "/sfm/Replica-REPoffice0_100/results.json",
    # "/sfm/TUM-room/results.json",
    # "/sfm/Replica-REProom1_100/results.json",
    # "/sfm/MipNerf-room/results.json",
    # "/sfm/Replica-REProom0_100/results.json",
    # "/sfm/TUM-computer-R/results.json",
    # "/sfm/TUM-computer-T/results.json",
    "/sfm/bal-93/results.json",
    # "/sfm/Replica-REPoffice1_100/results.json",
  #  "/sfm/MipNerf-kitchen/results.json",
 #   "/pgo/results.json",
    "/snl/intel_snl/results.json",
    "/snl/parking-garage_snl/results.json",
    "/snl/grid3D_snl/results.json",
    "/snl/MIT_snl/results.json",
   # "/snl/smallGrid3D_snl/results.json",
   # "/snl/M3500_snl/results.json",
    "/snl/city10000_snl/results.json",
   # "/snl/tinyGrid3D_snl/results.json",
   # "/snl/torus3D_snl/results.json",
    "/snl/sphere2500_snl/results.json",
]

# --- Labeling / normalization ---
FORMULATION_MAP = {0: "Explicit", 1: "Explicit VarPro", 2: "Implicit"}
FORM_ORDER = ["Implicit", "Explicit", "Explicit VarPro", "GTSAM"]  # fixed legend order

# ---- at top-level (config) ----
MAX_ITERS = 100      # e.g., 150; None = no cap
MAX_TIME_S = 100     # e.g., 0.75; None = no cap

RANK_TARGET = "rank5"

def normalize_formulation(value, source_path: str) -> str:
    """Map int codes and strings to canonical labels. Default GTSAM for data_nik if missing."""
    if isinstance(value, int):
        return FORMULATION_MAP.get(value, f"Form {value}")
    if isinstance(value, str):
        s = value.strip().lower().replace("_", " ").replace("-", " ")
        if s == "gtsam":
            return "GTSAM"
        if s == "explicit":
            return "Explicit"
        if s in {"explicit varpro", "explicit var pro", "varpro"}:
            return "Explicit VarPro"
        if s == "implicit":
            return "Implicit"
        return value
    if "data_nik" in source_path:
        return "GTSAM"
    return "Unknown"

RANK_PATTERN = re.compile(r"rank(\d+)", re.IGNORECASE)
def extract_rank(init_file: str) -> str:
    m = RANK_PATTERN.search(os.path.basename(init_file or ""))
    return f"rank{m.group(1)}" if m else "rank?"

def get_entry_rank(e, default=RANK_TARGET):
    """Resolve rank robustly (init_file -> rank field -> GTSAM fallback)."""
    r = extract_rank(e.get("init_file", ""))
    if r != "rank?":
        return r
    cand = e.get("rank")  # sometimes present as 5 or "rank5"
    if cand is not None:
        m = RANK_PATTERN.search(str(cand))
        if m:
            return f"rank{m.group(1)}"
    # fallback: many GTSAM logs omit rank in filename
    form_guess = normalize_formulation(e.get("formulation"), e.get("_source_path", ""))
    if form_guess == "GTSAM":
        return default
    return "rank?"

# Fixed colors for the four formulations
_palette = plt.get_cmap("tab10")
COLORS_FIXED = {
    "Implicit": _palette(0),
    "Explicit": _palette(1),
    "Explicit VarPro": _palette(2),
    "GTSAM": _palette(3),
}

def visualize_dataset(dataset_name: str, entries: list):
    """
    entries: list of dicts each with keys:
      - 'formulation' (int or str), 'init_file', 'times' (list), 'costs' (list)
    Only plots rank5, and only the four formulations in FORM_ORDER.
    Aggregates multiple runs (min/max band + median) and uses cumulative time.
    """
    if "bal-93" not in dataset_name:
        print(dataset_name)
        return
    groups = defaultdict(list)  # {formulation: [(cum_times, costs), ...]}
    for e in entries:
        form = normalize_formulation(e.get("formulation"), e.get("_source_path", ""))
        if form not in FORM_ORDER:
            continue
        rank = get_entry_rank(e, default=RANK_TARGET)
        if RANK_TARGET and rank != RANK_TARGET:
            print(dataset_name)
            print(form)
            print(rank)
            print()
            continue
        print(dataset_name)
        print(form)
        print(rank)
        print()

        times = np.array(e.get("times", []), dtype=float)
        costs = np.array(e.get("costs", []), dtype=float)
        if len(times) == 0 or len(costs) == 0:
            continue

        # Convert per-iterate durations -> cumulative time
        L0 = min(len(times), len(costs))
        cum_times_full = np.cumsum(times[:L0])

        # Apply caps
        L = L0
        if MAX_ITERS is not None:
            L = min(L, int(MAX_ITERS))
        if MAX_TIME_S is not None:
            idx = int(np.searchsorted(cum_times_full, float(MAX_TIME_S), side="right"))
            L = min(L, idx)

        if L <= 0:
            continue

        cum_times = cum_times_full[:L]
        groups[form].append((cum_times, costs[:L]))


    if not groups:
        print(f"[warn] No rank5 runs for dataset '{dataset_name}'.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 6.5))

    for form in FORM_ORDER:
        if form not in groups:
            continue
        runs = groups[form]

        # Align by iteration index: truncate to shared min length
        min_len = min(c.shape[0] for (_, c) in runs)
        cum_t_stack = np.array([t[:min_len] for (t, _) in runs])   # cumulative times per run
        costs_stack = np.array([c[:min_len] for (_, c) in runs])

        # Use the median cumulative time per iteration as x-axis
        times_x = np.median(cum_t_stack, axis=0)

        min_cost = costs_stack.min(axis=0)
        max_cost = costs_stack.max(axis=0)
        median_cost = np.median(costs_stack, axis=0)

        color = COLORS_FIXED[form]
        label = f"{form} ({RANK_TARGET})"

        # Left: costs vs iterations
        axs[0].fill_between(range(min_len), min_cost, max_cost, color=color, alpha=0.18)
        axs[0].plot(range(min_len), median_cost, color=color, label=label)

        # Right: costs vs cumulative time
        axs[1].fill_between(times_x, min_cost, max_cost, color=color, alpha=0.18)
        axs[1].plot(times_x, median_cost, color=color, label=label)

        # Add endpoint markers (iterations axis)
        axs[0].scatter(
            [0, min_len - 1],
            [median_cost[0], median_cost[-1]],
            marker='o',edgecolors=color, linewidths=1,
            zorder=4, label='_nolegend_'
        )

        # Add endpoint markers (cumulative time axis)
        axs[1].scatter(
            [times_x[0], times_x[-1]],
            [median_cost[0], median_cost[-1]],
            marker='o', edgecolors=color, linewidths=1,
            zorder=4, label='_nolegend_'
)

    # Axes formatting
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Cost")
    axs[0].set_yscale("log")
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    axs[0].set_title("Costs vs Iterations (rank5)")
    axs[0].text(0.5, 1.05, f"Dataset: {dataset_name}", fontsize=10, ha='center', transform=axs[0].transAxes)

    axs[1].set_xlabel("Cumulative time (s)")
    axs[1].set_ylabel("Cost")
    axs[1].set_yscale("log")
    axs[1].grid(True, which="both", ls="--", alpha=0.5)
    axs[1].set_title("Costs vs Cumulative Time (rank5)")
    axs[1].text(0.5, 1.05, f"Dataset: {dataset_name}", fontsize=10, ha='center', transform=axs[1].transAxes)

    # Shared legend (dedup & ordered)
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles += h; labels += l
    if handles:
        seen = set()
        ordered = []
        for form in FORM_ORDER:
            tag = f"{form} ({RANK_TARGET})"
            for h, l in zip(handles, labels):
                if l == tag and l not in seen:
                    ordered.append((h, l))
                    seen.add(l)
                    break
        axs[1].legend([h for h, _ in ordered], [l for _, l in ordered], loc="best")

    plt.tight_layout()
    plt.show()

def load_entries_from_file(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[error] Failed to parse {path}: {exc}")
        return []
    if not isinstance(data, list):
        print(f"[warn] {path} did not contain a list; skipping.")
        return []
    for e in data:
        if isinstance(e, dict):
            e["_source_path"] = path
    return data
# --- Helpers: capped cumulative time and aggregation ---

from matplotlib.patches import Rectangle

def _median_tick(ax, x, y, *, x_width=0.0, y_frac=0.00, color="k", z=6):
    # thicker in Y, thinner in X
    y1, y2 = ax.get_ylim()
    h = max((y2 - y1) * y_frac, 1e-12)
    ax.add_patch(Rectangle((x - x_width/2, y - h/2), x_width, h,
                           facecolor=color, edgecolor="none", zorder=z))
# --- One figure per task: grouped solid boxes (legend = formulations) ---
from matplotlib.patches import Patch

def visualize_task_runtime_shadedbox(task_name: str,
                                     dataset_to_entries: dict,
                                     *,
                                     rank_target=RANK_TARGET,
                                     y_log=False,
                                     group_width=0.82,   # total width of each dataset "cluster"
                                     bar_alpha=0.18):
    """
    Build a single figure for the given task (PGO / RA-SLAM / SNL / SFM):
      X-axis: datasets in that task
      Grouped per dataset: colored solid box for each formulation (with a subtle bar to median underneath)
      Legend: formulations
    """
    # 1) Collect datasets for this task + per-dataset times per formulation
    ds_names = []
    ds_times_by_form = []  # list of dicts: {form: [times]}

    for ds, entries in dataset_to_entries.items():
        if detect_task_for_entries(entries, ds) != task_name:
            continue
        times_dict = cumtimes_by_form(entries, rank_target=rank_target)
        if any(len(v) > 0 for v in times_dict.values()):
            ds_names.append(ds)
            ds_times_by_form.append(times_dict)

    if not ds_names:
        print(f"[warn] No datasets found for task '{task_name}'.")
        return

    # Forms we’ll actually show (those present anywhere across datasets)
    forms_present = [f for f in FORM_ORDER if any(f in d and len(d[f])>0 for d in ds_times_by_form)]
    if not forms_present:
        print(f"[warn] No formulations with data for task '{task_name}'.")
        return

    # 2) Layout: grouped positions per dataset with per-form offsets for formulations
    nD = len(ds_names)
    nF = len(forms_present)
    x_centers = np.arange(nD)  # cluster centers for datasets

    delta = group_width / max(nF, 1)
    # symmetric offsets around 0
    offsets = [(-group_width/2) + (i + 0.5)*delta for i in range(nF)]
    bar_width = delta * 0.70
    box_width = delta * 0.85

    fig, ax = plt.subplots(figsize=(max(9.0, 1.2*nD), 4.8), dpi=150)

    # 3) Draw per-formulation, across datasets: first bar-to-median, then solid boxes
    legend_handles = []
    for f_idx, form in enumerate(forms_present):
        color = COLORS_FIXED.get(form, None)
        # Gather data & positions for datasets where this form exists
        positions, data = [], []
        med_positions, med_values = [], []

        for di, times_dict in enumerate(ds_times_by_form):
            vals = times_dict.get(form, [])
            if not vals:
                continue
            pos = x_centers[di] + offsets[f_idx]
            positions.append(pos)
            data.append(np.asarray(vals, dtype=float))
            med_values.append(float(np.median(vals)))
            med_positions.append(pos)

        if not data:
            continue

        # subtle bars to medians (behind boxes)
        for pos, med in zip(med_positions, med_values):
            ax.bar(pos, med, width=bar_width, color=color, alpha=bar_alpha, linewidth=0, zorder=1)

        # solid colored boxes (IQR+whiskers)
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,    # keep clean; flip to True if you want outliers
            whis=1.5,
            zorder=3
        )
        for i in range(len(positions)):
            bp["boxes"][i].set_facecolor(color)
            bp["boxes"][i].set_edgecolor(color)
            bp["boxes"][i].set_linewidth(1.4)
            # hide the internal black median line
            bp["medians"][i].set_visible(False)
            # neutral whiskers/caps
            bp["whiskers"][2*i  ].set_color("#666"); bp["whiskers"][2*i  ].set_linewidth(1.0)
            bp["whiskers"][2*i+1].set_color("#666"); bp["whiskers"][2*i+1].set_linewidth(1.0)
            bp["caps"][2*i  ].set_color("#666");     bp["caps"][2*i  ].set_linewidth(1.0)
            bp["caps"][2*i+1].set_color("#666");     bp["caps"][2*i+1].set_linewidth(1.0)

        # legend item for this formulation
        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=form))

    # 4) Cosmetics
    ax.set_xticks(x_centers)
    ax.set_xticklabels(ds_names, rotation=20, ha="right")
    ax.set_ylabel("Cumulative time (s)")
    if y_log:
        ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", ls="--", lw=0.6, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    cap_iters = MAX_ITERS if MAX_ITERS is not None else "∞"
    cap_time  = MAX_TIME_S if MAX_TIME_S is not None else "∞"
    ax.set_title(f"Runtime by Dataset — {task_name} ({rank_target})\n"
                 f"caps: iters={cap_iters}, time={cap_time}s", pad=8)

    # Legend = formulations
    if legend_handles:
        ax.legend(handles=legend_handles, title="Formulation", frameon=False, loc="upper right", ncol=1)

    fig.tight_layout()
    plt.show()


# --- Convenience wrapper to render figures for multiple tasks ---
def visualize_all_tasks_runtime_shadedbox(dataset_to_entries: dict,
                                          tasks=("PGO","RA-SLAM","SNL","SFM"),
                                          rank_target=RANK_TARGET,
                                          y_log=False):
    for task in tasks:
        visualize_task_runtime_shadedbox(task, dataset_to_entries,
                                         rank_target=rank_target, y_log=y_log)




def cumtimes_by_form(entries, rank_target=RANK_TARGET):
    out = defaultdict(list)
    for e in entries:
        form = normalize_formulation(e.get("formulation"), e.get("_source_path", ""))
        if form not in FORM_ORDER:
            continue

        # parse rank, but fall back for GTSAM (many GTSAM JSONs lack 'init_file' rank)
        rank = extract_rank(e.get("init_file", ""))
        if rank == "rank?" and form == "GTSAM":
            rank = rank_target  # assume it's the same target rank you’re plotting

        if rank_target and rank != rank_target:
            continue

        tval = capped_cumulative_time(e.get("times", []), e.get("costs", []))
        if tval is not None and np.isfinite(tval):
            out[form].append(float(tval))
    return out


# --- Task detection (simple, path- and name-based) ---
def classify_task_from_path(path: str, dataset_name: str) -> str:
    p = (path or "").lower()
    d = (dataset_name or "").lower()
    if "/pgo/" in p or d.endswith("_pgo") or "pgo" in d:   return "PGO"
    if "/raslam/" in p or "raslam" in p or "raslam" in d:  return "RA-SLAM"
    if "/snl/" in p or d.endswith("_snl") or "snl" in d:   return "SNL"
    if "/sfm/" in p or "sfm" in d:                         return "SFM"
    return "UNKNOWN"

def detect_task_for_entries(entries: list, dataset_name: str) -> str:
    # Pick the first non-UNKNOWN classification seen across entry source paths.
    for e in entries:
        t = classify_task_from_path(e.get("_source_path",""), dataset_name)
        if t != "UNKNOWN":
            return t
    return "UNKNOWN"


# ---------- CSV Export Helpers (drop-in) ----------
import csv
import numpy as np

FORMS_CANON = ["Implicit", "Explicit", "Explicit VarPro", "GTSAM"]

def classify_task_from_path(path: str, dataset_name: str) -> str:
    """Classify into PGO | RA-SLAM | SNL | SFM from source path / dataset name."""
    p = (path or "").lower()
    d = (dataset_name or "").lower()
    if "/pgo/" in p or d.endswith("_pgo") or "pgo" in d:
        return "PGO"
    if "/raslam/" in p or "raslam" in p or "raslam" in d:
        return "RA-SLAM"
    if "/snl/" in p or d.endswith("_snl") or "snl" in d:
        return "SNL"
    if "/sfm/" in p or "sfm" in d:
        return "SFM"
    return "UNKNOWN"

def capped_cumulative_time(times, costs, max_iters=MAX_ITERS, max_time_s=MAX_TIME_S):
    """Return final cumulative time after applying caps. None if not computable."""
    if times is None or costs is None:
        return None
    t = np.asarray(times, dtype=float)
    c = np.asarray(costs, dtype=float)
    L0 = int(min(t.shape[0], c.shape[0]))
    if L0 == 0:
        return None
    cum = np.cumsum(t[:L0])
    L = L0
    if max_iters is not None:
        L = min(L, int(max_iters))
    if max_time_s is not None:
        idx = int(np.searchsorted(cum, float(max_time_s), side="right"))
        L = min(L, idx)
    if L <= 0:
        return None
    return float(cum[L - 1])

FORMS_CANON = ["Implicit", "Explicit", "Explicit VarPro", "GTSAM"]

def classify_task_from_path(path: str, dataset_name: str) -> str:
    p = (path or "").lower()
    d = (dataset_name or "").lower()
    if "/pgo/" in p or d.endswith("_pgo") or "pgo" in d: return "PGO"
    if "/raslam/" in p or "raslam" in p or "raslam" in d: return "RA-SLAM"
    if "/snl/" in p or d.endswith("_snl") or "snl" in d: return "SNL"
    if "/sfm/" in p or "sfm" in d: return "SFM"
    return "UNKNOWN"

def capped_cumulative_time_and_iters(times, costs, max_iters=MAX_ITERS, max_time_s=MAX_TIME_S):
    """Return (final cumulative time, capped iteration count). None if not computable."""
    if times is None or costs is None:
        return None, None
    t = np.asarray(times, dtype=float)
    c = np.asarray(costs, dtype=float)
    L0 = int(min(t.shape[0], c.shape[0]))
    if L0 == 0:
        return None, None
    cum = np.cumsum(t[:L0])
    L = L0
    if max_iters is not None:
        L = min(L, int(max_iters))
    if max_time_s is not None:
        idx = int(np.searchsorted(cum, float(max_time_s), side="right"))
        L = min(L, idx)
    if L <= 0:
        return None, None
    return float(cum[L - 1]), int(L)

def summarize_cumtimes(dataset_to_entries: dict,
                       rank_filter: str | None = None,
                       include_unknown_task: bool = False):
    """
    Aggregate capped cumulative time + iteration counts per (task, dataset, rank, formulation).
    Returns list of rows with time stats and median iterations.
    """
    groups = {}  # key -> {"times": [...], "iters":[...]}
    for ds, entries in dataset_to_entries.items():
        for e in entries:
            src  = e.get("_source_path", "")
            task = classify_task_from_path(src, ds)
            if (not include_unknown_task) and task == "UNKNOWN":
                continue
            form = normalize_formulation(e.get("formulation"), src)
            if form not in FORMS_CANON:
                continue
            rank = extract_rank(e.get("init_file", ""))
            if rank_filter and rank != rank_filter:
                continue

            tt, iters = capped_cumulative_time_and_iters(e.get("times", []), e.get("costs", []))
            if tt is None or not np.isfinite(tt):
                continue

            key = (task, ds, rank, form)
            bucket = groups.setdefault(key, {"times": [], "iters": []})
            bucket["times"].append(tt)
            if iters is not None:
                bucket["iters"].append(iters)

    rows = []
    for (task, ds, rank, form), g in groups.items():
        at = np.asarray(g["times"], dtype=float)
        ai = np.asarray(g["iters"], dtype=float) if g["iters"] else np.array([], dtype=float)
        rows.append({
            "task": task,
            "dataset": ds,
            "rank": rank,
            "formulation": form,
            # time stats
            "time_min_s": float(np.min(at)),
            "time_q25_s": float(np.percentile(at, 25)),
            "time_median_s": float(np.median(at)),
            "time_q75_s": float(np.percentile(at, 75)),
            "time_max_s": float(np.max(at)),
            # iterations (median across runs; blank if none)
            "iters_median": (int(np.median(ai)) if ai.size else ""),
            # caps
            "cap_iters": MAX_ITERS if MAX_ITERS is not None else float("inf"),
            "cap_time_s": MAX_TIME_S if MAX_TIME_S is not None else float("inf"),
        })
    rows.sort(key=lambda r: (r["task"], r["dataset"], r["rank"], FORMS_CANON.index(r["formulation"])))
    return rows

def write_csv_long(rows: list, out_path: str):
    """One row per (task,dataset,rank,formulation) with time stats + median iterations."""
    header = ["task","dataset","rank","formulation",
              "time_min_s","time_q25_s","time_median_s","time_q75_s","time_max_s",
              "iters_median","cap_iters","cap_time_s"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_csv_wide(rows: list, out_path: str):
    """
    One row per (task,dataset,rank) with per-formulation median time AND median iterations.
    Also includes Explicit vs Implicit improvement columns:
      - Explicit_vs_Implicit_impr_pct  = 100 * (Implicit - Explicit) / Implicit
      - Explicit_vs_Implicit_speedup_x = Implicit / Explicit
    """
    # group rows by (task,dataset,rank) -> {form: row}
    by_hdr = {}
    for r in rows:
        key = (r["task"], r["dataset"], r["rank"])
        by_hdr.setdefault(key, {})[r["formulation"]] = r

    header = ["task","dataset","rank"]
    for f in FORMS_CANON:
        header.append(f"{f}_time_median_s")
    for f in FORMS_CANON:
        header.append(f"{f}_iters_median")
    # rename the comparison to Implicit vs Explicit (Implicit better => positive)
    header += ["Implicit_vs_Explicit_impr_pct", "Implicit_vs_Explicit_speedup_x"]


    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for (task, ds, rank), fmap in sorted(by_hdr.items(), key=lambda k: k[0]):
            row = {"task": task, "dataset": ds, "rank": rank}
            # per-formulation medians
            for f in FORMS_CANON:
                row[f"{f}_time_median_s"] = fmap.get(f, {}).get("time_median_s", "")
            for f in FORMS_CANON:
                row[f"{f}_iters_median"] = fmap.get(f, {}).get("iters_median", "")
            # improvement Explicit vs Implicit
            imp = fmap.get("Implicit", {})
            exp = fmap.get("Explicit", {})
            t_imp = imp.get("time_median_s", None)
            t_exp = exp.get("time_median_s", None)
            if isinstance(t_imp, (int, float)) and isinstance(t_exp, (int, float)) and t_exp > 0 and t_imp > 0:
                # Baseline = Explicit
                # % improvement if you use Implicit instead of Explicit
                impr_pct = 100.0 * (t_exp - t_imp) / t_exp
                speedup  = t_exp / t_imp
                row["Implicit_vs_Explicit_impr_pct"]  = impr_pct
                row["Implicit_vs_Explicit_speedup_x"] = speedup
            else:
                row["Implicit_vs_Explicit_impr_pct"]  = ""
                row["Implicit_vs_Explicit_speedup_x"] = ""

def export_cumtime_csv(dataset_to_entries: dict,
                       long_out: str = "cumtime_summary_long.csv",
                       wide_out: str = "cumtime_summary_wide.csv",
                       rank_filter: str | None = None):
    rows = summarize_cumtimes(dataset_to_entries, rank_filter=rank_filter)
    write_csv_long(rows, long_out)
    write_csv_wide(rows, wide_out)
    print(f"[csv] Wrote:\n  - {long_out}\n  - {wide_out}")
# ---------- end CSV Export Helpers ----------




if __name__ == "__main__":
    dataset_to_entries = defaultdict(list)

    for sub in EXP_SUBDIRS:
        for base in BASE_DIRS:
            full = os.path.join(base, sub.lstrip("/"))
            for e in load_entries_from_file(full):
                ds_name = e.get("dataset_name") or os.path.basename(os.path.dirname(full))
                dataset_to_entries[ds_name].append(e)

    for ds, entries in dataset_to_entries.items():
        visualize_dataset(ds, entries)
    visualize_all_tasks_runtime_shadedbox(
    dataset_to_entries,
    tasks=("RA-SLAM","SNL","PGO","SFM"),   # choose the ones you want
    rank_target=RANK_TARGET,               # e.g., "rank5"
    y_log=True                           # or True if ranges span orders of magnitude
)

    export_cumtime_csv(
    dataset_to_entries,
    long_out="cumtime_summary_long.csv",
    wide_out="cumtime_summary_wide.csv",
    rank_filter="rank5")   
