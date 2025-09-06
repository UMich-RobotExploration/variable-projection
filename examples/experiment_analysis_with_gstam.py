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
    # "/sfm/bal-392/results.json",
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
    # "/sfm/bal-93/results.json",
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
    if "sfm" in dataset_name:
        return
    groups = defaultdict(list)  # {formulation: [(cum_times, costs), ...]}
    for e in entries:
        form = normalize_formulation(e.get("formulation"), e.get("_source_path", ""))
        if form not in FORM_ORDER:
            continue
        rank = extract_rank(e.get("init_file", ""))
        if rank != RANK_TARGET:
            continue

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
