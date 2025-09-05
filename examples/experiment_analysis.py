import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

BASE_DATA_DIR = "/home/alan/variable-projection/examples/data"
EXP_SUBDIRS = [
    "/raslam/factor_graph_small/results.json",
    "/raslam/single_drone/results.json",
    "/raslam/plaza2/results.json",
    "/sfm/bal-392/results.json",
    "/sfm/TUM-desk/results.json",
    "/sfm/MipNerf-garden/results.json",
    "/sfm/IMC-gate/results.json",
    "/sfm/IMC-temple/results.json",
    "/sfm/Replica-REPoffice0_100/results.json",
    "/sfm/TUM-room/results.json",
    "/sfm/Replica-REProom1_100/results.json",
    "/sfm/MipNerf-room/results.json",
    "/sfm/Replica-REProom0_100/results.json",
    "/sfm/TUM-computer-R/results.json",
    "/sfm/TUM-computer-T/results.json",
    "/sfm/bal-93/results.json",
    "/sfm/Replica-REPoffice1_100/results.json",
    "/sfm/MipNerf-kitchen/results.json",
    "/pgo/results.json",
    "/snl/intel_snl/results.json",
    "/snl/parking-garage_snl/results.json",
    "/snl/grid3D_snl/results.json",
    "/snl/MIT_snl/results.json",
    "/snl/smallGrid3D_snl/results.json",
    "/snl/M3500_snl/results.json",
    "/snl/city10000_snl/results.json",
    "/snl/tinyGrid3D_snl/results.json",
    "/snl/torus3D_snl/results.json",
    "/snl/sphere2500_snl/results.json",
]

COLOR_SCHEME = plt.get_cmap("tab10")
COLOR_KEYS = [(rank, form) for rank in ["rank3", "rank4", "rank5"] for form in ["Explicit", "Explicit VarPro", "Implicit"]]
COLORS = {key: COLOR_SCHEME(i) for i, key in enumerate(COLOR_KEYS)}


def visualize_data(data_fpath: str):
    if "sfm" not in data_fpath:
        return

    with open(data_fpath, "r") as f:
        data = json.load(f)

    # Group by (formulation, rank)
    groups = defaultdict(list)

    FORMULATION_MAP = {0: "Explicit", 1: "Explicit VarPro", 2: "Implicit"}

    for entry in data:
        formulation = FORMULATION_MAP[entry["formulation"]]
        # Extract rank from filename, e.g. ".../rank2_init1.txt"
        match = re.search(r"rank(\d+)", entry["init_file"])
        rank = f"rank{match.group(1)}" if match else "unknown"

        times = np.array(entry["times"])
        costs = np.array(entry["costs"])
        groups[(formulation, rank)].append((times, costs))

    # Plot
    plt.figure(figsize=(10, 6))

    for i, ((formulation, rank), runs) in enumerate(groups.items()):
        rank_num = int(rank.replace("rank", ""))
        if rank_num != 5:
            continue

        # Pad runs to equal length by truncating to min length
        min_len = min(len(costs) for _, costs in runs)
        times_stack = np.array([t[:min_len] for t, _ in runs])
        costs_stack = np.array([c[:min_len] for _, c in runs])

        # Assume times are consistent across runs (you mentioned times list is per-iterate)
        print("WARNING: Assuming times are consistent across runs -- should test this!")
        times = times_stack[0]

        # Compute min/max across runs
        min_cost = costs_stack.min(axis=0)
        max_cost = costs_stack.max(axis=0)

        # Compute median cost across runs
        median_cost = np.median(costs_stack, axis=0)

        # Plot shaded area
        color = COLORS[(rank, formulation)]
        label = f"{rank} ({formulation})"
        plt.fill_between(times, min_cost, max_cost, color=color, alpha=0.2)
        plt.plot(times, median_cost, color=color, label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("Cost")
    plt.yscale("log")  # log scale often useful for optimization costs
    plt.legend()
    plt.title("Solver Costs vs Time")
    # subtitle with dataset name
    dataset_name = data_fpath.split("/")[-2]
    plt.suptitle(f"Dataset: {dataset_name}", fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for subdir in EXP_SUBDIRS:
        data_fpath = BASE_DATA_DIR + subdir
        visualize_data(data_fpath)