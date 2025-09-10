import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

BASE_VARPRO_DATA_DIR = "/home/alan/variable-projection/examples/data"
BASE_GTSAM_DATA_DIR = "/home/alan/variable-projection/examples/data_nik"
EXP_SUBDIRS = [
    # "/raslam/factor_graph_small/results.json",
    # "/raslam/mrclam/mrclam2/results.json",
    # "/raslam/mrclam/mrclam4/results.json",
    # "/raslam/mrclam/mrclam6/results.json",
    # "/raslam/mrclam/mrclam7/results.json",
    # "/raslam/single_drone/results.json",
    # "/raslam/plaza2/results.json",
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
    "/sfm/bal-392/results.json",
    "/sfm/bal-1934/results.json",
    # "/sfm/Replica-REPoffice1_100/results.json",
    # "/sfm/MipNerf-kitchen/results.json",
    # "/pgo/results.json",
    # "/snl/intel_snl/results.json",
    # "/snl/parking-garage_snl/results.json",
    # "/snl/grid3D_snl/results.json",
    # "/snl/MIT_snl/results.json",
    # "/snl/smallGrid3D_snl/results.json",
    # "/snl/M3500_snl/results.json",
    # "/snl/city10000_snl/results.json",
    # "/snl/tinyGrid3D_snl/results.json",
    # "/snl/torus3D_snl/results.json",
    # "/snl/sphere2500_snl/results.json",
]

# Define a color scheme for the plots using a standard matplotlib colormap.
COLOR_SCHEME = plt.get_cmap("tab10")
num_colors = COLOR_SCHEME.N
# Create a list of keys to assign unique colors to each combination of rank and formulation.
COLOR_KEYS = [(rank, form) for rank in ["rank5"] for form in ["Explicit", "Explicit VarPro", "Implicit", "GTSAM"]]
# Generate a dictionary mapping each key to a color from the color scheme.
COLORS = {key: COLOR_SCHEME(i % num_colors) for i, key in enumerate(COLOR_KEYS)}

# Optional fallback colors
def color_for(key):
    """
    Retrieves a color for a given (formulation, rank) key.
    This function provides a consistent color for each data series in the plots.
    """
    # key is (formulation, rank)
    try :
        return COLORS[key[1], key[0]]
    except KeyError:
        # If the key is not in the pre-defined color map, use a fallback color.
        base = {
            "Explicit": "#1f77b4",
            "Explicit VarPro": "#ff7f0e",
            "Implicit": "#2ca02c",
            "GTSAM": "#d62728",
        }
        return base.get(key[0], "#9467bd")

def _ensure_strictly_increasing(t):
    """
    Ensures that the time values are strictly increasing.
    This is a preprocessing step to prevent issues with interpolation,
    which requires monotonically increasing sample points.
    Duplicate time values are adjusted by a small epsilon.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("times must be 1-D")
    eps = np.finfo(float).eps
    for i in range(1, t.size):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + max(1e-12, abs(t[i-1]) * 1e-12 + eps)
    return t

def get_groups_from_data(data_fpath: str) -> dict:
    """
    Loads experimental data from a JSON file and groups it by formulation and rank.
    Each group contains a list of runs, where each run is a tuple of (times, costs).
    """
    try:
        with open(data_fpath, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {data_fpath}")
        return {}

    groups = defaultdict(list)
    FORMULATION_MAP = {0: "Explicit", 1: "Explicit VarPro", 2: "Implicit", "gtsam": "GTSAM"}

    for entry in data:
        form_raw = entry.get("formulation")
        formulation = FORMULATION_MAP.get(form_raw, f"Unknown({form_raw})")

        init_file = entry.get("init_file", "")
        m = re.search(r"rank(\d+)", init_file)
        rank = f"rank{m.group(1)}" if m else "unknown"

        times = np.asarray(entry.get("times", []), dtype=float)
        costs = np.asarray(entry.get("costs", []), dtype=float)

        # Ensure times and costs arrays have the same length.
        L = min(times.size, costs.size)
        times, costs = times[:L], costs[:L]

        # For GTSAM data, the reported times are per-iteration, so we compute a cumulative sum.
        if formulation == "GTSAM" and times.size > 0:
            times = np.cumsum(times)
            # Prepend a t=0 sample to align with other formulations that start at iteration 0.
            times = np.concatenate(([0.0], times))
            costs = np.concatenate(([costs[0]], costs))

        # Sort data points by time and ensure time is strictly increasing for interpolation.
        if times.size > 1:
            order = np.argsort(times)
            times, costs = times[order], costs[order]
            times = _ensure_strictly_increasing(times)

        if times.size and costs.size:
            groups[(formulation, rank)].append((times, costs))

    return groups

def _common_time_grid(runs, n_points=400, mode="linspace"):
    """
    Constructs a common time grid for a set of runs.
    This grid is used to interpolate the cost data from different runs
    so they can be aggregated (e.g., to compute median and percentiles).
    The grid spans the time interval where all runs overlap.
    """
    tmins = [t[0] for (t, _) in runs if t.size]
    tmaxs = [t[-1] for (t, _) in runs if t.size]
    if not tmins or not tmaxs:
        return None
    t0 = max(tmins) # The latest start time among all runs.
    t1 = min(tmaxs) # The earliest end time among all runs.
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return None
    if mode == "linspace":
        # Create a linearly spaced grid of points.
        return np.linspace(t0, t1, n_points)
    elif mode == "union":
        # Create a grid from the union of all timestamps within the overlap interval.
        T = np.unique(np.concatenate([t[(t >= t0) & (t <= t1)] for (t, _) in runs]))
        # Downsample if the number of points is too large.
        if T.size > 2000:
            idx = np.linspace(0, T.size - 1, 2000).round().astype(int)
            T = T[idx]
        return T
    else:
        raise ValueError("Unknown grid mode")

def _interp_run_to_grid(times, costs, grid):
    """
    Interpolates the cost data of a single run onto a common time grid.
    This allows for direct comparison and aggregation of costs across different runs at the same time points.
    """
    if times.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    # Ensure times and costs have the same length.
    L = min(times.size, costs.size)
    times, costs = times[:L], costs[:L]
    times = _ensure_strictly_increasing(times)
    # np.interp performs linear interpolation.
    return np.interp(grid, times, costs)

def visualize_data(varpro_data_fpath: str, gtsam_data_fpath: str = ""):
    """
    Main function to generate and display plots comparing solver performance.
    It creates two subplots:
    1. Cost vs. Iterations: Shows how the cost function value evolves with each solver iteration.
    2. Cost vs. Time: Shows how the cost function value evolves over wall-clock time.
    """
    # Load and group data from VarPro and (optionally) GTSAM result files.
    group_varpro = get_groups_from_data(varpro_data_fpath)
    groups = dict(group_varpro)
    if gtsam_data_fpath:
        groups.update(get_groups_from_data(gtsam_data_fpath))

    # If no data is available, exit early.
    if not groups:
        print(f"No data to visualize for {varpro_data_fpath} and {gtsam_data_fpath}.")
        return

    # Create a figure with two subplots, side-by-side.
    fig, axs = plt.subplots(1, 2, figsize=(11, 6))

    def get_med_upper_lower(arr, q_lo=None, q_hi=None):
        """
        Computes the median and specified percentiles of an array along the first axis.
        This is used to summarize the performance across multiple runs.
        """
        med = np.median(arr, axis=0)
        lo = np.min(arr, axis=0)
        hi = np.max(arr, axis=0)

        if q_lo:
            lo = np.nanpercentile(arr, q_lo, axis=0)
        if q_hi:
            hi = np.nanpercentile(arr, q_hi, axis=0)
        return med, lo, hi

    # Iterate over each group of runs (grouped by formulation and rank).
    for (formulation, rank), runs in groups.items():
        # --- Data Filtering ---
        # Example filter: only process runs for a specific rank (e.g., rank 5).
        try:
            rank_num = int(str(rank).replace("rank", ""))
        except ValueError:
            continue
        if rank_num != 5:
            continue
        if not runs:
            continue

        # --- Panel 1: Cost vs. Iterations ---
        # This plot shows the convergence of the solver in terms of iterations.
        # It helps to understand the algorithmic efficiency, independent of hardware.

        # To compare runs, they are aligned by iteration count.
        # All runs are truncated to the length of the shortest run.
        min_iter_len = min(min(len(t), len(c)) for t, c in runs)
        if min_iter_len < 2:
            continue
        # Stack costs from all runs for this group into a 2D numpy array.
        costs_iter = np.stack([c[:min_iter_len] for _, c in runs], axis=0)
        iters = np.arange(min_iter_len)

        # --- Data Presented: Median and Percentiles of Cost per Iteration ---
        # Instead of plotting every run, we plot the median cost at each iteration,
        # with a shaded region representing the upper and lower bounds. Can also do percentiles.
        # This gives a statistical summary of the performance across multiple runs.
        c_med_iter, c_lo_iter, c_hi_iter = get_med_upper_lower(costs_iter)


        # --- Plotting the Data for Panel 1 ---
        color = color_for((formulation, rank))
        label = f"{rank} ({formulation})"
        # The shaded area represents the variability of the cost.
        axs[0].fill_between(iters, c_lo_iter, c_hi_iter, alpha=0.18, label=None, color=color)
        # The solid line is the median cost over all runs.
        axs[0].plot(iters, c_med_iter, label=label, color=color)

        # --- Panel 2: Cost vs. Time ---
        # This plot shows the convergence of the solver in terms of wall-clock time.
        # It provides a practical measure of performance.

        # Create a common time grid for interpolation.
        grid = _common_time_grid(runs, n_points=500, mode="linspace")
        if grid is None:
            # If runs do not overlap in time, plot them individually.
            for (t, c) in runs:
                axs[1].plot(t, c, alpha=0.35, color=color)
        else:
            # Interpolate each run's cost data onto the common time grid.
            Cs = np.stack([_interp_run_to_grid(t, c, grid) for (t, c) in runs], axis=0)

            # --- Data Presented: Median and Percentiles of Cost over Time ---
            # Similar to the first plot, we calculate the median and bounds of
            # the interpolated costs at each point in the time grid.
            c_med, c_lo, c_hi = get_med_upper_lower(Cs, q_lo=10, q_hi=90)

            # --- Plotting the Data for Panel 2 ---
            # The shaded area shows the range of costs over time.
            axs[1].fill_between(grid, c_lo, c_hi, alpha=0.18, color=color)
            # The solid line shows the median cost over time.
            axs[1].plot(grid, c_med, color=color, label=label)

    # --- Styling and Final Touches for the Plots ---
    # Configure axes, labels, titles, and legends for clarity.

    # Panel 1: Cost vs. Iterations
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Cost")
    axs[0].set_yscale("log") # Log scale for cost is common for optimization problems.
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    axs[0].set_title("Solver Costs vs Iterations")
    axs[0].legend()

    # Panel 2: Cost vs. Time
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Cost")
    axs[1].set_yscale("log")
    axs[1].grid(True, which="both", ls="--", alpha=0.5)
    axs[1].set_title("Solver Costs vs Time")
    axs[1].legend()

    # At the top of the Figure, display the dataset name extracted from the file path.
    dataset_name = varpro_data_fpath.split("/")[-2] if "/" in varpro_data_fpath else varpro_data_fpath
    fig.suptitle(f"Dataset: {dataset_name}", fontsize=16)

    # Adjust layout to prevent labels from overlapping and display the plot.
    fig.tight_layout()

    # save to /tmp/varpro/dataset_name.png
    out_fpath = f"/tmp/varpro/{dataset_name.replace('/', '_')}.png"
    print(f"Saving figure to {out_fpath}...")
    import os
    os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
    fig.savefig(out_fpath)

    plt.show()


if __name__ == "__main__":
    # This part of the script executes when run from the command line.
    # It iterates through a list of specified experiment subdirectories.
    for subdir in EXP_SUBDIRS:
        # Construct the full paths to the data files.
        varpro_data_fpath = BASE_VARPRO_DATA_DIR + subdir
        gtsam_data_fpath = BASE_GTSAM_DATA_DIR + subdir

        print(f"Processing {varpro_data_fpath} and {gtsam_data_fpath}...")
        # Call the main visualization function for each experiment.
        visualize_data(varpro_data_fpath, gtsam_data_fpath)