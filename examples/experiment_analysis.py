import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

import os
HOMEDIR = os.path.expanduser("~")
BASE_VARPRO_DATA_DIR = f"{HOMEDIR}/variable-projection/examples/data"
BASE_GTSAM_DATA_DIR = f"{HOMEDIR}/variable-projection/examples/data_nik"
EXP_SUBDIRS = [
    # "/raslam/tiers/results.json",
    # "/raslam/mrclam/mrclam2/results.json",
    # "/raslam/mrclam/mrclam4/results.json",
    # "/raslam/mrclam/mrclam6/results.json",
    # "/raslam/mrclam/mrclam7/results.json",
    "/raslam/single_drone/results.json",
    # "/raslam/plaza2/results.json",
    # "/raslam/plaza1/results.json",
#     "/sfm/TUM-desk/results.json",
     "/sfm/MipNerf-garden/results.json",
#     "/sfm/IMC-gate/results.json",
#     "/sfm/IMC-temple/results.json",
#     "/sfm/IMC-rome/results.json",
#     "/sfm/Replica-REPoffice0/results.json",
#     "/sfm/Replica-REPoffice0_100/results.json",
#     #"/sfm/Replica-REPoffice1/results.json",
#     "/sfm/Replica-REPoffice1_100/results.json",
#     #"/sfm/Replica-REProom0/results.json",
#     "/sfm/Replica-REProom0_100/results.json",
#    # "/sfm/Replica-REProom1/results.json",
#     "/sfm/Replica-REProom1_100/results.json",
#     "/sfm/TUM-room/results.json",
#     "/sfm/MipNerf-room/results.json",
#     "/sfm/TUM-computer-R/results.json",
#     "/sfm/TUM-computer-T/results.json",
#     "/sfm/bal-93/results.json",
#     "/sfm/bal-392/results.json",
#    "/sfm/bal-1934/results.json",
#     "/sfm/Replica-REPoffice1_100/results.json",
#     "/sfm/MipNerf-kitchen/results.json",
   # "/pgo/results.json",
   # "/snl/intel_snl/results.json",
    # "/snl/parking-garage_snl/results.json",
    # "/snl/grid3D_snl/results.json",
    "/snl/MIT_snl/results.json",
    # #"/snl/smallGrid3D_snl/results.json",
    # "/snl/M3500_snl/results.json",
    # "/snl/city10000_snl/results.json",
    # #"/snl/tinyGrid3D_snl/results.json",
    # "/snl/torus3D_snl/results.json",
    # "/snl/sphere2500_snl/results.json",
    "/pgo/intel/results.json",
    # "/pgo/parking-garage/results.json",
    # "/pgo/grid3D/results.json",
    # "/pgo/MIT/results.json",
    # #"/snl/smallGrid3D_snl/results.json",
    # "/pgo/M3500/results.json",
    # "/pgo/city10000/results.json",
    # #"/snl/tinyGrid3D_snl/results.json",
    # "/pgo/torus3D/results.json",
    # "/pgo/sphere2500/results.json",
]

# Define a color scheme for the plots using a standard matplotlib colormap.
COLOR_SCHEME = plt.get_cmap("tab10")
num_colors = COLOR_SCHEME.N
# Create a list of keys to assign unique colors to each combination of rank and formulation.
COLOR_KEYS = [(rank, form) for rank in ["rank5"] for form in ["Explicit", "Explicit VarPro", "Implicit", "GTSAM"]]
# Generate a dictionary mapping each key to a color from the color scheme.
COLORS = {key: COLOR_SCHEME(i % num_colors) for i, key in enumerate(COLOR_KEYS)}
def _implicit_target_cost(groups, rank="rank5", q=0.5):
    """Return target C* from Implicit group (median of final costs)."""
    key = ("Implicit", rank)
    runs = groups.get(key, [])
    last_costs = [c[-1] for (_, c) in runs if len(c) > 0]
    if not last_costs:
        # fallback: across all runs/methods
        last_costs = [c[-1] for rr in groups.values() for (_, c) in rr if len(c) > 0]
    if not last_costs:
        return None
    arr = np.asarray(last_costs, dtype=float)
    return float(np.nanmedian(arr) if q == 0.5 else np.nanpercentile(arr, q*100.0))

def _first_index_within_pct(costs, target, pct):
    """First index k where costs[k] <= (1+pct)*target; else len(costs)."""
    if target is None or len(costs) == 0:
        return len(costs)
    thr = (1.0 + pct) * target
    idx = np.argmax(costs <= thr)  # returns 0 if first is True, else 0
    if (costs <= thr).any():
        return int(np.where(costs <= thr)[0][0])
    return len(costs)

def _trim_groups_to_target(groups, target_cost, pct, rank="rank5", buffer=5):
    """Trim all runs in all methods to first time they reach within pct of
    target_cost. Buffers the trimming a bit so we have some extra points to show
     convergence behavior."""
    trimmed = {}
    for (formulation, rnk), runs in groups.items():
        new_runs = []
        for (t, c) in runs:
            k = _first_index_within_pct(c, target_cost, pct) + buffer
            t2 = t[:k+1] if k < len(c) else t
            c2 = c[:k+1] if k < len(c) else c
            new_runs.append((t2, c2))
        trimmed[(formulation, rnk)] = new_runs
    return trimmed


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
# --- Paste this whole function (and the imports) over your current version ---

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

def visualize_data(varpro_data_fpath: str, gtsam_data_fpath: str = ""):
    """
    Generate two readability-optimized plots (for LaTeX):
      1) Cost vs Iterations  2) Cost vs Time
    Saves both PNG (300 dpi) and PDF (vector) with tight bounding box.
    """
    # -------- Paper-friendly defaults --------
    mpl.rcParams.update({
        "font.size": 20,            # base
        "font.serif": "Times New Roman",
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 18,
        "lines.linewidth": 2.2,
        "lines.markersize": 5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,         # embed TrueType; safer in some LaTeX setups
        "ps.fonttype": 42,
    })
    # Optional: uncomment if you have LaTeX installed and want LaTeX text rendering
    # mpl.rcParams.update({"text.usetex": True})

    # -------- Load/group data (your existing helpers) --------
    group_varpro = get_groups_from_data(varpro_data_fpath)
    groups = dict(group_varpro)
    if gtsam_data_fpath:
        groups.update(get_groups_from_data(gtsam_data_fpath))

    if not groups:
        print(f"No data to visualize for {varpro_data_fpath} and {gtsam_data_fpath}.")
        return

    converge_pct = 0.01  # 1%
    target_C = _implicit_target_cost(groups, rank="rank5", q=0.5)
    if target_C is not None:
        groups = _trim_groups_to_target(groups, target_C, converge_pct, rank="rank5")
        print(f"[convergence] Target C* (Implicit, rank5) = {target_C:.6g}; "
              f"cut when cost <= {(1+converge_pct)*target_C:.6g}")
    else:
        print("[convergence] No target C* available (no runs); skipping trim.")

    # -------- Wider figure; share y so log ticks align --------
    fig, axs = plt.subplots(
        1, 2,
        figsize=(14.8, 5.2),      # wider & a touch shorter
        constrained_layout=False,
        sharey=True               # align log y-ticks across panels
    )

    # Consistent palette + emphasis for Ours (Implicit)
    palette = {"Explicit": "C0", "Explicit VarPro": "C1", "Implicit": "C2", "GTSAM": "C3"}
    def style_for(method):
        if method == "Implicit":
            return dict(color=palette[method], lw=3.0, zorder=4, band_alpha=0.14)
        return dict(color=palette.get(method, "C7"), lw=2.0, zorder=3, band_alpha=0.10)

    def get_med_upper_lower(arr, q_lo=None, q_hi=None):
        med = np.median(arr, axis=0)
        lo = np.min(arr, axis=0)
        hi = np.max(arr, axis=0)
        if q_lo is not None:
            lo = np.nanpercentile(arr, q_lo, axis=0)
        if q_hi is not None:
            hi = np.nanpercentile(arr, q_hi, axis=0)
        return med, lo, hi
    # Legend text mapping
    LEGEND_LABELS = {
        "Implicit": "Ours",           # (Reduced)
        "Explicit": "Original",
        "Explicit VarPro": "Orig. + VP",
        "GTSAM": "GTSAM",
    }
    # -------- Plot each formulation --------
    for (formulation, rank), runs in groups.items():
        try:
            rank_num = int(str(rank).replace("rank", ""))
        except ValueError:
            continue
        if rank_num != 5 or not runs:
            continue

        sty = style_for(formulation)
        color, lw, z = sty["color"], sty["lw"], sty["zorder"]
        band_alpha = sty["band_alpha"]

        # ----- Panel 1: Cost vs Iterations -----
        min_iter_len = min(min(len(t), len(c)) for t, c in runs)
        if min_iter_len < 2:
            continue
        costs_iter = np.stack([c[:min_iter_len] for _, c in runs], axis=0)
        iters = np.arange(min_iter_len)
        c_med_iter, c_lo_iter, c_hi_iter = get_med_upper_lower(costs_iter, q_lo=10, q_hi=90)

        axs[0].fill_between(iters, c_lo_iter, c_hi_iter, alpha=band_alpha, color=color, linewidth=0)
        axs[0].plot(iters, c_med_iter,
            label=LEGEND_LABELS.get(formulation, formulation),
            color=color, lw=lw, zorder=z)
        # ----- Panel 2: Cost vs Time -----
        grid = _common_time_grid(runs, n_points=500, mode="linspace")
        if grid is None:
            for (t, c) in runs:
                axs[1].plot(t, c, alpha=0.35, color=color, lw=1.5)
        else:
            Cs = np.stack([_interp_run_to_grid(t, c, grid) for (t, c) in runs], axis=0)
            c_med, c_lo, c_hi = get_med_upper_lower(Cs, q_lo=10, q_hi=90)
            axs[1].fill_between(grid, c_lo, c_hi, alpha=band_alpha, color=color, linewidth=0)
            axs[1].plot(grid, c_med, color=color, lw=lw, zorder=z)

    # -------- Styling --------
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.tick_params(axis="both", which="major", length=6, width=1)
        ax.tick_params(axis="both", which="minor", length=3, width=0.8)

        # Better log ticks
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(1, 2, 5), numticks=12))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
    

    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Cost")
    axs[0].set_title("Solver Costs vs Iterations")

    axs[1].set_xlabel("Time (s)")
    axs[1].set_title("Solver Costs vs Time")
    axs[1].legend(loc="upper right", frameon=False)


    # -------- Single, figure-level legend --------
    # Dedupe handles/labels collected from the left axis
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Reserve space: more bottom, a bit less top (tweak if needed)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.24, wspace=0.08)

    # Add a small, empty axes for the legend at the bottom
    leg_ax = fig.add_axes([0.05, 0.02, 0.90, 0.12])  # [left, bottom, width, height] in figure coords
    leg_ax.axis("off")
    leg_ax.legend(
        by_label.values(), by_label.keys(),
        ncol=len(by_label),
        loc="center",
        frameon=False,
    )

   # -------- Title + Save --------
    dataset_name = varpro_data_fpath.split("/")[-2] if "/" in varpro_data_fpath else varpro_data_fpath
    # fig.suptitle(dataset_name, fontsize=22, fontweight="bold", y=1.12)

    out_dir = "/home/nikolas/variable-projection/pics"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, dataset_name.replace("/", "_"))
    png_path = f"{base}.png"
    pdf_path = f"{base}.pdf"
    svg_path = f"{base}.svg"

    print(f"Saving figure to {png_path}, {pdf_path}, and {svg_path} ...")


    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)  # or: format="svg"


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