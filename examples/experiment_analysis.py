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
    "/raslam/mrclam/mrclam4/results.json",
    # "/raslam/mrclam/mrclam6/results.json",
    # "/raslam/mrclam/mrclam7/results.json",
    # "/raslam/single_drone/results.json",
    # "/raslam/plaza2/results.json",
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

COLOR_SCHEME = plt.get_cmap("tab10")
num_colors = COLOR_SCHEME.N
COLOR_KEYS = [(rank, form) for rank in ["rank5"] for form in ["Explicit", "Explicit VarPro", "Implicit", "GTSAM"]]
COLORS = {key: COLOR_SCHEME(i % num_colors) for i, key in enumerate(COLOR_KEYS)}

# Optional fallback colors
def color_for(key):
    # key is (formulation, rank)
    try :
        return COLORS[key[1], key[0]]
    except KeyError:
        base = {
            "Explicit": "#1f77b4",
            "Explicit VarPro": "#ff7f0e",
            "Implicit": "#2ca02c",
            "GTSAM": "#d62728",
        }
        return base.get(key[0], "#9467bd")

def _ensure_strictly_increasing(t):
    """Make times strictly increasing (fix duplicates with tiny eps)."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("times must be 1-D")
    eps = np.finfo(float).eps
    for i in range(1, t.size):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + max(1e-12, abs(t[i-1]) * 1e-12 + eps)
    return t

def get_groups_from_data(data_fpath: str) -> dict:
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

        # Keep lengths consistent
        L = min(times.size, costs.size)
        times, costs = times[:L], costs[:L]

        # GTSAM: make times cumulative, and prepend initial sample so lengths stay equal
        if formulation == "GTSAM" and times.size > 0:
            times = np.cumsum(times)
            # align with an initial "t=0, cost=cost[0]" sample
            times = np.concatenate(([0.0], times))
            costs = np.concatenate(([costs[0]], costs))

        # Ensure strictly increasing times for interpolation stability
        if times.size > 1:
            order = np.argsort(times)
            times, costs = times[order], costs[order]
            times = _ensure_strictly_increasing(times)

        if times.size and costs.size:
            groups[(formulation, rank)].append((times, costs))

    return groups

def _common_time_grid(runs, n_points=400, mode="linspace"):
    """
    Build a common time grid over the intersection of all runs' time ranges.
    """
    tmins = [t[0] for (t, _) in runs if t.size]
    tmaxs = [t[-1] for (t, _) in runs if t.size]
    if not tmins or not tmaxs:
        return None
    t0 = max(tmins)
    t1 = min(tmaxs)
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return None
    if mode == "linspace":
        return np.linspace(t0, t1, n_points)
    elif mode == "union":
        # union of all time stamps, clipped to [t0,t1], then unique & sorted
        T = np.unique(np.concatenate([t[(t >= t0) & (t <= t1)] for (t, _) in runs]))
        # cap cardinality if it explodes
        if T.size > 2000:
            # downsample uniformly
            idx = np.linspace(0, T.size - 1, 2000).round().astype(int)
            T = T[idx]
        return T
    else:
        raise ValueError("Unknown grid mode")

def _interp_run_to_grid(times, costs, grid):
    """
    Linearly interpolate costs(t) onto 'grid'.
    Extrapolation is clamped to endpoints (np.interp behavior).
    """
    if times.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    # safety: equalize length
    L = min(times.size, costs.size)
    times, costs = times[:L], costs[:L]
    times = _ensure_strictly_increasing(times)
    return np.interp(grid, times, costs)

def visualize_data(varpro_data_fpath: str, gtsam_data_fpath: str = ""):
    group_varpro = get_groups_from_data(varpro_data_fpath)
    groups = dict(group_varpro)
    if gtsam_data_fpath:
        groups.update(get_groups_from_data(gtsam_data_fpath))

    # Two panels
    fig, axs = plt.subplots(1, 2, figsize=(11, 6))

    for (formulation, rank), runs in groups.items():
        # Filter example: only rank5; remove this if you want all ranks
        try:
            rank_num = int(str(rank).replace("rank", ""))
        except ValueError:
            continue
        if rank_num != 5:
            continue
        if not runs:
            continue

        # -------- Panel 1: Costs vs Iterations (no need to interpolate) --------
        # Align by iterations: truncate each run to min length
        min_iter_len = min(min(len(t), len(c)) for t, c in runs)
        if min_iter_len < 2:
            continue
        costs_iter = np.stack([c[:min_iter_len] for _, c in runs], axis=0)
        iters = np.arange(min_iter_len)

        c_med_iter = np.nanmedian(costs_iter, axis=0)
        c_lo_iter = np.nanpercentile(costs_iter, 10, axis=0)
        c_hi_iter = np.nanpercentile(costs_iter, 90, axis=0)

        color = color_for((formulation, rank))
        label = f"{rank} ({formulation})"
        axs[0].fill_between(iters, c_lo_iter, c_hi_iter, alpha=0.18, label=None, color=color)
        axs[0].plot(iters, c_med_iter, label=label, color=color)

        # -------- Panel 2: Costs vs Time (interpolate to common grid) --------
        grid = _common_time_grid(runs, n_points=500, mode="linspace")
        if grid is None:
            # Not enough overlap; fall back to plotting individual runs
            for (t, c) in runs:
                axs[1].plot(t, c, alpha=0.35, color=color)
        else:
            Cs = np.stack([_interp_run_to_grid(t, c, grid) for (t, c) in runs], axis=0)
            c_med = np.nanmedian(Cs, axis=0)
            c_lo = np.nanpercentile(Cs, 10, axis=0)
            c_hi = np.nanpercentile(Cs, 90, axis=0)

            axs[1].fill_between(grid, c_lo, c_hi, alpha=0.18, color=color)
            axs[1].plot(grid, c_med, color=color, label=label)

    # ---- Styling ----
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Cost")
    axs[0].set_yscale("log")
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    axs[0].set_title("Solver Costs vs Iterations")
    axs[0].legend()

    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Cost")
    axs[1].set_yscale("log")
    axs[1].grid(True, which="both", ls="--", alpha=0.5)
    axs[1].set_title("Solver Costs vs Time")
    axs[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    for subdir in EXP_SUBDIRS:
        varpro_data_fpath = BASE_VARPRO_DATA_DIR + subdir
        gtsam_data_fpath = BASE_GTSAM_DATA_DIR + subdir

        print(f"Processing {varpro_data_fpath} and {gtsam_data_fpath}...")
        visualize_data(varpro_data_fpath, gtsam_data_fpath)
