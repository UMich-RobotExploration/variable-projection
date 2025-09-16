#!/usr/bin/env python3
"""
Export table metrics to CSV for your paper.

What it does
------------
For each dataset JSON (VarPro + optional GTSAM), we:
1) Group runs by formulation (Explicit, Explicit VarPro, Implicit, GTSAM) for rank5.
2) Compute the convergence target as the median of the final Implicit costs (rank5).
3) For every formulation (including Implicit itself), compute the *first* time and
   iteration index where the cost reaches within 1% of that Implicit target.
4) Aggregate across runs using the median (per method) for both time and iteration.
5) Write a consolidated CSV with columns ordered to match your LaTeX table macro:
   dataset, Ours(Red)_time, Original_time, Orig+VP_time, GTSAM_time,
            Ours(Red)_iters, Original_iters, Orig+VP_iters, GTSAM_iters,
            speedup_Orig_over_Ours, speedup_OrigVP_over_Ours, speedup_GTSAM_over_Ours

Assumptions
-----------
- "Ours (Reduced)" is mapped to the "Implicit" formulation in your JSON.
  If you need a different mapping, change Ours_IS formulation below.
- JSON layout matches your existing scripts (see FORMULATION_MAP in get_groups_from_data).
- GTSAM times are "per-iteration" and must be cumulatively summed (handled here).
- We filter to rank5 by default; override with --rank rankK if needed.
- If a method fails to reach the threshold in any run, we record a blank field.
- Aggregation uses medians across runs that *did* reach the threshold.

Usage
-----
python export_table_metrics.py \
  --varpro-base  "$HOME/variable-projection/examples/data" \
  --gtsam-base   "$HOME/variable-projection/examples/data_nik" \
  --save         "/tmp/varpro/table_metrics.csv" \
  --rank         rank5

You can pass a file with subpaths (one per line) or rely on defaults:
  --subpaths-file /path/to/subpaths.txt

Defaults mirror your EXP_SUBDIRS.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np


# ---------- Defaults (mirror your script) ----------
DEFAULT_EXP_SUBDIRS = [

    # "/raslam/tiers/results.json",
    # "/raslam/mrclam/mrclam2/results.json",
    # "/raslam/mrclam/mrclam4/results.json",
    # "/raslam/mrclam/mrclam6/results.json",
    # "/raslam/mrclam/mrclam7/results.json",
    # "/raslam/single_drone/results.json",
    # "/raslam/plaza2/results.json",
    # "/raslam/plaza1/results.json",
    "/sfm/TUM-desk/results.json",
    "/sfm/MipNerf-garden/results.json",
    "/sfm/IMC-gate/results.json",
    "/sfm/IMC-temple/results.json",
    "/sfm/IMC-rome/results.json",
    #"/sfm/Replica-REPoffice0/results.json",
    "/sfm/Replica-REPoffice0_100/results.json",
    #"/sfm/Replica-REPoffice1/results.json",
    "/sfm/Replica-REPoffice1_100/results.json",
    #"/sfm/Replica-REProom0/results.json",
    "/sfm/Replica-REProom0_100/results.json",
   # "/sfm/Replica-REProom1/results.json",
    "/sfm/Replica-REProom1_100/results.json",
    "/sfm/TUM-room/results.json",
    "/sfm/MipNerf-room/results.json",
    "/sfm/TUM-computer-R/results.json",
    "/sfm/TUM-computer-T/results.json",
    "/sfm/bal-93/results.json",
    "/sfm/bal-392/results.json",
   "/sfm/bal-1934/results.json",
    "/sfm/Replica-REPoffice1_100/results.json",
    "/sfm/MipNerf-kitchen/results.json",
#    # "/pgo/results.json",
#     "/snl/intel_snl/results.json",
#     "/snl/parking-garage_snl/results.json",
#     "/snl/grid3D_snl/results.json",
#     "/snl/MIT_snl/results.json",
#     #"/snl/smallGrid3D_snl/results.json",
#     "/snl/M3500_snl/results.json",
#     "/snl/city10000_snl/results.json",
#     #"/snl/tinyGrid3D_snl/results.json",
#     "/snl/torus3D_snl/results.json",
#     "/snl/sphere2500_snl/results.json",
#     "/pgo/intel/results.json",
#     "/pgo/parking-garage/results.json",
#     "/pgo/grid3D/results.json",
#     "/pgo/MIT/results.json",
#     #"/snl/smallGrid3D_snl/results.json",
#     "/pgo/M3500/results.json",
#     "/pgo/city10000/results.json",
#     #"/snl/tinyGrid3D_snl/results.json",
#     "/pgo/torus3D/results.json",
#     "/pgo/sphere2500/results.json",
]

DEFAULT_VARPRO_BASE = os.path.expanduser("~/variable-projection/examples/data")
DEFAULT_GTSAM_BASE  = os.path.expanduser("~/variable-projection/examples/data_nik")

# Mapping from JSON "formulation" field to label
FORMULATION_MAP = {0: "Explicit", 1: "Explicit VarPro", 2: "Implicit", "gtsam": "GTSAM"}

# Which formulation to treat as "Ours (Reduced)"
OURS_FORMULATION = "Implicit"  # change here if needed ("Explicit VarPro", etc.)

# Convergence threshold relative to Implicit's final cost
CONVERGE_PCT = 0.01  # 1%


def _ensure_strictly_increasing(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float).copy()
    if t.ndim != 1:
        raise ValueError("times must be 1-D")
    eps = np.finfo(float).eps
    for i in range(1, t.size):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + max(1e-12, abs(t[i - 1]) * 1e-12 + eps)
    return t


def _first_index_within_pct(costs: np.ndarray, target: float, pct: float) -> Optional[int]:
    """Return first k with costs[k] <= (1+pct)*target, else None."""
    if target is None or costs.size == 0:
        return None
    thr = (1.0 + pct) * target
    mask = np.where(costs <= thr)[0]
    return int(mask[0]) if mask.size else None


def _implicit_target_cost(groups: Dict[Tuple[str, str], List[Tuple[np.ndarray, np.ndarray]]],
                          rank: str = "rank5",
                          q: float = 0.5) -> Optional[float]:
    key = ("Implicit", rank)
    runs = groups.get(key, [])
    last_costs = [c[-1] for (_, c) in runs if c.size > 0]
    if not last_costs:
        # fallback across all runs/methods
        last_costs = [c[-1] for rr in groups.values() for (_, c) in rr if len(c) > 0]
    if not last_costs:
        return None
    arr = np.asarray(last_costs, dtype=float)
    if q == 0.5:
        return float(np.nanmedian(arr))
    return float(np.nanpercentile(arr, q * 100.0))


def _dataset_label_from_subpath(subpath: str) -> str:
    """
    Try to produce a clean dataset label from the relative subpath.
    E.g., '/snl/city10000_snl/results.json' -> 'City10000'
          '/snl/parking-garage_snl/results.json' -> 'Garage'
    You can tweak this mapping as needed.
    """
    name = subpath.strip("/").split("/")[-2] if "/" in subpath else subpath
    # strip common suffixes
    name = re.sub(r"(_snl|_pgo|_sfm)$", "", name, flags=re.IGNORECASE)
    # a few friendly aliases
    aliases = {
        "city10000": "City10000",
        "grid3D": "Grid3D",
        "intel": "Intel",
        "M3500": "M3500",
        "MIT": "MIT",
        "parking-garage": "Garage",
        "sphere2500": "Sphere",
        "torus3D": "Torus",
    }
    for k, v in aliases.items():
        if name.lower() == k.lower():
            return v
    # default: title-ish
    name = re.sub(r"[_\-]+", " ", name).strip()
    return name.title()


def get_groups_from_data(data_fpath: str) -> Dict[Tuple[str, str], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Load experimental data from a JSON file and group by (formulation, rank).
    Each group is a list of runs: (times, costs).
    """
    if not os.path.exists(data_fpath):
        return {}
    with open(data_fpath, "r") as f:
        data = json.load(f)

    groups = defaultdict(list)
    for entry in data:
        form_raw = entry.get("formulation")
        formulation = FORMULATION_MAP.get(form_raw, f"Unknown({form_raw})")

        init_file = entry.get("init_file", "")
        m = re.search(r"rank(\d+)", init_file)
        rank = f"rank{m.group(1)}" if m else "unknown"

        times = np.asarray(entry.get("times", []), dtype=float)
        costs = np.asarray(entry.get("costs", []), dtype=float)

        L = min(times.size, costs.size)
        times, costs = times[:L], costs[:L]

        if formulation == "GTSAM" and times.size > 0:
            # accumulate per-iteration times
            times = np.cumsum(times)
            times = np.concatenate(([0.0], times))
            costs = np.concatenate(([costs[0]], costs))

        if times.size > 1:
            order = np.argsort(times)
            times, costs = times[order], costs[order]
            times = _ensure_strictly_increasing(times)

        if times.size and costs.size:
            groups[(formulation, rank)].append((times, costs))

    return groups


def _median_time_iter_to_threshold(
    runs: List[Tuple[np.ndarray, np.ndarray]],
    target_cost: Optional[float],
    pct: float
) -> Tuple[Optional[float], Optional[int]]:
    """
    Given runs [(t, c), ...], return median time and iteration where c <= (1+pct)*target_cost.
    If no run hits the threshold, returns (None, None).
    """
    if not runs or target_cost is None or not np.isfinite(target_cost):
        return None, None

    hit_times = []
    hit_iters = []

    for (t, c) in runs:
        if t.size == 0 or c.size == 0:
            continue
        k = _first_index_within_pct(c, target_cost, pct)
        if k is None:
            continue
        # guard bounds
        if k >= t.size:
            k = t.size - 1
        hit_times.append(float(t[k]))
        hit_iters.append(int(k))

    if not hit_times:
        return None, None

    # Aggregate by median
    med_time = float(np.median(np.asarray(hit_times)))
    med_iter = int(np.median(np.asarray(hit_iters)))
    return med_time, med_iter


def _fmt(val, ndigits=2):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return ""
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    return f"{float(val):.{ndigits}f}"


def export_csv(
    varpro_base: str,
    gtsam_base: str,
    subpaths: List[str],
    save_path: str,
    rank: str = "rank5",
    converge_pct: float = CONVERGE_PCT,
) -> None:
    """
    Aggregate across provided subpaths and save a CSV.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # CSV header (ordered to match your LaTeX macro \tableRow)
    header = [
        "Dataset",
        "Ours(Red)_time", "Original_time", "Orig+VP_time", "GTSAM_time",
        "Ours(Red)_iters", "Original_iters", "Orig+VP_iters", "GTSAM_iters",
        "speedup_Orig_over_Ours", "speedup_OrigVP_over_Ours", "speedup_GTSAM_over_Ours",
    ]

    rows = []

    for sub in subpaths:
        ds_label = _dataset_label_from_subpath(sub)

        varpro_path = os.path.join(varpro_base, sub.lstrip("/"))
        gtsam_path  = os.path.join(gtsam_base,  sub.lstrip("/"))

        # load groups (VarPro + optional GTSAM)
        groups_v = get_groups_from_data(varpro_path)
        groups   = dict(groups_v)
        groups_g = get_groups_from_data(gtsam_path)
        # merge
        for k, v in groups_g.items():  # k=(formulation,rank)
            groups.setdefault(k, []).extend(v)

        # filter rank
        groups_rank = {k: v for k, v in groups.items() if k[1] == rank}
        if not groups_rank:
            # no data for this dataset/rank
            rows.append([ds_label] + [""] * (len(header) - 1))
            continue

        # compute Implicit target
        target_C = _implicit_target_cost(groups_rank, rank=rank, q=0.5)

        # Collect per-method stats
        methods = OrderedDict([
            (OURS_FORMULATION, "Ours(Red)"),       # Implicit -> Ours
            ("Explicit",       "Original"),
            ("Explicit VarPro","Orig+VP"),
            ("GTSAM",          "GTSAM"),
        ])

        # medians
        times = {alias: None for alias in methods.values()}
        iters = {alias: None for alias in methods.values()}

        for (formulation, alias) in methods.items():
            runs = groups_rank.get((formulation, rank), [])
            med_t, med_k = _median_time_iter_to_threshold(runs, target_C, converge_pct)
            times[alias] = med_t
            iters[alias] = med_k

        # speedups: baseline / ours
        def safe_ratio(baseline, ours):
            if baseline is None or ours is None or ours == 0:
                return None
            return float(baseline) / float(ours)

        spd_orig_over_ours   = safe_ratio(times["Original"], times["Ours(Red)"])
        spd_origvp_over_ours = safe_ratio(times["Orig+VP"], times["Ours(Red)"])
        spd_gtsam_over_ours  = safe_ratio(times["GTSAM"],   times["Ours(Red)"])

        row = [
            ds_label,
            _fmt(times["Ours(Red)"]), _fmt(times["Original"]), _fmt(times["Orig+VP"]), _fmt(times["GTSAM"]),
            _fmt(iters["Ours(Red)"], 0), _fmt(iters["Original"], 0), _fmt(iters["Orig+VP"], 0), _fmt(iters["GTSAM"], 0),
            _fmt(spd_orig_over_ours), _fmt(spd_origvp_over_ours), _fmt(spd_gtsam_over_ours),
        ]
        rows.append(row)

    # write CSV
    with open(save_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print(f"Wrote CSV to: {save_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Export table metrics CSV (time/iters to within 1% of Implicit final cost).")
    ap.add_argument("--varpro-base", type=str, default=DEFAULT_VARPRO_BASE, help="Base path for VarPro data JSONs")
    ap.add_argument("--gtsam-base",  type=str, default=DEFAULT_GTSAM_BASE,  help="Base path for GTSAM data JSONs")
    ap.add_argument("--save",        type=str, default="tables/table_metrics.csv", help="Output CSV path")
    ap.add_argument("--rank",        type=str, default="rank5", help="Rank to filter (e.g., rank5)")
    ap.add_argument("--subpaths-file", type=str, default="", help="Optional file listing relative JSON subpaths (one per line)")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.subpaths_file:
        if not os.path.exists(args.subpaths_file):
            print(f"[error] subpaths file not found: {args.subpaths_file}", file=sys.stderr)
            sys.exit(2)
        with open(args.subpaths_file, "r") as f:
            subpaths = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    else:
        subpaths = list(DEFAULT_EXP_SUBDIRS)

    export_csv(
        varpro_base=args.varpro_base,
        gtsam_base=args.gtsam_base,
        subpaths=subpaths,
        save_path=args.save,
        rank=args.rank,
        converge_pct=CONVERGE_PCT,
    )


if __name__ == "__main__":
    main()
