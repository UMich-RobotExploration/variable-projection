"""Aggregate per-dataset JSON files from the cosmobench/nebula IRLS-GNC sweep
into a single CSV + a single combined JSON for table generation.

Inputs:  examples/data/analysis/cosmobench_irls_gnc_full/*.json
Outputs: examples/data/analysis/cosmobench_irls_gnc_full/summary.csv
         examples/data/analysis/cosmobench_irls_gnc_full/summary.json
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

OUT_DIR = Path("examples/data/analysis/cosmobench_irls_gnc_full")
METHODS = ["explicit", "expvp", "impl"]
LABEL = {"explicit": "Explicit", "expvp": "ExpVP", "impl": "Implicit"}


def mean_or_nan(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    recs = [json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("*.json"))
            if p.name not in {"summary.json"}]
    print(f"loaded {len(recs)} per-dataset records")

    csv_path = OUT_DIR / "summary.csv"
    fields = (
        ["dataset", "n_poses", "n_edges", "n_outliers_truth", "n_robots"]
        + [
            f"{LABEL[m]}_{k}"
            for m in METHODS
            for k in [
                "outer_iters", "inner_iters_total", "wall_s", "precompute_s",
                "inner_cost", "robust_cost",
                "ate_global_rmse_m", "ate_per_robot_mean_m",
            ]
        ]
    )
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in recs:
            row = [r["dataset"], r["n_poses"], r["n_edges"],
                    r["n_outliers_truth"], r["n_robots"]]
            for m in METHODS:
                d = r["methods"][m]
                if d.get("status") != "ok":
                    row += [""] * 8
                    continue
                per_r = list(d.get("ate_per_robot_aligned_rmse_m", {}).values())
                row += [
                    d["outer_iters"], d["inner_iters_total"], f"{d['wall_s']:.4f}",
                    f"{d['precompute_s']:.6f}", f"{d['inner_cost']:.4f}",
                    f"{d['robust_cost']:.4f}",
                    f"{d['ate_global_rmse_m']:.4f}",
                    f"{mean_or_nan(per_r):.4f}",
                ]
            w.writerow(row)
    print(f"wrote {csv_path}")

    # Aggregate medians
    agg = {m: {k: [] for k in ["outer_iters", "inner_iters_total", "wall_s",
                                "precompute_s", "inner_cost", "robust_cost",
                                "ate_global_rmse_m", "ate_per_robot_mean_m"]}
            for m in METHODS}
    for r in recs:
        for m in METHODS:
            d = r["methods"][m]
            if d.get("status") != "ok":
                continue
            agg[m]["outer_iters"].append(d["outer_iters"])
            agg[m]["inner_iters_total"].append(d["inner_iters_total"])
            agg[m]["wall_s"].append(d["wall_s"])
            agg[m]["precompute_s"].append(d["precompute_s"])
            agg[m]["inner_cost"].append(d["inner_cost"])
            agg[m]["robust_cost"].append(d["robust_cost"])
            agg[m]["ate_global_rmse_m"].append(d["ate_global_rmse_m"])
            per_r = list(d.get("ate_per_robot_aligned_rmse_m", {}).values())
            agg[m]["ate_per_robot_mean_m"].append(mean_or_nan(per_r))

    combined = {
        "n_datasets": len(recs),
        "kernel": "gm",
        "gnc": True,
        "init": "odometry",
        "priors": "stripped",
        "datasets": [r["dataset"] for r in recs],
        "per_method_aggregate": {
            LABEL[m]: {
                "median": {k: statistics.median(v) for k, v in agg[m].items() if v},
                "mean":   {k: statistics.mean(v)   for k, v in agg[m].items() if v},
                "n_runs": len(agg[m]["wall_s"]),
            } for m in METHODS
        },
        "per_dataset": recs,
    }
    json_path = OUT_DIR / "summary.json"
    json_path.write_text(json.dumps(combined, indent=2))
    print(f"wrote {json_path}")

    # Pretty print medians
    print("\nMedian over all datasets:")
    print(f"{'method':10s} {'outer':>7s} {'inner':>7s} {'wall_s':>9s} "
          f"{'precomp_s':>10s} {'robust_cost':>12s} {'ATE_global':>11s} {'ATE_perRobot':>13s}")
    for m in METHODS:
        a = combined["per_method_aggregate"][LABEL[m]]["median"]
        print(f"{LABEL[m]:10s} {a['outer_iters']:7.0f} {a['inner_iters_total']:7.0f} "
              f"{a['wall_s']:9.2f} {a['precompute_s']:10.4f} "
              f"{a['robust_cost']:12.1f} {a['ate_global_rmse_m']:11.2f} "
              f"{a['ate_per_robot_mean_m']:13.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
