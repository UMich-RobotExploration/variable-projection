#!/usr/bin/env python3
"""LaTeX rows for the robust IRLS-GNC table (tab:cosmobench_irls_gnc).

Reads the same two sweeps as plot_robust_pareto.py and emits one
`\\rowCB{name}{{ours}{orig}{vp}{gtsam}}{{iters...}}` line per dataset, grouped
CosmoBench Wi-Fi / CosmoBench Radio / Nebula in the table's row order.

The table reports a **single trial** (--seed, default 0 = the noiseless
odometry init every method shares), not a median over inits.  A method is
reported if its global ATE is within --ate-tol (default 5%) of the best ATE
any of the four methods reached on that dataset for that trial, *or* its GM
robust cost is within --cost-tol (default 1%) of the best robust cost -- two
runs at the same minimum of the objective can sit a few percent apart in ATE.
Otherwise both its runtime and its iteration cell are `\\mredx`, as are runs
that did not finish (timeout / crash).

Runtime is solver `wall_s`; iterations are total inner iterations summed over
the GNC outer loop (`inner_iters_total` for ours, `inner` for the SESync
baseline) -- the same quantities the old table used.

Usage:
  .venv/bin/python examples/runners/make_robust_table.py
  .venv/bin/python examples/runners/make_robust_table.py --seed 0 --ate-tol 0.05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_robust_pareto import IRLS_DIR, SESYNC_DIR, rel  # noqa: E402

# Column order: Ours / Original / Orig.+VP / GTSAM.
COLUMNS = ["impl", "explicit", "expvp", "gtsam"]

WIFI_RADIO = [
    ("Kittredge Loop", "kittredge_loop"),
    ("Main Campus", "main_campus"),
    ("KTH R3-00", "kth_r3_00_d06_d09_d10"),
    ("KTH R3-01", "kth_r3_01_n01_n04_n05"),
    ("KTH R4-00", "kth_r4_00_d06_d10_n04_n05"),
    ("NTU R3-00", "ntu_r3_00_d01_n04_n08"),
    ("NTU R3-01", "ntu_r3_01_n04_n08_n13"),
    ("NTU R3-02", "ntu_r3_02_d02_n04_n13"),
    ("NTU R4-00", "ntu_r4_00_d01_n04_n08_n13"),
    ("NTU R5-00", "ntu_r5_00_d01_d02_n04_n08_n13"),
]
GROUPS = [
    ("\\wifiMultiRow", [(n, f"{s}_wifi") for n, s in WIFI_RADIO]),
    ("\\proradioMultiRow", [(n, f"{s}_proradio") for n, s in WIFI_RADIO]),
    ("\\nebulaMultiRow", [("Finals", "finals"),
                          ("Kentucky UG", "kentucky_underground"),
                          ("Tunnel", "tunnel"), ("Urban", "urban")]),
]


def trial(irls_dir: Path, sesync_dir: Path, dataset: str, seed: int) -> dict:
    """method -> {status, wall_s, ate, iters} for one dataset and one seed."""
    d = json.loads((irls_dir / f"{dataset}.json").read_text())
    g = json.loads((sesync_dir / f"{dataset}.json").read_text())
    out = {}
    for k in ("impl", "explicit", "expvp"):
        r = d["methods"][k]["seeds"].get(str(seed), {})
        out[k] = {"status": r.get("status", "missing"), "wall_s": r.get("wall_s"),
                  "ate": r.get("ate_global_rmse_m"),
                  "cost": r.get("robust_cost"),
                  "iters": r.get("inner_iters_total")}
    r = g["runs"].get(str(seed), {})
    out["gtsam"] = {"status": r.get("status", "missing"), "wall_s": r.get("wall_s"),
                    "ate": r.get("ate_global_rmse_m"),
                    "cost": r.get("robust_cost"), "iters": r.get("inner")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0,
                    help="init seed to report (default 0: noiseless odometry)")
    ap.add_argument("--ate-tol", type=float, default=0.05,
                    help="relative ATE tolerance vs the per-dataset best "
                         "(default 0.05 = within 5%%)")
    ap.add_argument("--cost-tol", type=float, default=0.01,
                    help="relative robust-cost tolerance vs the per-dataset "
                         "best; a run within it also counts as converged "
                         "(default 0.01 = within 1%%)")
    ap.add_argument("--irls-dir", type=Path, default=IRLS_DIR)
    ap.add_argument("--sesync-dir", type=Path, default=SESYNC_DIR)
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the rows to this file")
    args = ap.parse_args()

    lines, notes = [], []
    for g_idx, (multirow, datasets) in enumerate(GROUPS):
        if g_idx:
            lines.append("\\midrule")
        lines.append(multirow)
        for label, ds in datasets:
            t = trial(args.irls_dir, args.sesync_dir, ds, args.seed)
            ok = {k: v for k, v in t.items()
                  if v["status"] == "ok" and v["wall_s"] and v["ate"]}
            best = min(v["ate"] for v in ok.values())
            best_cost = min(v["cost"] for v in ok.values())
            times, iters = [], []
            for k in COLUMNS:
                v = t[k]
                if k in ok and v["ate"] > best * (1.0 + args.ate_tol) \
                        and v["cost"] <= best_cost * (1.0 + args.cost_tol):
                    notes.append(f"{ds:40s} {k:8s} kept on cost: ATE "
                                 f"{v['ate'] / best:.3f}x, cost "
                                 f"{v['cost'] / best_cost:.4f}x best")
                if k in ok and (v["ate"] <= best * (1.0 + args.ate_tol)
                                or v["cost"] <= best_cost * (1.0 + args.cost_tol)):
                    times.append(f"{{{v['wall_s']:.2f}}}")
                    iters.append(f"{{{int(v['iters'])}}}")
                else:
                    times.append("{\\mredx}")
                    iters.append("{\\mredx}")
                    why = (f"ATE {v['ate'] / best:.3f}x, cost "
                           f"{v['cost'] / best_cost:.4f}x best" if k in ok
                           else f"status={v['status']}")
                    notes.append(f"{ds:40s} {k:8s} {why}")
            name = f"{{{label}}}"
            run = "{" + "".join(times) + "}"
            it = "{" + "".join(iters) + "}"
            lines.append(f"  & \\rowCB{name:17s}{run:36s}{it}")

    text = "\n".join(lines) + "\n"
    print(text)
    print(f"% seed {args.seed}, ATE tol {args.ate_tol:.0%}, cost tol "
          f"{args.cost_tol:.0%}; \\mredx / cost-kept cells:",
          file=sys.stderr)
    for n in notes:
        print(f"%   {n}", file=sys.stderr)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {rel(args.out)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
