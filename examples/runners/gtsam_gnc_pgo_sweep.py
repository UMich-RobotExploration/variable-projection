#!/usr/bin/env python3
"""Sweep GTSAM's GNC-GM solver over the same PGO datasets that the VarPro IRLS
sweep covers, from the byte-identical odometry init.

For each dataset in examples/data/pgo_outliers/ (default) — or
examples/data/pgo/ when --clean is set — this runs:

  build/bin/gtsam_gnc_pgo <pyfg> <init.tum> <final.tum> [--init-seed S ...]

parses the GTSAM_GNC_RESULT summary line, and writes one JSON
(gtsam_gnc_gm.json) under examples/data/analysis/pgo_outlier_gtsam/ (or
pgo_outlier_gtsam_randinit/ for seeded runs).

Output JSON layout:
  - single-init: {dataset_key: {metrics...}}
  - randinit:    {dataset_key: {seed_str: {metrics...}}}

Metrics: outer/inner iters are NOT exposed by GTSAM's GNC summary (it bundles
them internally), so we record:
  initial_cost, final_cost, inliers, outliers, total_s, dim, edges.

Usage:
  python3 examples/gtsam_gnc_pgo_sweep.py
  python3 examples/gtsam_gnc_pgo_sweep.py --seeds 1 2 3 4 5
  python3 examples/gtsam_gnc_pgo_sweep.py --clean      # use uncorrupted PGO data
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent.parent
BIN = REPO / "build" / "bin" / "gtsam_gnc_pgo"

OUTLIER_ROOT = REPO / "examples" / "data" / "pgo_outliers"
CLEAN_ROOT = REPO / "examples" / "data" / "pgo"

ANALYSIS_OUT = REPO / "examples" / "data" / "analysis" / "pgo_outlier_gtsam"
ANALYSIS_OUT_RANDINIT = REPO / "examples" / "data" / "analysis" / "pgo_outlier_gtsam_randinit"
ANALYSIS_OUT_CLEAN = REPO / "examples" / "data" / "analysis" / "pgo_gtsam"

RESULT_RE = re.compile(r"^GTSAM_GNC_RESULT\s+(.+)$", re.MULTILINE)


def parse_result(stdout: str) -> dict | None:
    m = RESULT_RE.search(stdout)
    if not m:
        return None
    out: dict = {}
    for tok in m.group(1).split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            if k in ("dim", "init_seed", "edges", "inliers", "outliers"):
                out[k] = int(v)
            elif k in ("total_s", "final_cost", "initial_cost"):
                out[k] = float(v)
            else:
                out[k] = v
        except ValueError:
            out[k] = v
    return out


def count_poses(pyfg: Path) -> int:
    n = 0
    with pyfg.open() as f:
        for raw in f:
            if raw.startswith("VERTEX_SE2 ") or raw.startswith("VERTEX_SE3:QUAT "):
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", action="store_true",
                     help="Run on examples/data/pgo/ (uncorrupted) instead of "
                          "examples/data/pgo_outliers/.")
    ap.add_argument("--barc-prob", type=float, default=0.99)
    ap.add_argument("--mu-step", type=float, default=1.4)
    ap.add_argument("--max-iters", type=int, default=100)
    ap.add_argument("--rel-tol", type=float, default=1e-5)
    ap.add_argument("--max-poses", type=int, default=None,
                     help="skip datasets whose VERTEX count exceeds this")
    ap.add_argument("--skip", type=str, nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                     help="run a sweep over these init-seed values; output goes to "
                          "pgo_outlier_gtsam_randinit/ and JSON is nested by seed")
    ap.add_argument("--init-noise-rot-deg", type=float, default=2.0)
    ap.add_argument("--init-noise-trans", type=float, default=0.05)
    ap.add_argument("--tum-dir", type=Path,
                     default=Path("/tmp/pgo_gtsam_tums"))
    args = ap.parse_args()

    if not BIN.exists():
        print(f"missing {BIN} — build with `cmake --build build --target gtsam_gnc_pgo`")
        return 1

    seeds = list(args.seeds) if args.seeds is not None else [0]
    randinit = args.seeds is not None
    if args.clean:
        data_root = CLEAN_ROOT
        out_dir = ANALYSIS_OUT_CLEAN
    else:
        data_root = OUTLIER_ROOT
        out_dir = ANALYSIS_OUT_RANDINIT if randinit else ANALYSIS_OUT
    if not data_root.exists():
        print(f"missing data root {data_root}")
        return 1

    args.tum_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    skip_set = set(args.skip)
    candidates = sorted([d for d in data_root.iterdir() if d.is_dir()])
    datasets: list[Path] = []
    for ds in candidates:
        if ds.name in skip_set:
            continue
        pyfg = next(ds.glob("*.pyfg"), None)
        if pyfg is None:
            continue
        if args.max_poses is not None and count_poses(pyfg) > args.max_poses:
            continue
        datasets.append(ds)
    if not datasets:
        print(f"no datasets matched under {data_root}")
        return 1
    print(f"running on {len(datasets)} datasets: "
          + ", ".join(d.name for d in datasets))

    results: dict[str, dict] = {}
    t_sweep = time.time()
    for ds in datasets:
        pyfg = next(ds.glob("*.pyfg"))
        ds_key = ds.name
        if randinit:
            results.setdefault(ds_key, {})
        for seed in seeds:
            seed_suffix = f"_s{seed}" if randinit else ""
            init_tum = args.tum_dir / f"{ds_key}{seed_suffix}_init.tum"
            final_tum = args.tum_dir / f"{ds_key}{seed_suffix}_final.tum"
            cmd = [str(BIN), str(pyfg), str(init_tum), str(final_tum),
                    "--barc-prob", str(args.barc_prob),
                    "--mu-step", str(args.mu_step),
                    "--max-iters", str(args.max_iters),
                    "--rel-tol", str(args.rel_tol)]
            if randinit:
                cmd += ["--init-seed", str(seed),
                        "--init-noise-rot-deg", str(args.init_noise_rot_deg),
                        "--init-noise-trans", str(args.init_noise_trans)]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            wall = time.time() - t0
            label = f"{ds_key}{seed_suffix}"
            if proc.returncode != 0:
                print(f"  FAIL  {label:<26}  exit={proc.returncode}  wall={wall:.1f}s")
                metrics = {"error": proc.stderr.strip()[:300],
                           "exit_code": proc.returncode}
            else:
                r = parse_result(proc.stdout)
                if r is None:
                    print(f"  PARSE  {label:<26}  no GTSAM_GNC_RESULT in stdout")
                    metrics = {"error": "no GTSAM_GNC_RESULT parsed",
                               "stdout_tail": proc.stdout.strip()[-200:]}
                else:
                    metrics = {
                        "dim": r.get("dim"),
                        "edges": r.get("edges"),
                        "inliers": r.get("inliers"),
                        "outliers": r.get("outliers"),
                        "initial_cost": r.get("initial_cost"),
                        "final_cost": r.get("final_cost"),
                        "total_s": r.get("total_s"),
                    }
                    print(f"  {label:<26}  in/out={r.get('inliers'):>5}/{r.get('outliers'):<4} "
                          f"init_cost={r.get('initial_cost', 0.0):>10.3g} "
                          f"final={r.get('final_cost', 0.0):>10.3g} "
                          f"wall={r.get('total_s', 0.0):>6.2f}s")
            if randinit:
                results[ds_key][str(seed)] = metrics
            else:
                results[ds_key] = metrics

    elapsed = time.time() - t_sweep
    print(f"\nsweep took {elapsed:.1f}s wall over {len(datasets)} datasets x "
          f"{len(seeds)} seed(s)")
    out_path = out_dir / "gtsam_gnc_gm.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path.relative_to(REPO)} ({len(results)} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
