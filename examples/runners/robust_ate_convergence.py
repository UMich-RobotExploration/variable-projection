#!/usr/bin/env python3
"""ATE-vs-runtime convergence traces for the robust (IRLS + GNC) experiments.

Runs the four requested cosmobench datasets under each method with
--dump-iterates, so every outer GNC iteration leaves a .tum behind, then
computes global ATE for each of those iterates against ground truth. The
result is a real convergence curve: ATE on the y-axis against cumulative
solver wall time on the x-axis.

This needs instrumentation that the stored sweep results do not have -- they
hold only one final ATE and one wall_s per method. irls_robust.cpp and
~/varProj-gtsam's SESync_GNC_example.cpp both now emit a per-iteration
`... _ITER k=<k> t=<cumulative_s> ...` line and honour --dump-iterates; dump
I/O is subtracted from `t`, so the last point matches each binary's own
published wall time.

Methods:
  Original / Original + V.P. / Ours   examples/irls_robust.cpp
  GTSAM                               ~/varProj-gtsam SESync_GNC_example,
                                      the residual-matched (chordal SEsyncFactor)
                                      configuration -- the only GTSAM variant
                                      that optimises the same objective we do.

ATE is `ate_global_rmse_m`: one Umeyama alignment over all robots, matching
the numbers in the existing tables. Priors are stripped, as in the sweep.

Usage:
  python3 examples/robust_ate_convergence.py [--datasets a b] [--reuse]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# run_cosmobench_irls_gnc_sweep.py lives one level up, at examples/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_cosmobench_irls_gnc_sweep import (  # noqa: E402
    compute_ate, parse_pyfg_vertices, parse_tum_xyz, strip_priors_to,
)

REPO = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = REPO / "examples" / "data" / "analysis"
OUT_JSON = ANALYSIS_DIR / "robust_ate_convergence.json"

IRLS_BIN = Path(os.environ.get("VARPRO_IRLS_BIN",
                                REPO / "build" / "bin" / "irls_robust"))
SESYNC_BIN = Path(os.environ.get(
    "VARPRO_SESYNC_BIN",
    Path.home() / "varProj-gtsam" / "cmake-build-default" / "bin"
    / "SESync_GNC_example"))

DATASETS = {
    "kittredge_loop_wifi": "cosmobench/wifi",
    "kittredge_loop_proradio": "cosmobench/proradio",
    "main_campus_wifi": "cosmobench/wifi",
    "main_campus_proradio": "cosmobench/proradio",
}

# method key -> (label, binary kind, formulation flag)
METHODS = [
    ("explicit", "Original", "irls"),
    ("expvp", "Original + V.P.", "irls"),
    ("impl", "Ours", "irls"),
    ("gtsam", "GTSAM", "sesync"),
]

ITER_RE = re.compile(
    r"(?:IRLS|SESYNC_GNC)_ITER\s+k=(\d+)\s+t=(\S+)\s+c2_eff=(\S+)\s+"
    r"robust_cost=(\S+)\s+inner_cost=(\S+)\s+inner_iters=(\d+)")


def run_traced(cmd: list[str], dump_dir: Path, timeout_s: float):
    """Run a solver with --dump-iterates; return (iters, stdout)."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    for stale in dump_dir.glob("iter_*.tum"):
        stale.unlink()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}):\n"
                           f"{proc.stderr[-2000:]}")
    iters = []
    for m in ITER_RE.finditer(proc.stdout):
        iters.append({"k": int(m.group(1)), "t_s": float(m.group(2)),
                      "c2_eff": float(m.group(3)),
                      "robust_cost": float(m.group(4)),
                      "inner_cost": float(m.group(5)),
                      "inner_iters": int(m.group(6))})
    if not iters:
        raise RuntimeError(f"no _ITER lines from {cmd[0]}; "
                           "is the instrumented binary built?")
    return iters, proc.stdout


def trace_for(method: str, kind: str, pyfg_no_priors: Path, scratch: Path,
              gt_xyz: np.ndarray, robot_letters: np.ndarray, timeout_s: float):
    """Per-iteration (t, ATE) for one method on one dataset."""
    dump_dir = scratch / f"dump_{method}"
    if kind == "irls":
        cmd = [str(IRLS_BIN), str(pyfg_no_priors),
               str(scratch / f"init_{method}.tum"),
               str(scratch / f"final_{method}.tum"),
               "--kernel", "gm", "--formulation", method, "--gnc",
               "--dump-iterates", str(dump_dir)]
    else:
        cmd = [str(SESYNC_BIN), "3", "3", str(pyfg_no_priors),
               str(scratch / f"final_{method}.tum"), "--gnc",
               "--dump-iterates", str(dump_dir)]

    iters, _out = run_traced(cmd, dump_dir, timeout_s)

    pts = []
    for rec in iters:
        tum = dump_dir / f"iter_{rec['k']:03d}.tum"
        if not tum.exists():
            continue
        est = parse_tum_xyz(tum)
        if est.shape[0] != gt_xyz.shape[0]:
            raise RuntimeError(f"{tum.name}: {est.shape[0]} rows vs "
                               f"{gt_xyz.shape[0]} gt rows")
        ate = compute_ate(est, gt_xyz, robot_letters)
        pts.append({**rec, "ate_global_rmse_m": ate["ate_global_rmse_m"]})
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--scratch", type=Path,
                    default=Path(os.environ.get("TMPDIR", "/tmp"))
                    / "robust_ate_convergence")
    args = ap.parse_args()

    for b, what in ((IRLS_BIN, "irls_robust"), (SESYNC_BIN, "SESync_GNC")):
        if not b.exists():
            print(f"missing {what} binary: {b}", file=sys.stderr)
            return 1

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"ate_metric": "ate_global_rmse_m", "datasets": {}}
    t0 = time.time()

    for name in args.datasets:
        sub = DATASETS[name]
        pyfg = REPO / "examples" / "data" / sub / f"{name}.pyfg"
        if not pyfg.exists():
            print(f"skip {name}: no {pyfg}", file=sys.stderr)
            continue
        scratch = args.scratch / name
        scratch.mkdir(parents=True, exist_ok=True)
        pyfg_np = scratch / f"{name}_no_priors.pyfg"
        n_stripped = strip_priors_to(pyfg, pyfg_np)

        names, gt_xyz = parse_pyfg_vertices(pyfg_np)
        robot_letters = np.array([n[0] for n in names])

        entry = {"n_poses": len(names), "n_priors_stripped": n_stripped,
                 "methods": {}}
        print(f"\n=== {name} ({len(names)} poses, "
              f"{n_stripped} priors stripped) ===")
        for method, label, kind in METHODS:
            ts = time.time()
            try:
                pts = trace_for(method, kind, pyfg_np, scratch,
                                gt_xyz, robot_letters, args.timeout)
            except Exception as exc:               # noqa: BLE001
                print(f"  {label:<18} FAILED: {exc}", file=sys.stderr)
                entry["methods"][method] = {"label": label,
                                            "status": f"failed: {exc}"}
                continue
            entry["methods"][method] = {"label": label, "status": "ok",
                                        "points": pts}
            print(f"  {label:<18} {len(pts):>2} iters, "
                  f"t={pts[-1]['t_s']:>7.2f}s  "
                  f"ATE {pts[0]['ate_global_rmse_m']:>7.2f} -> "
                  f"{pts[-1]['ate_global_rmse_m']:.3f} m "
                  f"({time.time() - ts:.0f}s wall)")
        # t=0 anchor. All four methods start from the same chained-odometry
        # init -- irls_robust writes it per formulation and the three agree
        # exactly, and SESync_GNC_example's odometryInit chains the same
        # measurements -- so one shared value anchors every curve.
        init_tum = scratch / "init_impl.tum"
        if init_tum.exists():
            est = parse_tum_xyz(init_tum)
            if est.shape[0] == gt_xyz.shape[0]:
                entry["init_ate_global_rmse_m"] = compute_ate(
                    est, gt_xyz, robot_letters)["ate_global_rmse_m"]
                print(f"  {'odometry init':<18} "
                      f"ATE {entry['init_ate_global_rmse_m']:.2f} m (t=0)")
        out["datasets"][name] = entry

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\ntotal {time.time() - t0:.0f}s")
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
