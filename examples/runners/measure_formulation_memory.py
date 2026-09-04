#!/usr/bin/env python3
"""Peak-RAM measurement per formulation on the standard benchmark datasets.

Peak RSS is a *per-process* quantity, so one formulation is isolated per
process via the VARPRO_FORMULATION env var that paper_experiments.cpp reads
(see its getFormulationsToSweep). Each dataset is also isolated: the sweep
discovers "any leaf dir holding a .pyfg" under abs_data_path, so we point it at
a scratch dir holding symlinks to one dataset's .pyfg and inits/. Symlinking
rather than copying keeps the real dataset dirs untouched, and leaving
cached_results/ out of the scratch copy forces a real solve instead of a
cache hit.

Peak memory is allocated up front (the operator, and for Dense the p^2 reduced
system), so runs are capped at VARPRO_MAX_ITERATIONS=1 -- the peak is reached
during precompute and is unaffected by how long the solve then runs.

Measurement is wait4() rusage (GNU time's %M), which is exact regardless of
how briefly the process lives -- the small datasets finish in under 20 ms, so
polling /proc/<pid>/status misses VmHWM and reports ~0. `timeout` is nested
inside `time` so a killed run still reports its peak.

Dense is subject to the config's max_dense_gb: runs that would exceed it are
recorded as skipped with the dense_gb they would have needed, matching the
standard sweep's behaviour rather than inventing a 40 GB allocation.

Output: examples/data/analysis/formulation_memory.json

Usage:
  python3 examples/measure_formulation_memory.py [--datasets NAME ...]
                                                 [--max-dense-gb 8]
                                                 [--timeout 600]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DATA = REPO / "examples" / "data"
ANALYSIS = REPO / "examples" / "data" / "analysis"
OUT_JSON = ANALYSIS / "formulation_memory.json"
BIN = Path(os.environ.get("VARPRO_PAPER_BIN", REPO / "build" / "bin"
                          / "paper_experiments"))

# Same exclusions the standard sweep uses (examples/config.json).
SKIP = ("cosmobench", "nebula", "pgo_outliers", "sphere_sweep")
FAMILIES = ("pgo", "raslam", "snl", "sfm")
FORMULATIONS = ["Explicit", "ExplicitVarPro", "Implicit", "Dense"]
RANK = 5


def list_datasets() -> list[Path]:
    """Leaf dirs holding a .pyfg, minus the skipped families."""
    out = []
    for pyfg in sorted(DATA.rglob("*.pyfg")):
        rel = pyfg.relative_to(DATA).as_posix()
        if any(s in rel for s in SKIP):
            continue
        if pyfg.stem.endswith("_no_outliers"):
            continue
        if pyfg.parent not in out:
            out.append(pyfg.parent)
    return out


def family_of(ds: Path) -> str:
    rel = ds.relative_to(DATA).as_posix()
    return next((f for f in FAMILIES if rel.startswith(f)), "?")


def stage(ds: Path, scratch: Path) -> Path:
    """Symlink one dataset into an isolated tree; return the abs_data_path."""
    root = scratch / "data"
    if root.exists():
        shutil.rmtree(root)
    leaf = root / ds.name
    (leaf / "inits").mkdir(parents=True, exist_ok=True)
    for pyfg in ds.glob("*.pyfg"):
        (leaf / pyfg.name).symlink_to(pyfg.resolve())
    src_inits = ds / "inits"
    if src_inits.is_dir():
        for f in src_inits.iterdir():
            (leaf / "inits" / f.name).symlink_to(f.resolve())
    return root


def peak_rss_kb(cmd: list[str], env: dict[str, str], timeout_s: float):
    """Run cmd, return (peak_rss_kb, status, output_tail).

    Peak RSS comes from wait4() rusage via GNU time's %M, not from polling
    /proc: the small datasets finish in under 20 ms, so any poll interval
    misses VmHWM entirely and reports ~0. `timeout` sits *inside* `time` so
    that a killed run still gets its rusage reported -- ru_maxrss is the max
    over the child and its reaped descendants.
    """
    wrapped = ["/usr/bin/time", "-f", "PEAKKB %M",
               "timeout", "-s", "KILL", str(int(timeout_s))] + cmd
    proc = subprocess.run(wrapped, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True)
    peak = 0
    for ln in proc.stderr.splitlines():
        if ln.startswith("PEAKKB "):
            try:
                peak = int(ln.split()[1])
            except (ValueError, IndexError):
                pass
    # timeout exits 137 when it SIGKILLs; time propagates the child's status.
    if proc.returncode == 137:
        status = "timeout"
    elif proc.returncode == 0:
        status = "ok"
    else:
        status = f"exit {proc.returncode}"
    tail = (proc.stdout + proc.stderr)[-1500:]
    return peak, status, tail


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--max-dense-gb", type=float, default=8.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--scratch", type=Path,
                    default=Path(os.environ.get("TMPDIR", "/tmp"))
                    / "formulation_memory")
    args = ap.parse_args()

    if not BIN.exists():
        print(f"missing binary: {BIN}", file=sys.stderr)
        return 1

    datasets = list_datasets()
    if args.datasets:
        want = set(args.datasets)
        datasets = [d for d in datasets if d.name in want]
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    out: dict[str, object] = {
        "metric": "peak_rss_mb",
        "note": ("wait4 ru_maxrss per process; one formulation per process "
                 "via VARPRO_FORMULATION; VARPRO_MAX_ITERATIONS=1"),
        "max_dense_gb": args.max_dense_gb,
        "rank": RANK,
        "datasets": {},
    }
    print(f"measuring {len(datasets)} datasets x {len(FORMULATIONS)} "
          f"formulations\n")
    t_all = time.time()

    for i, ds in enumerate(datasets, 1):
        root = stage(ds, args.scratch)
        cfg = {
            "verbose": False,
            "min_rank": RANK, "max_rank": RANK,
            "abs_data_path": str(root) + "/",
            "num_inits": 1,
            "scale_reg_weight": 0.01,
            "max_dense_gb": args.max_dense_gb,
            "skip_substrings": [],
            "sfm_use_scaled_stiefel": False,
        }
        entry = {"family": family_of(ds),
                 "path": ds.relative_to(REPO).as_posix(), "methods": {}}
        print(f"[{i}/{len(datasets)}] {ds.name} ({entry['family']})")
        for form in FORMULATIONS:
            cfg_path = args.scratch / f"config_{form}.json"
            cfg_path.write_text(json.dumps({**cfg, "formulations": [form]}))
            env = {**os.environ, "VARPRO_FORMULATION": form,
                   "VARPRO_MAX_ITERATIONS": "1"}
            ts = time.time()
            kb, status, tail = peak_rss_kb([str(BIN), str(cfg_path)], env,
                                           args.timeout)
            rec: dict[str, object] = {"peak_rss_mb": round(kb / 1024.0, 1),
                                      "status": status,
                                      "wall_s": round(time.time() - ts, 1)}
            if "dense_too_large" in tail:
                rec["status"] = "dense_too_large"
            entry["methods"][form] = rec
            flag = "" if rec["status"] == "ok" else f"  [{rec['status']}]"
            print(f"    {form:<16} {rec['peak_rss_mb']:>10.1f} MB"
                  f"  ({rec['wall_s']:>6.1f}s){flag}")
        out["datasets"][ds.name] = entry
        OUT_JSON.write_text(json.dumps(out, indent=2))

    print(f"\ntotal {time.time() - t_all:.0f}s")
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
