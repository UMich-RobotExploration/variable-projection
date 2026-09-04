#!/usr/bin/env python3
"""Sweep the corrupted PGO datasets through IRLS (no GNC, odom init only).

For every dataset in examples/data/pgo_outliers/ and every method combo
(formulation × kernel), run the IRLS solver, parse the IRLS_RESULT line, and
record:
  - outer (IRLS) iterations
  - total inner solver iterations
  - total wall-clock time (s)
  - first Cholesky precompute time (s) — the initial fillImplicitFormulationMatrices

One JSON per method is emitted under examples/data/analysis/pgo_outlier_irls/:
  explicit_gm.json, expvp_gm.json, impl_gm.json
  explicit_tls.json, expvp_tls.json, impl_tls.json

Each JSON is keyed by "<dataset>_o<pct>" (e.g. "M3500_o30") and maps to a dict
with the four metrics + final robust cost.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO / "examples" / "data" / "pgo_outliers"
BIN = REPO / "build" / "bin" / "irls_robust"
ANALYSIS_DIR = REPO / "examples" / "data" / "analysis" / "pgo_outlier_irls"
ANALYSIS_DIR_RANDINIT = REPO / "examples" / "data" / "analysis" / "pgo_outlier_irls_randinit"

FORMULATIONS = ["explicit", "expvp", "impl"]
DEFAULT_KERNELS = ["gm", "tls"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]

RESULT_RE = re.compile(r"^IRLS_RESULT\s+(.+)$", re.MULTILINE)


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
            if k in ("dim", "outer", "inner", "gnc"):
                out[k] = int(v)
            elif k in ("total_s", "final_cost", "robust_cost", "precompute_s"):
                out[k] = float(v)
            else:
                out[k] = v
        except ValueError:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-irls", type=int, default=30)
    ap.add_argument("--c2", type=float, default=25.0)
    ap.add_argument("--rel-tol", type=float, default=1e-4)
    ap.add_argument("--max-poses", type=int, default=None,
                     help="skip datasets whose VERTEX_SE2/SE3:QUAT count exceeds this")
    ap.add_argument("--skip", type=str, nargs="*", default=[],
                     help="dataset folder names to skip (e.g. city10000_o10)")
    ap.add_argument("--kernels", nargs="+", choices=("gm", "tls"),
                     default=DEFAULT_KERNELS,
                     help="which robust kernels to sweep (default: both)")
    ap.add_argument("--tum-dir", type=Path, default=Path("/tmp/pgo_irls_tums"),
                     help="directory for per-combo TUM outputs (default /tmp/pgo_irls_tums)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                     help="run a sweep over these init-seed values (each seed perturbs the "
                          "odometry init differently). When set, outputs go to "
                          "pgo_outlier_irls_randinit/ and JSON is nested by seed.")
    ap.add_argument("--init-noise-rot-deg", type=float, default=2.0,
                     help="per-edge rotation noise std (degrees) when --seeds is used")
    ap.add_argument("--init-noise-trans", type=float, default=0.05,
                     help="per-edge translation noise std when --seeds is used")
    args = ap.parse_args()
    kernels = list(args.kernels)
    seeds = list(args.seeds) if args.seeds is not None else [0]
    randinit = args.seeds is not None
    out_dir = ANALYSIS_DIR_RANDINIT if randinit else ANALYSIS_DIR

    if not BIN.exists():
        print(f"missing {BIN} — build with `cmake --build build --target irls_robust`")
        return 1
    if not DATA_ROOT.exists():
        print(f"missing {DATA_ROOT} — generate with examples/inject_pgo_outliers.py")
        return 1

    args.tum_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    skip_set = set(args.skip)
    candidate_datasets = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])

    def count_poses(pyfg: Path) -> int:
        n = 0
        with pyfg.open() as f:
            for raw in f:
                if raw.startswith("VERTEX_SE2 ") or raw.startswith("VERTEX_SE3:QUAT "):
                    n += 1
        return n

    datasets: list[Path] = []
    for ds in candidate_datasets:
        if ds.name in skip_set:
            continue
        if args.max_poses is not None:
            pyfg = next(ds.glob("*.pyfg"), None)
            if pyfg is None or count_poses(pyfg) > args.max_poses:
                continue
        datasets.append(ds)
    if not datasets:
        print(f"no datasets matched filters under {DATA_ROOT}")
        return 1
    print(f"running on {len(datasets)} datasets: "
           + ", ".join(d.name for d in datasets))

    # method_key -> {dataset_key: metrics dict}    (single-init, original layout)
    # method_key -> {dataset_key: {seed_str: metrics}}    (randinit layout)
    results: dict[str, dict[str, dict]] = {
        f"{form}_{kernel}": {} for form in FORMULATIONS for kernel in kernels
    }

    t_sweep = time.time()
    for ds in datasets:
        pyfg = next(ds.glob("*.pyfg"), None)
        if pyfg is None:
            continue
        ds_key = ds.name  # e.g. "M3500_o30"
        for form in FORMULATIONS:
            for kernel in kernels:
                method_key = f"{form}_{kernel}"
                if randinit:
                    results[method_key].setdefault(ds_key, {})
                for seed in seeds:
                    seed_suffix = f"_s{seed}" if randinit else ""
                    init_tum = args.tum_dir / f"{ds_key}{seed_suffix}_init.tum"
                    final_tum = args.tum_dir / f"{ds_key}{seed_suffix}_{method_key}.tum"
                    cmd = [str(BIN), str(pyfg), str(init_tum), str(final_tum),
                            "--kernel", kernel,
                            "--formulation", form,
                            "--c2", str(args.c2),
                            "--max-irls", str(args.max_irls),
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
                        print(f"  FAIL  {label:<26} {method_key:<14}  "
                              f"exit={proc.returncode}  wall={wall:.1f}s")
                        metrics = {
                            "error": proc.stderr.strip()[:300],
                            "exit_code": proc.returncode,
                        }
                    else:
                        r = parse_result(proc.stdout)
                        if r is None:
                            print(f"  PARSE  {label:<26} {method_key:<14}  "
                                  f"no IRLS_RESULT in stdout")
                            metrics = {
                                "error": "no IRLS_RESULT parsed",
                                "stdout_tail": proc.stdout.strip()[-200:],
                            }
                        else:
                            metrics = {
                                "outer_iters": r.get("outer"),
                                "inner_iters": r.get("inner"),
                                "total_s": r.get("total_s"),
                                "precompute_s": r.get("precompute_s"),
                                "final_cost": r.get("final_cost"),
                                "robust_cost": r.get("robust_cost"),
                                "dim": r.get("dim"),
                            }
                            print(f"  {label:<26} {method_key:<14}  "
                                  f"outer={r.get('outer'):>3} inner={r.get('inner'):>5} "
                                  f"prep={r.get('precompute_s', 0.0):.4f}s  "
                                  f"total={r.get('total_s', 0.0):>7.2f}s  "
                                  f"robust={r.get('robust_cost', 0.0):.3g}")
                    if randinit:
                        results[method_key][ds_key][str(seed)] = metrics
                    else:
                        results[method_key][ds_key] = metrics

    elapsed = time.time() - t_sweep
    n_combos = len(FORMULATIONS) * len(kernels) * len(seeds)
    print(f"\nsweep took {elapsed:.1f}s wall over {len(datasets)} datasets x "
          f"{n_combos} combos")

    for method_key, dataset_results in results.items():
        out_path = out_dir / f"{method_key}.json"
        out_path.write_text(json.dumps(dataset_results, indent=2))
        print(f"wrote {out_path.relative_to(REPO)} "
              f"({len(dataset_results)} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
