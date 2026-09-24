"""Sweep IRLS-GNC (GM kernel) across all cosmobench + nebula datasets.

For each dataset, with priors stripped (the JRL priors have variance 1e-8/1e-6
which makes the Hessian ill-conditioned and crashes the inner solver):

  1. Run `build/bin/irls_robust --gnc --kernel gm --formulation {explicit, expvp, impl}`
     (pass --include-dense to add the `dense` baseline, which forms the reduced
     system explicitly and re-forms it every IRLS outer iteration)
     once per init seed, from the same odometry init.
  2. Compute ATE (global Umeyama + per-robot Umeyama) against ground truth.
     Results are JSON only -- this script collects data and renders nothing.
  3. Write a per-dataset JSON with all the metrics for the table.
  4. Render the 4-panel trajectory plot (GT + 3 methods, per-robot aligned).

Multiple inits (--seeds): seed 0 is noiseless odometry; seed > 0 perturbs every
sequential odometry edge by a random SE(d) with std --init-noise-rot-deg /
--init-noise-trans before chaining, so the drift is deterministic per seed and
independent of the formulation. Every formulation sees the *same* init for a
given seed, and the init is written once per (dataset, seed) to
`<out>/inits/<dataset>/init_s<seed>.tum` so the GTSAM/SESync baseline can
consume the identical iterate via its --init-tum flag.

`dense` is run only for the first seed: at a 70 s median wall (vs 2.3 s for
`impl`) it would dominate the sweep cost, and it reaches the same optimum as
`impl`, so its ATE distribution over inits adds nothing.

Results are written **immediately** after each dataset finishes so you can
inspect progress without waiting for the whole sweep.

Usage:
  python3 examples/run_cosmobench_irls_gnc_sweep.py \
      --out examples/data/analysis/cosmobench_irls_gnc_full

  # the 5-init configuration used for the paper
  python3 examples/run_cosmobench_irls_gnc_sweep.py \
      --seeds 0 1 2 3 4 --init-noise-rot-deg 0.5 --init-noise-trans 0.02 \
      --include-dense \
      --out examples/data/analysis/cosmobench_irls_gnc_5init
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


REPO = Path(__file__).resolve().parent.parent
IRLS_BIN = Path(os.environ.get("VARPRO_IRLS_BIN",
                                 REPO / "build" / "bin" / "irls_robust"))

DATASETS_ROOTS = [
    REPO / "examples" / "data" / "cosmobench" / "wifi",
    REPO / "examples" / "data" / "cosmobench" / "proradio",
    REPO / "examples" / "data" / "nebula",
]

# The three formulations the paper compares. "dense" (explicitly-formed Schur
# complement) is opt-in via --include-dense: it is a memory/time baseline, and
# under IRLS it re-forms the p x p reduced system on every outer iteration.
METHODS = ["explicit", "expvp", "impl"]
DENSE_METHOD = "dense"

# Labels used in summary.json / summary.csv, i.e. the names the paper table uses.
METHOD_LABELS = {"explicit": "Explicit", "expvp": "ExpVP",
                 "impl": "Implicit", "dense": "Dense"}

# Scalars aggregated across init seeds. Every one of these is a per-run scalar
# in the JSON, so median/min/max over seeds is well defined.
AGG_METRICS = ["outer_iters", "inner_iters_total", "wall_s", "precompute_s",
               "total_precompute_s", "inner_cost", "robust_cost",
               "ate_global_rmse_m", "ate_per_robot_mean_m"]

IRLS_RESULT_RE = re.compile(
    r"IRLS_RESULT\s+kernel=(\S+)\s+form=(\S+)\s+gnc=(\d+)\s+init_seed=(\d+)\s+"
    r"dim=(\d+)\s+outer=(\d+)\s+inner=(\d+)\s+final_cost=(\S+)\s+"
    r"robust_cost=(\S+)\s+total_s=(\S+)\s+precompute_s=(\S+)"
)
# Emitted only by builds that know about the Dense formulation; parsed
# separately so older result lines still match the main regex.
TOTAL_PRECOMPUTE_RE = re.compile(r"total_precompute_s=(\S+)")
DENSE_GB_RE = re.compile(r"dense_gb=(\S+)")


# ---------------------------------------------------------------------------
# PyFG parsing helpers
# ---------------------------------------------------------------------------


def strip_priors_to(pyfg_in: Path, pyfg_out: Path) -> int:
    """Write pyfg_in to pyfg_out with all VERTEX_SE3:QUAT:PRIOR lines dropped.

    Returns the number of dropped prior lines.
    """
    src = pyfg_in.read_text().splitlines()
    n_priors = 0
    out_lines: List[str] = []
    for line in src:
        if line.startswith("VERTEX_SE3:QUAT:PRIOR"):
            n_priors += 1
            continue
        out_lines.append(line)
    pyfg_out.write_text("\n".join(out_lines) + "\n")
    return n_priors


def parse_pyfg_vertices(pyfg: Path) -> Tuple[List[str], np.ndarray]:
    """Return (names_in_file_order, gt_xyz_Nx3)."""
    names, xyz = [], []
    for ln in pyfg.read_text().splitlines():
        parts = ln.split()
        if parts and parts[0] == "VERTEX_SE3:QUAT":
            names.append(parts[2])
            xyz.append([float(parts[3]), float(parts[4]), float(parts[5])])
    return names, np.asarray(xyz, dtype=float)


def count_edges(pyfg: Path) -> int:
    n = 0
    for ln in pyfg.read_text().splitlines():
        if ln.startswith("EDGE_SE3:QUAT"):
            n += 1
    return n


def parse_tum_xyz(tum: Path) -> np.ndarray:
    rows = [list(map(float, ln.split())) for ln in tum.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    return np.asarray(rows, dtype=float)[:, 1:4]


# ---------------------------------------------------------------------------
# Alignment + RMSE
# ---------------------------------------------------------------------------


def umeyama_no_scale(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = src - src.mean(0)
    B = dst - dst.mean(0)
    U, _S, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = Vt.T @ D @ U.T
    t = dst.mean(0) - R @ src.mean(0)
    return R, t


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def compute_ate(est_xyz: np.ndarray, gt_xyz: np.ndarray,
                robot_letters: np.ndarray) -> Dict[str, object]:
    """Compute global + per-robot ATE numbers used in the report."""
    # global alignment
    R, t = umeyama_no_scale(est_xyz, gt_xyz)
    est_g = est_xyz @ R.T + t
    overall_global = rmse(est_g, gt_xyz)
    per_robot_global = {}
    per_robot_aligned = {}
    for r in sorted(set(robot_letters)):
        m = robot_letters == r
        per_robot_global[r] = rmse(est_g[m], gt_xyz[m])
        Rr, tr = umeyama_no_scale(est_xyz[m], gt_xyz[m])
        per_robot_aligned[r] = rmse(est_xyz[m] @ Rr.T + tr, gt_xyz[m])
    return {
        "ate_global_rmse_m": overall_global,
        "ate_global_per_robot_rmse_m": per_robot_global,
        "ate_per_robot_aligned_rmse_m": per_robot_aligned,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def parse_irls_result(stdout: str) -> Optional[Dict[str, object]]:
    for line in stdout.splitlines()[::-1]:
        m = IRLS_RESULT_RE.search(line)
        if m:
            return {
                "kernel": m.group(1),
                "form": m.group(2),
                "gnc": bool(int(m.group(3))),
                "init_seed": int(m.group(4)),
                "dim": int(m.group(5)),
                "outer_iters": int(m.group(6)),
                "inner_iters_total": int(m.group(7)),
                "inner_cost": float(m.group(8)),
                "robust_cost": float(m.group(9)),
                "wall_s": float(m.group(10)),
                "precompute_s": float(m.group(11)),
                # Total precompute across all IRLS outer iterations (IRLS calls
                # updateProblemData() once per iteration). Falls back to the
                # single-shot number for binaries that predate the field.
                "total_precompute_s": (
                    float(tp.group(1))
                    if (tp := TOTAL_PRECOMPUTE_RE.search(line))
                    else float(m.group(11))
                ),
                "dense_gb": (
                    float(dg.group(1))
                    if (dg := DENSE_GB_RE.search(line))
                    else 0.0
                ),
            }
    return None


def rel_to_repo(path: Path) -> str:
    """Repo-relative posix path, falling back to absolute if outside the repo."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path.resolve())


def run_one_method(pyfg_no_priors: Path, formulation: str, tum_dir: Path,
                    timeout_s: float, use_gpu: bool = False,
                    seed: int = 0, noise_rot_deg: float = 0.5,
                    noise_trans: float = 0.02,
                    init_tum: Optional[Path] = None
                    ) -> Tuple[Dict[str, object], Path]:
    # One init per (dataset, seed), shared by every formulation: chainOdometry()
    # runs before setFormulation(), so the iterate depends only on the seed and
    # the noise levels. Each formulation rewrites the same bytes here, and the
    # file is what the SESync baseline reads back via --init-tum.
    if init_tum is None:
        init_tum = tum_dir / f"init_s{seed}.tum"
    final_tum = tum_dir / f"final_{formulation}_s{seed}.tum"
    cmd = [
        str(IRLS_BIN), str(pyfg_no_priors), str(init_tum), str(final_tum),
        "--kernel", "gm", "--formulation", formulation, "--gnc",
        "--init-seed", str(seed),
        "--init-noise-rot-deg", str(noise_rot_deg),
        "--init-noise-trans", str(noise_trans),
    ]
    if use_gpu:
        # Runs each IRLS inner solve on the GPU RTR solver. The operator is
        # rebuilt every outer iteration because reweighting rewrites the
        # covariances, so small problems are dominated by that setup cost.
        cmd.append("--gpu")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=timeout_s, check=False)
        wall = time.time() - t0
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_s": time.time() - t0}, final_tum

    stats = parse_irls_result(result.stdout)
    if result.returncode != 0 or stats is None:
        # Capture the last few lines of stderr/stdout for the JSON record.
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-6:])
        return {
            "status": "crashed",
            "returncode": result.returncode,
            "wall_s": wall,
            "tail": tail,
        }, final_tum
    stats["status"] = "ok"
    return stats, final_tum


# ---------------------------------------------------------------------------
# Aggregation across init seeds
# ---------------------------------------------------------------------------


def add_per_robot_mean(stats: Dict[str, object]) -> None:
    """Flatten the per-robot aligned ATEs to the single mean the table quotes."""
    per_robot = stats.get("ate_per_robot_aligned_rmse_m")
    if isinstance(per_robot, dict) and per_robot:
        stats["ate_per_robot_mean_m"] = float(np.mean(list(per_robot.values())))


def _spread(vals: List[float]) -> Dict[str, float]:
    return {"median": statistics.median(vals), "min": min(vals),
            "max": max(vals), "n": len(vals)}


def aggregate_runs(runs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """median/min/max of each table metric over the *ok* seeds of one method."""
    ok = [s for s in runs.values() if s.get("status") == "ok"]
    out: Dict[str, object] = {}
    for key in AGG_METRICS:
        vals = [float(s[key]) for s in ok
                if isinstance(s.get(key), (int, float))]
        if vals:
            out[key] = _spread(vals)
    return out


def median_over_seeds(method_record: Dict[str, object], key: str) -> Optional[float]:
    """Per-dataset representative value for one method: median across seeds."""
    agg = method_record.get("aggregate", {}) if method_record else {}
    entry = agg.get(key) if isinstance(agg, dict) else None
    return entry["median"] if isinstance(entry, dict) else None


def write_summary(out_dir: Path) -> None:
    """Rebuild summary.json + summary.csv from the per-dataset JSONs.

    Two levels of aggregation, kept separate on purpose:

      per-dataset   median over the init seeds (`aggregate` in each JSON)
      per-method    median/mean over datasets of that per-dataset median,
                    plus `median_pooled` over every (dataset, seed) run and
                    `median_seed_range` (median over datasets of max-min across
                    seeds), which is the number that says whether the init
                    actually mattered.
    """
    paths = sorted(p for p in out_dir.glob("*.json") if p.stem != "summary")
    if not paths:
        print(f"no per-dataset JSONs in {out_dir}, skipping summary")
        return
    records = [json.loads(p.read_text()) for p in paths]

    forms: List[str] = []
    for rec in records:
        for form in rec.get("methods", {}):
            if form not in forms:
                forms.append(form)

    per_method: Dict[str, object] = {}
    for form in forms:
        label = METHOD_LABELS.get(form, form)
        med_by_metric: Dict[str, List[float]] = defaultdict(list)
        pooled: Dict[str, List[float]] = defaultdict(list)
        ranges: Dict[str, List[float]] = defaultdict(list)
        n_ok = n_runs = n_datasets = 0
        ate_all: List[float] = []
        for rec in records:
            mr = rec.get("methods", {}).get(form)
            if not mr:
                continue
            n_datasets += 1
            n_ok += mr.get("n_ok", 0)
            n_runs += mr.get("n_runs", 0)
            for metric, entry in mr.get("aggregate", {}).items():
                med_by_metric[metric].append(entry["median"])
                ranges[metric].append(entry["max"] - entry["min"])
            for run in mr.get("seeds", {}).values():
                if run.get("status") != "ok":
                    continue
                for metric in AGG_METRICS:
                    if isinstance(run.get(metric), (int, float)):
                        pooled[metric].append(float(run[metric]))
                if isinstance(run.get("ate_global_rmse_m"), (int, float)):
                    ate_all.append(float(run["ate_global_rmse_m"]))
        per_method[label] = {
            "median": {m: statistics.median(v) for m, v in med_by_metric.items()},
            "mean": {m: float(np.mean(v)) for m, v in med_by_metric.items()},
            "median_pooled": {m: statistics.median(v) for m, v in pooled.items()},
            "median_seed_range": {m: statistics.median(v)
                                   for m, v in ranges.items()},
            "n_datasets": n_datasets,
            "n_runs": n_runs,
            "n_ok": n_ok,
            # Robustness readout: how many individual inits landed near the
            # good optimum, over every (dataset, seed) pair.
            "n_runs_ate_under_3m": sum(1 for a in ate_all if a < 3.0),
            "n_runs_with_ate": len(ate_all),
        }

    first = records[0]
    summary = {
        "n_datasets": len(records),
        "kernel": first.get("kernel"),
        "gnc": first.get("gnc"),
        "init": first.get("init"),
        "init_noise": first.get("init_noise"),
        "seeds": first.get("seeds"),
        "priors": first.get("priors", "stripped"),
        "datasets": [r["dataset"] for r in records],
        "per_method_aggregate": per_method,
        "per_dataset": {
            r["dataset"]: {
                "n_poses": r.get("n_poses"),
                "n_edges": r.get("n_edges"),
                "n_outliers_truth": r.get("n_outliers_truth"),
                "methods": {f: mr.get("aggregate", {})
                             for f, mr in r.get("methods", {}).items()},
            }
            for r in records
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Wide per-dataset CSV: median/min/max across seeds for the two columns the
    # table reads (wall, ATE), medians only for the rest.
    head = ["dataset", "n_poses", "n_edges", "n_outliers_truth", "n_robots"]
    for form in forms:
        label = METHOD_LABELS.get(form, form)
        head += [f"{label}_n_ok",
                 f"{label}_wall_s_med", f"{label}_wall_s_min", f"{label}_wall_s_max",
                 f"{label}_ate_global_med", f"{label}_ate_global_min",
                 f"{label}_ate_global_max",
                 f"{label}_outer_iters_med", f"{label}_inner_iters_med",
                 f"{label}_robust_cost_med", f"{label}_ate_per_robot_mean_med"]
    lines = [",".join(head)]
    for rec in records:
        row = [rec["dataset"], str(rec.get("n_poses", "")),
               str(rec.get("n_edges", "")), str(rec.get("n_outliers_truth", "")),
               str(rec.get("n_robots", ""))]
        for form in forms:
            mr = rec.get("methods", {}).get(form, {})
            agg = mr.get("aggregate", {})

            def cell(metric: str, stat: str = "median", fmt: str = "{:.4f}") -> str:
                entry = agg.get(metric)
                return fmt.format(entry[stat]) if isinstance(entry, dict) else ""

            row += [str(mr.get("n_ok", "")),
                    cell("wall_s"), cell("wall_s", "min"), cell("wall_s", "max"),
                    cell("ate_global_rmse_m"), cell("ate_global_rmse_m", "min"),
                    cell("ate_global_rmse_m", "max"),
                    cell("outer_iters", fmt="{:.1f}"),
                    cell("inner_iters_total", fmt="{:.1f}"),
                    cell("robust_cost"), cell("ate_per_robot_mean_m")]
        lines.append(",".join(row))
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n")
    print(f"wrote {rel_to_repo(out_dir / 'summary.json')} and "
          f"{rel_to_repo(out_dir / 'summary.csv')}")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def list_datasets() -> List[Path]:
    out: List[Path] = []
    for root in DATASETS_ROOTS:
        if not root.exists():
            continue
        for p in sorted(root.glob("*.pyfg")):
            # We sweep on the *with-outlier* files (which is the .pyfg without
            # the `_no_outliers` suffix). Each cosmobench/nebula dataset has
            # both variants — we only want the one that exercises the robust
            # kernel.
            if p.stem.endswith("_no_outliers"):
                continue
            out.append(p)
    return out


def sweep(out_dir: Path, scratch_dir: Path, timeout_s: float,
          include_dense: bool = False, use_gpu: bool = False,
          seeds: Optional[List[int]] = None, noise_rot_deg: float = 0.5,
          noise_trans: float = 0.02,
          only: Optional[List[str]] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds) if seeds else [0]

    methods = METHODS + ([DENSE_METHOD] if include_dense else [])

    datasets = list_datasets()
    if only:
        wanted = set(only)
        datasets = [p for p in datasets if p.stem in wanted]
        missing = wanted - {p.stem for p in datasets}
        if missing:
            print(f"warning: no such dataset(s): {', '.join(sorted(missing))}",
                   file=sys.stderr)
    print(f"sweeping {len(datasets)} datasets into {out_dir} "
          f"(methods: {', '.join(methods)}; seeds: "
          f"{', '.join(str(s) for s in seeds)}; "
          f"noise {noise_rot_deg} deg / {noise_trans} m per odometry edge, "
          f"seed 0 = noiseless)\n")
    t_start = time.time()

    for idx, pyfg in enumerate(datasets, start=1):
        ds_t0 = time.time()
        name = pyfg.stem
        rel = pyfg.relative_to(REPO).as_posix()

        ds_scratch = scratch_dir / name
        ds_scratch.mkdir(parents=True, exist_ok=True)
        pyfg_no_priors = ds_scratch / f"{name}_no_priors.pyfg"
        n_priors_stripped = strip_priors_to(pyfg, pyfg_no_priors)

        # Inits live with the results, not in scratch: the SESync baseline
        # sweep reads them back to start from the identical iterate.
        init_dir = out_dir / "inits" / name
        init_dir.mkdir(parents=True, exist_ok=True)

        # Source metadata
        names, gt_xyz = parse_pyfg_vertices(pyfg_no_priors)
        robot_letters = np.array([n[0] for n in names])
        poses_per_robot = {r: int((robot_letters == r).sum())
                            for r in sorted(set(robot_letters))}
        n_edges_full = count_edges(pyfg)
        n_edges_clean = None
        clean_path = pyfg.with_name(pyfg.stem + "_no_outliers.pyfg")
        if clean_path.exists():
            n_edges_clean = count_edges(clean_path)

        record: Dict[str, object] = {
            "dataset": name,
            "source_pyfg": rel,
            "stripped_priors": True,
            "n_priors_stripped": n_priors_stripped,
            "kernel": "gm",
            "gnc": True,
            "init": "odometry" if seeds == [0] else "odometry+noise",
            "init_noise": {"rot_deg": noise_rot_deg, "trans": noise_trans,
                            "note": "seed 0 is noiseless"},
            "seeds": seeds,
            "n_poses": int(len(names)),
            "n_edges": int(n_edges_full),
            "n_edges_inlier_truth": (int(n_edges_clean) if n_edges_clean is not None
                                       else None),
            "n_outliers_truth": (int(n_edges_full - n_edges_clean)
                                   if n_edges_clean is not None else None),
            "n_robots": len(poses_per_robot),
            "robots": list(poses_per_robot.keys()),
            "poses_per_robot": poses_per_robot,
            "methods": {},
        }

        per_method_status = []
        for form in methods:
            # Dense runs on the first seed only -- see the module docstring.
            form_seeds = seeds[:1] if form == DENSE_METHOD else seeds
            runs: Dict[str, Dict[str, object]] = {}
            for seed in form_seeds:
                stats, final_tum = run_one_method(
                    pyfg_no_priors, form, ds_scratch, timeout_s,
                    use_gpu=use_gpu, seed=seed, noise_rot_deg=noise_rot_deg,
                    noise_trans=noise_trans,
                    init_tum=init_dir / f"init_s{seed}.tum")
                stats["seed"] = seed
                if stats.get("status") == "ok" and final_tum.exists():
                    est_xyz = parse_tum_xyz(final_tum)
                    if est_xyz.shape[0] == gt_xyz.shape[0]:
                        stats.update(compute_ate(est_xyz, gt_xyz, robot_letters))
                    else:
                        stats["ate_warning"] = (
                            f"tum rows {est_xyz.shape[0]} != gt rows {gt_xyz.shape[0]}"
                        )
                add_per_robot_mean(stats)
                runs[str(seed)] = stats
            ok = [s for s in runs.values() if s.get("status") == "ok"]
            record["methods"][form] = {
                "seeds": runs,
                "n_runs": len(runs),
                "n_ok": len(ok),
                "aggregate": aggregate_runs(runs),
            }
            bad = [s.get("status", "?") for s in runs.values()
                   if s.get("status") != "ok"]
            mark = "ok" if not bad else "/".join(sorted(set(bad)))
            per_method_status.append(f"{form}:{mark}({len(ok)}/{len(runs)})")

        ds_wall = time.time() - ds_t0
        record["dataset_wall_s"] = ds_wall

        json_path = out_dir / f"{name}.json"
        json_path.write_text(json.dumps(record, indent=2))
        elapsed = time.time() - t_start
        print(f"[{idx:2d}/{len(datasets)}] {name}  "
              f"({record['n_poses']} poses, {record['n_edges']} edges, "
              f"{record['n_outliers_truth']} outliers)  "
              f"{'  '.join(per_method_status)}  "
              f"ds={ds_wall:6.1f}s  total={elapsed/60:5.1f}min",
              flush=True)

    total = time.time() - t_start
    print(f"\nsweep finished: {len(datasets)} datasets in {total/60:.1f} min")
    write_summary(out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                     default=REPO / "examples" / "data" / "analysis"
                             / "cosmobench_irls_gnc_full",
                     help="output directory for per-dataset JSON + plots")
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/cosmobench_irls_sweep"),
                     help="scratch directory for stripped-prior pyfg + TUM outputs")
    ap.add_argument("--timeout", type=float, default=600.0,
                     help="per-method timeout in seconds (default 600)")
    ap.add_argument("--gpu", action="store_true",
                     help="run each IRLS inner solve on the GPU RTR solver "
                          "(irls_robust --gpu); requires an ENABLE_GPU build")
    ap.add_argument("--include-dense", action="store_true",
                     help="also run the `dense` formulation (explicitly-formed "
                          "Schur complement). Off by default: it holds a p x p "
                          "matrix and re-forms it every IRLS outer iteration. "
                          "Runs on the first seed only.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                     help="init seeds to run per dataset (default: 0, the "
                          "noiseless odometry init). Seed > 0 perturbs each "
                          "odometry edge before chaining; all formulations see "
                          "the same init for a given seed.")
    ap.add_argument("--init-noise-rot-deg", type=float, default=0.5,
                     help="per-edge rotation noise std in degrees for seeds > 0 "
                          "(default 0.5)")
    ap.add_argument("--init-noise-trans", type=float, default=0.02,
                     help="per-edge translation noise std for seeds > 0 "
                          "(default 0.02)")
    ap.add_argument("--datasets", nargs="+", default=None,
                     help="restrict the sweep to these dataset stems "
                          "(default: all cosmobench + nebula datasets)")
    ap.add_argument("--summary-only", action="store_true",
                     help="rebuild summary.json/summary.csv from the per-dataset "
                          "JSONs already in --out, without running anything")
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.scratch = args.scratch.resolve()
    if args.summary_only:
        write_summary(args.out)
        return 0
    if not IRLS_BIN.exists():
        print(f"missing binary: {IRLS_BIN}", file=sys.stderr)
        return 2
    sweep(args.out, args.scratch, args.timeout, args.include_dense,
           use_gpu=args.gpu, seeds=args.seeds,
           noise_rot_deg=args.init_noise_rot_deg,
           noise_trans=args.init_noise_trans, only=args.datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
