"""GTSAM GNC-GM counterpart to run_cosmobench_irls_gnc_sweep.py.

For every cosmobench (wifi + proradio) and nebula dataset:

  1. Run `build/bin/gtsam_gnc_pgo` with GNC-GM from the odometry init.
  2. Compute ATE (global Umeyama + per-robot Umeyama) against the GT poses
     stored in the pyfg VERTEX lines.
  3. Write a per-dataset JSON with all the metrics for the table.

The GTSAM binary silently skips `VERTEX_SE3:QUAT:PRIOR` lines, so we do NOT
need to pre-strip priors the way the IRLS sweep does — the effective input
graph is the same (BetweenFactors only, first vertex anchored).

Metrics emitted mirror the IRLS sweep where possible:
  status, dim, init_seed, n_edges, inliers, outliers, initial_cost, final_cost,
  wall_s, ate_global_rmse_m, ate_global_per_robot_rmse_m,
  ate_per_robot_aligned_rmse_m.

GNC outer/inner iteration counts are NOT exposed by GTSAM's GncOptimizer
summary, so they are omitted (unlike the IRLS record which has them).

Usage:
  python3 examples/run_cosmobench_gtsam_gnc_sweep.py \
      --out examples/data/analysis/cosmobench_gtsam_gnc_full
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
from typing import Dict, List, Optional, Tuple

import numpy as np


REPO = Path(__file__).resolve().parent.parent.parent
# gtsam_gnc_pgo must be built with ENABLE_VECTORIZATION=OFF: -march=native
# changes Eigen's alignment relative to the installed libgtsam and corrupts the
# heap inside gtsam::NoiseModelFactor::error. Override the path when the default
# build dir has it compiled the wrong way.
SESYNC_BIN = (Path(os.environ["VARPRO_SESYNC_BIN"])
              if os.environ.get("VARPRO_SESYNC_BIN") else None)
BIN = Path(os.environ.get("VARPRO_GTSAM_BIN",
                           REPO / "build" / "bin" / "gtsam_gnc_pgo"))

DATASETS_ROOTS = [
    REPO / "examples" / "data" / "cosmobench" / "wifi",
    REPO / "examples" / "data" / "cosmobench" / "proradio",
    REPO / "examples" / "data" / "nebula",
]

GTSAM_RESULT_RE = re.compile(r"^(?:GTSAM_GNC_RESULT|SESYNC_GNC_RESULT)\s+(.+)$",
                              re.MULTILINE)


# ---------------------------------------------------------------------------
# PyFG parsing helpers
# ---------------------------------------------------------------------------


def parse_pyfg_vertices(pyfg: Path) -> Tuple[List[str], np.ndarray]:
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
# Alignment + RMSE (identical to the IRLS sweep)
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


def parse_gtsam_result(stdout: str) -> Optional[Dict[str, object]]:
    m = GTSAM_RESULT_RE.search(stdout)
    if not m:
        return None
    out: Dict[str, object] = {}
    for tok in m.group(1).split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k in ("dim", "init_seed", "edges", "inliers", "outliers",
                 "outer", "inner"):
            try: out[k] = int(v)
            except ValueError: out[k] = v
        elif k in ("initial_cost", "final_cost", "robust_cost", "total_s"):
            try: out[k] = float(v)
            except ValueError: out[k] = v
        else:
            out[k] = v
    # Normalize keys to what the rest of the script expects.
    out["wall_s"] = out.pop("total_s", None)
    out["n_edges_seen"] = out.pop("edges", None)
    return out


def run_one(pyfg: Path, tum_dir: Path, timeout_s: float,
             strip_priors: bool = False,
             extra_args: Optional[List[str]] = None) -> Tuple[Dict[str, object], Path]:
    init_tum = tum_dir / "init.tum"
    final_tum = tum_dir / "final.tum"
    # VARPRO_SESYNC_BIN selects the SE-Sync (chordal) GNC driver, which
    # minimises the *same* objective as VarPro's irls_robust -- see
    # SESync_GNC_example.cpp. It takes <d> <p> <pyfg> <final.tum> and needs no
    # init file (it chains odometry internally, as irls_robust does) and no
    # --strip-priors (it only reads pose-pose edges).
    if SESYNC_BIN:
        cmd = [str(SESYNC_BIN), "3", "3", str(pyfg), str(final_tum), "--gnc"]
        if extra_args:
            cmd.extend(extra_args)
    else:
        cmd = [str(BIN), str(pyfg), str(init_tum), str(final_tum)]
        if strip_priors:
            cmd.append("--strip-priors")
        if extra_args:
            cmd.extend(extra_args)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=timeout_s, check=False)
        wall = time.time() - t0
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_s": time.time() - t0}, final_tum

    stats = parse_gtsam_result(result.stdout)
    if result.returncode != 0 or stats is None:
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
# Main sweep
# ---------------------------------------------------------------------------


def list_datasets() -> List[Path]:
    out: List[Path] = []
    for root in DATASETS_ROOTS:
        if not root.exists():
            continue
        for p in sorted(root.glob("*.pyfg")):
            if p.stem.endswith("_no_outliers"):
                continue
            out.append(p)
    return out


def sweep(out_dir: Path, scratch_dir: Path, timeout_s: float,
           strip_priors: bool = False,
           extra_args: Optional[List[str]] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    datasets = list_datasets()
    print(f"sweeping {len(datasets)} datasets into {out_dir}\n")
    t_start = time.time()

    for idx, pyfg in enumerate(datasets, start=1):
        ds_t0 = time.time()
        name = pyfg.stem
        rel = pyfg.relative_to(REPO).as_posix()

        ds_scratch = scratch_dir / name
        ds_scratch.mkdir(parents=True, exist_ok=True)

        names, gt_xyz = parse_pyfg_vertices(pyfg)
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
            "method": "gtsam_gnc_gm",
            "kernel": "gm",
            "gnc": True,
            "init": "odometry",
            "n_poses": int(len(names)),
            "n_edges": int(n_edges_full),
            "n_edges_inlier_truth": (int(n_edges_clean) if n_edges_clean is not None
                                       else None),
            "n_outliers_truth": (int(n_edges_full - n_edges_clean)
                                   if n_edges_clean is not None else None),
            "n_robots": len(poses_per_robot),
            "robots": list(poses_per_robot.keys()),
            "poses_per_robot": poses_per_robot,
        }

        stats, final_tum = run_one(pyfg, ds_scratch, timeout_s,
                                     strip_priors=strip_priors,
                                     extra_args=extra_args)
        if stats.get("status") == "ok" and final_tum.exists():
            est_xyz = parse_tum_xyz(final_tum)
            if est_xyz.shape[0] == gt_xyz.shape[0]:
                stats.update(compute_ate(est_xyz, gt_xyz, robot_letters))
            else:
                stats["ate_warning"] = (
                    f"tum rows {est_xyz.shape[0]} != gt rows {gt_xyz.shape[0]}"
                )
        record["result"] = stats

        ds_wall = time.time() - ds_t0
        record["dataset_wall_s"] = ds_wall

        json_path = out_dir / f"{name}.json"
        json_path.write_text(json.dumps(record, indent=2))
        elapsed = time.time() - t_start
        mark = stats.get("status", "?")
        ate_str = ""
        if "ate_global_rmse_m" in stats:
            ate_str = f"ATE={stats['ate_global_rmse_m']:.3f}m"
        print(f"[{idx:2d}/{len(datasets)}] {name}  "
              f"({record['n_poses']} poses, {record['n_edges']} edges, "
              f"truth_out={record['n_outliers_truth']})  "
              f"{mark} in/out={stats.get('inliers', '?')}/{stats.get('outliers', '?')}  "
              f"{ate_str}  ds={ds_wall:6.1f}s  total={elapsed/60:5.1f}min",
              flush=True)

    total = time.time() - t_start
    print(f"\nsweep finished: {len(datasets)} datasets in {total/60:.1f} min")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                     default=REPO / "examples" / "data" / "analysis"
                             / "cosmobench_gtsam_gnc_full",
                     help="output directory for per-dataset JSON")
    ap.add_argument("--scratch", type=Path,
                     default=Path("/tmp/cosmobench_gtsam_sweep"),
                     help="scratch dir for per-dataset TUM outputs")
    ap.add_argument("--timeout", type=float, default=600.0,
                     help="per-dataset timeout in seconds (default 600)")
    ap.add_argument("--strip-priors", action="store_true",
                     help="pass --strip-priors to gtsam_gnc_pgo (apples-to-apples "
                          "with the VarPro IRLS sweep, which strips pyfg priors)")
    ap.add_argument("--gtsam-args", default="",
                     help="extra flags forwarded verbatim to gtsam_gnc_pgo, "
                          "whitespace-separated. Use this to select GTSAM's own "
                          "GncOptimizer and match the IRLS hyperparameters, e.g. "
                          "--gtsam-args '--use-gtsam-gnc --barc-prob 0.999659 "
                          "--mu-step 1.4 --max-iters 20 --rel-tol 1e-4'")
    args = ap.parse_args()
    if SESYNC_BIN and not SESYNC_BIN.exists():
        print(f"missing SE-Sync GNC binary: {SESYNC_BIN}", file=sys.stderr)
        return 2
    if not SESYNC_BIN and not BIN.exists():
        print(f"missing binary: {BIN} — build with "
              f"`cmake --build build --target gtsam_gnc_pgo`",
              file=sys.stderr)
        return 2
    sweep(args.out, args.scratch, args.timeout, strip_priors=args.strip_priors,
           extra_args=args.gtsam_args.split() or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
