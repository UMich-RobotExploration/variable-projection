"""Sweep IRLS-GNC (GM kernel) across all cosmobench + nebula datasets.

For each dataset, with priors stripped (the JRL priors have variance 1e-8/1e-6
which makes the Hessian ill-conditioned and crashes the inner solver):

  1. Run `build/bin/irls_robust --gnc --kernel gm --formulation {explicit, expvp, impl}`
     from the same odometry init.
  2. Compute ATE (global Umeyama + per-robot Umeyama) against ground truth.
  3. Write a per-dataset JSON with all the metrics for the table.
  4. Render the 4-panel trajectory plot (GT + 3 methods, per-robot aligned).

Results are written **immediately** after each dataset finishes so you can
inspect progress without waiting for the whole sweep.

Usage:
  python3 examples/run_cosmobench_irls_gnc_sweep.py \
      --out examples/data/analysis/cosmobench_irls_gnc_full
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


REPO = Path(__file__).resolve().parent.parent
IRLS_BIN = REPO / "build" / "bin" / "irls_robust"
PLOT_SCRIPT = REPO / "examples" / "plot_cosmobench_irls.py"

DATASETS_ROOTS = [
    REPO / "examples" / "data" / "cosmobench" / "wifi",
    REPO / "examples" / "data" / "cosmobench" / "proradio",
    REPO / "examples" / "data" / "nebula",
]

METHODS = ["explicit", "expvp", "impl"]

IRLS_RESULT_RE = re.compile(
    r"IRLS_RESULT\s+kernel=(\S+)\s+form=(\S+)\s+gnc=(\d+)\s+init_seed=(\d+)\s+"
    r"dim=(\d+)\s+outer=(\d+)\s+inner=(\d+)\s+final_cost=(\S+)\s+"
    r"robust_cost=(\S+)\s+total_s=(\S+)\s+precompute_s=(\S+)"
)


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
            }
    return None


def run_one_method(pyfg_no_priors: Path, formulation: str, tum_dir: Path,
                    timeout_s: float) -> Tuple[Dict[str, object], Path]:
    init_tum = tum_dir / f"init_{formulation}.tum"
    final_tum = tum_dir / f"final_{formulation}.tum"
    cmd = [
        str(IRLS_BIN), str(pyfg_no_priors), str(init_tum), str(final_tum),
        "--kernel", "gm", "--formulation", formulation, "--gnc",
    ]
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


def render_plot(pyfg_no_priors: Path, tum_explicit: Path, tum_expvp: Path,
                tum_impl: Path, out_png: Path, per_robot_align: bool) -> None:
    cmd = [
        sys.executable, str(PLOT_SCRIPT),
        "--pyfg", str(pyfg_no_priors),
        "--tum-explicit", str(tum_explicit),
        "--tum-expvp", str(tum_expvp),
        "--tum-impl", str(tum_impl),
        "-o", str(out_png),
    ]
    if per_robot_align:
        cmd.append("--per-robot-align")
    subprocess.run(cmd, check=True, capture_output=True)


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


def sweep(out_dir: Path, scratch_dir: Path, timeout_s: float) -> None:
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
        pyfg_no_priors = ds_scratch / f"{name}_no_priors.pyfg"
        n_priors_stripped = strip_priors_to(pyfg, pyfg_no_priors)

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
            "methods": {},
        }

        per_method_status = []
        method_tums: Dict[str, Path] = {}
        for form in METHODS:
            stats, final_tum = run_one_method(pyfg_no_priors, form, ds_scratch,
                                                timeout_s)
            method_tums[form] = final_tum
            if stats.get("status") == "ok":
                # ATE
                if final_tum.exists():
                    est_xyz = parse_tum_xyz(final_tum)
                    if est_xyz.shape[0] == gt_xyz.shape[0]:
                        ate = compute_ate(est_xyz, gt_xyz, robot_letters)
                        stats.update(ate)
                    else:
                        stats["ate_warning"] = (
                            f"tum rows {est_xyz.shape[0]} != gt rows {gt_xyz.shape[0]}"
                        )
            record["methods"][form] = stats
            mark = "ok" if stats.get("status") == "ok" else stats.get("status", "?")
            per_method_status.append(f"{form}:{mark}")

        # Plot (per-robot aligned) if all three methods finished
        plot_ok = all(record["methods"][f].get("status") == "ok" for f in METHODS)
        if plot_ok:
            plot_perRobot = out_dir / f"{name}_perRobot.png"
            plot_global = out_dir / f"{name}_global.png"
            try:
                render_plot(pyfg_no_priors, method_tums["explicit"],
                            method_tums["expvp"], method_tums["impl"],
                            plot_perRobot, per_robot_align=True)
                render_plot(pyfg_no_priors, method_tums["explicit"],
                            method_tums["expvp"], method_tums["impl"],
                            plot_global, per_robot_align=False)
                record["plot_perRobot"] = plot_perRobot.relative_to(REPO).as_posix()
                record["plot_global"] = plot_global.relative_to(REPO).as_posix()
            except subprocess.CalledProcessError as e:
                record["plot_error"] = e.stderr.decode("utf-8", errors="replace")[:400]
        else:
            record["plot_error"] = "skipped (one or more methods failed)"

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
    args = ap.parse_args()
    if not IRLS_BIN.exists():
        print(f"missing binary: {IRLS_BIN}", file=sys.stderr)
        return 2
    sweep(args.out, args.scratch, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
