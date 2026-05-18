#!/usr/bin/env python3
"""Grid3D-style sweep: synthetic PGO trajectory + pose-landmark observations.

This is a drop-in replacement for sfm_sweep.py that uses a more realistic
problem structure than the random-sphere SfM:

  * Poses follow a meandering 3D random walk (like the real Grid3D
    dataset — a long trajectory with smooth turns).
  * Pose-pose measurements: sequential odometry + spatially-driven loop
    closures (~1.5 per pose, matching Grid3D's LC density).
  * Landmarks: scattered through the trajectory's bounding box.
  * Pose-landmark measurements: 3D relative-translation observations in
    each observing pose's body frame, written as EDGE_SE3_XYZ.

Two sweep axes (same names + grid as the SfM sweep so the plot code can
re-use the existing aggregate / sweep_plots pipeline):

  size  - total (P + L) at fixed pose:landmark ratio (1:5)
  ratio - pose:landmark ratio at fixed pose count (P = 1000 ≈ the
          "default Grid3D" we use as the ratio-sweep base)

The on-disk layout matches sfm_sweep.py, just in a different root:
  examples/data/grid3d_sweep/<axis>/<scenario_name>/scenario.pyfg
  examples/data/grid3d_sweep/<axis>/<scenario_name>/meta.json
  examples/data/grid3d_sweep/<axis>/<scenario_name>/cached_results/...
  examples/data/grid3d_sweep/<axis>/<scenario_name>/inits/...

Subcommands match sfm_sweep.py: generate / run / aggregate.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SWEEP_ROOT = REPO / "examples" / "data" / "grid3d_sweep"
CONFIG_PATH = REPO / "examples" / "config.json"
ANALYSIS_DIR = REPO / "examples" / "data" / "analysis"

# Noise / generator parameters. Picked to roughly match the real Grid3D
# (translation σ ≈ 0.1 m, rotation σ ≈ 0.05 rad in each Euler component)
# while keeping the problem solvable with the default solver tolerances.
SIGMA_T_ODOM  = 0.1   # m, translational noise on pose-pose measurements
SIGMA_R_ODOM  = 0.05  # rad, rotational noise on pose-pose measurements
SIGMA_T_LMK   = 0.05  # m, translational noise on pose-landmark observations
STEP_SIZE     = 1.0   # m per pose along the body-x direction
LC_RADIUS     = 3.0   # m, spatial radius for considering loop closures
LC_MIN_TIME_GAP = 20  # poses; LCs require |i-j| >= this
LC_PROB       = 0.35  # probability of accepting a candidate LC pair
OBS_PER_LANDMARK = 30 # observations per landmark (same as sfm_sweep)
RANK          = 5

# Sweep grids — match sfm_sweep.py so we can compare side-by-side.
SIZE_GRID = [2_000, 3_000, 5_000, 7_500, 10_000, 15_000, 22_000]
RATIO_FIXED_POSE_TO_LANDMARK = (1, 5)

# "Default" Grid3D base for the ratio sweep. Fewer poses than the real
# 8000-pose Grid3D because the 100× landmark blow-up at ratio 1:100 would
# otherwise produce 800k landmarks (intractable).
RATIO_FIXED_POSES = 1000
RATIO_GRID = [(1, 1), (1, 2), (1, 3), (1, 5), (1, 10), (1, 20), (1, 50), (1, 100)]


FORMULATION_NAME = {0: "Explicit", 1: "ExplicitVarPro", 2: "Implicit"}


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    axis: str   # "size" or "ratio"
    name: str
    P: int      # number of poses
    L: int      # number of landmarks
    K: int      # observations per landmark
    seed: int

    @property
    def total_vars(self) -> int:
        return self.P + self.L


def make_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    base_seed = 42

    # Axis A: total size at fixed P:L = 1:5
    a, b = RATIO_FIXED_POSE_TO_LANDMARK
    for total in SIZE_GRID:
        P = max(1, total * a // (a + b))
        L = total - P
        scenarios.append(Scenario(
            axis="size",
            name=f"size_{total // 1000}k",
            P=P, L=L,
            K=OBS_PER_LANDMARK,
            seed=base_seed + total,
        ))

    # Axis B: P:L ratio at fixed P (= "default Grid3D" base, RATIO_FIXED_POSES)
    for (a, b) in RATIO_GRID:
        # P is fixed; L is set so L/P = b/a (i.e. 1:b means L = b*P).
        P = RATIO_FIXED_POSES
        L = P * b // a
        scenarios.append(Scenario(
            axis="ratio",
            name=f"ratio_1_{b}",
            P=P, L=L,
            K=OBS_PER_LANDMARK,
            seed=base_seed + 1000 + b,
        ))

    return scenarios


# ---------------------------------------------------------------------------
# Grid3D-style trajectory generation
# ---------------------------------------------------------------------------

def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """3D rotation matrix from axis-angle (axis must be unit-norm)."""
    K = np.array([[0.0,    -axis[2],  axis[1]],
                  [axis[2], 0.0,     -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _random_small_rotation(rng: np.random.Generator, sigma_rad: float) -> np.ndarray:
    """Random rotation perturbation with each axis-angle component ~N(0, σ)."""
    omega = rng.normal(0.0, sigma_rad, 3)
    ang = np.linalg.norm(omega)
    if ang < 1e-12:
        return np.eye(3)
    return _rodrigues(omega / ang, ang)


def generate_trajectory(n_poses: int, rng: np.random.Generator
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Random walk with smooth turns. Returns (Rs, ts) of shapes
    (n, 3, 3) and (n, 3)."""
    Rs = np.empty((n_poses, 3, 3))
    ts = np.empty((n_poses, 3))
    Rs[0] = np.eye(3)
    ts[0] = np.zeros(3)
    # Smoothly-varying angular velocity drives the turn rate.
    omega = np.zeros(3)
    for i in range(1, n_poses):
        omega += rng.normal(0.0, 0.08, 3)
        omega *= 0.92  # decay → keeps turns from spiralling away
        ang = float(np.linalg.norm(omega))
        if ang > 1e-9:
            R_step = _rodrigues(omega / ang, ang)
        else:
            R_step = np.eye(3)
        Rs[i] = Rs[i - 1] @ R_step
        ts[i] = ts[i - 1] + Rs[i - 1] @ np.array([STEP_SIZE, 0.0, 0.0])
    return Rs, ts


def generate_pgo_edges(Rs: np.ndarray, ts: np.ndarray,
                        rng: np.random.Generator
                        ) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Return [(i, j, R_ij, t_ij), ...] — odometry + loop closures."""
    n = len(Rs)
    edges = []

    # Odometry (every adjacent pair).
    for i in range(n - 1):
        R_ij = Rs[i].T @ Rs[i + 1]
        t_ij = Rs[i].T @ (ts[i + 1] - ts[i])
        R_ij = R_ij @ _random_small_rotation(rng, SIGMA_R_ODOM)
        t_ij = t_ij + rng.normal(0.0, SIGMA_T_ODOM, 3)
        edges.append((i, i + 1, R_ij, t_ij))

    # Loop closures: spatially close pose pairs with a time-gap floor.
    # Use a KD-tree-free O(n²) check; n ≤ ~5000 so it's fast enough.
    for i in range(n):
        d = np.linalg.norm(ts - ts[i], axis=1)
        for j in range(i + LC_MIN_TIME_GAP, n):
            if d[j] >= LC_RADIUS:
                continue
            if rng.random() >= LC_PROB:
                continue
            R_ij = Rs[i].T @ Rs[j]
            t_ij = Rs[i].T @ (ts[j] - ts[i])
            R_ij = R_ij @ _random_small_rotation(rng, SIGMA_R_ODOM)
            t_ij = t_ij + rng.normal(0.0, SIGMA_T_ODOM, 3)
            edges.append((i, j, R_ij, t_ij))
    return edges


def generate_landmarks_and_obs(Rs: np.ndarray, ts: np.ndarray,
                                n_landmarks: int, K_obs: int,
                                rng: np.random.Generator
                                ) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray]]]:
    """Place landmarks across the trajectory bbox; each landmark observed by
    K_obs random poses. Returns (landmarks (L, 3), [(pose_idx, lmk_idx, t_il), ...])."""
    n = len(Rs)
    # Bounding box with slight padding so landmarks can be a little
    # outside the trajectory hull.
    mn, mx = ts.min(axis=0), ts.max(axis=0)
    pad = 0.2 * np.maximum(mx - mn, 1.0)
    landmarks = rng.uniform(mn - pad, mx + pad, (n_landmarks, 3))

    K_eff = min(K_obs, n)
    obs = []
    for lj in range(n_landmarks):
        cam_idxs = rng.choice(n, size=K_eff, replace=False)
        for ci in cam_idxs:
            t_il = Rs[ci].T @ (landmarks[lj] - ts[ci])
            t_il = t_il + rng.normal(0.0, SIGMA_T_LMK, 3)
            obs.append((int(ci), int(lj), t_il))
    return landmarks, obs


# ---------------------------------------------------------------------------
# pyfg writing
# ---------------------------------------------------------------------------

def matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """(qx, qy, qz, qw) from a 3x3 rotation matrix — shake-pulled from the
    sfm_sweep generator so output matches its convention."""
    t = R.trace()
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


# Covariance lines: 6x6 upper triangle for pose-pose (xyz then rpy),
# 6 entries for pose-landmark (just xyz). Values picked to roughly match
# the real Grid3D file (σ_t ≈ 0.1 m → var 0.01; σ_r ≈ 0.05 → var 0.0025).
_COV_POSE_POSE_UT = (
    "0.010000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 "
    "0.010000000 0.000000000 0.000000000 0.000000000 0.000000000 "
    "0.010000000 0.000000000 0.000000000 0.000000000 "
    "0.002500000 0.000000000 0.000000000 "
    "0.002500000 0.000000000 "
    "0.002500000"
)
_COV_POSE_LMK_UT = "0.002500000 0.000000000 0.000000000 0.002500000 0.000000000 0.002500000"


def write_pyfg(out_path: Path, Rs: np.ndarray, ts: np.ndarray,
                edges: list, landmarks: np.ndarray, obs: list) -> None:
    with out_path.open("w") as f:
        # Pose vertices.
        for i, (R, t) in enumerate(zip(Rs, ts)):
            qx, qy, qz, qw = matrix_to_quat(R)
            f.write(
                f"VERTEX_SE3:QUAT 0.000000 A{i} "
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )
        # Landmark vertices.
        for j, lm in enumerate(landmarks):
            f.write(
                f"VERTEX_XYZ 0.0 L{j} {lm[0]:.6f} {lm[1]:.6f} {lm[2]:.6f}\n"
            )
        # Pose-pose edges (odom + LCs).
        for (i, j, R_ij, t_ij) in edges:
            qx, qy, qz, qw = matrix_to_quat(R_ij)
            f.write(
                f"EDGE_SE3:QUAT 0.0 A{i} A{j} "
                f"{t_ij[0]:.6f} {t_ij[1]:.6f} {t_ij[2]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f} {_COV_POSE_POSE_UT}\n"
            )
        # Pose-landmark observations.
        for (ci, lj, t_il) in obs:
            f.write(
                f"EDGE_SE3_XYZ 0.0 A{ci} L{lj} "
                f"{t_il[0]:.6f} {t_il[1]:.6f} {t_il[2]:.6f} {_COV_POSE_LMK_UT}\n"
            )


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

def generate_scenario(scenario: Scenario, out_root: Path) -> Path:
    rng = np.random.default_rng(scenario.seed)
    out_dir = out_root / scenario.axis / scenario.name
    out_dir.mkdir(parents=True, exist_ok=True)

    Rs, ts = generate_trajectory(scenario.P, rng)
    pose_edges = generate_pgo_edges(Rs, ts, rng)
    landmarks, obs = generate_landmarks_and_obs(
        Rs, ts, scenario.L, scenario.K, rng)

    pyfg_path = out_dir / f"{scenario.name}.pyfg"
    write_pyfg(pyfg_path, Rs, ts, pose_edges, landmarks, obs)

    K_eff = min(scenario.K, scenario.P)
    meta = {
        "axis": scenario.axis,
        "scenario": scenario.name,
        "poses": scenario.P,
        "landmarks": scenario.L,
        "obs_per_landmark": K_eff,
        "total_vars": scenario.total_vars,
        "total_observations": scenario.L * K_eff,
        "pose_to_landmark_ratio": scenario.L / max(scenario.P, 1),
        "pose_pose_edges": len(pose_edges),
        "odom_edges": scenario.P - 1,
        "loop_closure_edges": len(pose_edges) - (scenario.P - 1),
        "seed": scenario.seed,
        "sigma_t_odom": SIGMA_T_ODOM,
        "sigma_r_odom": SIGMA_R_ODOM,
        "sigma_t_lmk": SIGMA_T_LMK,
        "rank": RANK,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def cmd_generate(args: argparse.Namespace) -> int:
    if SWEEP_ROOT.exists() and not args.force:
        print(f"{SWEEP_ROOT.relative_to(REPO)} already exists. Pass --force to overwrite.")
        return 1
    if SWEEP_ROOT.exists():
        shutil.rmtree(SWEEP_ROOT)
    scenarios = make_scenarios()
    for sc in scenarios:
        out = generate_scenario(sc, SWEEP_ROOT)
        # Read back so we can print actual edge counts.
        meta = json.loads((out / "meta.json").read_text())
        print(f"  {out.relative_to(REPO)}: "
              f"P={sc.P} L={sc.L} odom={meta['odom_edges']} "
              f"LC={meta['loop_closure_edges']} obs={meta['total_observations']}")
    print(f"done. {len(scenarios)} scenarios under {SWEEP_ROOT.relative_to(REPO)}")
    return 0


# ---------------------------------------------------------------------------
# Run wrapper (identical to sfm_sweep.py, just with grid3d data path)
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    paper = REPO / "build" / "bin" / "paper_experiments"
    gpu_paper = REPO / "build" / "bin" / "gpu_paper_experiments"
    need_cpu = not args.gpu_only
    need_gpu = not args.cpu_only
    if need_cpu and not paper.exists():
        print(f"missing {paper}; build it first")
        return 1
    if need_gpu and not gpu_paper.exists():
        print(f"missing {gpu_paper}; build it first (cmake -DENABLE_GPU=ON)")
        return 1
    if not SWEEP_ROOT.exists():
        print(f"{SWEEP_ROOT} not found; run `generate` first")
        return 1

    backup = CONFIG_PATH.with_suffix(".json.swap-backup")
    shutil.copy(CONFIG_PATH, backup)
    cfg = json.loads(CONFIG_PATH.read_text())
    cfg["abs_data_path"] = str(SWEEP_ROOT) + "/"
    cfg["min_rank"] = RANK
    cfg["max_rank"] = RANK
    cfg["num_inits"] = args.num_inits
    cfg["verbose"] = False
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

    try:
        if need_cpu:
            print("=== CPU sweep (paper_experiments) ===")
            subprocess.run([str(paper)], cwd=REPO, check=True)
        if need_gpu:
            print("=== GPU sweep (gpu_paper_experiments) ===")
            subprocess.run([str(gpu_paper)], cwd=REPO, check=True)
    finally:
        shutil.move(str(backup), str(CONFIG_PATH))
    return 0


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def cmd_aggregate(args: argparse.Namespace) -> int:
    if not SWEEP_ROOT.exists():
        print(f"{SWEEP_ROOT} not found.")
        return 1

    out: dict = {
        "rank": RANK,
        "obs_per_landmark": OBS_PER_LANDMARK,
        "sigma_t_odom": SIGMA_T_ODOM,
        "sigma_r_odom": SIGMA_R_ODOM,
        "sigma_t_lmk": SIGMA_T_LMK,
        "scenarios": [],
    }
    for axis_dir in sorted(SWEEP_ROOT.iterdir()):
        if not axis_dir.is_dir():
            continue
        for scn_dir in sorted(axis_dir.iterdir()):
            meta_p = scn_dir / "meta.json"
            cache_dir = scn_dir / "cached_results"
            if not meta_p.exists():
                continue
            meta = json.loads(meta_p.read_text())
            runs: list[dict] = []
            if cache_dir.exists():
                for cache_file in sorted(cache_dir.iterdir()):
                    if not cache_file.name.startswith(("results_rank", "gpu_results_rank")):
                        continue
                    try:
                        arr = json.loads(cache_file.read_text())
                    except json.JSONDecodeError:
                        print(f"  warning: skipping {cache_file}")
                        continue
                    is_gpu = cache_file.name.startswith("gpu_results_rank")
                    for r in arr:
                        times = r.get("times") or []
                        costs = r.get("costs") or []
                        finite_costs = [c for c in costs if isinstance(c, (int, float)) and math.isfinite(c)]
                        formulation = FORMULATION_NAME.get(r.get("formulation"), str(r.get("formulation")))
                        runs.append({
                            "backend": r.get("backend", "Gpu" if is_gpu else "Cpu"),
                            "formulation": formulation,
                            "init_file": r.get("init_file"),
                            "iterations": len(costs),
                            "elapsed_s": times[-1] if times else None,
                            "final_cost": finite_costs[-1] if finite_costs else None,
                            "converged": bool(finite_costs) and len(finite_costs) == len(costs),
                            "times": times,
                            "costs": costs,
                        })
            out["scenarios"].append({**meta, "runs": runs})

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / "grid3d_sweep_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    n_scen = len(out["scenarios"])
    n_runs = sum(len(s["runs"]) for s in out["scenarios"])
    print(f"wrote {out_path}  ({n_scen} scenarios, {n_runs} runs)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="emit synthetic .pyfg files")
    g.add_argument("--force", action="store_true",
                    help=f"overwrite {SWEEP_ROOT.relative_to(REPO)} if it exists")
    r = sub.add_parser("run", help="run CPU + GPU paper_experiments on the sweep")
    r.add_argument("--num-inits", type=int, default=5)
    r.add_argument("--cpu-only", action="store_true")
    r.add_argument("--gpu-only", action="store_true")
    sub.add_parser("aggregate",
                    help="dump per-scenario runs to data/analysis/grid3d_sweep_results.json")
    args = parser.parse_args()
    if args.cmd == "generate":
        sys.exit(cmd_generate(args))
    if args.cmd == "run":
        sys.exit(cmd_run(args))
    if args.cmd == "aggregate":
        sys.exit(cmd_aggregate(args))


if __name__ == "__main__":
    main()
