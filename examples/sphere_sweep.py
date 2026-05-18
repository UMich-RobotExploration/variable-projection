#!/usr/bin/env python3
"""Sphere-style synthetic sweep, faithfully porting the g2o create_sphere
example (R. Kuemmerle et al., 2011) and augmenting it with landmarks.

Trajectory: poses spiral from the south pole of a radius-R sphere up to the
north pole — `num_laps` circumferential loops, each with `nodes_per_level`
poses. Total poses P = num_laps * nodes_per_level.

Pose-pose measurements:
  * Odometry: every consecutive pair (i, i+1).
  * Loop closures: for each pose at (level f-1, node n), three edges to
    (level f, nodes n-1, n, n+1) — clamped at the level boundary (no
    'n+1' from the last level). Yields ~3 LC/pose, matching the original
    g2o output.

Noise (matches g2o defaults):
  * Translation: σ = 0.01 m per axis.
  * Rotation: g2o-style quaternion noise — sample (q_x, q_y, q_z) ~
    N(0, σ_r^2 I), set q_w = 1 - ||q_xyz|| (clamped at 0), normalise.
    σ_r = 0.005 per axis by default.

Landmark augmentation (not present in the g2o example):
  * `n_landmarks` landmarks sampled uniformly in a box that wraps the
    sphere with 20% padding.
  * Each landmark is observed by `obs_per_landmark` randomly-chosen poses.
  * Observations are 3-D relative translations in each pose's body frame
    (EDGE_SE3_XYZ), with translational σ = 0.05 m.

Two sweep axes (same names + grid as the SfM sweep so the plot code reuses
the existing aggregate / sweep_plots pipeline):

  size  - total (P + L) at fixed pose:landmark ratio (1:5)
  ratio - pose:landmark ratio at fixed P (= 990 poses, 33 laps × 30 nodes)
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
SWEEP_ROOT = REPO / "examples" / "data" / "sphere_sweep"
CONFIG_PATH = REPO / "examples" / "config.json"
ANALYSIS_DIR = REPO / "examples" / "data" / "analysis"

# Generator parameters (match the g2o create_sphere defaults).
RADIUS = 100.0
SIGMA_T_ODOM = 0.01   # m — g2o default translational noise
SIGMA_R_ODOM = 0.005  # rad — g2o default rotational noise (per-axis)
SIGMA_T_LMK  = 0.05   # m — translational noise on landmark observations
NODES_PER_LEVEL = 30  # circumferential resolution of the spiral
OBS_PER_LANDMARK = 30
RANK = 5

# Sweep grids — same as the SfM / grid3d sweeps for cross-comparison.
SIZE_GRID = [2_000, 3_000, 5_000, 7_500, 10_000, 15_000, 22_000]
RATIO_FIXED_POSE_TO_LANDMARK = (1, 5)
RATIO_FIXED_POSES = 33 * NODES_PER_LEVEL    # 990 poses for the ratio sweep
RATIO_GRID = [(1, 1), (1, 2), (1, 3), (1, 5), (1, 10), (1, 20), (1, 50), (1, 100)]

FORMULATION_NAME = {0: "Explicit", 1: "ExplicitVarPro", 2: "Implicit"}


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    axis: str          # "size" or "ratio"
    name: str
    num_laps: int
    nodes_per_level: int
    n_landmarks: int
    K: int             # observations per landmark
    seed: int

    @property
    def P(self) -> int:
        return self.num_laps * self.nodes_per_level

    @property
    def total_vars(self) -> int:
        return self.P + self.n_landmarks


def make_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    base_seed = 42
    npl = NODES_PER_LEVEL

    # Axis A: total size at fixed P:L = 1:5
    a, b = RATIO_FIXED_POSE_TO_LANDMARK
    for total in SIZE_GRID:
        target_P = max(npl * 2, total * a // (a + b))
        # Round num_laps so num_laps * npl is the closest match to target_P
        num_laps = max(2, round(target_P / npl))
        P = num_laps * npl
        L = total - P
        if L < 0:
            L = 0
        scenarios.append(Scenario(
            axis="size",
            name=f"size_{total // 1000}k",
            num_laps=num_laps,
            nodes_per_level=npl,
            n_landmarks=L,
            K=OBS_PER_LANDMARK,
            seed=base_seed + total,
        ))

    # Axis B: P:L ratio at fixed P = RATIO_FIXED_POSES (≈ default sphere)
    P_fixed = RATIO_FIXED_POSES
    num_laps_fixed = P_fixed // npl
    for (a, b) in RATIO_GRID:
        L = P_fixed * b // a
        scenarios.append(Scenario(
            axis="ratio",
            name=f"ratio_1_{b}",
            num_laps=num_laps_fixed,
            nodes_per_level=npl,
            n_landmarks=L,
            K=OBS_PER_LANDMARK,
            seed=base_seed + 1000 + b,
        ))
    return scenarios


# ---------------------------------------------------------------------------
# g2o-style sphere trajectory + edges
# ---------------------------------------------------------------------------

def _rotz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _roty(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def generate_sphere_trajectory(num_laps: int, nodes_per_level: int,
                                radius: float = RADIUS
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Faithful port of the pose construction in g2o create_sphere:
        rotz angle = -π + 2*n*π / nodes_per_level
        roty angle = -½π + id * π / (num_laps * nodes_per_level)
        rot       = R_z @ R_y          (so the body-x axis points outward)
        t         = rot @ (radius, 0, 0)

    g2o uses a post-increment `id++` for the y-axis angle, so the FIRST
    pose has id=1 in the formula (not 0). We reproduce that.
    """
    n = num_laps * nodes_per_level
    Rs = np.empty((n, 3, 3))
    ts = np.empty((n, 3))
    idx = 0
    for f in range(num_laps):
        for nn in range(nodes_per_level):
            id_after_inc = idx + 1                 # mimics `setId(id++)`
            angle_z = -math.pi + 2.0 * nn * math.pi / nodes_per_level
            angle_y = (-0.5 * math.pi
                        + id_after_inc * math.pi
                        / (num_laps * nodes_per_level))
            rot = _rotz(angle_z) @ _roty(angle_y)
            Rs[idx] = rot
            ts[idx] = rot @ np.array([radius, 0.0, 0.0])
            idx += 1
    return Rs, ts


def _g2o_rotation_noise(rng: np.random.Generator,
                         sigma_r: float) -> np.ndarray:
    """g2o quaternion-style rotation noise: sample q_xyz ~ N(0, σ²I),
    set q_w = 1 - ||q_xyz|| (clamped at 0), normalise. Returns the
    corresponding 3×3 rotation matrix."""
    q_xyz = rng.normal(0.0, sigma_r, 3)
    qw = 1.0 - float(np.linalg.norm(q_xyz))
    if qw < 0.0:
        qw = 0.0
    qx, qy, qz = q_xyz
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = (v / norm for v in (qw, qx, qy, qz))
    # Quaternion -> rotation matrix (Hamiltonian convention, Eigen-like).
    return np.array([
        [1 - 2 * (qy * qy + qz * qz),   2 * (qx * qy - qz * qw),       2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),       1 - 2 * (qx * qx + qz * qz),   2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),       2 * (qy * qz + qx * qw),       1 - 2 * (qx * qx + qy * qy)],
    ])


def generate_sphere_edges(Rs: np.ndarray, ts: np.ndarray,
                           num_laps: int, nodes_per_level: int,
                           rng: np.random.Generator
                           ) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Faithful port of the edge construction in g2o create_sphere:
        - sequential odometry edges
        - 3 LCs per pose at (level f-1, node n) -> (level f, node n-1/n/n+1)
          with the 'last level + n+1' boundary case skipped.
    Then applies g2o's noise model (additive Gaussian on translation,
    quaternion-perturbation on rotation, both pre-multiplied as in the
    original code).
    """
    n_total = num_laps * nodes_per_level
    raw: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    # Odometry — every adjacent pair.
    for i in range(1, n_total):
        R_ij = Rs[i - 1].T @ Rs[i]
        t_ij = Rs[i - 1].T @ (ts[i] - ts[i - 1])
        raw.append((i - 1, i, R_ij, t_ij))

    # Loop closures — across-level wraparound, matching g2o:
    #   from = vertices[(f-1)*npl + nn]
    #   for offset in {-1, 0, +1}:  to = vertices[f*npl + nn + offset]
    #   skip (f == num_laps-1, offset == +1)
    for f in range(1, num_laps):
        for nn in range(nodes_per_level):
            i = (f - 1) * nodes_per_level + nn
            for offset in (-1, 0, 1):
                if f == num_laps - 1 and offset == 1:
                    continue
                target_nn = nn + offset
                # g2o doesn't bounds-check target_nn but the indexing wraps
                # around the level naturally. We clamp instead of wrap so
                # we don't accidentally edge to the wrong row.
                if target_nn < 0 or target_nn >= nodes_per_level:
                    continue
                j = f * nodes_per_level + target_nn
                R_ij = Rs[i].T @ Rs[j]
                t_ij = Rs[i].T @ (ts[j] - ts[i])
                raw.append((i, j, R_ij, t_ij))

    # Apply g2o noise model: t_noisy = t + Δt, R_noisy = R · R_noise
    out: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for (i, j, R_ij, t_ij) in raw:
        R_noise = _g2o_rotation_noise(rng, SIGMA_R_ODOM)
        t_noise = rng.normal(0.0, SIGMA_T_ODOM, 3)
        out.append((i, j, R_ij @ R_noise, t_ij + t_noise))
    return out


def generate_landmarks_and_obs(Rs: np.ndarray, ts: np.ndarray,
                                n_landmarks: int, K_obs: int,
                                rng: np.random.Generator
                                ) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray]]]:
    """Place landmarks in a box wrapping the sphere (with 20% padding).
    Each landmark observed by K_obs random poses."""
    if n_landmarks == 0:
        return np.zeros((0, 3)), []
    n = len(Rs)
    mn, mx = ts.min(axis=0), ts.max(axis=0)
    pad = 0.2 * np.maximum(mx - mn, 1.0)
    landmarks = rng.uniform(mn - pad, mx + pad, (n_landmarks, 3))
    K_eff = min(K_obs, n)
    obs: list[tuple[int, int, np.ndarray]] = []
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

def _matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
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


# Pose-pose covariance: diag(σ_t², σ_t², σ_t², σ_r², σ_r², σ_r²). Written as the
# upper triangle of a 6×6 symmetric matrix in row-major order (21 entries).
def _build_pose_pose_cov_ut() -> str:
    var_t = SIGMA_T_ODOM ** 2
    var_r = SIGMA_R_ODOM ** 2
    cov = np.zeros((6, 6))
    for i in range(3):
        cov[i, i] = var_t
        cov[i + 3, i + 3] = var_r
    parts = []
    for i in range(6):
        for j in range(i, 6):
            parts.append(f"{cov[i, j]:.9f}")
    return " ".join(parts)


_COV_POSE_POSE_UT = _build_pose_pose_cov_ut()
_COV_POSE_LMK_UT = " ".join([f"{SIGMA_T_LMK ** 2:.9f}",
                              "0.000000000", "0.000000000",
                              f"{SIGMA_T_LMK ** 2:.9f}", "0.000000000",
                              f"{SIGMA_T_LMK ** 2:.9f}"])


def write_pyfg(out_path: Path, Rs: np.ndarray, ts: np.ndarray,
                edges: list, landmarks: np.ndarray, obs: list) -> None:
    with out_path.open("w") as f:
        for i, (R, t) in enumerate(zip(Rs, ts)):
            qx, qy, qz, qw = _matrix_to_quat(R)
            f.write(
                f"VERTEX_SE3:QUAT 0.000000 A{i} "
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )
        for j, lm in enumerate(landmarks):
            f.write(
                f"VERTEX_XYZ 0.0 L{j} {lm[0]:.6f} {lm[1]:.6f} {lm[2]:.6f}\n"
            )
        for (i, j, R_ij, t_ij) in edges:
            qx, qy, qz, qw = _matrix_to_quat(R_ij)
            f.write(
                f"EDGE_SE3:QUAT 0.0 A{i} A{j} "
                f"{t_ij[0]:.6f} {t_ij[1]:.6f} {t_ij[2]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f} {_COV_POSE_POSE_UT}\n"
            )
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

    Rs, ts = generate_sphere_trajectory(
        scenario.num_laps, scenario.nodes_per_level, RADIUS)
    pose_edges = generate_sphere_edges(
        Rs, ts, scenario.num_laps, scenario.nodes_per_level, rng)
    landmarks, obs = generate_landmarks_and_obs(
        Rs, ts, scenario.n_landmarks, scenario.K, rng)

    pyfg_path = out_dir / f"{scenario.name}.pyfg"
    write_pyfg(pyfg_path, Rs, ts, pose_edges, landmarks, obs)

    P = scenario.P
    K_eff = min(scenario.K, P)
    meta = {
        "axis": scenario.axis,
        "scenario": scenario.name,
        "poses": P,
        "landmarks": scenario.n_landmarks,
        "obs_per_landmark": K_eff,
        "total_vars": scenario.total_vars,
        "total_observations": scenario.n_landmarks * K_eff,
        "pose_to_landmark_ratio": (scenario.n_landmarks / max(P, 1)),
        "pose_pose_edges": len(pose_edges),
        "odom_edges": P - 1,
        "loop_closure_edges": len(pose_edges) - (P - 1),
        "num_laps": scenario.num_laps,
        "nodes_per_level": scenario.nodes_per_level,
        "radius": RADIUS,
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
        meta = json.loads((out / "meta.json").read_text())
        print(f"  {out.relative_to(REPO)}: "
              f"laps={sc.num_laps} npl={sc.nodes_per_level} "
              f"P={sc.P} L={sc.n_landmarks} "
              f"odom={meta['odom_edges']} LC={meta['loop_closure_edges']} "
              f"obs={meta['total_observations']}")
    print(f"done. {len(scenarios)} scenarios under {SWEEP_ROOT.relative_to(REPO)}")
    return 0


# ---------------------------------------------------------------------------
# Run wrapper (identical to grid3d_sweep.py, retargeted at sphere data path)
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
        "radius": RADIUS,
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
    out_path = ANALYSIS_DIR / "sphere_sweep_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    n_scen = len(out["scenarios"])
    n_runs = sum(len(s["runs"]) for s in out["scenarios"])
    print(f"wrote {out_path}  ({n_scen} scenarios, {n_runs} runs)")
    return 0


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
                    help="dump per-scenario runs to data/analysis/sphere_sweep_results.json")
    args = parser.parse_args()
    if args.cmd == "generate":
        sys.exit(cmd_generate(args))
    if args.cmd == "run":
        sys.exit(cmd_run(args))
    if args.cmd == "aggregate":
        sys.exit(cmd_aggregate(args))


if __name__ == "__main__":
    main()
