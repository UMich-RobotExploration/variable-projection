#!/usr/bin/env python3
"""SfM sweep harness: synthetic dataset generation + run wrapper + JSON export.

Two sweep axes, both at relaxation rank 5:
  size  - total (P + L) varied at fixed pose:landmark ratio (1:5)
  ratio - pose:landmark ratio varied at fixed total (P + L)

Connectivity (avg observations per landmark) is held fixed across both axes.

Subcommands:
  generate  - emit synthetic .pyfg files into examples/data/sfm_sweep/<axis>/<id>/
  run       - run paper_experiments + gpu_paper_experiments on the sweep tree
  aggregate - read per-init result JSONs and write sweep_results.json

Typical workflow:
  python examples/sfm_sweep.py generate
  python examples/sfm_sweep.py run
  python examples/sfm_sweep.py aggregate
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
SWEEP_ROOT = REPO / "examples" / "data" / "sfm_sweep"
CONFIG_PATH = REPO / "examples" / "config.json"
ANALYSIS_DIR = REPO / "examples" / "data" / "analysis"

# Connectivity is held fixed across both axes (the user explicitly skipped a
# connectivity sweep). 30 observations per landmark sits roughly in the middle
# of the existing SfM benchmark suite (bal-93: ~28, Replica-REProom1: ~91).
OBS_PER_LANDMARK = 30
RANK = 5
NOISE_SIGMA = 0.005  # bearing-residual noise in normalised camera coordinates

# Axis A: total var count, fixed pose:landmark = 1:5. 50k matches the largest
# existing SfM dataset (Replica-REProom1: 49,759 vars). Roughly log-spaced.
SIZE_GRID = [2_000, 3_000, 5_000, 7_500, 10_000, 15_000, 22_000]
RATIO_FIXED_POSE_TO_LANDMARK = (1, 5)

# Axis B: pose:landmark ratio, fixed total = 10k. Roughly log-spaced.
TOTAL_FOR_RATIO = 10_000
RATIO_GRID = [(1, 1), (1, 2), (1, 3), (1, 5), (1, 10), (1, 20), (1, 50), (1, 100)]


# Formulation enum used in the result JSONs (see VarPro::Formulation).
FORMULATION_NAME = {0: "Explicit", 1: "ExplicitVarPro", 2: "Implicit"}


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    axis: str   # "size" or "ratio"
    name: str   # used as both directory name and pyfg basename
    P: int      # number of camera poses (constrained variables)
    L: int      # number of landmark points (unconstrained variables)
    K: int      # observations per landmark
    seed: int

    @property
    def total_vars(self) -> int:
        return self.P + self.L

    @property
    def total_obs(self) -> int:
        return self.L * min(self.K, self.P)


def make_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    base_seed = 42

    # Axis A: total size at fixed P:L = 1:5
    a, b = RATIO_FIXED_POSE_TO_LANDMARK
    for total in SIZE_GRID:
        P = total * a // (a + b)
        L = total - P
        scenarios.append(Scenario(
            axis="size",
            name=f"size_{total // 1000}k",
            P=P, L=L,
            K=OBS_PER_LANDMARK,
            seed=base_seed + total,
        ))

    # Axis B: P:L ratio at fixed total
    for (a, b) in RATIO_GRID:
        P = TOTAL_FOR_RATIO * a // (a + b)
        L = TOTAL_FOR_RATIO - P
        scenarios.append(Scenario(
            axis="ratio",
            name=f"ratio_1_{b}",
            P=P, L=L,
            K=OBS_PER_LANDMARK,
            seed=base_seed + 1000 + b,
        ))

    return scenarios


# ---------------------------------------------------------------------------
# Synthetic SfM scene
# ---------------------------------------------------------------------------

def look_at(cam_pos: np.ndarray,
            target: np.ndarray = np.zeros(3),
            up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Return a 3x3 rotation matrix R_world_from_cam (cam-Z forward, cam-Y down)."""
    z = target - cam_pos
    z /= np.linalg.norm(z) + 1e-30
    x = np.cross(up, z)
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    else:
        x /= nx
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Return (qx, qy, qz, qw) from a 3x3 rotation matrix."""
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


def generate_scenario(scenario: Scenario, out_root: Path) -> Path:
    rng = np.random.default_rng(scenario.seed)
    out_dir = out_root / scenario.axis / scenario.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cameras on a sphere of radius R_cam around origin, looking at origin.
    # Sample (theta, phi) avoiding the poles to keep the up-vector well-defined.
    R_cam = 5.0
    theta = rng.uniform(0.1, math.pi - 0.1, scenario.P)
    phi = rng.uniform(0.0, 2.0 * math.pi, scenario.P)
    cam_t = np.column_stack([
        R_cam * np.sin(theta) * np.cos(phi),
        R_cam * np.sin(theta) * np.sin(phi),
        R_cam * np.cos(theta),
    ])
    cam_R = np.stack([look_at(cam_t[i]) for i in range(scenario.P)], axis=0)

    # Landmarks uniform in a unit cube around origin (well inside the camera shell).
    landmarks = rng.uniform(-1.0, 1.0, (scenario.L, 3))

    # Visibility: each landmark observed by min(K, P) random cameras (without replacement).
    K_eff = min(scenario.K, scenario.P)

    # Write .pyfg
    pyfg_path = out_dir / f"{scenario.name}.pyfg"
    cov6 = "1.0 0.0 0.0 1.0 0.0 1.0"
    with pyfg_path.open("w") as f:
        for i in range(scenario.P):
            qx, qy, qz, qw = matrix_to_quat(cam_R[i])
            tx, ty, tz = cam_t[i]
            f.write(
                f"VERTEX_SE3:QUAT 0.000000 A{i} "
                f"{tx:.6f} {ty:.6f} {tz:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )
        for j in range(scenario.L):
            lx, ly, lz = landmarks[j]
            f.write(f"VERTEX_XYZ 0.0 L{j} {lx:.6f} {ly:.6f} {lz:.6f}\n")
        for j in range(scenario.L):
            cam_idxs = rng.choice(scenario.P, size=K_eff, replace=False)
            for i in cam_idxs:
                R = cam_R[i]
                p_cam = R.T @ (landmarks[j] - cam_t[i])
                p_cam = p_cam + rng.normal(0.0, NOISE_SIGMA, 3)
                f.write(
                    f"EDGE_SE3_XYZ 0.0 A{i} L{j} "
                    f"{p_cam[0]:.6f} {p_cam[1]:.6f} {p_cam[2]:.6f} "
                    f"{cov6}\n"
                )

    meta = {
        "axis": scenario.axis,
        "scenario": scenario.name,
        "poses": scenario.P,
        "landmarks": scenario.L,
        "obs_per_landmark": K_eff,
        "total_vars": scenario.total_vars,
        "total_observations": scenario.L * K_eff,
        "pose_to_landmark_ratio": scenario.L / scenario.P,
        "seed": scenario.seed,
        "noise_sigma": NOISE_SIGMA,
        "rank": RANK,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def cmd_generate(args: argparse.Namespace) -> int:
    if SWEEP_ROOT.exists() and not args.force:
        print(f"{SWEEP_ROOT} already exists. Pass --force to overwrite.")
        return 1
    if SWEEP_ROOT.exists():
        shutil.rmtree(SWEEP_ROOT)
    for sc in make_scenarios():
        out = generate_scenario(sc, SWEEP_ROOT)
        print(f"  {out.relative_to(REPO)}: P={sc.P}  L={sc.L}  obs={sc.total_obs}")
    print(f"done. {len(make_scenarios())} scenarios under {SWEEP_ROOT.relative_to(REPO)}")
    return 0


# ---------------------------------------------------------------------------
# Run wrapper
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
            print(f"=== CPU sweep (paper_experiments) ===")
            subprocess.run([str(paper)], cwd=REPO, check=True)
        if need_gpu:
            print(f"=== GPU sweep (gpu_paper_experiments) ===")
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
        "noise_sigma": NOISE_SIGMA,
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
                        print(f"  warning: skipping unreadable {cache_file}")
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
    out_path = ANALYSIS_DIR / "sweep_results.json"
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
    r.add_argument("--num-inits", type=int, default=5,
                   help="random initializations per (scenario, formulation) [default 5]")
    r.add_argument("--cpu-only", action="store_true")
    r.add_argument("--gpu-only", action="store_true")

    sub.add_parser("aggregate",
                   help="dump per-scenario runs to data/analysis/sweep_results.json")

    args = parser.parse_args()
    if args.cmd == "generate":
        sys.exit(cmd_generate(args))
    if args.cmd == "run":
        sys.exit(cmd_run(args))
    if args.cmd == "aggregate":
        sys.exit(cmd_aggregate(args))


if __name__ == "__main__":
    main()
