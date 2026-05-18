#!/usr/bin/env python3
"""Measure only the precompute step (`fillImplicitFormulationMatrices()`) on
every standard-dataset directory, grouped by problem type (PGO / SNL / SfM /
RA-SLAM). No solver runs — only parse + `updateProblemData()`.

Reads each .pyfg, records the precompute time reported by
`Problem::getImplicitPrecomputeTimeS()`, and writes the aggregate to
examples/data/analysis/standard_precompute_times.json.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "examples" / "data"
ANALYSIS_DIR = REPO / "examples" / "data" / "analysis"
BIN = REPO / "build" / "bin" / "benchmark_precompute"

# Map of problem type → dataset root. Skip the SfM `clean_sfm_formats.py`
# script that lives next to the dataset directories.
PROBLEM_ROOTS = {
    "PGO":     DATA_ROOT / "pgo",
    "SNL":     DATA_ROOT / "snl",
    "SfM":     DATA_ROOT / "sfm",
    "RA-SLAM": DATA_ROOT / "raslam",
}


def list_datasets(root: Path) -> list[Path]:
    """Return every directory under `root` that contains a .pyfg file. Walks
    one level deeper when the immediate child has no .pyfg of its own (e.g.
    raslam/mrclam contains mrclam2/, mrclam4/, ... each with its own .pyfg)."""
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if any(c.suffix == ".pyfg" for c in p.iterdir()):
            out.append(p)
            continue
        for sub in sorted(p.iterdir()):
            if sub.is_dir() and any(c.suffix == ".pyfg" for c in sub.iterdir()):
                out.append(sub)
    return out


def parse_output(stdout: str) -> dict[str, dict]:
    """Pull tab-separated rows out of benchmark_precompute stdout, ignoring
    chatter like 'Regularized Cholesky preconditioner ...' that some datasets
    log to stdout. Expects 7 columns:
       path  cpu_precompute_s  gpu_upload_s  wall_s  poses  landmarks  ranges
    """
    rows: dict[str, dict] = {}
    for line in stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 7:
            continue
        path, cpu, gpu_up, wall, poses, lmks, ranges = parts
        if path == "path":  # header
            continue
        try:
            cpu_f, gpu_up_f = float(cpu), float(gpu_up)
            rows[path] = {
                "precompute_s": cpu_f,
                "gpu_upload_s": gpu_up_f,
                "gpu_precompute_s": cpu_f + gpu_up_f,
                "wall_s": float(wall),
                "poses": int(poses),
                "landmarks": int(lmks),
                "ranges": int(ranges),
            }
        except ValueError:
            continue
    return rows


def main() -> int:
    if not BIN.exists():
        print(f"missing {BIN}; build with `cmake --build build --target benchmark_precompute`")
        return 1

    results: dict[str, dict] = {}
    for problem, root in PROBLEM_ROOTS.items():
        if not root.exists():
            print(f"skip {problem}: {root} missing")
            continue
        datasets = list_datasets(root)
        if not datasets:
            print(f"skip {problem}: no .pyfg datasets under {root}")
            continue
        print(f"=== {problem} ({len(datasets)} datasets) ===")
        proc = subprocess.run(
            [str(BIN), "--gpu", *[str(d) for d in datasets]],
            capture_output=True, text=True,
        )
        rows = parse_output(proc.stdout)
        if proc.returncode != 0:
            print(f"  benchmark_precompute exited {proc.returncode}; "
                  f"got {len(rows)}/{len(datasets)} rows")
            if proc.stderr.strip():
                print("  stderr:")
                for line in proc.stderr.splitlines():
                    print(f"    {line}")
        per_dataset: dict[str, dict] = {}
        for d in datasets:
            r = rows.get(str(d))
            if r is None:
                print(f"  {d.name:<25} (no output)")
                continue
            print(f"  {d.name:<25} cpu={r['precompute_s']:.6f}s  "
                  f"gpu={r['gpu_precompute_s']:.6f}s  "
                  f"poses={r['poses']} lmks={r['landmarks']} ranges={r['ranges']}")
            per_dataset[d.name] = r
        results[problem] = per_dataset

    out_path = ANALYSIS_DIR / "standard_precompute_times.json"
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
