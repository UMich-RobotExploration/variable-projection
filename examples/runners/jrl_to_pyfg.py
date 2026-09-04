"""Convert JRL (DanMcGann/jrl) datasets to PyFG text format (MarineRoboticsGroup/PyFactorGraph).

For each input .jrl file we emit two PyFG files:
  - <name>.pyfg            — all measurements, matching the source JRL graph
  - <name>_no_outliers.pyfg — same graph with ground-truth outlier factors removed

Only the PriorFactorPose3 and BetweenFactorPose3 measurement types are handled
(the COSMO-Bench + Nebula datasets only use these). Cross-robot BetweenFactor
edges become loop-closure-style EDGE_SE3:QUAT entries between robot frames.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np


# PyFG record tags (3D only; all our inputs are Pose3).
POSE_TYPE_3D = "VERTEX_SE3:QUAT"
POSE_PRIOR_3D = "VERTEX_SE3:QUAT:PRIOR"
REL_POSE_POSE_TYPE_3D = "EDGE_SE3:QUAT"

# Same precision PyFactorGraph's own writer uses, so the files round-trip.
TIME_FPREC = 9
TRANSLATION_FPREC = 9
QUATERNION_FPREC = 7
COVARIANCE_FPREC = 9

# PyFG reserves 'L' for landmarks. Build the same map JRL → PyFG.
_ROBOT_CHARS = [chr(ord("A") + i) for i in range(26) if chr(ord("A") + i) != "L"]


def _gtsam_key_to_name(key: int) -> str:
    """Decode a GTSAM symbol (uint64) into a PyFG variable name like 'A12'."""
    char_byte = (key >> 56) & 0xFF
    index = key & ((1 << 56) - 1)
    jrl_char = chr(char_byte)

    if jrl_char.isalpha():
        pyfg_char = jrl_char.upper()
    else:
        # Numeric robot id (shouldn't happen in COSMO-Bench / Nebula but be safe).
        pyfg_char = _ROBOT_CHARS[char_byte % len(_ROBOT_CHARS)]
    if pyfg_char == "L":
        # Avoid collision with PyFG's landmark prefix.
        raise ValueError(f"JRL key {key} maps to reserved PyFG letter 'L'")
    return f"{pyfg_char}{index}"


def _jrl_quat_to_xyzw(rotation: List[float]) -> Tuple[float, float, float, float]:
    """JRL stores quaternions as [w, x, y, z]; PyFG wants (x, y, z, w)."""
    qw, qx, qy, qz = rotation
    return qx, qy, qz, qw


def _row_major_to_upper_triangle(cov_row_major: List[float], size: int) -> List[float]:
    """JRL covariance is a flat 6x6 row-major matrix.

    PyFG's reader expects the values in the order produced by
    get_list_column_major_from_symmetric_matrix: iterate i in 0..size, j in i..size,
    emit mat[i, j]. (The helper's name says 'column major' but the loop emits the
    upper triangle row-by-row.) We rebuild the matrix and serialize identically so
    a PyFG reader can rebuild the exact same covariance.

    For 6x6 Pose3 covariances we also reorder blocks: JRL/GTSAM uses tangent
    order [rotation; translation], but PyFG (via SE-Sync conventions) reads the
    top-left 3x3 as translation and the bottom-right 3x3 as rotation. We swap
    the two 3-blocks so the recovered precisions are physically correct.
    """
    if len(cov_row_major) != size * size:
        raise ValueError(
            f"covariance has {len(cov_row_major)} entries, expected {size * size}"
        )
    mat = np.asarray(cov_row_major, dtype=float).reshape(size, size)
    if size == 6:
        # Permutation swapping the first 3 axes with the last 3.
        perm = np.array([3, 4, 5, 0, 1, 2])
        mat = mat[np.ix_(perm, perm)]
    # Symmetrize tiny numerical asymmetries so the PyFG _check_symmetric passes.
    mat = 0.5 * (mat + mat.T)
    vals: List[float] = []
    for i in range(size):
        for j in range(i, size):
            vals.append(float(mat[i, j]))
    return vals


def _fmt_cov(cov_row_major: List[float], size: int) -> str:
    elems = _row_major_to_upper_triangle(cov_row_major, size)
    parts = [f"{v:.{COVARIANCE_FPREC}f}" for v in elems]
    # PyFG's writer scrubs "-0." to "0." in the noise field — match it so the
    # files are byte-comparable to a round-trip through PyFG.
    return " ".join(parts).replace("-0. ", "0. ")


def _gather_groundtruth(
    jrl: dict,
) -> Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
    """Collect a single ground-truth pose per variable name.

    JRL stores ground-truth per robot, and shared variables (observed by multiple
    robots) appear in each robot's list. The format guarantees those duplicates
    are identical, so we take whichever copy we see first.
    """
    gt: Dict[
        str, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]
    ] = {}
    for _robot, values in jrl.get("groundtruth", {}).items():
        for v in values:
            if v.get("type") != "Pose3":
                raise ValueError(f"Unsupported groundtruth type: {v.get('type')}")
            name = _gtsam_key_to_name(v["key"])
            if name in gt:
                continue
            tx, ty, tz = v["translation"]
            quat_xyzw = _jrl_quat_to_xyzw(v["rotation"])
            gt[name] = ((float(tx), float(ty), float(tz)), quat_xyzw)
    return gt


def _name_sort_key(name: str) -> Tuple[str, int]:
    """Sort variable names by robot letter, then by numeric index."""
    # Names are <letter><digits>; index can be large so split on first non-letter.
    i = 0
    while i < len(name) and name[i].isalpha():
        i += 1
    return name[:i], int(name[i:])


def _write_pyfg(
    out_path: Path,
    pose_lines: List[str],
    prior_lines: List[str],
    edge_lines: List[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for line in pose_lines:
            f.write(line + "\n")
        for line in prior_lines:
            f.write(line + "\n")
        for line in edge_lines:
            f.write(line + "\n")


def convert_jrl_file(jrl_path: Path, out_dir: Path) -> Tuple[Path, Path, dict]:
    """Convert a single JRL file to (full, no-outliers) PyFG files. Returns paths + stats."""
    with open(jrl_path, "r") as f:
        jrl = json.load(f)

    gt = _gather_groundtruth(jrl)

    # Pose vertex lines (ground-truth values), sorted for deterministic output.
    pose_lines: List[str] = []
    # PyFG vertex lines do not carry a timestamp from JRL ground-truth, so we
    # use 0.0 — the optimization back-ends in this project don't read it for
    # pose chains derived from PGO files.
    for name in sorted(gt.keys(), key=_name_sort_key):
        (tx, ty, tz), (qx, qy, qz, qw) = gt[name]
        pose_lines.append(
            f"{POSE_TYPE_3D} 0.{'0' * TIME_FPREC} {name} "
            f"{tx:.{TRANSLATION_FPREC}f} {ty:.{TRANSLATION_FPREC}f} {tz:.{TRANSLATION_FPREC}f} "
            f"{qx:.{QUATERNION_FPREC}f} {qy:.{QUATERNION_FPREC}f} "
            f"{qz:.{QUATERNION_FPREC}f} {qw:.{QUATERNION_FPREC}f}"
        )

    outlier_factor_ids: Dict[str, Set[Tuple[int, int]]] = {}
    for robot, ids in (jrl.get("outlier_factors") or {}).items():
        outlier_factor_ids[robot] = {(int(a), int(b)) for a, b in ids}

    prior_lines_all: List[str] = []
    prior_lines_clean: List[str] = []
    edge_lines_all: List[str] = []
    edge_lines_clean: List[str] = []

    n_prior = 0
    n_between = 0
    n_outliers = 0
    n_skipped_dup_between = 0

    # Track every BetweenFactor as an unordered (name1, name2) pair so we drop
    # the second appearance of inter-robot loop closures (JRL stores them once
    # per robot; PyFG's Problem treats unordered duplicates as the same edge
    # and rejects the second).
    seen_between_pairs: Set[frozenset] = set()

    for robot, entries in jrl.get("measurements", {}).items():
        outliers = outlier_factor_ids.get(robot, set())
        for entry_idx, entry in enumerate(entries):
            stamp_ns = int(entry.get("stamp", 0))
            stamp_s = stamp_ns * 1e-9
            for meas_idx, m in enumerate(entry["measurements"]):
                fid = (entry_idx, meas_idx)
                mtype = m["type"]
                is_outlier = fid in outliers

                if mtype == "PriorFactorPose3":
                    name = _gtsam_key_to_name(m["key"])
                    tx, ty, tz = m["prior"]["translation"]
                    qx, qy, qz, qw = _jrl_quat_to_xyzw(m["prior"]["rotation"])
                    cov_str = _fmt_cov(m["covariance"], size=6)
                    line = (
                        f"{POSE_PRIOR_3D} {stamp_s:.{TIME_FPREC}f} {name} "
                        f"{tx:.{TRANSLATION_FPREC}f} {ty:.{TRANSLATION_FPREC}f} {tz:.{TRANSLATION_FPREC}f} "
                        f"{qx:.{QUATERNION_FPREC}f} {qy:.{QUATERNION_FPREC}f} "
                        f"{qz:.{QUATERNION_FPREC}f} {qw:.{QUATERNION_FPREC}f} {cov_str}"
                    )
                    prior_lines_all.append(line)
                    if not is_outlier:
                        prior_lines_clean.append(line)
                    n_prior += 1

                elif mtype == "BetweenFactorPose3":
                    name1 = _gtsam_key_to_name(m["key1"])
                    name2 = _gtsam_key_to_name(m["key2"])
                    pair_key = frozenset({name1, name2})
                    if pair_key in seen_between_pairs:
                        # Same loop closure already reported by the other robot.
                        # PyFG's Problem dedup is unordered, so we'd be rejected
                        # at load time. Drop the second copy here.
                        n_skipped_dup_between += 1
                        continue
                    seen_between_pairs.add(pair_key)
                    tx, ty, tz = m["measurement"]["translation"]
                    qx, qy, qz, qw = _jrl_quat_to_xyzw(m["measurement"]["rotation"])
                    cov_str = _fmt_cov(m["covariance"], size=6)
                    line = (
                        f"{REL_POSE_POSE_TYPE_3D} {stamp_s:.{TIME_FPREC}f} {name1} {name2} "
                        f"{tx:.{TRANSLATION_FPREC}f} {ty:.{TRANSLATION_FPREC}f} {tz:.{TRANSLATION_FPREC}f} "
                        f"{qx:.{QUATERNION_FPREC}f} {qy:.{QUATERNION_FPREC}f} "
                        f"{qz:.{QUATERNION_FPREC}f} {qw:.{QUATERNION_FPREC}f} {cov_str}"
                    )
                    edge_lines_all.append(line)
                    if not is_outlier:
                        edge_lines_clean.append(line)
                    n_between += 1

                else:
                    raise ValueError(
                        f"Unsupported measurement type in {jrl_path.name}: {mtype}"
                    )

                if is_outlier:
                    n_outliers += 1

    base = jrl_path.stem
    full_path = out_dir / f"{base}.pyfg"
    clean_path = out_dir / f"{base}_no_outliers.pyfg"
    _write_pyfg(full_path, pose_lines, prior_lines_all, edge_lines_all)
    _write_pyfg(clean_path, pose_lines, prior_lines_clean, edge_lines_clean)

    stats = {
        "name": base,
        "robots": jrl.get("robots"),
        "poses": len(pose_lines),
        "priors": n_prior,
        "between": n_between,
        "outliers": n_outliers,
        "skipped_unordered_duplicates": n_skipped_dup_between,
        "dropped_from_clean": n_outliers,
    }
    return full_path, clean_path, stats


def _iter_jrl_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for dirpath, _dirs, files in os.walk(root):
        for fname in sorted(files):
            if fname.endswith(".jrl"):
                yield Path(dirpath) / fname


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        type=Path,
        help="Root directory containing .jrl files (recurses), or a single .jrl file.",
    )
    p.add_argument(
        "output",
        type=Path,
        help="Output directory. Subdirectory structure mirrors the input root.",
    )
    args = p.parse_args(argv)

    input_root = args.input
    output_root = args.output

    if not input_root.exists():
        print(f"error: input {input_root} does not exist", file=sys.stderr)
        return 2

    if input_root.is_file():
        # Single file: emit directly into output_root.
        _, _, stats = convert_jrl_file(input_root, output_root)
        print(
            f"{stats['name']}: poses={stats['poses']} priors={stats['priors']} "
            f"between={stats['between']} outliers={stats['outliers']}"
        )
        return 0

    total = 0
    for jrl_path in _iter_jrl_files(input_root):
        rel = jrl_path.parent.relative_to(input_root)
        out_dir = output_root / rel
        _, _, stats = convert_jrl_file(jrl_path, out_dir)
        print(
            f"{rel}/{stats['name']}: robots={stats['robots']} poses={stats['poses']} "
            f"priors={stats['priors']} between={stats['between']} "
            f"outliers={stats['outliers']}"
        )
        total += 1
    print(f"\nConverted {total} JRL file(s) into {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
