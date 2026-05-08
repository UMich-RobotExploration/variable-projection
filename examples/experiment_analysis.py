#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FORMULATION_MAP = {
    0: "Explicit",
    1: "Explicit VarPro",
    2: "Implicit",
    "gtsam": "GTSAM",
}

TASK_ORDER = ["PGO", "RA-SLAM", "SNL", "SfM"]
CPU_METHODS = ["Implicit", "Explicit", "Explicit VarPro", "GTSAM"]
GPU_METHODS = ["Implicit", "Explicit", "Explicit VarPro"]
RANK_TARGET = "rank5"
CONVERGENCE_TOL = 0.01
RANK_PATTERN = re.compile(r"rank(\d+)", re.IGNORECASE)

# GPU outputs (table + speedup plots) drop SNL and the MR.CLAM RA-SLAM datasets.
GPU_EXCLUDED_TASKS = {"SNL"}
GPU_EXCLUDED_KEYS = {"mrclam2", "mrclam4", "mrclam6", "mrclam7"}


def gpu_specs() -> list["DatasetSpec"]:
    return [
        spec
        for spec in DATASET_SPECS
        if spec.task not in GPU_EXCLUDED_TASKS and spec.key not in GPU_EXCLUDED_KEYS
    ]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    task: str


DATASET_SPECS = [
    DatasetSpec("intel", "Intel", "PGO"),
    DatasetSpec("parking-garage", "Garage", "PGO"),
    DatasetSpec("grid3D", "Grid3D", "PGO"),
    DatasetSpec("MIT", "MIT", "PGO"),
    DatasetSpec("M3500", "M3500", "PGO"),
    DatasetSpec("city10000", "City10000", "PGO"),
    DatasetSpec("torus3D", "Torus", "PGO"),
    DatasetSpec("sphere2500", "Sphere", "PGO"),
    DatasetSpec("tiers", "TIERS", "RA-SLAM"),
    DatasetSpec("single_drone", "Single Drone", "RA-SLAM"),
    DatasetSpec("plaza2", "Plaza2", "RA-SLAM"),
    DatasetSpec("plaza1", "Plaza1", "RA-SLAM"),
    DatasetSpec("mrclam2", "MR.CLAM2", "RA-SLAM"),
    DatasetSpec("mrclam4", "MR.CLAM4", "RA-SLAM"),
    DatasetSpec("mrclam6", "MR.CLAM6", "RA-SLAM"),
    DatasetSpec("mrclam7", "MR.CLAM7", "RA-SLAM"),
    DatasetSpec("intel_snl", "Intel", "SNL"),
    DatasetSpec("parking-garage_snl", "Garage", "SNL"),
    DatasetSpec("grid3D_snl", "Grid3D", "SNL"),
    DatasetSpec("MIT_snl", "MIT", "SNL"),
    DatasetSpec("M3500_snl", "M3500", "SNL"),
    DatasetSpec("city10000_snl", "City10000", "SNL"),
    DatasetSpec("torus3D_snl", "Torus", "SNL"),
    DatasetSpec("sphere2500_snl", "Sphere", "SNL"),
    DatasetSpec("bal-93", "BAL-93", "SfM"),
    DatasetSpec("bal-392", "BAL-392", "SfM"),
    DatasetSpec("bal-1934", "BAL-1934", "SfM"),
    DatasetSpec("IMC-gate", "IMC Gate", "SfM"),
    DatasetSpec("IMC-temple", "IMC Temple", "SfM"),
    DatasetSpec("IMC-rome", "IMC Rome", "SfM"),
    DatasetSpec("Replica-REPoffice0_100", "Rep. Office0-100", "SfM"),
    DatasetSpec("Replica-REPoffice1_100", "Rep. Office1-100", "SfM"),
    DatasetSpec("Replica-REProom0_100", "Rep. Room0-100", "SfM"),
    DatasetSpec("Replica-REProom1_100", "Rep. Room1-100", "SfM"),
    DatasetSpec("MipNerf-garden", "Mip-NeRF Garden", "SfM"),
    DatasetSpec("MipNerf-room", "Mip-NeRF Room", "SfM"),
    DatasetSpec("MipNerf-kitchen", "Mip-NeRF Kitchen", "SfM"),
    DatasetSpec("TUM-room", "TUM Room", "SfM"),
    DatasetSpec("TUM-desk", "TUM Desk", "SfM"),
    DatasetSpec("TUM-computer-R", "TUM Comp-R", "SfM"),
    DatasetSpec("TUM-computer-T", "TUM Comp-T", "SfM"),
]

DATASET_BY_KEY = {spec.key: spec for spec in DATASET_SPECS}

# These values are taken directly from the CPU table in the paper draft the user
# provided. They are treated as the authoritative GTSAM reference for the CPU
# export because no local GTSAM results repository exists in this workspace.
GTSAM_REFERENCE = {
    "intel": {"time_s": 0.63, "iters": 12},
    "parking-garage": {"time_s": 7.51, "iters": 63},
    "grid3D": {"time_s": None, "iters": None},
    "MIT": {"time_s": 0.28, "iters": 11},
    "M3500": {"time_s": 2.85, "iters": 22},
    "city10000": {"time_s": 18.43, "iters": 29},
    "torus3D": {"time_s": 37.24, "iters": 36},
    "sphere2500": {"time_s": 14.68, "iters": 39},
    "tiers": {"time_s": None, "iters": None},
    "single_drone": {"time_s": 4.91, "iters": 57},
    "plaza2": {"time_s": 4.73, "iters": 57},
    "plaza1": {"time_s": 10.91, "iters": 63},
    "mrclam2": {"time_s": 11.59, "iters": 18},
    "mrclam4": {"time_s": 8.02, "iters": 15},
    "mrclam6": {"time_s": 5.89, "iters": 25},
    "mrclam7": {"time_s": 10.32, "iters": 36},
    "intel_snl": {"time_s": None, "iters": None},
    "parking-garage_snl": {"time_s": None, "iters": None},
    "grid3D_snl": {"time_s": None, "iters": None},
    "MIT_snl": {"time_s": 0.21, "iters": 12},
    "M3500_snl": {"time_s": None, "iters": None},
    "city10000_snl": {"time_s": None, "iters": None},
    "torus3D_snl": {"time_s": None, "iters": None},
    "sphere2500_snl": {"time_s": None, "iters": None},
    "bal-93": {"time_s": 10.09, "iters": 52},
    "bal-392": {"time_s": None, "iters": None},
    "bal-1934": {"time_s": None, "iters": None},
    "IMC-gate": {"time_s": None, "iters": None},
    "IMC-temple": {"time_s": None, "iters": None},
    "IMC-rome": {"time_s": None, "iters": None},
    "Replica-REPoffice0_100": {"time_s": 8.39, "iters": 10},
    "Replica-REPoffice1_100": {"time_s": 3.14, "iters": 9},
    "Replica-REProom0_100": {"time_s": 12.00, "iters": 9},
    "Replica-REProom1_100": {"time_s": 13.63, "iters": 13},
    "MipNerf-garden": {"time_s": 19.76, "iters": 12},
    "MipNerf-room": {"time_s": None, "iters": None},
    "MipNerf-kitchen": {"time_s": None, "iters": None},
    "TUM-room": {"time_s": None, "iters": None},
    "TUM-desk": {"time_s": None, "iters": None},
    "TUM-computer-R": {"time_s": None, "iters": None},
    "TUM-computer-T": {"time_s": None, "iters": None},
}

def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parent / "data"
    parser = argparse.ArgumentParser(
        description="Export CPU/GPU experiment tables and speedup plots."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=data_dir,
        help="Root directory containing <dataset>/cached_results/ subfolders. "
             "CPU and GPU per-init result JSONs are loaded directly from those, "
             "avoiding any stale top-level aggregate file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_dir / "analysis",
        help="Directory for CSV exports and plots.",
    )
    return parser.parse_args()


def normalize_formulation(value: Any) -> str | None:
    if value in FORMULATION_MAP:
        return FORMULATION_MAP[value]
    if isinstance(value, str):
        normalized = value.strip().lower().replace("_", " ").replace("-", " ")
        if normalized == "explicit":
            return "Explicit"
        if normalized in {"explicit varpro", "explicit var proj", "varpro"}:
            return "Explicit VarPro"
        if normalized == "implicit":
            return "Implicit"
        if normalized == "gtsam":
            return "GTSAM"
    return None


def extract_rank(init_file: str) -> str:
    match = RANK_PATTERN.search(init_file or "")
    if not match:
        return "unknown"
    return f"rank{match.group(1)}"


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [entry for entry in data if isinstance(entry, dict)]


def load_from_cached_results(data_root: Path, gpu: bool) -> list[dict[str, Any]]:
    """Walk data_root for per-init cached results and concatenate them.

    Each dataset writes one JSON per (formulation, init) pair into
    <dataset>/cached_results/. CPU files are named ``results_rank<N>_<form>_init<i>.json``;
    GPU files are prefixed ``gpu_results_...``. Each file is a list with one entry.

    Reading these directly (instead of the aggregate ``experiment_results.json``)
    avoids the staleness problem where the aggregate is written once at the end of
    a sweep and may not reflect recent re-runs of individual datasets.
    """
    prefix = "gpu_results_rank" if gpu else "results_rank"
    entries: list[dict[str, Any]] = []
    skipped = 0
    for results_file in data_root.rglob(f"{prefix}*.json"):
        if results_file.parent.name != "cached_results":
            continue
        try:
            with results_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        if isinstance(data, list):
            entries.extend(e for e in data if isinstance(e, dict))
        elif isinstance(data, dict):
            entries.append(data)
    if skipped:
        print(f"Warning: skipped {skipped} unreadable cached result file(s) under {data_root}")
    return entries


def extract_run_trace(
    entry: dict[str, Any],
    formulation: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    times = np.asarray(entry.get("times", []), dtype=float)
    costs = np.asarray(entry.get("costs", []), dtype=float)
    length = min(times.size, costs.size)
    if length == 0:
        return None

    times = times[:length]
    costs = costs[:length]

    if formulation == "GTSAM":
        times = np.cumsum(times)

    finite_mask = np.isfinite(times) & np.isfinite(costs)
    if not finite_mask.any():
        return None
    bad_indices = np.flatnonzero(~finite_mask)
    if bad_indices.size:
        # Treat a run with trailing NaNs/nulls as failed rather than silently
        # truncating it. The paper table renders those cases as missing.
        return None

    return times, costs


def first_threshold_hit(
    times: np.ndarray,
    costs: np.ndarray,
    target_cost: float,
) -> tuple[float, int] | None:
    if costs.size == 0:
        return None

    threshold = (1.0 + CONVERGENCE_TOL) * target_cost
    hit_indices = np.flatnonzero(costs <= threshold)
    if hit_indices.size == 0:
        return None

    first_hit = int(hit_indices[0])
    return float(times[first_hit]), first_hit


def aggregate_results(
    entries: list[dict[str, Any]],
    method_order: list[str],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    grouped: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
        spec.key: {
            method: []
            for method in method_order
        }
        for spec in DATASET_SPECS
    }

    for entry in entries:
        dataset_key = entry.get("dataset_name")
        if dataset_key not in DATASET_BY_KEY:
            continue

        if extract_rank(entry.get("init_file", "")) != RANK_TARGET:
            continue

        formulation = normalize_formulation(entry.get("formulation"))
        if formulation not in method_order:
            continue

        trace = extract_run_trace(entry, formulation)
        if trace is None:
            continue

        grouped[dataset_key][formulation].append(trace)

    summary: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for spec in DATASET_SPECS:
        dataset_summary: dict[str, dict[str, float | int | None]] = {}
        implicit_runs = grouped[spec.key]["Implicit"]
        implicit_final_costs = np.asarray(
            [costs[-1] for _, costs in implicit_runs if costs.size],
            dtype=float,
        )
        implicit_target = (
            float(np.median(implicit_final_costs))
            if implicit_final_costs.size
            else None
        )

        for method in method_order:
            runs = grouped[spec.key][method]
            if implicit_target is None:
                dataset_summary[method] = {"time_s": None, "iters": None}
                continue

            successful_runs = [
                hit
                for times, costs in runs
                if (hit := first_threshold_hit(times, costs, implicit_target)) is not None
            ]
            if not successful_runs:
                dataset_summary[method] = {"time_s": None, "iters": None}
                continue

            times = np.asarray([runtime for runtime, _ in successful_runs], dtype=float)
            iters = np.asarray([iterations for _, iterations in successful_runs], dtype=float)
            dataset_summary[method] = {
                "time_s": float(np.median(times)),
                "iters": int(np.median(iters)) if iters.size else None,
            }
        summary[spec.key] = dataset_summary

    return summary


def attach_gtsam_reference(
    summary: dict[str, dict[str, dict[str, float | int | None]]]
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    for spec in DATASET_SPECS:
        reference = GTSAM_REFERENCE.get(spec.key, {"time_s": None, "iters": None})
        summary[spec.key]["GTSAM"] = {
            "time_s": reference["time_s"],
            "iters": reference["iters"],
        }
    return summary
def safe_speedup(
    baseline_time: float | int | None,
    target_time: float | int | None,
) -> float | None:
    if baseline_time is None or target_time is None:
        return None
    baseline = float(baseline_time)
    target = float(target_time)
    if baseline <= 0.0 or target <= 0.0:
        return None
    return baseline / target


def format_float(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def format_int(value: int | float | None) -> str:
    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.2f}"


def write_cpu_table(
    path: Path,
    cpu_summary: dict[str, dict[str, dict[str, float | int | None]]],
) -> None:
    header = [
        "task",
        "dataset",
        "ours_time_s",
        "original_time_s",
        "orig_vp_time_s",
        "gtsam_time_s",
        "ours_iters",
        "original_iters",
        "orig_vp_iters",
        "gtsam_iters",
        "speedup_original_over_ours",
        "speedup_orig_vp_over_ours",
        "speedup_gtsam_over_ours",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()

        for spec in DATASET_SPECS:
            methods = cpu_summary[spec.key]
            ours = methods["Implicit"]
            original = methods["Explicit"]
            orig_vp = methods["Explicit VarPro"]
            gtsam = methods["GTSAM"]
            writer.writerow(
                {
                    "task": spec.task,
                    "dataset": spec.label,
                    "ours_time_s": format_float(ours["time_s"]),
                    "original_time_s": format_float(original["time_s"]),
                    "orig_vp_time_s": format_float(orig_vp["time_s"]),
                    "gtsam_time_s": format_float(gtsam["time_s"]),
                    "ours_iters": format_int(ours["iters"]),
                    "original_iters": format_int(original["iters"]),
                    "orig_vp_iters": format_int(orig_vp["iters"]),
                    "gtsam_iters": format_int(gtsam["iters"]),
                    "speedup_original_over_ours": format_float(
                        safe_speedup(original["time_s"], ours["time_s"])
                    ),
                    "speedup_orig_vp_over_ours": format_float(
                        safe_speedup(orig_vp["time_s"], ours["time_s"])
                    ),
                    "speedup_gtsam_over_ours": format_float(
                        safe_speedup(gtsam["time_s"], ours["time_s"])
                    ),
                }
            )


def write_gpu_table(
    path: Path,
    gpu_summary: dict[str, dict[str, dict[str, float | int | None]]],
) -> None:
    header = [
        "task",
        "dataset",
        "ours_time_s",
        "original_time_s",
        "orig_vp_time_s",
        "ours_iters",
        "original_iters",
        "orig_vp_iters",
        "speedup_original_over_ours",
        "speedup_orig_vp_over_ours",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()

        for spec in gpu_specs():
            methods = gpu_summary[spec.key]
            ours = methods["Implicit"]
            original = methods["Explicit"]
            orig_vp = methods["Explicit VarPro"]
            writer.writerow(
                {
                    "task": spec.task,
                    "dataset": spec.label,
                    "ours_time_s": format_float(ours["time_s"]),
                    "original_time_s": format_float(original["time_s"]),
                    "orig_vp_time_s": format_float(orig_vp["time_s"]),
                    "ours_iters": format_int(ours["iters"]),
                    "original_iters": format_int(original["iters"]),
                    "orig_vp_iters": format_int(orig_vp["iters"]),
                    "speedup_original_over_ours": format_float(
                        safe_speedup(original["time_s"], ours["time_s"])
                    ),
                    "speedup_orig_vp_over_ours": format_float(
                        safe_speedup(orig_vp["time_s"], ours["time_s"])
                    ),
                }
            )


def filter_specs_with_data(
    series: list[tuple[str, dict[str, float | None]]]
) -> list[DatasetSpec]:
    filtered = []
    for spec in DATASET_SPECS:
        if any(series_map.get(spec.key) is not None for _, series_map in series):
            filtered.append(spec)
    return filtered


def add_task_separators(ax: plt.Axes, specs: list[DatasetSpec]) -> None:
    if not specs:
        return

    start = 0
    for index in range(len(specs) - 1):
        if specs[index].task != specs[index + 1].task:
            ax.axvline(index + 0.5, color="0.75", linewidth=1.0, zorder=0)
            center = (start + index) / 2.0
            ax.text(
                center,
                1.02,
                specs[index].task,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
            start = index + 1

    center = (start + len(specs) - 1) / 2.0
    ax.text(
        center,
        1.02,
        specs[-1].task,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def save_speedup_plot(
    output_base: Path,
    title: str,
    series: list[tuple[str, dict[str, float | None]]],
) -> None:
    specs = filter_specs_with_data(series)
    if not specs:
        return

    x = np.arange(len(specs), dtype=float)
    colors = ["C0", "C1", "C2", "C3", "C4"]
    total_width = 0.82
    bar_width = total_width / max(len(series), 1)

    positive_values = []
    for _, series_map in series:
        for spec in specs:
            value = series_map.get(spec.key)
            if value is not None and math.isfinite(value) and value > 0.0:
                positive_values.append(value)
    if not positive_values:
        return

    # Snug log bounds around the actual data so bars above 1 (the regime we
    # care about) aren't visually compressed by an empty decade below. Always
    # include 1.0 so the no-speedup reference line is drawn.
    y_min = min(positive_values)
    y_max = max(positive_values)
    log_pad = 0.05
    y_floor = 10 ** (math.log10(min(y_min, 1.0)) - log_pad)
    y_top = 10 ** (math.log10(max(y_max, 1.0)) + log_pad)

    fig, ax = plt.subplots(figsize=(max(14.0, len(specs) * 0.45), 6.5), dpi=150)

    for idx, (label, series_map) in enumerate(series):
        xs = []
        heights = []
        for spec_index, spec in enumerate(specs):
            value = series_map.get(spec.key)
            if value is None or not math.isfinite(value) or value <= 0.0:
                continue
            x_offset = -total_width / 2.0 + (idx + 0.5) * bar_width
            xs.append(x[spec_index] + x_offset)
            heights.append(value - y_floor)
        if xs and heights:
            ax.bar(
                xs,
                heights,
                width=bar_width * 0.9,
                bottom=y_floor,
                label=label,
                color=colors[idx % len(colors)],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(y_floor, y_top)
    ax.set_ylabel("Runtime Improvement Factor (baseline / ours)")
    # Lift the title clear of the task-section labels rendered at y=1.02 in
    # axes-fraction coordinates by add_task_separators.
    ax.set_title(title, pad=28)
    ax.set_xticks(x)
    ax.set_xticklabels([spec.label for spec in specs], rotation=60, ha="right")
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=min(len(series), 3), loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_task_separators(ax, specs)
    fig.tight_layout()

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def build_cpu_speedup_series(
    cpu_summary: dict[str, dict[str, dict[str, float | int | None]]]
) -> list[tuple[str, dict[str, float | None]]]:
    original_vs_ours = {}
    orig_vp_vs_ours = {}
    gtsam_vs_ours = {}

    for spec in DATASET_SPECS:
        methods = cpu_summary[spec.key]
        ours = methods["Implicit"]["time_s"]
        original_vs_ours[spec.key] = safe_speedup(
            methods["Explicit"]["time_s"],
            ours,
        )
        orig_vp_vs_ours[spec.key] = safe_speedup(
            methods["Explicit VarPro"]["time_s"],
            ours,
        )
        gtsam_vs_ours[spec.key] = safe_speedup(
            methods["GTSAM"]["time_s"],
            ours,
        )

    return [
        ("Original / Ours", original_vs_ours),
        ("Orig. + VP / Ours", orig_vp_vs_ours),
        ("GTSAM / Ours", gtsam_vs_ours),
    ]


def build_gpu_speedup_series(
    gpu_summary: dict[str, dict[str, dict[str, float | int | None]]]
) -> list[tuple[str, dict[str, float | None]]]:
    original_vs_ours = {}
    orig_vp_vs_ours = {}

    for spec in gpu_specs():
        methods = gpu_summary[spec.key]
        ours = methods["Implicit"]["time_s"]
        original_vs_ours[spec.key] = safe_speedup(
            methods["Explicit"]["time_s"],
            ours,
        )
        orig_vp_vs_ours[spec.key] = safe_speedup(
            methods["Explicit VarPro"]["time_s"],
            ours,
        )

    return [
        ("Original / Ours", original_vs_ours),
        ("Orig. + VP / Ours", orig_vp_vs_ours),
    ]


def build_gpu_vs_cpu_speedup_series(
    cpu_summary: dict[str, dict[str, dict[str, float | int | None]]],
    gpu_summary: dict[str, dict[str, dict[str, float | int | None]]],
) -> list[tuple[str, dict[str, float | None]]]:
    explicit = {}
    explicit_varpro = {}
    implicit = {}

    for spec in gpu_specs():
        explicit[spec.key] = safe_speedup(
            cpu_summary[spec.key]["Explicit"]["time_s"],
            gpu_summary[spec.key]["Explicit"]["time_s"],
        )
        explicit_varpro[spec.key] = safe_speedup(
            cpu_summary[spec.key]["Explicit VarPro"]["time_s"],
            gpu_summary[spec.key]["Explicit VarPro"]["time_s"],
        )
        implicit[spec.key] = safe_speedup(
            cpu_summary[spec.key]["Implicit"]["time_s"],
            gpu_summary[spec.key]["Implicit"]["time_s"],
        )

    return [
        ("CPU / GPU (Original)", explicit),
        ("CPU / GPU (Orig. + VP)", explicit_varpro),
        ("CPU / GPU (Ours)", implicit),
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cpu_entries = load_from_cached_results(args.data_root, gpu=False)
    gpu_entries = load_from_cached_results(args.data_root, gpu=True)
    print(f"Loaded {len(cpu_entries)} CPU and {len(gpu_entries)} GPU per-init results "
          f"from {args.data_root}")

    cpu_summary = attach_gtsam_reference(aggregate_results(cpu_entries, GPU_METHODS))
    gpu_summary = aggregate_results(gpu_entries, GPU_METHODS)

    cpu_table_path = args.output_dir / "cpu_table.csv"
    gpu_table_path = args.output_dir / "gpu_table.csv"
    write_cpu_table(cpu_table_path, cpu_summary)
    write_gpu_table(gpu_table_path, gpu_summary)

    save_speedup_plot(
        args.output_dir / "cpu_speedups",
        "CPU Speedups Relative to Ours",
        build_cpu_speedup_series(cpu_summary),
    )
    save_speedup_plot(
        args.output_dir / "gpu_speedups",
        "GPU Speedups Relative to Ours",
        build_gpu_speedup_series(gpu_summary),
    )
    save_speedup_plot(
        args.output_dir / "gpu_vs_cpu_speedups",
        "CPU-to-GPU Speedup by Method",
        build_gpu_vs_cpu_speedup_series(cpu_summary, gpu_summary),
    )

    print(f"Wrote {cpu_table_path}")
    print(f"Wrote {gpu_table_path}")
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
