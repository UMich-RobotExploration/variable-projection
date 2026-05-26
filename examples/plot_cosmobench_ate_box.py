"""Box-and-whisker plot comparing per-dataset ATE across solvers.

Reads:
  examples/data/analysis/cosmobench_irls_gnc_full/<ds>.json   (VarPro IRLS: 3 formulations)
  examples/data/analysis/cosmobench_gtsam_gnc_full_noprior/<ds>.json   (GTSAM IRLS-GM, zero priors)

For each dataset we collect ATE (global Umeyama RMSE, meters) for the four
methods that appear in the LaTeX table:
  * Ours   ← IRLS formulation "impl"      (Implicit / Schur-eliminated translations)
  * Orig.  ← IRLS formulation "explicit"  (joint (R, t) on Stiefel relaxation)
  * O+VP   ← IRLS formulation "expvp"     (Explicit + VarPro)
  * GTSAM  ← GTSAM IRLS-GM loop with priors stripped (apples-to-apples)

We emit two PNGs:
  cosmobench_ate_box.png      — linear-y, with crashed-run / large-ATE points
                                drawn as fliers
  cosmobench_ate_box_log.png  — log-y, makes the tuhh_* / kth_r4_00 outliers
                                visible without compressing the bulk of the
                                distribution
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent
IRLS_DIR = REPO / "examples/data/analysis/cosmobench_irls_gnc_full"
GTSAM_DIR = REPO / "examples/data/analysis/cosmobench_gtsam_gnc_full_noprior"

METHODS = [
    ("Ours",       "impl"),
    ("Orig.",      "explicit"),
    ("Orig + V.P.", "expvp"),
]

# Datasets excluded from the LaTeX table — the tuhh_r3_* runs land in a
# pathological local minimum on both VarPro and zero-prior GTSAM (ATE ~105m)
# and skew the box plot tails without saying anything new about the solvers.
EXCLUDE = {
    "tuhh_r3_00_day_wifi",
    "tuhh_r3_01_night_wifi",
    "tuhh_r3_00_day_proradio",
    "tuhh_r3_01_night_proradio",
}


def collect() -> Dict[str, List[float]]:
    ate: Dict[str, List[float]] = {m: [] for m, _ in METHODS}
    ate["GTSAM"] = []
    datasets = sorted(p.stem for p in GTSAM_DIR.glob("*.json")
                       if p.stem not in EXCLUDE)
    for ds in datasets:
        irls_path = IRLS_DIR / f"{ds}.json"
        gtsam_path = GTSAM_DIR / f"{ds}.json"
        if not irls_path.exists() or not gtsam_path.exists():
            continue
        irls = json.loads(irls_path.read_text())
        gtsam = json.loads(gtsam_path.read_text())
        methods = irls.get("methods", {})
        for label, key in METHODS:
            m = methods.get(key, {})
            v = m.get("ate_global_rmse_m") if m.get("status") == "ok" else None
            if v is not None:
                ate[label].append(float(v))
        gres = gtsam.get("result", {})
        v = gres.get("ate_global_rmse_m") if gres.get("status") == "ok" else None
        if v is not None:
            ate["GTSAM"].append(float(v))
    return ate


def plot_box(ate: Dict[str, List[float]], out_path: Path, log: bool) -> None:
    labels = ["Ours", "Orig.", "Orig + V.P.", "GTSAM"]
    data = [ate[l] for l in labels]
    # Tableau-ish discrete palette, one solid color per box.
    palette = ["#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    bp = ax.boxplot(
        data,
        labels=labels,
        widths=0.55,
        patch_artist=True,
        showmeans=False,
        # Hide the median line entirely (reference plot has no median bar).
        medianprops=dict(linewidth=0),
        whiskerprops=dict(color="black", linewidth=0.9),
        capprops=dict(color="black", linewidth=0.9),
        boxprops=dict(linewidth=0.9),
        flierprops=dict(marker="+", markersize=5, markeredgewidth=0.9,
                         linestyle="none"),
    )
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_alpha(0.85)
    # Color the flier markers to match their box.
    for flier, color in zip(bp["fliers"], palette):
        flier.set_markeredgecolor(color)

    ax.set_ylabel("ATE (m)")
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.5, 500)
    ax.grid(axis="y", which="both", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path.relative_to(REPO)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                     default=REPO / "examples" / "data" / "analysis")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ate = collect()
    for label, vals in ate.items():
        print(f"  {label:>6}: n={len(vals):2d}  "
              f"min={min(vals):.2f}  median={np.median(vals):.2f}  "
              f"p75={np.percentile(vals, 75):.2f}  max={max(vals):.2f}")

    plot_box(ate, args.out_dir / "cosmobench_ate_box.png", log=False)
    plot_box(ate, args.out_dir / "cosmobench_ate_box_log.png", log=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
