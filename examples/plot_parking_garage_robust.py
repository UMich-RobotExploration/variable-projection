#!/usr/bin/env python3
"""Box-whisker of robust objective + trajectory overlays for parking-garage_o{10,20,30}.

Reads:
  - examples/data/analysis/pgo_outlier_irls_randinit/{explicit,expvp,impl}_gm.json
    (robust_cost per (dataset, seed) cell)
  - /tmp/pgo_irls_tums/parking-garage_o{rate}_s{seed}_{method}_gm.tum
    (final SE(3) trajectories from the 5-seed random-init sweep)

Writes two PNGs into examples/data/analysis/pgo_outlier_irls_randinit/.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO         = Path(__file__).resolve().parent.parent
JSON_DIR     = REPO / "examples" / "data" / "analysis" / "pgo_outlier_irls_randinit"
TUM_DIR      = Path("/tmp/pgo_irls_tums")
OUT_DIR      = JSON_DIR

METHODS      = ["explicit", "expvp", "impl"]
METHOD_LABEL = {"explicit": "Explicit", "expvp": "Orig.+VP", "impl": "Implicit (Ours)"}
METHOD_COLOR = {"explicit": "tab:red", "expvp": "tab:orange", "impl": "tab:blue"}
RATES        = [10, 20, 30]
SEEDS        = [1, 2, 3, 4, 5]


def load_tum(path: Path) -> np.ndarray:
    rows = [list(map(float, ln.split())) for ln in path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    return np.asarray(rows)


def main() -> int:
    # ------------------------------------------------------------------
    # Load robust_cost per (rate, method, seed)
    # ------------------------------------------------------------------
    jsons = {m: json.loads((JSON_DIR / f"{m}_gm.json").read_text())
             for m in METHODS}
    # costs[rate][method] = list of 5 robust costs (one per seed)
    costs = {r: {m: [jsons[m][f"parking-garage_o{r}"][str(s)]["robust_cost"]
                       for s in SEEDS]
                  for m in METHODS}
              for r in RATES}

    # ------------------------------------------------------------------
    # Figure 1: 1x3 box-whisker, one panel per outlier rate
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, rate in zip(axes, RATES):
        box_data = [costs[rate][m] for m in METHODS]
        bp = ax.boxplot(box_data, labels=[METHOD_LABEL[m] for m in METHODS],
                        patch_artist=True, widths=0.55, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="black",
                                       markeredgecolor="black", markersize=6))
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(METHOD_COLOR[m])
            patch.set_alpha(0.55)
        # overlay individual seed values (jittered) so spread + agreement
        # are visible even when whiskers are tight.
        rng = np.random.default_rng(0)
        for i, m in enumerate(METHODS, start=1):
            xs = i + rng.uniform(-0.08, 0.08, size=len(costs[rate][m]))
            ax.scatter(xs, costs[rate][m], color="black", s=18, zorder=3,
                       edgecolor="white", linewidth=0.5)
        ax.set_title(f"parking-garage, {rate}% outliers", fontsize=11)
        ax.set_ylabel("robust objective (GM)") if rate == RATES[0] else None
        ax.grid(True, axis="y", alpha=0.3)
        # Tight y-axis around the data, with a little padding
        all_vals = [v for m in METHODS for v in costs[rate][m]]
        lo, hi = min(all_vals), max(all_vals)
        pad = (hi - lo) * 0.15 if hi > lo else max(hi * 0.001, 1.0)
        ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle("Robust GM objective across 5 random-init seeds — per outlier rate",
                  fontsize=12)
    fig.tight_layout()
    out1 = OUT_DIR / "parking_garage_robust_box.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"wrote {out1}")

    # ------------------------------------------------------------------
    # Figure 2: 1x3 3D trajectory overlays per outlier rate.
    # For each (rate, method) pick the seed whose robust_cost is closest to
    # the median (= "typical" run). Bold that seed; draw the other 4 seeds
    # with low alpha so the spread of solutions is visible.
    # ------------------------------------------------------------------
    # For trajectory clarity, plot ONLY the median-cost seed per method
    # (parking-garage is naturally a busy 3D curve; overlaying all 5 seeds
    # per method makes the figure unreadable).
    fig2 = plt.figure(figsize=(18, 7.0))
    for j, rate in enumerate(RATES, start=1):
        ax = fig2.add_subplot(1, 3, j, projection="3d")
        all_xyz = []
        for m in METHODS:
            cs = costs[rate][m]
            med = statistics.median(cs)
            best_seed = SEEDS[int(np.argmin([abs(c - med) for c in cs]))]
            tum = TUM_DIR / f"parking-garage_o{rate}_s{best_seed}_{m}_gm.tum"
            if not tum.exists():
                continue
            data = load_tum(tum)
            xyz = data[:, 1:4]
            all_xyz.append(xyz)
            color = METHOD_COLOR[m]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "-",
                    color=color, linewidth=1.3,
                    label=f"{METHOD_LABEL[m]}  (seed {best_seed},  "
                          f"cost {cs[best_seed-1]:.0f})",
                    zorder=3)
            # mark start and end
            ax.scatter(*xyz[0],  color=color, marker="o", s=35,
                       edgecolor="black", linewidth=0.5, zorder=4)
            ax.scatter(*xyz[-1], color=color, marker="s", s=35,
                       edgecolor="black", linewidth=0.5, zorder=4)
        if all_xyz:
            all_xyz = np.concatenate(all_xyz, axis=0)
            mid = (all_xyz.max(0) + all_xyz.min(0)) / 2.0
            half = max((all_xyz.max(0) - all_xyz.min(0)).max() / 2.0, 1e-6) * 1.05
            ax.set_xlim(mid[0] - half, mid[0] + half)
            ax.set_ylim(mid[1] - half, mid[1] + half)
            ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"parking-garage, {rate}% outliers", fontsize=11)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
    fig2.suptitle("parking-garage — recovered trajectories (median-cost seed per method)",
                   fontsize=13, y=0.98)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    out2 = OUT_DIR / "parking_garage_robust_trajectories.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"wrote {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
