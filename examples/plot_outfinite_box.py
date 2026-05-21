"""Box-and-whisker of final objective values across initializations on the
outfinite (RA-SLAM) dataset, one box per formulation.

Reads:  examples/data/raslam/outfinite/cached_results/results_rank5_*.json
Writes: examples/data/raslam/outfinite/outfinite_final_cost_box.png
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("examples/data/raslam/outfinite/cached_results")
OUT_PNG  = Path("examples/data/raslam/outfinite/outfinite_final_cost_box.png")

# formulation int code -> (display label, color)
FORM_META = {
    0: ("Explicit",        "tab:red"),
    1: ("Explicit + VarPro", "tab:orange"),
    2: ("Implicit (Ours)", "tab:blue"),
}
# Display order (left -> right): Explicit, ExpVP, Implicit
ORDER = [0, 1, 2]


def main() -> int:
    finals = defaultdict(list)
    for p in sorted(DATA_DIR.glob("results_rank5_*.json")):
        d = json.loads(p.read_text())[0]
        finals[d["formulation"]].append(d["costs"][-1])

    for code in ORDER:
        runs = finals[code]
        label, _ = FORM_META[code]
        print(f"{label:22s} n={len(runs)}  median={np.median(runs):.4g}  "
              f"min={min(runs):.4g}  max={max(runs):.4g}")

    data_raw = [finals[code] for code in ORDER]
    labels = [FORM_META[code][0] for code in ORDER]
    colors = [FORM_META[code][1] for code in ORDER]

    # Normalize each value by the best (= median Implicit) cost so the y-axis
    # measures "x times worse than the best solution found." With log scaling
    # this puts all three methods on a compact 10^0 .. 10^2 range, with major
    # decades labeled 10^k (mathtext) and the inter-decade minor gridlines
    # visible — matches the style of the reference figure.
    impl_med = float(np.median(finals[2]))
    data = [np.asarray(d) / impl_med for d in data_raw]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bp = ax.boxplot(
        data, labels=labels, patch_artist=True, widths=0.55,
        showmeans=False, showfliers=False,
        medianprops=dict(color="black", linewidth=1.6),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)

    ax.set_yscale("log")
    # "Even" y-axis: center on the geometric mean of the min/max of the data
    # and extend symmetrically in log-space so the populated region (Implicit
    # at ~10^0, Explicit at ~10^1.7) sits in the middle of the plot rather
    # than crammed at the extremes.
    all_vals = np.concatenate(data)
    lo, hi = float(all_vals.min()), float(all_vals.max())
    log_mid = 0.5 * (np.log10(lo) + np.log10(hi))
    log_half = 0.5 * (np.log10(hi) - np.log10(lo)) + 0.35   # pad ~0.35 decades
    ax.set_ylim(10 ** (log_mid - log_half), 10 ** (log_mid + log_half))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0,
                                            subs=np.arange(2, 10) * 0.1,
                                            numticks=20))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="major", length=5)
    ax.tick_params(axis="y", which="minor", length=3)
    ax.grid(True, axis="y", which="major", color="0.55", linewidth=0.9)
    ax.grid(True, axis="y", which="minor", color="0.75", linewidth=0.5)
    ax.set_axisbelow(True)

    ax.set_ylabel("Final objective  /  best Implicit cost")
    ax.set_title("outfinite (RA-SLAM, rank 5) — final objective across 5 inits",
                 fontsize=12)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
