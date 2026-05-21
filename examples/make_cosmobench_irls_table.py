"""Emit a LaTeX table for the cosmobench + nebula IRLS-GNC sweep.

Reads examples/data/analysis/cosmobench_irls_gnc_full/*.json and produces a
table with columns:

  Precompute (s)   | Solver time (s)        | Solver iters         | ATE (m)
  Ours / O / O+VP  | Ours / O / O+VP / GT   | Ours / O / O+VP / GT | Ours/O/O+VP
                                                                   | Improvement (runtime, iters)

The GTSAM cells are emitted as `\\mredx` placeholders (we didn't run GTSAM
for these datasets — preserves the format the existing tables use).

Output goes to stdout (paste into the paper) and a sidecar .tex file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

OUT_DIR = Path("examples/data/analysis/cosmobench_irls_gnc_full")
TEX_OUT = OUT_DIR / "cosmobench_irls_table.tex"

# Order: wifi block, proradio block, nebula block.
WIFI = [
    "kittredge_loop_wifi",
    "main_campus_wifi",
    "kth_r3_00_d06_d09_d10_wifi",
    "kth_r3_01_n01_n04_n05_wifi",
    "kth_r4_00_d06_d10_n04_n05_wifi",
    "ntu_r3_00_d01_n04_n08_wifi",
    "ntu_r3_01_n04_n08_n13_wifi",
    "ntu_r3_02_d02_n04_n13_wifi",
    "ntu_r4_00_d01_n04_n08_n13_wifi",
    "ntu_r5_00_d01_d02_n04_n08_n13_wifi",
    "tuhh_r3_00_day_wifi",
    "tuhh_r3_01_night_wifi",
]
PRORADIO = [n.replace("_wifi", "_proradio") for n in WIFI]
NEBULA = ["finals", "kentucky_underground", "tunnel", "urban"]


def short_label(name: str) -> str:
    """Compact human-readable label."""
    return (
        name.replace("_wifi", "")
        .replace("_proradio", "")
        .replace("_", "\\_")
    )


def fmt_pc(s: float) -> str:
    # Precompute is sub-millisecond on these datasets — format in ms.
    return f"{s * 1000:.2f}"


def fmt_solver(s: float) -> str:
    return f"{s:.2f}"


def fmt_int(n: int) -> str:
    return str(int(n))


def fmt_ate(m: float) -> str:
    return f"{m:.2f}"


def load_record(name: str) -> dict:
    return json.loads((OUT_DIR / f"{name}.json").read_text())


def row(rec: dict) -> str:
    label = short_label(rec["dataset"])
    impl = rec["methods"]["impl"]
    ex = rec["methods"]["explicit"]
    vp = rec["methods"]["expvp"]
    cells = [
        # precompute (ms) — Ours, Original, Orig+VP
        fmt_pc(impl["precompute_s"]),
        fmt_pc(ex["precompute_s"]),
        fmt_pc(vp["precompute_s"]),
        # solver time (s) — Ours, Original, Orig+VP, GTSAM
        fmt_solver(impl["wall_s"]),
        fmt_solver(ex["wall_s"]),
        fmt_solver(vp["wall_s"]),
        "\\mredx",
        # iterations — Ours, Original, Orig+VP, GTSAM
        fmt_int(impl["inner_iters_total"]),
        fmt_int(ex["inner_iters_total"]),
        fmt_int(vp["inner_iters_total"]),
        "\\mredx",
        # ATE (m) — Ours, Original, Orig+VP (global Umeyama)
        fmt_ate(impl["ate_global_rmse_m"]),
        fmt_ate(ex["ate_global_rmse_m"]),
        fmt_ate(vp["ate_global_rmse_m"]),
    ]
    return f"    \\tableRow{{{label}}}{{" + "}{".join(cells) + "}"


def block(title_macro: str, names: List[str], leading_amp: bool) -> str:
    rows = [row(load_record(n)) for n in names]
    out = []
    if leading_amp:
        out.append(f"            {title_macro}    & {rows[0].split(' ', 1)[1]}")
    else:
        out.append(f"            {title_macro}\n            & {rows[0].split(' ', 1)[1]}")
    for r in rows[1:]:
        out.append(f"                            & {r.split(' ', 1)[1]}")
    return "\n".join(out)


HEADER = r"""% --- new tableRow macro: precompute (3) + solver time (4) + iters (4) + ATE (3) + improvements ---
\renewcommand{\tableRow}[15]{
    #1
    % --- Precompute (ms): Ours / Original / Orig+VP ---
    & \boldifless{#2}{#3}{#4}
    & \boldifless{#3}{#2}{#4}
    & \boldifless{#4}{#2}{#3}
    % --- Solver time (s): Ours / Original / Orig+VP / GTSAM ---
    & \boldifless{#5}{#6}{#7}
    & \boldifless{#6}{#5}{#7}
    & \boldifless{#7}{#5}{#6}
    & #8
    % --- Solver iterations: Ours / Original / Orig+VP / GTSAM ---
    & #9 & #10 & #11 & #12
    % --- ATE (m): Ours / Original / Orig+VP ---
    & \boldifless{#13}{#14}{#15}
    & \boldifless{#14}{#13}{#15}
    & \boldifless{#15}{#13}{#14}
    % --- Improvement factors (Orig/Ours, Orig+VP/Ours) ---
    & \speedupCalc{#6}{#5}\,/\,\speedupCalc{#7}{#5}
    & \iterCalc{#10}{#9}\,/\,\iterCalc{#11}{#9}
    \\%
}

\def\wifiMultiRow{
    \multirow[c]{12}{*}{\rotatebox[origin=c]{90}{\textbf{Cosmobench Wi-Fi}}}
}
\def\proradioMultiRow{
    \multirow[c]{12}{*}{\rotatebox[origin=c]{90}{\textbf{Cosmobench Pro-Radio}}}
}
\def\nebulaMultiRow{
    \multirow[c]{4}{*}{\rotatebox[origin=c]{90}{\textbf{Nebula}}}
}

\begin{table*}[t]
    \vspace{0.5em}
    \centering
    \caption{
        \textbf{Robust IRLS-GNC (Geman--McClure) on COSMO-Bench and Nebula multi-robot
        PGO datasets.}
        All runs use the same odometry-chained initialization; pose priors are
        stripped because the dataset priors have variance $10^{-8}$/$10^{-6}$ and
        make the unweighted Hessian ill-conditioned. Precompute times are in
        milliseconds, solver times in seconds; ATE is the global RMSE (m) after a
        rigid Umeyama alignment to the ground-truth pose chain. GTSAM cells are
        marked (\mredx) where we have not yet collected a measurement.
        Runtime improvement factor is $\frac{\text{Baseline}}{\text{Ours}}$;
        iteration improvement is $\frac{\text{Baseline iters}}{\text{Ours iters}}$.
        Values $>1$ indicate Ours required less time/iterations than the baseline;
        values $>1$ are green, values $<1$ are red.
        Each row's fastest precompute, solver time, and lowest ATE are bolded.
    }
    \label{tab:cosmobench_irls_gnc}
    \resizebox{\linewidth}{!}{%
        \begin{tabular}{ll ccc cccc cccc ccc cc}
            \toprule
                & & \multicolumn{3}{c}{\textbf{Precompute (ms)} $\downarrow$}
                  & \multicolumn{4}{c}{\textbf{Solver time (s)} $\downarrow$}
                  & \multicolumn{4}{c}{\textbf{Solver iters} $\downarrow$}
                  & \multicolumn{3}{c}{\textbf{ATE (m)} $\downarrow$}
                  & \multicolumn{2}{c}{\textbf{Improvement} $\uparrow$} \\
            \cmidrule(lr){3-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13} \cmidrule(lr){14-16} \cmidrule(lr){17-18}
                & Dataset
                & Ours & Orig. & O+VP
                & Ours & Orig. & O+VP & GTSAM
                & Ours & Orig. & O+VP & GTSAM
                & Ours & Orig. & O+VP
                & \makecell[c]{Runtime \\{\scriptsize(Orig.\,/\,O+VP)}}
                & \makecell[c]{Iterations \\{\scriptsize(Orig.\,/\,O+VP)}} \\
            \midrule"""


FOOTER = r"""            \bottomrule
        \end{tabular}
    }
    \vspace{-1em}
\end{table*}
"""


def main() -> int:
    parts = [HEADER]
    parts.append("            % ================= Cosmobench Wi-Fi =================")
    parts.append(block(r"\wifiMultiRow", WIFI, leading_amp=True))
    parts.append(r"            \midrule")
    parts.append("            % ================= Cosmobench Pro-Radio =================")
    parts.append(block(r"\proradioMultiRow", PRORADIO, leading_amp=True))
    parts.append(r"            \midrule")
    parts.append("            % ================= Nebula =================")
    parts.append(block(r"\nebulaMultiRow", NEBULA, leading_amp=True))
    parts.append(FOOTER)
    table = "\n".join(parts)
    TEX_OUT.write_text(table)
    print(table)
    print(f"\n% wrote {TEX_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
