#!/usr/bin/env python3
"""Run the IRLS robust-cost solver in every combination of robust kernel
(Geman-McClure, Truncated Least Squares) and formulation (Explicit,
Explicit+VarPro, Implicit) on a single PGO dataset, then print a comparison
table.

Usage:
    python3 irls_compare.py <pyfg> [--c2 X] [--max-irls K] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "build" / "bin" / "irls_robust"

KERNELS = ["gm", "tls"]
FORMULATIONS = ["explicit", "expvp", "impl"]

KERNEL_LABEL = {"gm": "Geman-McClure", "tls": "Truncated LS"}
FORM_LABEL = {"explicit": "Original", "expvp": "Original+VP", "impl": "Ours"}

RESULT_RE = re.compile(r"^IRLS_RESULT\s+(.+)$", re.MULTILINE)


def parse_result(stdout: str) -> dict[str, str] | None:
    m = RESULT_RE.search(stdout)
    if not m:
        return None
    out: dict[str, str] = {}
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pyfg", type=Path, help="path to .pyfg dataset")
    ap.add_argument("--c2", type=float, default=25.0,
                     help="robust kernel scale parameter (squared); default 25 "
                          "(≈4-sigma for 6-DOF chi-squared residuals)")
    ap.add_argument("--max-irls", type=int, default=20)
    ap.add_argument("--rel-tol", type=float, default=1e-4)
    ap.add_argument("--gnc", action="store_true",
                     help="anneal the kernel scale from gnc_init * c2 down to c2")
    ap.add_argument("--gnc-init", type=float, default=64.0,
                     help="initial multiplier on c2 when --gnc is set (default 64)")
    ap.add_argument("--gnc-shrink", type=float, default=1.4,
                     help="per-outer-iter shrink factor for the GNC schedule")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="directory for per-combo TUM outputs (default: /tmp)")
    args = ap.parse_args()

    if not BIN.exists():
        print(f"missing binary: {BIN} — build with `cmake --build build --target irls_robust`")
        return 1
    if not args.pyfg.exists():
        print(f"missing dataset: {args.pyfg}")
        return 1

    out_dir = args.out_dir or Path("/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.pyfg.stem

    init_tum = out_dir / f"{stem}_irls_init.tum"
    results: list[tuple[str, str, dict[str, str] | None, str]] = []
    for kernel in KERNELS:
        for form in FORMULATIONS:
            final_tum = out_dir / f"{stem}_irls_{kernel}_{form}.tum"
            cmd = [str(BIN), str(args.pyfg), str(init_tum), str(final_tum),
                    "--kernel", kernel,
                    "--formulation", form,
                    "--c2", str(args.c2),
                    "--max-irls", str(args.max_irls),
                    "--rel-tol", str(args.rel_tol)]
            if args.gnc:
                cmd += ["--gnc",
                         "--gnc-init", str(args.gnc_init),
                         "--gnc-shrink", str(args.gnc_shrink)]
            print(f">>> {kernel:3s} / {form:8s}", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"    failed (exit {proc.returncode})")
                print("    stderr:", proc.stderr.strip()[:300])
                results.append((kernel, form, None, proc.stderr))
                continue
            r = parse_result(proc.stdout)
            results.append((kernel, form, r, proc.stdout))

    print("\nDataset:", args.pyfg)
    header = f"{'Kernel':<14}{'Formulation':<14}{'Outer':>6}{'Inner':>7}" \
              f"{'Inner cost':>14}{'Robust cost':>14}{'Wall (s)':>11}"
    print(header)
    print("-" * len(header))
    for kernel, form, r, _stdout in results:
        klabel = KERNEL_LABEL[kernel]
        flabel = FORM_LABEL[form]
        if r is None:
            print(f"{klabel:<14}{flabel:<14}{'FAIL':>6}")
            continue
        print(f"{klabel:<14}{flabel:<14}"
               f"{r.get('outer', '-'):>6}"
               f"{r.get('inner', '-'):>7}"
               f"{float(r.get('final_cost', 'nan')):>14.4g}"
               f"{float(r.get('robust_cost', 'nan')):>14.4g}"
               f"{float(r.get('total_s', 'nan')):>11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
