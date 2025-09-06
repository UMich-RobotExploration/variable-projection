#!/usr/bin/env python3
import argparse, math, re
from pathlib import Path

FILENAME_RE = re.compile(r"^rank([3-5])_init([1-9]|10)\.txt$")

def parse_args():
    ap = argparse.ArgumentParser(description="List init files with bad VERTEX_POINT/POSE values.")
    ap.add_argument("--base", type=Path, default=Path("."),
                    help="Base dir (e.g., .../StiefelManifold/data/sfm).")
    ap.add_argument("--abs-threshold", type=float, default=1e6,
                    help="Flag |value| >= this threshold (default: 1e6).")
    return ap.parse_args()

def file_is_target(path: Path) -> bool:
    return bool(FILENAME_RE.match(path.name))

def line_has_issue(line: str, abs_threshold: float) -> bool:
    s = line.lstrip()
    if not (s.startswith("VERTEX_POINT") or s.startswith("VERTEX_POSE")):
        return False
    toks = line.strip().split()
    if len(toks) < 3:
        return True  # malformed
    for t in toks[2:]:
        try:
            v = float(t)
        except ValueError:
            return True  # non-numeric (e.g., '-nan' that failed, etc.)
        if not math.isfinite(v) or abs(v) >= abs_threshold:
            return True
    return False

def main():
    args = parse_args()
    base = args.base.resolve()
    for ds_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for init_name in ("init", "inits"):
            init_dir = ds_dir / init_name
            if not init_dir.is_dir():
                continue
            for f in sorted(p for p in init_dir.iterdir() if p.is_file() and file_is_target(p)):
                try:
                    with f.open("r", encoding="utf-8", errors="replace") as fh:
                        if any(line_has_issue(line, args.abs_threshold) for line in fh):
                            rel = f.relative_to(base)
                            print(f"/{rel.as_posix()}")
                        else :
                            pass
                            print(f"OK: {f}")
                except Exception:
                    rel = f.relative_to(base)
                    print(f"/{rel.as_posix()}")

if __name__ == "__main__":
    main()


