#!/usr/bin/env python3
# Merge dynamic data from stray data dirs into canonical <repo>/data.
# Usage:
#   python scripts/merge_data_once.py                # default (merges backend/data if exists)
#   python scripts/merge_data_once.py --dry-run      # preview only
#   python scripts/merge_data_once.py --strays path1 path2 ...

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

# ---- Repo layout ----
THIS = Path(__file__).resolve()
REPO = THIS.parents[1]                  # .../your-repo
CANON = REPO / "data"                   # canonical data dir
DEFAULT_STRAYS = [REPO / "backend" / "data"]  # common stray

DYNAMIC = {"gold.jsonl", "silver.jsonl", "unlabeled.jsonl"}
STATIC  = {"eval_gold.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "challenge.jsonl"}
ALL     = list(DYNAMIC | STATIC)

def read_jsonl(p: Path):
    rows = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return rows

def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)

def ts_key(r):
    t = r.get("ts")
    if not t:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        # handle ...Z
        if t.endswith("Z"):
            return datetime.fromisoformat(t.replace("Z", "+00:00"))
        return datetime.fromisoformat(t)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)

def prio(r):
    src = (r.get("source") or "").lower()
    # user_feedback (GOLD) > silver > inference (UNLABELED) > else
    return {"user_feedback": 3, "silver": 2, "inference": 1}.get(src, 0)

def fp_key(filename: str, r: dict) -> str:
    # For dynamic files use fingerprint; for static, exact content key
    if filename in DYNAMIC:
        return str(r.get("fp") or "")
    return json.dumps(r, sort_keys=True, ensure_ascii=False)

def merge_file(name: str, canon_dir: Path, stray_dirs: list[Path], dry_run: bool = False, verbose: bool = True):
    buckets = {}

    # Canonical first
    canon_rows = read_jsonl(canon_dir / name)
    for r in canon_rows:
        buckets.setdefault(fp_key(name, r), []).append(r)

    # Then each stray
    for d in stray_dirs:
        rows = read_jsonl(d / name)
        for r in rows:
            buckets.setdefault(fp_key(name, r), []).append(r)

    # Choose the best row per key
    merged = []
    for _, rs in buckets.items():
        if not rs:
            continue
        # Prefer higher priority, then newer timestamp
        rs.sort(key=lambda x: (prio(x), ts_key(x)), reverse=True)
        merged.append(rs[0])

    before = len(canon_rows)
    after = len(merged)

    if verbose:
        print(f"{name}: canon={before} -> merged={after} (from {len(stray_dirs)} stray dir(s))")

    if dry_run:
        return

    if merged:
        write_jsonl(canon_dir / name, merged)

def main():
    parser = argparse.ArgumentParser(description="Merge stray data/ folders into canonical <repo>/data.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write; only print changes.")
    parser.add_argument("--strays", nargs="*", type=str, default=None,
                        help="Additional stray data dirs to merge (default scans backend/data).")
    args = parser.parse_args()

    stray_dirs = []
    if args.strays:
        stray_dirs = [Path(p).resolve() for p in args.strays if Path(p).exists()]
    else:
        stray_dirs = [d for d in DEFAULT_STRAYS if d.exists()]

    if not stray_dirs:
        print("No stray data directories found. Nothing to merge.")
        return

    print(f"Canonical data dir: {CANON}")
    print("Stray data dirs:", ", ".join(str(d) for d in stray_dirs))
    CANON.mkdir(parents=True, exist_ok=True)

    for name in ALL:
        merge_file(name, CANON, stray_dirs, dry_run=args.dry_run)

    print("Done.")

if __name__ == "__main__":
    main()
