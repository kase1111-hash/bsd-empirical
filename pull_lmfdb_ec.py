#!/usr/bin/env python3
"""pull_lmfdb_ec.py — bulk pull of elliptic curves over Q with BSD invariants.

Source of truth: John Cremona's ecdata repo (the same flat files LMFDB ingests).
Files are ASCII, binned by conductor range (allcurves.100000-109999, ...),
which makes the parse embarrassingly parallel.

Workflow:
    1. git clone (or pull) JohnCremona/ecdata into --data-dir.
    2. Parse allcurves + allbsd + degphi per range, join on (N, iso, num).
    3. Concat -> Parquet (full corpus) + CSV (with ainvs stringified) + JSON meta.

Usage:
    python pull_lmfdb_ec.py --validate                # show first rows + parse check
    python pull_lmfdb_ec.py --quick                   # 1-2 ranges, ~30 sec, smoke test
    python pull_lmfdb_ec.py --conductor-max 100000    # small corpus, fast
    python pull_lmfdb_ec.py                           # full corpus, ~5-10 min parse
    python pull_lmfdb_ec.py --no-git                  # skip git, assume data present
    python pull_lmfdb_ec.py --workers 4               # explicit worker count

Schema notes (verified by --validate against ecdata master 2026-05):
    allcurves: <N> <iso> <num> <ainvs> <rank> <torsion_order>
    allbsd:    <N> <iso> <num> <ainvs> <rank> <torsion_order>
               <tamagawa_prod> <Omega> <L*> <Reg> <Sha_an>
               where L* = L^(r)(E,1) / r!  (Cremona convention)
    degphi:    <N> <iso> <num> <deg_phi> <{extra}> <ainvs>

Run with --validate FIRST on any new ecdata clone to confirm column positions.

Author: Kase Branham - Independent Researcher
"""

import os
import re
import json
import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    parallel_map, summary_table, interpret, Timer, C, debug,
)


# =====================================================================
#  CONFIGURATION
# =====================================================================

ECDATA_REPO = "https://github.com/JohnCremona/ecdata.git"
DEFAULT_DATA_DIR = Path("./ecdata")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

# Subdirs to parse. Each contains files named "<subdir>.NNNNNN-NNNNNN".
# Order matters for the join: allcurves is the spine.
SUBDIRS = ("allcurves", "allbsd", "degphi")

RANGE_RE = re.compile(r"^[a-z]+\.(\d+)-(\d+)$")


# =====================================================================
#  PARSERS
# =====================================================================

def _ainvs(s):
    """Parse '[0,1,1,-2,0]' -> [0,1,1,-2,0]."""
    return [int(x) for x in s.strip("[]").split(",") if x]


def parse_allcurves(line):
    """allcurves.RANGE: N iso num ainvs rank torsion_order"""
    p = line.split()
    return {
        "conductor":     int(p[0]),
        "iso":           p[1],
        "curve_number":  int(p[2]),
        "ainvs":         _ainvs(p[3]),
        "rank":          int(p[4]),
        "torsion_order": int(p[5]),
    }


def parse_allbsd(line):
    """allbsd.RANGE: N iso num ainvs rank torsion_order tamagawa Omega L* Reg Sha_an

    11 fields total. p[3]=ainvs, p[4]=rank, p[5]=torsion_order all duplicate
    allcurves and are dropped at merge time.
    """
    p = line.split()
    return {
        "conductor":         int(p[0]),
        "iso":               p[1],
        "curve_number":      int(p[2]),
        "tamagawa_product":  int(p[6]),
        "real_period":       float(p[7]),
        "L_value":           float(p[8]),   # L^(r)(E,1) / r!
        "regulator":         float(p[9]),
        "sha_an":            float(p[10]),
    }


def parse_degphi(line):
    """degphi.RANGE: N iso num deg_phi {extra} ainvs

    deg_phi is at position 3, NOT at the end. The {extra} field is typically '{}'
    (an empty dict placeholder) and ainvs comes last.
    """
    p = line.split()
    return {
        "conductor":    int(p[0]),
        "iso":          p[1],
        "curve_number": int(p[2]),
        "deg_phi":      int(p[3]),
    }


PARSERS = {
    "allcurves": parse_allcurves,
    "allbsd":    parse_allbsd,
    "degphi":    parse_degphi,
}


def parse_file(path, parser):
    """Parse one ecdata file. Returns (rows, n_skipped)."""
    rows, skipped = [], 0
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(parser(line))
            except (ValueError, IndexError):
                skipped += 1
    return rows, skipped


# =====================================================================
#  WORKER
# =====================================================================

def worker(item):
    """Process one conductor range with a plain-dict join. No pandas in workers
    (Windows multiprocessing + pandas object-dtype merges produce spurious
    allocation failures even for tiny arrays). DataFrame is built once in reduce."""
    try:
        debug("worker: range=%s", item["range"])
        data_dir = Path(item["data_dir"])
        rng = item["range"]

        # Parse each subdir to a dict keyed by (conductor, iso, curve_number).
        indexed = {}
        skipped_total = 0
        for sub in SUBDIRS:
            path = data_dir / sub / f"{sub}.{rng}"
            if not path.exists():
                continue
            rows, skipped = parse_file(path, PARSERS[sub])
            skipped_total += skipped
            indexed[sub] = {
                (r["conductor"], r["iso"], r["curve_number"]): r
                for r in rows
            }

        if "allcurves" not in indexed:
            return {"range": rng, "n": 0, "skipped": skipped_total, "rows": []}

        # Spine = allcurves; merge in allbsd + degphi fields by key.
        BSD_FIELDS = ("tamagawa_product", "real_period",
                      "L_value", "regulator", "sha_an")
        bsd_index = indexed.get("allbsd", {})
        deg_index = indexed.get("degphi", {})

        merged = []
        for key, base in indexed["allcurves"].items():
            row = dict(base)
            if key in bsd_index:
                bsd = bsd_index[key]
                for f in BSD_FIELDS:
                    row[f] = bsd.get(f)
            if key in deg_index:
                row["deg_phi"] = deg_index[key].get("deg_phi")
            row["label"] = f"{row['conductor']}{row['iso']}{row['curve_number']}"
            merged.append(row)

        return {
            "range":   rng,
            "n":       len(merged),
            "skipped": skipped_total,
            "rows":    merged,
        }
    except Exception as e:
        return {"error": str(e), "range": item.get("range", "?")}


# =====================================================================
#  GIT / FS
# =====================================================================

def ensure_ecdata(data_dir: Path, no_git: bool):
    """Clone or pull JohnCremona/ecdata. Skip if --no-git."""
    if no_git:
        if not data_dir.exists():
            raise FileNotFoundError(f"--no-git but {data_dir} missing")
        print(f"  Using existing data at {data_dir} (--no-git)")
        return

    if not data_dir.exists():
        section(f"Cloning ecdata -> {data_dir}")
        debug("git clone --depth=1 %s %s", ECDATA_REPO, data_dir)
        # --depth=1 is ~2 GB instead of full history; plenty for parsing.
        subprocess.run(
            ["git", "clone", "--depth=1", ECDATA_REPO, str(data_dir)],
            check=True,
        )
    else:
        section(f"Pulling latest ecdata in {data_dir}")
        debug("git -C %s pull --ff-only", data_dir)
        subprocess.run(
            ["git", "-C", str(data_dir), "pull", "--ff-only"],
            check=False,  # tolerate "already up to date" / shallow-clone quirks
        )


def discover_ranges(data_dir: Path):
    """Scan allcurves/ for range files. Returns sorted [(lo, hi, 'lo-hi'), ...]."""
    ac = data_dir / "allcurves"
    if not ac.is_dir():
        raise FileNotFoundError(f"{ac} not found - bad data_dir?")
    ranges = []
    for p in ac.iterdir():
        m = RANGE_RE.match(p.name)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            ranges.append((lo, hi, p.name.split(".", 1)[1]))
    ranges.sort()
    return ranges


def validate(data_dir: Path):
    """Show first 3 lines of each file type for the smallest range, try parsing."""
    section("VALIDATE - schema sanity check")
    ranges = discover_ranges(data_dir)
    if not ranges:
        print(C.fail("  no range files found"))
        return
    _, _, rng = ranges[0]
    print(f"  Smallest range: {rng}\n")
    for sub in SUBDIRS:
        path = data_dir / sub / f"{sub}.{rng}"
        if not path.exists():
            print(C.warn(f"  {sub}.{rng}: (missing)"))
            continue
        with open(path) as f:
            lines = [next(f, "").rstrip() for _ in range(3)]
        print(C.info(f"  {sub}.{rng}:"))
        for ln in lines:
            print(f"    {ln}")
        if lines and lines[0]:
            try:
                parsed = PARSERS[sub](lines[0])
                print(f"    {C.ok('parsed OK:')} {parsed}")
            except Exception as e:
                print(f"    {C.fail('parse FAILED:')} {e}")
        print()


# =====================================================================
#  WORK LIST
# =====================================================================

def build_work_list(args, data_dir):
    ranges = discover_ranges(data_dir)
    print(f"  Found {len(ranges)} conductor-range files")

    items = []
    for lo, _hi, rng in ranges:
        if args.conductor_max is not None and lo > args.conductor_max:
            continue
        items.append({"range": rng, "data_dir": str(data_dir)})

    if args.preset == "quick":
        items = items[:2]
        print(f"  --quick: keeping first {len(items)} range(s)")
    elif args.n_samples and args.n_samples < len(items):
        items = items[:args.n_samples]
        print(f"  --n-samples: keeping first {len(items)} range(s)")

    return items


# =====================================================================
#  REDUCE
# =====================================================================

def analyze(results, output_dir: Path):
    section("REDUCE")
    errors = [r for r in results if "error" in r]
    good   = [r for r in results if "error" not in r]

    if errors:
        print(C.fail(f"  {len(errors)} ranges failed:"))
        for e in errors[:5]:
            print(f"    {e}")

    skipped_total = sum(r.get("skipped", 0) for r in good)
    if skipped_total:
        print(C.warn(f"  {skipped_total} malformed lines skipped across all files"))

    # Concat without exploding memory: build per-range DFs, then concat once.
    dfs = []
    for r in good:
        if r["rows"]:
            dfs.append(pd.DataFrame(r["rows"]))
    if not dfs:
        print(C.fail("  no rows parsed - aborting"))
        return
    df = pd.concat(dfs, ignore_index=True)
    n = len(df)
    print(f"  {C.ok(f'{n:,}')} curves parsed across {len(good)} ranges")

    # ---- Summary ---------------------------------------------------
    section("CORPUS SUMMARY")
    rank_counts = df["rank"].value_counts().sort_index()
    rows = [(int(rk), int(c), f"{100*c/n:.2f}%")
            for rk, c in rank_counts.items()]
    summary_table(
        rows, ["rank", "count", "pct"],
        title="Rank distribution",
        fmt=[">4d", ">12,d", ">8s"],
    )

    cond_min = int(df["conductor"].min())
    cond_max = int(df["conductor"].max())
    print(f"\n  Conductor range: {cond_min:,} - {cond_max:,}")
    print(f"  Mean rank:       {df['rank'].mean():.4f}")

    if "sha_an" in df.columns:
        sha = df["sha_an"].dropna()
        if len(sha):
            print(f"  Sha_an min/median/max: "
                  f"{sha.min():.3f} / {sha.median():.3f} / {sha.max():.3f}")
            sha_nontrivial = (sha > 1.5).sum()  # round(sha) >= 2
            print(f"  Curves with Sha_an >= 2: {sha_nontrivial:,} "
                  f"({100*sha_nontrivial/len(sha):.2f}%)")

    # ---- Save ------------------------------------------------------
    section("OUTPUT FILES")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ainvs is a list of arbitrary-precision Python ints. At high conductor
    # the minimal-model coefficients can exceed int64, which crashes pyarrow.
    # Store as a string both in Parquet and CSV; downstream readers can
    # ast.literal_eval to restore the list.
    df["ainvs"] = df["ainvs"].apply(
        lambda a: "[" + ",".join(map(str, a)) + "]"
    )

    parquet_path = output_dir / "ec_corpus.parquet"
    debug("writing parquet: %s", parquet_path)
    df.to_parquet(parquet_path, index=False)
    sz_mb = parquet_path.stat().st_size / 1e6
    print(f"  {C.info('parquet:')} {parquet_path}  ({sz_mb:.1f} MB)")

    csv_path = output_dir / "ec_corpus.csv"
    debug("writing csv: %s", csv_path)
    df.to_csv(csv_path, index=False)
    print(f"  {C.info('csv:    ')} {csv_path}")

    meta = {
        "n_curves":          n,
        "n_ranges":          len(good),
        "n_errors":          len(errors),
        "n_skipped_lines":   skipped_total,
        "conductor_min":     cond_min,
        "conductor_max":     cond_max,
        "mean_rank":         float(df["rank"].mean()),
        "rank_distribution": {int(k): int(v) for k, v in rank_counts.items()},
        "columns":           list(df.columns),
    }
    meta_path = output_dir / "ec_corpus_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  {C.info('meta:   ')} {meta_path}")

    # ---- Interpret -------------------------------------------------
    interpret(
        "Mean rank vs empirical expectation",
        float(df["rank"].mean()),
        [
            (0.70, "low - corpus skewed to small conductor"),
            (0.90, "typical LMFDB corpus range"),
            (1.20, "high - possible selection bias"),
        ],
        "very high - investigate sampling",
    )


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_standard_args(parser)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="local ecdata clone path (default ./ecdata)")
    parser.add_argument("--no-git", action="store_true",
                        help="skip git clone/pull; assume data already present")
    parser.add_argument("--validate", action="store_true",
                        help="print first rows of each file type and exit")
    parser.add_argument("--conductor-max", type=int, default=None,
                        help="only process ranges whose lower bound <= this value")
    args = parser.parse_args()
    resolve_args(args)

    # kase_utils default output is "results"; redirect to a topic-specific dir
    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)

    banner("Pull LMFDB Elliptic Curves (Cremona ecdata)", {
        "Data dir":      str(args.data_dir),
        "Output dir":    str(output_dir),
        "Workers":       args.workers,
        "No git":        args.no_git,
        "Validate only": args.validate,
        "Conductor max": args.conductor_max,
        "Preset":        args.preset,
    })

    # ---- 1. Fetch --------------------------------------------------
    step(1, 4, "Fetching ecdata...")
    ensure_ecdata(args.data_dir, args.no_git)

    if args.validate:
        validate(args.data_dir)
        return

    # ---- 2. Work list ----------------------------------------------
    step(2, 4, "Building work list...")
    work_list = build_work_list(args, args.data_dir)
    if not work_list:
        print(C.fail("  no ranges to process - check --conductor-max"))
        return

    # ---- 3. Parse --------------------------------------------------
    step(3, 4, f"Parsing {len(work_list)} ranges...")
    with Timer("Parse"):
        results = parallel_map(worker, work_list, args.workers, desc="Ranges")

    # ---- 4. Reduce -------------------------------------------------
    step(4, 4, "Reducing and saving...")
    analyze(results, output_dir)


if __name__ == "__main__":
    mp.freeze_support()
    main()
