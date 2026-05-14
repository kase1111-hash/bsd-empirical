#!/usr/bin/env python3
"""add_aplist.py - parse Cremona aplist files and add a_p coefficients to corpus.

Reads the aplist/ directory of the Cremona ecdata repository, where each
line records the Dirichlet a_p coefficients of the L-function L(E, s) for
an elliptic curve E. The a_p values are class-level invariants (all curves
in an isogeny class share the same L-function); the script deduplicates by
(conductor, isogeny class) and stores the a_p sequence per curve in the
output parquet for downstream convenience.

For an elliptic curve E/Q with good reduction at p:
    a_p = p + 1 - #E(F_p)        (Hasse bound: |a_p| <= 2*sqrt(p))
For bad primes p (p | N):
    a_p in {-1, 0, 1}              (depending on reduction type)

The a_p sequence enables:
    - Sato-Tate empirics (a_p / 2*sqrt(p) distribution)
    - L-function reconstruction
    - Modular form coincidences and lifts
    - Direct probing of local arithmetic at each prime

Output columns (added to existing parquet):
    aps_list     list[int]    -- the full a_p sequence
    n_aps        int          -- length of the sequence
    a2 a3 a5 a7 a11 a13       -- individual a_p for first six primes

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet
Output: lmfdb_ec_out/ec_corpus_with_aps.parquet

Cremona's aplist file format varies; this script auto-detects per-class
(N iso aps...) vs per-curve (N iso num aps...) layout by inspecting the
first 100 non-comment lines.

Author: Kase Branham - Independent Researcher
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, Timer, C, debug, parallel_map,
)


DEFAULT_ECDATA     = Path("./ecdata")
DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_spectrum.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

# Primes exposed as individual columns for fast access
EXPOSED_PRIMES = [2, 3, 5, 7, 11, 13]

# Order of primes in Cremona's a_p sequence (must match the file convention)
PRIME_ORDER  = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIME_ORDER)}

# Cremona's aplist gives a_p for the first 25 primes (2 through 97).
# Records may contain additional Atkin-Lehner signature tokens after
# the a_p list, of the form '+(p)' or '-(p)' where p is a bad prime
# greater than 97. These must NOT be parsed as a_p values.
N_APS_PER_RECORD = 25
AL_SIGNATURE_PATTERN = re.compile(r'^[+-]\(\d+\)$')


# =====================================================================
#  TOKEN PARSING
# =====================================================================

def _parse_ap_token(tok):
    """Parse one a_p token from a Cremona aplist file.

    For bad primes, Cremona uses single-character shorthand:
        '+' means a_p = +1   (split multiplicative reduction)
        '-' means a_p = -1   (non-split multiplicative reduction)
        '0' means a_p = 0    (additive reduction; same as integer 0)

    For good primes, a_p is an integer (signed).
    """
    if tok == "+":
        return 1
    if tok == "-":
        return -1
    return int(tok)


# =====================================================================
#  FORMAT DETECTION
# =====================================================================

def detect_per_curve_format(filepath, sample_size=100):
    """Inspect first sample_size non-comment lines.

    Returns True if the third token is consistently a small positive
    integer (1-30), suggesting per-curve format with curve_number.
    Otherwise returns False (assumed per-class format).

    The heuristic relies on a_2 being bounded in absolute value by
    2*sqrt(2) ~ 2.83, so |a_2| <= 2 for good reduction at 2. If the third
    token is >= 3 in any inspected line, it cannot be a_2 and must be a
    curve number.
    """
    try:
        with open(filepath, "r") as f:
            sampled = 0
            saw_large_third = False
            for line in f:
                if sampled >= sample_size:
                    break
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                sampled += 1
                try:
                    # + and - are a_p shorthand for ±1, never curve numbers
                    if parts[2] in ("+", "-"):
                        continue
                    third = int(parts[2])
                    if 3 <= third <= 30:
                        saw_large_third = True
                except ValueError:
                    pass
            return saw_large_third
    except (IOError, OSError):
        return False


# =====================================================================
#  PARSING
# =====================================================================

def parse_aplist_file(filepath):
    """Parse one aplist file. Returns dict {(N, iso): [a_p, ...]}.

    Auto-detects per-class vs per-curve format. For per-curve format,
    deduplicates on (N, iso); first occurrence wins (a_p is a class
    invariant, so all occurrences should agree).
    """
    is_per_curve = detect_per_curve_format(filepath)
    aps_start = 3 if is_per_curve else 2

    result = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                parts = line.split()
                if len(parts) < aps_start + 1:
                    continue
                try:
                    N   = int(parts[0])
                    iso = parts[1]
                    # Cap at the first N_APS_PER_RECORD tokens, and stop
                    # at any Atkin-Lehner signature like '+(101)'.
                    candidate_tokens = parts[aps_start:aps_start + N_APS_PER_RECORD]
                    aps = []
                    for tok in candidate_tokens:
                        if AL_SIGNATURE_PATTERN.match(tok):
                            break
                        aps.append(_parse_ap_token(tok))
                    if not aps:
                        continue
                    if (N, iso) not in result:
                        result[(N, iso)] = aps
                except (ValueError, IndexError):
                    continue
    except (IOError, OSError) as e:
        debug("Error reading %s: %s", filepath, e)
        return {}

    return result


def parse_for_worker(filepath_str):
    """Worker entry point for parallel_map."""
    return parse_aplist_file(Path(filepath_str))


# =====================================================================
#  MAIN
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_standard_args(p)
    p.add_argument("--ecdata", type=Path, default=DEFAULT_ECDATA,
                   help="Path to cloned ecdata repository")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                   help="Input corpus parquet")
    p.add_argument("--aplist-dir", type=Path, default=None,
                   help="Override aplist directory (default: ecdata/aplist)")
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aplist_dir is None:
        args.aplist_dir = args.ecdata / "aplist"

    banner("Parse Cremona aplist files and join a_p coefficients", {
        "Ecdata":        str(args.ecdata),
        "Aplist dir":    str(args.aplist_dir),
        "Input corpus":  str(args.corpus),
        "Output dir":    str(output_dir),
        "Workers":       args.workers,
    })

    # ---- 1. Find aplist files -------------------------------------
    step(1, 4, "Locating aplist files...")
    if not args.aplist_dir.exists():
        print(C.fail(f"  Directory not found: {args.aplist_dir}"))
        print(C.fail(f"  Verify ecdata is cloned with --recursive and"
                     f" the aplist/ directory is populated."))
        return

    aplist_files = sorted(args.aplist_dir.glob("aplist*"))
    if not aplist_files:
        # Fallback: any file in the directory
        aplist_files = sorted(f for f in args.aplist_dir.iterdir()
                              if f.is_file())

    print(f"  Found {len(aplist_files):,} aplist files")
    if len(aplist_files) == 0:
        print(C.fail(f"  No files in {args.aplist_dir}"))
        return
    for f in aplist_files[:5]:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name}  ({size_kb:,.0f} KB)")
    if len(aplist_files) > 5:
        print(f"    ... and {len(aplist_files) - 5} more")

    # Detect format on the first file as a quick sanity report
    sample_file = aplist_files[0]
    is_per_curve = detect_per_curve_format(sample_file)
    print(f"  Format detected from {sample_file.name}: "
          f"{'per-curve (with curve_number)' if is_per_curve else 'per-class'}")

    # ---- 2. Parse all files in parallel ---------------------------
    step(2, 4, "Parsing aplist files in parallel...")
    with Timer("Parsing"):
        per_file_dicts = parallel_map(
            parse_for_worker,
            [str(f) for f in aplist_files],
            n_workers=args.workers,
        )

    # Merge per-file dicts
    print("  Merging per-file results...")
    aps_dict = {}
    for d in per_file_dicts:
        if isinstance(d, dict):
            aps_dict.update(d)

    print(f"  Total isogeny classes with a_p list: {len(aps_dict):,}")

    if not aps_dict:
        print(C.fail("  No a_p data parsed. Check file format."))
        return

    # Length statistics
    sample_lengths = [len(v) for v in list(aps_dict.values())[:5000]]
    print(f"  a_p sequence length: median {int(np.median(sample_lengths))}, "
          f"min {min(sample_lengths)}, max {max(sample_lengths)}")

    # ---- 3. Join to corpus ---------------------------------------
    step(3, 4, "Joining a_p data onto corpus...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")

    # Build (N, iso) key
    corpus["_join_key"] = list(zip(
        corpus["conductor"].astype(int),
        corpus["iso"],
    ))

    print("  Mapping a_p lists onto rows...")
    with Timer("Mapping"):
        corpus["aps_list"] = corpus["_join_key"].map(aps_dict)

    # Coverage check
    n_with_aps = int(corpus["aps_list"].notna().sum())
    pct = 100.0 * n_with_aps / len(corpus) if len(corpus) > 0 else 0.0
    print(f"  Curves with a_p list: {n_with_aps:,} of {len(corpus):,}  "
          f"({pct:.2f}%)")
    if n_with_aps < len(corpus):
        n_missing = len(corpus) - n_with_aps
        print(C.warn(f"  {n_missing:,} curves have no a_p list. Their"
                     f" (N, iso) keys may not appear in the aplist files."))

    # Length column
    corpus["n_aps"] = corpus["aps_list"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # Extract individual a_p columns for first six primes
    print("  Extracting individual a_p columns for primes "
          f"{EXPOSED_PRIMES}...")
    for prime in EXPOSED_PRIMES:
        idx = PRIME_TO_IDX[prime]
        col = f"a{prime}"
        corpus[col] = corpus["aps_list"].apply(
            lambda x, idx=idx: (x[idx]
                                if isinstance(x, list) and len(x) > idx
                                else None)
        )
        corpus[col] = pd.to_numeric(corpus[col], errors="coerce")

    # Drop join key
    corpus = corpus.drop(columns=["_join_key"])

    # ---- 4. Save --------------------------------------------------
    step(4, 4, "Saving enriched parquet...")
    out_path = output_dir / "ec_corpus_with_aps.parquet"
    print(f"  Writing to {out_path}...")
    with Timer("Write"):
        table = pa.Table.from_pandas(corpus, preserve_index=False)
        pq.write_table(table, out_path, compression="snappy")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Parquet: {out_path}  ({size_mb:.1f} MB, "
          f"{len(corpus):,} rows, {len(corpus.columns)} columns)")

    # ---- Summary stats -------------------------------------------
    section("a_p SUMMARY STATISTICS")
    rows = []
    for prime in EXPOSED_PRIMES:
        col = f"a{prime}"
        valid = corpus[col].dropna()
        if len(valid) > 0:
            hasse = 2 * np.sqrt(prime)
            in_bound = int((valid.abs() <= hasse).sum())
            pct_in_bound = 100.0 * in_bound / len(valid)
            rows.append((
                prime,
                int(len(valid)),
                float(valid.min()),
                float(valid.max()),
                float(valid.mean()),
                float(valid.std()),
                f"|a_p| <= 2√p ({hasse:.2f})",
                pct_in_bound,
            ))
    summary_table(
        rows,
        ["prime p", "n", "min a_p", "max a_p", "mean", "std",
         "Hasse bound", "% in bound"],
        title="Individual a_p columns",
        fmt=[">8d", ">11,d", ">10.0f", ">10.0f", ">8.3f", ">7.3f",
             "<22s", ">11.3f"],
    )

    section("VERIFICATION")
    print("  Check 1: All exposed a_p satisfy the Hasse bound |a_p| <= 2*sqrt(p)?")
    all_pass = all(r[7] >= 99.9 for r in rows)
    if all_pass:
        print(C.ok("    PASS: all primes have >= 99.9% in Hasse bound."))
        print(C.ok("    (Bad reduction primes give |a_p| <= 1 which also passes.)"))
    else:
        print(C.warn("    Some primes show < 99.9% in bound; check parsing."))

    print()
    print("  Check 2: Coverage")
    print(f"    {pct:.2f}% of curves have an a_p list assigned.")


if __name__ == "__main__":
    main()
