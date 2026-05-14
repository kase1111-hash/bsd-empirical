#!/usr/bin/env python3
"""add_allisog.py - parse Cremona allisog files and add rational isogeny
matrix data to the corpus.

The Cremona allisog file format (one line per isogeny class):

    <N> <iso> <num> <optimal_ainvs> <all_ainvs_list> <isogeny_matrix>

Example for class 11a:
    11 a 1 [0,-1,1,-10,-20] [[0,-1,1,-10,-20],[0,-1,1,-7820,-263580],[0,-1,1,0,0]] [[1,5,5],[5,1,25],[5,25,1]]

The isogeny matrix entry [i][j] gives the degree of the rational isogeny
from curve i to curve j: 1 on the diagonal, a prime or prime power off it.

For 11a's matrix [[1,5,5],[5,1,25],[5,25,1]]:
    11a1 - 11a2 :  5-isogeny
    11a1 - 11a3 :  5-isogeny
    11a2 - 11a3 : 25-isogeny (composition of two 5-isogenies)

By Mazur's theorem, the only prime degrees that occur over Q are
p in MAZUR_PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 37, 43, 67, 163}.

For each class this script extracts:
    has_p_isog (bool) for each p in MAZUR_PRIMES (12 columns)
    isog_degrees_str  : JSON list of distinct non-1 entries in the matrix
    isog_matrix_str   : JSON-serialized full matrix
    max_isog_degree   : max entry in the matrix
    n_isog_primes     : count of distinct Mazur primes dividing any entry

Replaces the torsion-witness proxy used in Paper 1 with EXACT isogeny-graph
information, which sharpens the Sha-vs-isogeny analysis and enables
investigation of higher-degree isogenies (p in {11, 13, 17, 19, 37, ...}).

Input:  lmfdb_ec_out/ec_corpus_with_cm.parquet
Output: lmfdb_ec_out/ec_corpus_with_isog.parquet

Author: Kase Branham - Independent Researcher
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, Timer, C, debug, parallel_map,
)


DEFAULT_ECDATA       = Path("./ecdata")
DEFAULT_CORPUS       = Path("./lmfdb_ec_out/ec_corpus_with_cm.parquet")
DEFAULT_OUTPUT_DIR   = Path("./lmfdb_ec_out")

# Mazur's theorem: rational p-isogenies of elliptic curves over Q exist
# iff p is in this set. (For p in {37, 43, 67, 163}, isogenies exist only
# for a finite set of j-invariants.)
MAZUR_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 37, 43, 67, 163]


# =====================================================================
#  LINE PARSING
# =====================================================================

def parse_allisog_line(line):
    """Parse one allisog line.

    Returns (N, iso, matrix) where matrix is list[list[int]], or None if
    the line is malformed.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()
    if len(parts) < 6:
        return None

    try:
        N   = int(parts[0])
        iso = parts[1]
        # parts[2] is curve_num (always 1 in observed files)
        # parts[3] is optimal_ainvs   (we don't use it)
        # parts[4] is all_ainvs_list  (we don't use it)
        # parts[5] is isogeny matrix
        matrix = ast.literal_eval(parts[5])
        if not isinstance(matrix, list) or not all(isinstance(r, list) for r in matrix):
            return None
        return (N, iso, matrix)
    except (ValueError, SyntaxError, IndexError):
        return None


def parse_allisog_file(filepath):
    """Parse one allisog file. Returns dict {(N, iso): matrix}."""
    result = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                parsed = parse_allisog_line(line)
                if parsed is None:
                    continue
                N, iso, matrix = parsed
                if (N, iso) not in result:
                    result[(N, iso)] = matrix
    except (IOError, OSError) as e:
        debug("Error reading %s: %s", filepath, e)
        return {}
    return result


def parse_for_worker(filepath_str):
    """Worker entry point for parallel_map."""
    return parse_allisog_file(Path(filepath_str))


# =====================================================================
#  MATRIX ANALYSIS
# =====================================================================

def analyze_matrix(matrix):
    """Extract structural data from an isogeny matrix.

    Returns dict with:
        class_size, all_non_unit_degrees, primes_in_matrix, max_degree
    """
    K = len(matrix)
    all_degrees = set()
    for row in matrix:
        for d in row:
            if d > 1:
                all_degrees.add(int(d))

    primes_seen = set()
    for d in all_degrees:
        for p in MAZUR_PRIMES:
            if d % p == 0:
                primes_seen.add(p)

    return {
        "class_size": K,
        "all_degrees": sorted(all_degrees),
        "primes_in_matrix": sorted(primes_seen),
        "max_degree": max(all_degrees) if all_degrees else 1,
    }


# =====================================================================
#  MAIN
# =====================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(p)
    p.add_argument("--ecdata", type=Path, default=DEFAULT_ECDATA)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--allisog-dir", type=Path, default=None,
                   help="Override allisog directory (default: ecdata/allisog)")
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.allisog_dir is None:
        args.allisog_dir = args.ecdata / "allisog"

    banner("Parse Cremona allisog files and add isogeny matrix data", {
        "Ecdata":       str(args.ecdata),
        "Allisog dir":  str(args.allisog_dir),
        "Input corpus": str(args.corpus),
        "Output dir":   str(output_dir),
        "Workers":      args.workers,
    })

    # ---- 1. Find files ------------------------------------------
    step(1, 5, "Locating allisog files...")
    if not args.allisog_dir.exists():
        print(C.fail(f"  Directory not found: {args.allisog_dir}"))
        return

    allisog_files = sorted(args.allisog_dir.glob("allisog*"))
    if not allisog_files:
        allisog_files = sorted(f for f in args.allisog_dir.iterdir() if f.is_file())
    print(f"  Found {len(allisog_files):,} allisog files")
    for f in allisog_files[:3]:
        print(f"    {f.name}  ({f.stat().st_size/1024:,.0f} KB)")
    if len(allisog_files) > 3:
        print(f"    ... and {len(allisog_files) - 3} more")

    # ---- 2. Parse files in parallel -----------------------------
    step(2, 5, "Parsing allisog files in parallel...")
    with Timer("Parsing"):
        per_file_dicts = parallel_map(
            parse_for_worker,
            [str(f) for f in allisog_files],
            n_workers=args.workers,
        )

    print("  Merging per-file results...")
    matrix_dict = {}
    for d in per_file_dicts:
        if isinstance(d, dict):
            matrix_dict.update(d)
    print(f"  Total isogeny classes with matrix: {len(matrix_dict):,}")

    if not matrix_dict:
        print(C.fail("  No matrix data parsed. Check file format."))
        return

    # ---- 3. Analyze matrices -----------------------------------
    step(3, 5, "Analyzing matrices...")
    print(f"  Computing structural summaries for {len(matrix_dict):,} classes...")
    with Timer("Analysis"):
        analysis = {key: analyze_matrix(M) for key, M in matrix_dict.items()}

    # Quick stats on class sizes and primes seen
    class_sizes = [a["class_size"] for a in analysis.values()]
    prime_counts = {}
    for a in analysis.values():
        for p in a["primes_in_matrix"]:
            prime_counts[p] = prime_counts.get(p, 0) + 1
    print(f"  Class size distribution: min {min(class_sizes)}, "
          f"max {max(class_sizes)}, "
          f"mean {sum(class_sizes)/len(class_sizes):.2f}")
    print(f"  Classes with at least one non-trivial isogeny: "
          f"{sum(1 for a in analysis.values() if a['primes_in_matrix']):,}")

    # ---- 4. Join to corpus -------------------------------------
    step(4, 5, "Joining isogeny data onto corpus...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")

    # Build join key
    corpus["_jk"] = list(zip(corpus["conductor"].astype(int), corpus["iso"]))

    # For each Mazur prime, build a boolean column
    print("  Computing has_p_isog columns for Mazur primes...")
    primes_dict = {key: set(a["primes_in_matrix"]) for key, a in analysis.items()}
    for p in MAZUR_PRIMES:
        col = f"has_{p}_isog"
        corpus[col] = corpus["_jk"].apply(
            lambda jk, p=p: (jk in primes_dict
                             and p in primes_dict[jk])
        )

    # Max isogeny degree
    print("  Computing max_isog_degree...")
    max_degrees = {key: a["max_degree"] for key, a in analysis.items()}
    corpus["max_isog_degree"] = corpus["_jk"].apply(
        lambda jk: max_degrees.get(jk, 1)
    )

    # Number of distinct Mazur primes
    print("  Computing n_isog_primes...")
    corpus["n_isog_primes"] = corpus["_jk"].apply(
        lambda jk: len(primes_dict.get(jk, set()))
    )

    # Full matrix and degree-set as JSON strings (for downstream fidelity)
    print("  Serializing isog_matrix_str and isog_degrees_str...")
    corpus["isog_matrix_str"] = corpus["_jk"].apply(
        lambda jk: (json.dumps(matrix_dict[jk])
                    if jk in matrix_dict else None)
    )
    corpus["isog_degrees_str"] = corpus["_jk"].apply(
        lambda jk: (json.dumps(analysis[jk]["all_degrees"])
                    if jk in analysis else None)
    )

    # Class-size sanity check
    print("  Sanity check: class_size from matrix vs class_size from groupby...")
    matrix_class_sizes = {key: a["class_size"] for key, a in analysis.items()}
    corpus["class_size_from_matrix"] = corpus["_jk"].apply(
        lambda jk: matrix_class_sizes.get(jk, 0)
    )
    # Compare against derived class sizes (count of curves per (N, iso))
    grouped = corpus.groupby(["conductor", "iso"]).size().rename("class_size_derived")
    corpus = corpus.merge(grouped, on=["conductor", "iso"], how="left")
    mismatch = (
        (corpus["class_size_from_matrix"] != corpus["class_size_derived"])
        & (corpus["class_size_from_matrix"] > 0)
    )
    n_mismatch = int(mismatch.sum())
    print(f"  Class-size mismatches: {n_mismatch:,}")
    if n_mismatch > 0:
        print(C.warn("  (Mismatch suggests the corpus contains curves not "
                     "listed in allisog, or vice versa.)"))

    corpus = corpus.drop(columns=["_jk"])

    # ---- 5. Save -----------------------------------------------
    step(5, 5, "Saving enriched parquet...")
    out_path = output_dir / "ec_corpus_with_isog.parquet"
    print(f"  Writing to {out_path}...")
    with Timer("Write"):
        table = pa.Table.from_pandas(corpus, preserve_index=False)
        pq.write_table(table, out_path, compression="snappy")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Parquet: {out_path}  ({size_mb:.1f} MB, "
          f"{len(corpus):,} rows, {len(corpus.columns)} columns)")

    # ---- Summary stats -----------------------------------------
    section("MAZUR PRIME ISOGENY DISTRIBUTION")
    print("  Number of CURVES whose isogeny class has a rational p-isogeny:")
    print()
    rows = []
    for p in MAZUR_PRIMES:
        col = f"has_{p}_isog"
        n = int(corpus[col].sum())
        pct = 100.0 * n / len(corpus)
        rows.append((p, n, pct))
    summary_table(
        rows,
        ["prime p", "n curves", "pct"],
        title="Rational p-isogeny presence by Mazur prime",
        fmt=[">8d", ">12,d", ">10.4f"],
    )

    section("VERIFICATION")
    n_with_matrix = int(corpus["isog_matrix_str"].notna().sum())
    pct_match = 100.0 * n_with_matrix / len(corpus)
    print(f"  Curves with matrix attached: {n_with_matrix:,} of "
          f"{len(corpus):,} ({pct_match:.2f}%)")
    if pct_match > 99.5:
        print(C.ok("  PASS: ~100% of curves have an isogeny matrix."))
    else:
        print(C.warn(f"  {len(corpus) - n_with_matrix:,} curves are missing matrices."))

    print()
    print("  Class-size cross-check:")
    if n_mismatch == 0:
        print(C.ok("  PASS: matrix class sizes match derived class sizes."))
    else:
        print(C.warn(f"  {n_mismatch:,} mismatches; see above."))


if __name__ == "__main__":
    main()
