#!/usr/bin/env python3
"""add_jinvariants.py - compute j-invariant and Weierstrass invariants for
every curve and add to the corpus.

For an elliptic curve y² + a_1 x y + a_3 y = x³ + a_2 x² + a_4 x + a_6,
the Weierstrass invariants and j-invariant are defined by:

    b_2 = a_1² + 4 a_2
    b_4 = 2 a_4 + a_1 a_3
    b_6 = a_3² + 4 a_6
    b_8 = a_1² a_6 - a_1 a_3 a_4 + 4 a_2 a_6 + a_2 a_3² - a_4²
    c_4 = b_2² - 24 b_4
    c_6 = -b_2³ + 36 b_2 b_4 - 216 b_6
    Δ   = -b_2² b_8 - 8 b_4³ - 27 b_6² + 9 b_2 b_4 b_6
    j   = c_4³ / Δ

The j-invariant is a complete invariant for elliptic curves over Q-bar; two
curves over Q-bar are isomorphic iff they share j. Over Q, two curves with
the same j are twists of each other.

For each curve, this script computes c_4, c_6, Δ, and j (as reduced
fraction), then verifies the new data agrees with the previous CM detection
(13 CM j-invariants). It also produces a quick analysis of how the Mazur
exceptional prime isogeny classes cluster by j-invariant, validating Paper 1
v6 §6's family-vs-isolated regime hypothesis.

All large integers (c_4, c_6, Δ, j numerator, j denominator) are stored as
strings for parquet portability — Python int arithmetic is arbitrary
precision, but pyarrow's int64 max is ~9.2e18, which c_4³ can exceed at
even modest conductor.

Input:  lmfdb_ec_out/ec_corpus_with_isog.parquet
Output: lmfdb_ec_out/ec_corpus_with_jinv.parquet

Author: Kase Branham - Independent Researcher
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, Timer, C, debug, parallel_map,
)


DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

CHUNK_SIZE = 30_000  # rows per parallel worker

# 13 CM j-invariants (for verification cross-check)
CM_J_INVARIANTS = {
    0:                       -3,
    1728:                    -4,
    -3375:                   -7,
    8000:                    -8,
    -32768:                  -11,
    54000:                   -12,
    287496:                  -16,
    -884736:                 -19,
    -12288000:               -27,
    16581375:                -28,
    -884736000:              -43,
    -147197952000:           -67,
    -262537412640768000:     -163,
}

MAZUR_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 37, 43, 67, 163]


# =====================================================================
#  AINVS PARSING (shared with add_cm.py logic)
# =====================================================================

def normalize_ainvs(x):
    """Convert an ainvs value to [a1, a2, a3, a4, a6]."""
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]
    if isinstance(x, str):
        s = x.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [int(v) for v in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        s = s.strip("[]() ").replace(" ", "")
        return [int(v) for v in s.split(",") if v]
    raise ValueError(f"Unrecognized ainvs format: {type(x).__name__}")


# =====================================================================
#  INVARIANT COMPUTATION
# =====================================================================

def compute_invariants(ainvs):
    """From ainvs [a1, a2, a3, a4, a6], return (c4, c6, delta, j_num, j_den)
    with j = j_num / j_den in reduced form (j_den >= 1).
    """
    a1, a2, a3, a4, a6 = ainvs
    b2 = a1 * a1 + 4 * a2
    b4 = 2 * a4 + a1 * a3
    b6 = a3 * a3 + 4 * a6
    b8 = (a1 * a1 * a6 - a1 * a3 * a4
          + 4 * a2 * a6 + a2 * a3 * a3 - a4 * a4)
    c4 = b2 * b2 - 24 * b4
    c6 = -b2 ** 3 + 36 * b2 * b4 - 216 * b6
    delta = (-b2 * b2 * b8 - 8 * b4 ** 3 - 27 * b6 ** 2
             + 9 * b2 * b4 * b6)

    if delta == 0:
        return (c4, c6, delta, 0, 1)

    j_num = c4 ** 3
    j_den = delta
    g = math.gcd(abs(j_num), abs(j_den))
    if g > 1:
        j_num //= g
        j_den //= g
    if j_den < 0:
        j_num = -j_num
        j_den = -j_den
    return (c4, c6, delta, j_num, j_den)


def compute_for_chunk(ainvs_chunk):
    """Process a chunk of ainvs values; returns list of result tuples."""
    out = []
    for x in ainvs_chunk:
        try:
            ainvs = normalize_ainvs(x)
            c4, c6, delta, j_num, j_den = compute_invariants(ainvs)
            out.append((str(c4), str(c6), str(delta),
                        str(j_num), str(j_den)))
        except Exception:
            out.append(("", "", "", "", ""))
    return out


def compute_for_chunk_indexed(arg):
    """Wrapper that tags the chunk result with its submission index.

    parallel_map may return results in worker-completion order rather than
    submission order; the index lets us re-sort to the correct order.
    """
    idx, ainvs_chunk = arg
    return (idx, compute_for_chunk(ainvs_chunk))


# =====================================================================
#  MAIN
# =====================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(p)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("Compute j-invariants and Weierstrass invariants for every curve", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
        "Workers":    args.workers,
    })

    # ---- 1. Load corpus -----------------------------------------
    step(1, 5, "Loading corpus...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")

    # ---- 2. Compute invariants in parallel chunks --------------
    step(2, 5, f"Computing invariants in parallel chunks of ~{CHUNK_SIZE:,}...")
    ainvs_list = corpus["ainvs"].tolist()
    n_chunks = (len(ainvs_list) + CHUNK_SIZE - 1) // CHUNK_SIZE
    # Tag each chunk with its index so we can re-sort after parallel_map
    chunks_indexed = [(i, ainvs_list[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE])
                      for i in range(n_chunks)]
    print(f"  Chunks: {n_chunks}")

    with Timer("Invariant computation"):
        chunk_results_indexed = parallel_map(
            compute_for_chunk_indexed,
            chunks_indexed,
            n_workers=args.workers,
        )

    # Re-sort by index so chunks are in original submission order
    print("  Re-sorting chunks by index (parallel_map may scramble order)...")
    chunk_results_indexed.sort(key=lambda x: x[0])

    # Flatten in correct order
    print("  Flattening chunk results...")
    all_results = []
    for idx, chunk in chunk_results_indexed:
        all_results.extend(chunk)
    print(f"  Total rows processed: {len(all_results):,}")

    # ---- 3. Add columns to corpus ------------------------------
    step(3, 5, "Adding invariant columns to corpus...")
    corpus["c4_str"]     = [r[0] for r in all_results]
    corpus["c6_str"]     = [r[1] for r in all_results]
    corpus["delta_str"]  = [r[2] for r in all_results]
    corpus["j_num_str"]  = [r[3] for r in all_results]
    corpus["j_den_str"]  = [r[4] for r in all_results]

    # Derived columns
    print("  Computing derived j-invariant columns...")

    def j_str_from_pair(num_s, den_s):
        if not num_s or not den_s:
            return ""
        if den_s == "1":
            return num_s
        return f"{num_s}/{den_s}"

    def j_approx_from_pair(num_s, den_s):
        if not num_s or not den_s:
            return float("nan")
        try:
            n = int(num_s); d = int(den_s)
            if d == 0:
                return float("nan")
            return float(n) / float(d)
        except (ValueError, OverflowError):
            try:
                # Use log10 for huge numbers
                n = int(num_s); d = int(den_s)
                sign = 1 if (n >= 0) == (d > 0) else -1
                ln = math.log10(abs(n)) if n != 0 else -math.inf
                ld = math.log10(abs(d))
                return sign * 10 ** (ln - ld) if (ln - ld) < 308 else sign * float("inf")
            except Exception:
                return float("nan")

    corpus["j_str"] = [j_str_from_pair(n, d)
                       for n, d in zip(corpus["j_num_str"],
                                       corpus["j_den_str"])]
    corpus["j_approx"] = [j_approx_from_pair(n, d)
                          for n, d in zip(corpus["j_num_str"],
                                          corpus["j_den_str"])]
    corpus["j_is_integer"] = corpus["j_den_str"] == "1"

    # ---- 4. Verification: CM cross-check ----------------------
    step(4, 5, "Cross-checking against CM detection...")
    section("CM CROSS-CHECK")

    cm_str_set = {str(j): D for j, D in CM_J_INVARIANTS.items()}
    print("  Predicted: every is_cm=True curve has j_str in the 13-element CM set.")
    print("  Predicted: every curve with j_str in CM set has is_cm=True.")
    print()

    # Direction 1: is_cm=True curves should have CM j-invariants
    cm_curves = corpus[corpus["is_cm"]]
    cm_jstr_set = set(cm_curves["j_str"].unique())
    expected_cm = set(cm_str_set.keys())
    extra_in_cm = cm_jstr_set - expected_cm
    missing_in_cm = expected_cm - cm_jstr_set

    print(f"  CM curves: {len(cm_curves):,}")
    print(f"  Distinct j_str among CM curves: {len(cm_jstr_set)}")
    if not extra_in_cm:
        print(C.ok("  ✓ All CM curves have j_str in the 13-element CM set."))
    else:
        print(C.warn(f"  Extra j_str values in CM subset: {sorted(extra_in_cm)[:5]}"))
    if not missing_in_cm:
        print(C.ok("  ✓ All 13 CM j-invariants are represented among CM curves."))
    else:
        print(C.warn(f"  CM j-invariants absent from corpus: {sorted(missing_in_cm)}"))

    # Direction 2: curves with CM j-invariants should have is_cm=True
    print()
    has_cm_j = corpus["j_str"].isin(expected_cm)
    n_has_cm_j = int(has_cm_j.sum())
    n_has_cm_j_and_is_cm = int((has_cm_j & corpus["is_cm"]).sum())
    n_has_cm_j_not_is_cm = n_has_cm_j - n_has_cm_j_and_is_cm
    print(f"  Curves with j_str in CM set: {n_has_cm_j:,}")
    print(f"  Of those, is_cm=True: {n_has_cm_j_and_is_cm:,}")
    if n_has_cm_j_not_is_cm == 0:
        print(C.ok("  ✓ Every curve with CM j-invariant is flagged is_cm."))
    else:
        print(C.warn(f"  {n_has_cm_j_not_is_cm:,} curves have CM j_str but is_cm=False."))

    # ---- 4b. Mazur exceptional prime j-clustering ---------------
    section("MAZUR EXCEPTIONAL PRIMES: j-INVARIANT CLUSTERING")
    print("  For p in {11, 17, 19, 37, 43, 67, 163} (X_0(p) genus ≥ 1):")
    print("  Predicted: classes with rational p-isogeny cluster at finitely")
    print("  many j-invariants — the rational points of X_0(p).")
    print()

    rows = []
    for prime in [11, 13, 17, 19, 37, 43, 67, 163]:
        # Classes with rational p-isogeny
        col = f"has_{prime}_isog"
        if col not in corpus.columns:
            continue
        subset = corpus[corpus[col]]
        n_curves = len(subset)
        # One class is size 2 for these primes, so n_curves = 2 * n_classes
        distinct_j = subset["j_str"].nunique()
        # Top j values
        top_j = subset["j_str"].value_counts().head(3)
        top_j_strs = []
        for j_val, n in top_j.items():
            # Truncate very long j-strings for display
            display_j = j_val if len(j_val) <= 30 else j_val[:27] + "..."
            top_j_strs.append(f"{display_j} ({n})")
        rows.append((prime, n_curves, distinct_j, "; ".join(top_j_strs)))

    summary_table(
        rows,
        ["prime", "n curves with p-isog", "distinct j", "top 3 j (count)"],
        title="j-invariant clustering of Mazur exceptional p-isogeny classes",
        fmt=[">7d", ">22,d", ">12d", "<70s"],
    )

    # ---- 5. Save -----------------------------------------------
    step(5, 5, "Saving enriched parquet...")
    out_path = output_dir / "ec_corpus_with_jinv.parquet"
    print(f"  Writing to {out_path}...")
    with Timer("Write"):
        table = pa.Table.from_pandas(corpus, preserve_index=False)
        pq.write_table(table, out_path, compression="snappy")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Parquet: {out_path}  ({size_mb:.1f} MB, "
          f"{len(corpus):,} rows, {len(corpus.columns)} columns)")

    # ---- Summary ------------------------------------------------
    section("SUMMARY")
    n_j_int = int(corpus["j_is_integer"].sum())
    print(f"  Curves with integer j-invariant: {n_j_int:,}  "
          f"({100.0 * n_j_int / len(corpus):.4f}%)")
    print(f"  Curves with fractional j:       {len(corpus) - n_j_int:,}  "
          f"({100.0 * (len(corpus) - n_j_int) / len(corpus):.4f}%)")
    print()
    print("  New columns added:")
    print("    c4_str, c6_str, delta_str  (Weierstrass invariants, big-int strings)")
    print("    j_num_str, j_den_str       (reduced j = num/den, big-int strings)")
    print("    j_str                      (display: 'n' if integer, else 'n/d')")
    print("    j_approx                   (float approximation)")
    print("    j_is_integer               (boolean)")


if __name__ == "__main__":
    main()
