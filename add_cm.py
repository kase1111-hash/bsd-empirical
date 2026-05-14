#!/usr/bin/env python3
"""add_cm.py - detect CM curves and add CM-discriminant column to corpus.

An elliptic curve E/Q has complex multiplication (CM) iff End(E_C) is
strictly larger than Z, i.e. End(E_C) = O for some order O in an imaginary
quadratic field K. Over Q, only orders of class number 1 occur as
endomorphism rings, giving exactly 13 j-invariants for CM curves:

    D = -3:   j = 0
    D = -4:   j = 1728
    D = -7:   j = -3375
    D = -8:   j = 8000
    D = -11:  j = -32768
    D = -12:  j = 54000                  (non-fundamental: Z[√-3])
    D = -16:  j = 287496                 (non-fundamental: Z[2i])
    D = -19:  j = -884736
    D = -27:  j = -12288000              (non-fundamental: index-3 in O_{-3})
    D = -28:  j = 16581375               (non-fundamental: index-2 in O_{-7})
    D = -43:  j = -884736000
    D = -67:  j = -147197952000
    D = -163: j = -262537412640768000

For each curve in the corpus, computes the j-invariant from the Weierstrass
ainvs and checks membership in the set of 13 CM j-values. Adds two columns:

    is_cm           : bool, True iff E has CM
    cm_discriminant : Int64 (nullable), the CM order discriminant D (< 0)

Pure Python arithmetic — uses Python's built-in arbitrary-precision int
for the bignum c_4^3 and disc computations.

The CM/non-CM split is essential for Sato-Tate empirics: the limit measure
is the semicircle for non-CM curves and a different (degenerate) measure
for CM curves.

Input:  lmfdb_ec_out/ec_corpus_with_aps.parquet
Output: lmfdb_ec_out/ec_corpus_with_cm.parquet

Author: Kase Branham - Independent Researcher
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, Timer, C, debug,
)


DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_aps.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")


# The 13 CM j-invariants for elliptic curves over Q.
# Map j-value -> CM order discriminant D (negative integer).
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


# =====================================================================
#  AINVS PARSING
# =====================================================================

def normalize_ainvs(x):
    """Convert an ainvs value (any plausible storage form) to [a1,a2,a3,a4,a6].

    Handles list, numpy array, JSON string, and Cremona-style bracket string.
    """
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]
    if isinstance(x, str):
        s = x.strip()
        # Try JSON first
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [int(v) for v in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        # Try bracket-stripped CSV
        s = s.strip("[]() ").replace(" ", "")
        return [int(v) for v in s.split(",") if v]
    raise ValueError(f"Unrecognized ainvs format: {type(x).__name__}: {x!r}")


# =====================================================================
#  CM DETECTION
# =====================================================================

def cm_discriminant_from_ainvs(ainvs):
    """Return CM order discriminant D if the curve has CM, else None.

    Computes c_4, Δ from Weierstrass coefficients via the standard formulas:

        b_2 = a_1^2 + 4 a_2
        b_4 = 2 a_4 + a_1 a_3
        b_6 = a_3^2 + 4 a_6
        b_8 = a_1^2 a_6 - a_1 a_3 a_4 + 4 a_2 a_6 + a_2 a_3^2 - a_4^2
        c_4 = b_2^2 - 24 b_4
        Δ   = -b_2^2 b_8 - 8 b_4^3 - 27 b_6^2 + 9 b_2 b_4 b_6
        j   = c_4^3 / Δ

    Then checks whether c_4^3 == j_CM * Δ for any of the 13 CM j-values.
    Uses integer comparison to avoid rational arithmetic.
    """
    a1, a2, a3, a4, a6 = ainvs
    b2 = a1 * a1 + 4 * a2
    b4 = 2 * a4 + a1 * a3
    b6 = a3 * a3 + 4 * a6
    b8 = (a1 * a1 * a6 - a1 * a3 * a4
          + 4 * a2 * a6 + a2 * a3 * a3 - a4 * a4)
    c4 = b2 * b2 - 24 * b4
    disc = (-b2 * b2 * b8 - 8 * b4 ** 3 - 27 * b6 ** 2
            + 9 * b2 * b4 * b6)

    if disc == 0:
        return None  # singular; should not occur for valid elliptic curves

    c4_cubed = c4 ** 3
    for j_cm, D in CM_J_INVARIANTS.items():
        if c4_cubed == j_cm * disc:
            return D
    return None


def detect_cm_for_chunk(ainvs_list):
    """Process a chunk of ainvs values; returns list of (is_cm, D)."""
    out = []
    for x in ainvs_list:
        try:
            ainvs = normalize_ainvs(x)
            D = cm_discriminant_from_ainvs(ainvs)
            out.append((D is not None, D))
        except (ValueError, TypeError):
            out.append((False, None))
    return out


# =====================================================================
#  MAIN
# =====================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(p)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                   help="Input corpus parquet")
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("Detect CM curves from ainvs and add cm_discriminant column", {
        "Input corpus":  str(args.corpus),
        "Output dir":    str(output_dir),
        "Workers":       args.workers,
    })

    # ---- 1. Load corpus -----------------------------------------
    step(1, 4, "Loading corpus...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")

    if "ainvs" not in corpus.columns:
        print(C.fail("  Column 'ainvs' not found in corpus."))
        print(C.fail(f"  Columns present: {list(corpus.columns)[:20]}..."))
        return

    # Inspect format
    sample = corpus["ainvs"].iloc[0]
    print(f"  ainvs sample type: {type(sample).__name__}")
    print(f"  ainvs sample value: {sample!r}")

    # ---- 2. Detect CM for every curve ---------------------------
    step(2, 4, "Detecting CM via j-invariant check...")
    ainvs_values = corpus["ainvs"].tolist()

    print(f"  Processing {len(ainvs_values):,} curves...")
    with Timer("CM detection"):
        results = detect_cm_for_chunk(ainvs_values)

    is_cm_arr   = np.array([r[0] for r in results], dtype=bool)
    cm_disc_arr = np.array([r[1] if r[1] is not None else 0 for r in results],
                           dtype=np.int64)

    corpus["is_cm"] = is_cm_arr
    # Use pandas nullable Int64 so non-CM rows show as <NA> instead of 0
    corpus["cm_discriminant"] = pd.array(
        [r[1] for r in results], dtype="Int64"
    )

    n_cm = int(is_cm_arr.sum())
    print(f"  CM curves: {n_cm:,} of {len(corpus):,}  "
          f"({100.0 * n_cm / len(corpus):.4f}%)")

    # ---- 3. Verification & breakdown ----------------------------
    step(3, 4, "Verifying and breaking down CM detection...")
    section("CM DISCRIMINANT DISTRIBUTION")

    breakdown = corpus[corpus["is_cm"]].groupby(
        "cm_discriminant", observed=False
    ).size().sort_index()

    rows = []
    for D, n in breakdown.items():
        j_value = next(j for j, d in CM_J_INVARIANTS.items() if d == int(D))
        rows.append((
            int(D),
            int(n),
            j_value,
        ))
    summary_table(
        rows,
        ["CM discriminant D", "n curves", "j-invariant"],
        title="CM curves by order discriminant (fewer ~ rarer in the corpus)",
        fmt=[">18d", ">12,d", ">22d"],
    )

    # Sanity check: verify expected total of 13 distinct discriminants
    n_distinct = corpus.loc[corpus["is_cm"], "cm_discriminant"].nunique()
    print()
    print(f"  Distinct CM discriminants seen: {n_distinct} of 13 possible")
    if n_distinct < 13:
        missing = sorted(set(CM_J_INVARIANTS.values())
                         - set(int(d) for d in breakdown.index))
        print(f"  Not observed: {missing}")
        print(C.warn("  (Some CM discriminants — D = -67, -163, etc — "
                     "produce curves only at very specific conductors. "
                     "Absence at corpus scale doesn't indicate a bug.)"))

    # ---- 4. Save ------------------------------------------------
    step(4, 4, "Saving enriched parquet...")
    out_path = output_dir / "ec_corpus_with_cm.parquet"
    print(f"  Writing to {out_path}...")
    with Timer("Write"):
        table = pa.Table.from_pandas(corpus, preserve_index=False)
        pq.write_table(table, out_path, compression="snappy")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Parquet: {out_path}  ({size_mb:.1f} MB, "
          f"{len(corpus):,} rows, {len(corpus.columns)} columns)")

    # ---- Summary ------------------------------------------------
    section("SUMMARY")
    print(f"  Total curves:      {len(corpus):,}")
    print(f"  CM curves:         {n_cm:,}  ({100.0 * n_cm / len(corpus):.4f}%)")
    print(f"  Non-CM curves:     {len(corpus) - n_cm:,}  "
          f"({100.0 * (len(corpus) - n_cm) / len(corpus):.4f}%)")
    print(f"  Distinct CM D:     {n_distinct} of 13 possible")
    print()
    print("  CM-aware downstream analyses now possible:")
    print("    - Sato-Tate testing (different limit for CM vs non-CM)")
    print("    - CM-segregated rank, Sha, and modular-degree analyses")
    print("    - Identification of CM specimens for case studies")


if __name__ == "__main__":
    main()
