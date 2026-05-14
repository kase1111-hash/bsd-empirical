#!/usr/bin/env python3
"""investigate_isog_sha_split.py - characterize the disjoint vs overlapping
classes for isogeny-Sha analysis at Mazur exceptional primes.

For p in {11, 13, 17, 19, 37, 43, 67, 163}, regression_isog_vs_sha_exact.py
found near-zero overlap between classes with rational p-isogeny and classes
with p^2 | |Sha|_an. The only intermediate case is p = 13 with 4 overlap
classes.

This script characterizes each prime's three populations:

    1. ISOG-ONLY: has rational p-isogeny but no p^2|Sha
    2. SHA-ONLY:  has p^2|Sha but no rational p-isogeny
    3. BOTH:      both (lists by name; small)

For the two-regime hypothesis (Cassels-Fisher mechanism at genus-0 X_0(p),
Selmer-excess mechanism at genus->=1 X_0(p)):

  - SHA-ONLY classes should be predominantly size-1 (no rational isogeny
    of any prime), consistent with the "isolated Sha regime" of Paper 1 §5.
  - ISOG-ONLY classes should be larger (typically size 2-8) and represent
    the X_0(p)-family without producing Sha at the corresponding prime.
  - BOTH classes (if any) should be examined directly for j-invariants
    and conductor structure to see whether they have any special features.

Input:  lmfdb_ec_out/ec_corpus_with_isog.parquet
Output: lmfdb_ec_out/isog_sha_split_<p>.csv  (per prime)

Author: Kase Branham - Independent Researcher
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, Timer, C,
)


DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

INVESTIGATE_PRIMES = [11, 13, 17, 19, 37, 43, 67, 163]


def class_size_breakdown(df, label, n_show=None):
    """Print class-size distribution for a sub-population."""
    if len(df) == 0:
        print(f"  {label}: empty")
        return
    sizes = df["class_size"].value_counts().sort_index()
    print(f"  {label} (n = {len(df):,}):")
    for sz, n in sizes.items():
        pct = 100.0 * n / len(df)
        bar = "█" * int(40 * pct / 100)
        print(f"    size {sz}: {n:>8,d}  ({pct:5.2f}%)  {bar}")
    if n_show:
        print(f"  Top {n_show} by sha_max:")
        top = df.nlargest(n_show, "sha_max")
        for _, row in top.iterrows():
            print(f"    N={int(row['conductor'])}, iso={row['iso']}, "
                  f"size={int(row['class_size'])}, "
                  f"rank={int(row['rank'])}, sha_max={int(row['sha_max'])}")


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

    banner("Investigate isog/Sha disjoint vs overlap at Mazur exceptional primes", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load corpus -----------------------------------------
    step(1, 3, "Loading corpus and building per-curve has_p²_sha flags...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")
    corpus["sha_int"] = corpus["sha_an"].round().astype(np.int64)

    for prime in INVESTIGATE_PRIMES:
        p2 = prime * prime
        corpus[f"curve_has_{p2}sha"] = (
            (corpus["sha_int"] > 0)
            & (corpus["sha_int"] % p2 == 0)
        )

    # ---- 2. Aggregate to class level ----------------------------
    step(2, 3, "Aggregating to class level...")
    agg_dict = {
        "sha_int":          "max",
        "rank":             "first",
        "tamagawa_product": "max",
    }
    for prime in INVESTIGATE_PRIMES:
        agg_dict[f"has_{prime}_isog"]          = "first"
        agg_dict[f"curve_has_{prime*prime}sha"] = "any"

    with Timer("Class aggregation"):
        classes = (corpus.groupby(["conductor", "iso"], as_index=False)
                          .agg(agg_dict))
        class_sizes = (corpus.groupby(["conductor", "iso"])
                              .size().rename("class_size").reset_index())
        classes = classes.merge(class_sizes, on=["conductor", "iso"])

    classes = classes.rename(columns={"sha_int": "sha_max"})
    print(f"  Classes: {len(classes):,}")

    # ---- 3. Per-prime investigation -----------------------------
    step(3, 3, "Per-prime investigation...")

    for prime in INVESTIGATE_PRIMES:
        p2 = prime * prime
        section(f"PRIME p = {prime}  (p² = {p2})")
        print(f"  X_0({prime}) modular curve genus: "
              f"{'0' if prime in {2,3,5,7,13} else '>=1'} "
              f"(determines whether the prime is in the 'family' regime)")
        print()

        feat = classes[f"has_{prime}_isog"].astype(bool)
        out  = classes[f"curve_has_{p2}sha"].astype(bool)

        isog_only  = classes[feat & ~out]
        sha_only   = classes[~feat & out]
        both       = classes[feat & out]
        neither    = classes[~feat & ~out]

        rows = [
            ("ISOG only",   len(isog_only)),
            ("SHA only",    len(sha_only)),
            ("BOTH",        len(both)),
            ("NEITHER",     len(neither)),
            ("Total",       len(classes)),
        ]
        summary_table(
            rows,
            ["population", "n classes"],
            title=f"Population breakdown at p = {prime}",
            fmt=["<14s", ">12,d"],
        )

        # ISOG-only class-size breakdown
        if len(isog_only) > 0:
            print()
            print(f"  ISOG-only at p = {prime} (has p-isog, no p²|Sha):")
            class_size_breakdown(isog_only, "  class sizes")

        # SHA-only class-size breakdown — KEY TEST of regime hypothesis
        if len(sha_only) > 0:
            print()
            print(f"  SHA-only at p = {prime} (has p²|Sha, no p-isog):")
            class_size_breakdown(sha_only, "  class sizes", n_show=5)
            # The two-regime hypothesis predicts most SHA-only are size-1
            size_1_frac = float((sha_only["class_size"] == 1).mean())
            print(f"  Fraction of SHA-only that are size-1: {100*size_1_frac:.1f}%")
            if size_1_frac > 0.75:
                print(C.ok("  ✓ Consistent with isolated-Sha regime hypothesis "
                           "(predominantly size-1)."))
            else:
                print(f"  (Two-regime hypothesis predicts >75% size-1; observed "
                      f"{100*size_1_frac:.1f}%.)")

        # BOTH classes — examine directly
        if len(both) > 0:
            print()
            print(f"  BOTH at p = {prime} (has p-isog AND p²|Sha): "
                  f"DIRECT LIST")
            for _, row in both.iterrows():
                print(f"    N={int(row['conductor'])}, "
                      f"iso={row['iso']}, "
                      f"size={int(row['class_size'])}, "
                      f"rank={int(row['rank'])}, "
                      f"sha_max={int(row['sha_max'])}, "
                      f"tam={int(row['tamagawa_product'])}")

        # Save per-prime CSV combining all four populations
        combined = pd.concat([
            isog_only.assign(population="isog_only"),
            sha_only.assign(population="sha_only"),
            both.assign(population="both"),
        ], ignore_index=True)
        cols_to_save = ["conductor", "iso", "rank", "class_size", "sha_max",
                        "tamagawa_product", "population",
                        f"has_{prime}_isog", f"curve_has_{p2}sha"]
        cols_to_save = [c for c in cols_to_save if c in combined.columns]
        out_csv = output_dir / f"isog_sha_split_p{prime}.csv"
        combined[cols_to_save].to_csv(out_csv, index=False)
        print(f"\n  Per-population CSV: {out_csv}")

    # ---- Cross-prime summary -----------------------------------
    section("CROSS-PRIME SUMMARY: TWO-REGIME HYPOTHESIS")
    print()
    print("  For each Mazur prime p, what fraction of SHA-only classes "
          "are size-1?")
    print("  Hypothesis: at genus-≥1 primes, SHA-only is overwhelmingly "
          "size-1 (isolated Sha regime).")
    print()
    rows = []
    for prime in INVESTIGATE_PRIMES:
        p2 = prime * prime
        feat = classes[f"has_{prime}_isog"].astype(bool)
        out  = classes[f"curve_has_{p2}sha"].astype(bool)
        sha_only = classes[~feat & out]
        n_sha_only = len(sha_only)
        n_size1 = int((sha_only["class_size"] == 1).sum()) if n_sha_only > 0 else 0
        frac_size1 = (100.0 * n_size1 / n_sha_only) if n_sha_only > 0 else 0.0
        genus = "0" if prime in {2,3,5,7,13} else ">=1"
        rows.append((prime, genus, n_sha_only, n_size1, frac_size1))
    summary_table(
        rows,
        ["prime", "X_0(p) genus", "n_sha_only", "n_size_1",
         "pct size-1"],
        title="Size-1 dominance of SHA-only classes by prime",
        fmt=[">7d", ">14s", ">12,d", ">12,d", ">12.2f"],
    )

    print()
    print("  Verdict: if SHA-only classes at p ≥ 11 are predominantly size-1,")
    print("  the two-regime hypothesis is empirically supported.")


if __name__ == "__main__":
    main()
