#!/usr/bin/env python3
"""catalog_sha_rich_conductors.py - find Sha-rich conductors corpus-wide.

For every distinct conductor N in the corpus, computes:
    - n_curves                  total curves at this N
    - n_sha_nontrivial          curves with Sha_an >= threshold (default 4)
    - max_sha                   max Sha_an observed at this N
    - n_classes                 number of distinct isogeny classes at N
    - n_classes_sha             classes with at least one non-trivial Sha curve
    - omega                     number of distinct prime factors of N
    - n_square_primes           number of primes p with p^2 | N
    - largest_prime             largest prime factor of N
    - factorization             string form of N

Then tests THE HYPOTHESIS from the 177450dr/177450fb investigation:
    Conductors with multiple p^2 factors are systematically more
    Sha-rich than squarefree or near-squarefree conductors.

The cross-tab by n_square_primes is the headline test. The Top-N tables
then ground the statistics in explicit examples.

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet
Output: lmfdb_ec_out/sha_rich_conductors.csv  (full per-conductor data)

Author: Kase Branham - Independent Researcher
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, Timer, C, debug,
)


DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_spectrum.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")


# =====================================================================
#  FACTORIZATION VIA SMALLEST-PRIME-FACTOR SIEVE
# =====================================================================

def build_spf_sieve(n_max):
    """SPF[i] = smallest prime factor of i for i in 2..n_max."""
    spf = np.zeros(n_max + 1, dtype=np.int32)
    for i in range(2, n_max + 1):
        if spf[i] == 0:
            spf[i:n_max + 1:i] = np.where(
                spf[i:n_max + 1:i] == 0, i, spf[i:n_max + 1:i]
            )
    return spf


def factor_from_spf(n, spf):
    """Factor n via SPF lookup. Returns dict {p: e}."""
    f = {}
    while n > 1:
        p = int(spf[n])
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
    return f


def fmt_factorization(f):
    return " * ".join(
        f"{p}^{e}" if e > 1 else str(p)
        for p, e in sorted(f.items())
    )


# =====================================================================
#  MAIN
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_standard_args(p)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--top-n", type=int, default=30,
                   help="Number of top conductors to report")
    p.add_argument("--sha-threshold", type=int, default=4,
                   help="Sha_an threshold for 'non-trivial' (default 4)")
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("Sha-rich conductors catalog", {
        "Corpus":        str(args.corpus),
        "Output dir":    str(output_dir),
        "Top-N":         args.top_n,
        "Sha threshold": args.sha_threshold,
    })

    # ---- 1. Load + aggregate by conductor -------------------------
    step(1, 4, "Loading and aggregating by conductor...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    corpus = corpus[corpus["sha_an"].notna()].copy()
    corpus["sha_int"] = corpus["sha_an"].round().astype(int)
    thr = args.sha_threshold
    print(f"  Loaded {len(corpus):,} curves with sha_an")

    by_N = corpus.groupby("conductor").agg(
        n_curves=("sha_int", "count"),
        n_sha_nontrivial=("sha_int", lambda s: int((s >= thr).sum())),
        max_sha=("sha_int", "max"),
        n_classes=("iso", "nunique"),
    ).reset_index()

    sha_classes_per_N = (
        corpus[corpus["sha_int"] >= thr]
        .groupby("conductor")["iso"]
        .nunique()
        .reset_index()
        .rename(columns={"iso": "n_classes_sha"})
    )
    by_N = by_N.merge(sha_classes_per_N, on="conductor", how="left")
    by_N["n_classes_sha"] = by_N["n_classes_sha"].fillna(0).astype(int)
    print(f"  Aggregated to {len(by_N):,} unique conductors")

    # ---- 2. Factor every conductor --------------------------------
    step(2, 4, "Factoring conductors...")
    N_max = int(by_N["conductor"].max())
    print(f"  Building SPF sieve up to {N_max:,}...")
    with Timer("SPF sieve"):
        spf = build_spf_sieve(N_max)
    print(f"  Factoring {len(by_N):,} conductors...")
    with Timer("Factoring"):
        omega       = []
        sq_count    = []
        largest_p   = []
        cube_count  = []  # primes with p^3 | N
        factor_strs = []
        for N in by_N["conductor"]:
            f = factor_from_spf(int(N), spf)
            omega.append(len(f))
            sq_count.append(sum(1 for e in f.values() if e >= 2))
            cube_count.append(sum(1 for e in f.values() if e >= 3))
            largest_p.append(max(f.keys()))
            factor_strs.append(fmt_factorization(f))
    by_N["omega"]           = omega
    by_N["n_square_primes"] = sq_count
    by_N["n_cube_primes"]   = cube_count
    by_N["largest_prime"]   = largest_p
    by_N["factorization"]   = factor_strs

    step(3, 4, "Cross-tabs and stats...")

    # ---- GLOBAL stats ---------------------------------------------
    section("GLOBAL STATISTICS")
    total_curves = int(by_N["n_curves"].sum())
    total_sha    = int(by_N["n_sha_nontrivial"].sum())
    print(f"  Total curves:                    {total_curves:,}")
    print(f"  Curves with Sha_an >= {thr}:           "
          f"{total_sha:,}  ({100*total_sha/total_curves:.2f}%)")
    print(f"  Unique conductors:               {len(by_N):,}")
    n_with_sha = int((by_N["n_sha_nontrivial"] > 0).sum())
    print(f"  Conductors hosting any Sha>={thr}:   "
          f"{n_with_sha:,}  ({100*n_with_sha/len(by_N):.2f}%)")
    n_with_high = int((by_N["max_sha"] >= 16).sum())
    print(f"  Conductors with max_sha >= 16:   {n_with_high:,}")
    n_with_higher = int((by_N["max_sha"] >= 64).sum())
    print(f"  Conductors with max_sha >= 64:   {n_with_higher:,}")

    # ---- HEADLINE TEST: Sha-richness by n_square_primes -----------
    section("HEADLINE TEST  --  Sha-richness by number of p^2 factors")
    print("  For each value of n_square_primes (count of distinct primes p with p^2 | N):")
    print(f"  rate = fraction of curves at those conductors with Sha_an >= {thr}")
    print()
    rows = []
    for sq in sorted(by_N["n_square_primes"].unique()):
        sub = by_N[by_N["n_square_primes"] == sq]
        n_N        = int(len(sub))
        tot_curves = int(sub["n_curves"].sum())
        tot_sha    = int(sub["n_sha_nontrivial"].sum())
        rate       = 100 * tot_sha / tot_curves if tot_curves > 0 else 0.0
        avg_max    = float(sub["max_sha"].mean())
        max_max    = int(sub["max_sha"].max())
        rows.append((
            int(sq), n_N, tot_curves, tot_sha,
            rate, avg_max, max_max,
        ))
    summary_table(
        rows,
        ["sq_primes", "n_conductors", "n_curves", "n_sha_nontriv",
         "rate %", "avg_max_sha", "max_max_sha"],
        title=f"Sha (>={thr}) rate by n_square_primes",
        fmt=[">10d", ">14,d", ">14,d", ">16,d",
             ">10.3f", ">14.2f", ">14d"],
    )

    # Chi-square test of independence: n_square_primes vs has-non-trivial-Sha
    # 2 x K contingency table
    print()
    print(C.info("  Chi-square test of independence (n_square_primes vs Sha-trivial/nontrivial):"))
    cont = []
    for sq in sorted(by_N["n_square_primes"].unique()):
        sub = by_N[by_N["n_square_primes"] == sq]
        tot_curves = int(sub["n_curves"].sum())
        tot_sha    = int(sub["n_sha_nontrivial"].sum())
        cont.append([tot_sha, tot_curves - tot_sha])
    cont_arr = np.array(cont).T  # 2 x K: rows are [nontriv, triv]
    chi2, pval, dof, expected = stats.chi2_contingency(cont_arr)
    print(f"    chi^2 = {chi2:.2f}    dof = {dof}    p-value = {pval:.3e}")
    if pval < 1e-10:
        print(C.ok("    Highly significant -- n_square_primes IS associated with Sha-richness."))
    elif pval < 0.001:
        print(C.ok("    Significant association."))
    else:
        print(C.warn("    No significant association detected."))

    # ---- TOP conductors by n_sha_nontrivial -----------------------
    section(f"TOP {args.top_n} CONDUCTORS BY n_sha_nontrivial")
    top = by_N.nlargest(args.top_n, "n_sha_nontrivial")
    rows = []
    for r in top.itertuples():
        rows.append((
            int(r.conductor), int(r.n_curves),
            int(r.n_sha_nontrivial), int(r.max_sha),
            int(r.n_square_primes), int(r.omega),
            r.factorization,
        ))
    summary_table(
        rows,
        ["N", "n_curves", "n_sha+", "max_sha",
         "sq_p", "omega", "factorization"],
        title=f"Top {args.top_n} most Sha-rich conductors",
        fmt=[">8d", ">9d", ">8d", ">9d", ">6d", ">6d", "<42s"],
    )

    # ---- TOP conductors by max_sha --------------------------------
    section(f"CONDUCTORS WITH max_sha >= 16  (sorted by max_sha)")
    top_extreme = by_N[by_N["max_sha"] >= 16].sort_values(
        ["max_sha", "n_sha_nontrivial"], ascending=[False, False]
    )
    print(f"  {len(top_extreme):,} conductors host at least one Sha_an >= 16 curve")
    rows = []
    for r in top_extreme.head(args.top_n).itertuples():
        rows.append((
            int(r.conductor), int(r.max_sha),
            int(r.n_sha_nontrivial), int(r.n_curves),
            int(r.n_square_primes), int(r.omega),
            r.factorization,
        ))
    summary_table(
        rows,
        ["N", "max_sha", "n_sha+", "n_curves",
         "sq_p", "omega", "factorization"],
        title=f"Top {args.top_n} by max_sha",
        fmt=[">8d", ">9d", ">8d", ">9d", ">6d", ">6d", "<42s"],
    )

    # ---- Squarefree counterexamples -------------------------------
    section("MOST SHA-RICH SQUAREFREE CONDUCTORS  (n_square_primes = 0)")
    print("  These are the counterexamples to the hypothesis -- conductors with")
    print("  no p^2 factor that nonetheless host non-trivial Sha. If the list")
    print("  is short and max_sha is low, the hypothesis holds.")
    print()
    sqfree = by_N[(by_N["n_square_primes"] == 0)
                  & (by_N["n_sha_nontrivial"] > 0)]
    print(f"  Squarefree conductors with any Sha>={thr}: {len(sqfree):,}")
    print(f"  Max max_sha among squarefree: {int(sqfree['max_sha'].max()) if len(sqfree)>0 else 0}")
    print()
    top_sqfree = sqfree.nlargest(min(20, len(sqfree)), "max_sha")
    rows = []
    for r in top_sqfree.itertuples():
        rows.append((
            int(r.conductor), int(r.max_sha),
            int(r.n_sha_nontrivial), int(r.omega),
            r.factorization,
        ))
    summary_table(
        rows,
        ["N", "max_sha", "n_sha+", "omega", "factorization"],
        title="Top squarefree Sha-rich conductors",
        fmt=[">8d", ">9d", ">8d", ">6d", "<42s"],
    )

    # ---- 4. Save --------------------------------------------------
    step(4, 4, "Saving...")
    out_csv = output_dir / "sha_rich_conductors.csv"
    save_cols = ["conductor", "factorization", "n_curves",
                 "n_sha_nontrivial", "max_sha", "n_classes",
                 "n_classes_sha", "omega", "n_square_primes",
                 "n_cube_primes", "largest_prime"]
    by_N[save_cols].sort_values(
        ["n_sha_nontrivial", "max_sha"], ascending=[False, False]
    ).to_csv(out_csv, index=False)
    print(f"  CSV: {out_csv}  ({len(by_N):,} rows)")


if __name__ == "__main__":
    main()
