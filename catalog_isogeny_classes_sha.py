#!/usr/bin/env python3
"""catalog_isogeny_classes_sha.py - cross-tab isogeny-prime structure vs |Sha|
prime parts across the entire corpus.

For each (conductor, iso) class, computes:
    - n_curves            class size
    - max_rank            max rank in class
    - has_p_isog          for p in {2,3,5,7}: does ANY curve in the class
                          have rational p-torsion? (sufficient witness for
                          a rational p-isogeny)
    - max_sha             max sha_an = max |Sha| across class
    - v_p(max_sha)        p-adic valuation of max_sha for p in {2,3,5,7}
    - n_sha_nontriv       curves with sha_an >= threshold

The HEADLINE TEST: for each prime p in {2, 3, 5, 7}, compute the rate at
which classes have p^2 | max_sha (i.e. |Sha[p]| nontrivial in the BSD-
predicted-order sense), split by whether the class has a rational
p-isogeny. If "p-isogeny enables Sha[p] growth" is the right mechanism,
the rate ratio should be large.

Detection of p-isogeny: a class has a rational p-isogeny IF some curve
in the class has p | |T| (a rational p-torsion point gives a rational
p-subgroup, hence a rational p-isogeny). This is sufficient but not
necessary -- 11-isogeny classes and similar exceptional classes may
have no torsion witness. Errors will dilute, not reverse, the test.

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet
Output: lmfdb_ec_out/isogeny_class_sha_catalog.csv

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

ISOGENY_PRIMES = [2, 3, 5, 7]


def factor_dict(n):
    """Trial-division factorization."""
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def fmt_factorization(n):
    if n == 1:
        return "1"
    return " * ".join(
        f"{p}^{e}" if e > 1 else str(p)
        for p, e in sorted(factor_dict(n).items())
    )


def v_p(n, p):
    """p-adic valuation of n."""
    if n <= 0:
        return 0
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(p)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--sha-threshold", type=int, default=4)
    p.add_argument("--top-n", type=int, default=20)
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("Isogeny class catalog: p-isogeny vs |Sha| p-parts", {
        "Corpus":        str(args.corpus),
        "Sha threshold": args.sha_threshold,
        "Output dir":    str(output_dir),
    })

    # ---- 1. Load and prepare per-curve flags ----------------------
    step(1, 4, "Loading and computing per-curve indicators...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    corpus = corpus[corpus["sha_an"].notna()].copy()
    corpus["sha_int"]      = corpus["sha_an"].round().astype(int)
    corpus["torsion_int"]  = corpus["torsion_order"].astype(int)
    corpus["rank_int"]     = corpus["rank"].astype(int)
    print(f"  Loaded {len(corpus):,} curves")

    # Per-curve indicators: does this curve have rational p-torsion?
    for prime in ISOGENY_PRIMES:
        corpus[f"has_{prime}_tors"] = (corpus["torsion_int"] % prime == 0)
    print(f"  Torsion-prime indicators built for primes {ISOGENY_PRIMES}")

    # ---- 2. Aggregate by isogeny class ----------------------------
    step(2, 4, "Aggregating per isogeny class...")
    with Timer("Class aggregation"):
        agg_dict = {
            "n_curves":      ("sha_int",     "count"),
            "max_rank":      ("rank_int",    "max"),
            "max_sha":       ("sha_int",     "max"),
            "mean_sha":      ("sha_int",     "mean"),
            "n_sha_nontriv": ("sha_int",
                              lambda s: int((s >= args.sha_threshold).sum())),
        }
        for prime in ISOGENY_PRIMES:
            agg_dict[f"has_{prime}_isog"] = (f"has_{prime}_tors", "any")
        df = (corpus.groupby(["conductor", "iso"])
              .agg(**agg_dict)
              .reset_index())
    print(f"  {len(df):,} isogeny classes catalogued")

    # Compute v_p(max_sha) for each class and each prime
    print("  Computing v_p(max_sha) for each prime...")
    for prime in ISOGENY_PRIMES:
        df[f"vp_{prime}"] = df["max_sha"].apply(lambda n, p=prime: v_p(int(n), p))
        df[f"sha_has_{prime}sq"] = df[f"vp_{prime}"] >= 2

    # Compose class label and torsion-prime string
    df["class"] = df["conductor"].astype(str) + df["iso"]
    df["isog_primes"] = df.apply(
        lambda r: "[" + ",".join(
            str(p) for p in ISOGENY_PRIMES if r[f"has_{p}_isog"]
        ) + "]",
        axis=1,
    )

    # ---- 3. Global stats + headline test --------------------------
    step(3, 4, "Cross-tabs and tests...")

    section("GLOBAL CLASS-LEVEL STATISTICS")
    n_classes = len(df)
    n_classes_with_sha = int((df["n_sha_nontriv"] > 0).sum())
    print(f"  Total isogeny classes:                    {n_classes:,}")
    print(f"  Classes hosting any Sha>={args.sha_threshold}:                  "
          f"{n_classes_with_sha:,}  "
          f"({100*n_classes_with_sha/n_classes:.2f}%)")
    print()
    for prime in ISOGENY_PRIMES:
        col_iso = f"has_{prime}_isog"
        col_sha = f"sha_has_{prime}sq"
        n_iso = int(df[col_iso].sum())
        n_sha = int(df[col_sha].sum())
        print(f"  Classes with rational {prime}-isogeny witness:  "
              f"{n_iso:>8,}  ({100*n_iso/n_classes:.2f}%)")
        print(f"  Classes with {prime}^2 | max_sha:                "
              f"{n_sha:>8,}  ({100*n_sha/n_classes:.2f}%)")

    # ---- HEADLINE TEST -------------------------------------------
    section("HEADLINE TEST: p-isogeny presence vs p^2 | max_sha")
    print("  For each prime p, partition classes by whether they have a rational")
    print("  p-isogeny (= some curve in the class has p | |T|). For each subset,")
    print("  compute the fraction where p^2 | max_sha (i.e. |Sha[p]| nontrivial).")
    print()

    headline_rows = []
    for prime in ISOGENY_PRIMES:
        col_iso = f"has_{prime}_isog"
        col_sha = f"sha_has_{prime}sq"

        with_iso = df[df[col_iso]]
        without_iso = df[~df[col_iso]]

        n_with_iso = len(with_iso)
        n_with_iso_sha = int(with_iso[col_sha].sum())
        rate_with = 100 * n_with_iso_sha / max(1, n_with_iso)

        n_without_iso = len(without_iso)
        n_without_iso_sha = int(without_iso[col_sha].sum())
        rate_without = 100 * n_without_iso_sha / max(1, n_without_iso)

        ratio = rate_with / max(1e-9, rate_without)

        # Chi-square test of independence
        cont = np.array([
            [n_with_iso_sha,    n_with_iso - n_with_iso_sha],
            [n_without_iso_sha, n_without_iso - n_without_iso_sha],
        ])
        try:
            chi2, pval, _, _ = stats.chi2_contingency(cont)
        except Exception:
            chi2, pval = float("nan"), float("nan")

        headline_rows.append((
            prime,
            n_with_iso, n_with_iso_sha, rate_with,
            n_without_iso, n_without_iso_sha, rate_without,
            ratio, chi2, pval,
        ))

    summary_table(
        headline_rows,
        ["p", "n_with_iso", "n_sha[p]", "rate%",
         "n_without_iso", "n_sha[p]", "rate%",
         "ratio", "chi^2", "p-value"],
        title="p-isogeny presence vs p^2 | max_sha",
        fmt=[">3d",
             ">11,d", ">10,d", ">7.3f",
             ">14,d", ">10,d", ">7.3f",
             ">8.2f", ">10.1f", ">10.2e"],
    )

    # ---- Class-size distribution ---------------------------------
    section("CLASS SIZE DISTRIBUTION")
    size_counts = df["n_curves"].value_counts().sort_index()
    size_rows = []
    for s, c in size_counts.items():
        sub = df[df["n_curves"] == s]
        max_max_sha = int(sub["max_sha"].max())
        avg_max_sha = float(sub["max_sha"].mean())
        n_nontriv = int((sub["n_sha_nontriv"] > 0).sum())
        size_rows.append((
            int(s), int(c), 100*int(c)/n_classes,
            avg_max_sha, max_max_sha,
            n_nontriv, 100*n_nontriv/int(c),
        ))
    summary_table(
        size_rows,
        ["size", "n_classes", "% all",
         "avg_max_sha", "max_max_sha",
         "n_w_sha+", "% w_sha+"],
        title="Class size vs Sha behaviour",
        fmt=[">5d", ">11,d", ">8.3f",
             ">13.2f", ">13d",
             ">10,d", ">10.2f"],
    )

    # ---- Top classes by max_sha, with full isogeny structure -----
    section(f"TOP {args.top_n} CLASSES BY max_sha")
    top = df.nlargest(args.top_n, "max_sha")
    top_rows = []
    for r in top.itertuples():
        top_rows.append((
            f"{int(r.conductor)}{r.iso}",
            int(r.n_curves),
            int(r.max_rank),
            r.isog_primes,
            int(r.max_sha),
            fmt_factorization(int(r.max_sha)),
        ))
    summary_table(
        top_rows,
        ["class", "size", "rk", "isog_p", "max_sha", "factored"],
        title=f"Top {args.top_n} by max_sha",
        fmt=["<14s", ">5d", ">5d", "<10s", ">10d", "<26s"],
    )

    # ---- Distinct max_sha values and their isog-prime profile ----
    section("MOST COMMON max_sha VALUES AND ASSOCIATED p-ISOG PROFILE")
    print("  For each max_sha value seen >= 100 times, report what fraction")
    print("  of classes carrying it have each p-isogeny witness.")
    print()

    sha_groups = df.groupby("max_sha").agg(
        n_classes_=("class", "count"),
        frac_2isog=("has_2_isog", "mean"),
        frac_3isog=("has_3_isog", "mean"),
        frac_5isog=("has_5_isog", "mean"),
        frac_7isog=("has_7_isog", "mean"),
    ).reset_index()
    sha_groups = sha_groups[sha_groups["n_classes_"] >= 50].sort_values(
        "max_sha"
    )
    common_rows = []
    for r in sha_groups.itertuples():
        common_rows.append((
            int(r.max_sha),
            fmt_factorization(int(r.max_sha)),
            int(r.n_classes_),
            100*float(r.frac_2isog),
            100*float(r.frac_3isog),
            100*float(r.frac_5isog),
            100*float(r.frac_7isog),
        ))
    summary_table(
        common_rows,
        ["max_sha", "factored", "n_classes",
         "%w/2isog", "%w/3isog", "%w/5isog", "%w/7isog"],
        title="Sha values vs isogeny-prime witness rates",
        fmt=[">8d", "<14s", ">11,d",
             ">10.2f", ">10.2f", ">10.2f", ">10.2f"],
    )

    # ---- 4. Save -------------------------------------------------
    step(4, 4, "Saving...")
    out_csv = output_dir / "isogeny_class_sha_catalog.csv"
    save_cols = ["class", "conductor", "iso", "n_curves", "max_rank",
                 "isog_primes",
                 "has_2_isog", "has_3_isog", "has_5_isog", "has_7_isog",
                 "max_sha", "mean_sha", "n_sha_nontriv",
                 "vp_2", "vp_3", "vp_5", "vp_7",
                 "sha_has_2sq", "sha_has_3sq",
                 "sha_has_5sq", "sha_has_7sq"]
    df[save_cols].sort_values(
        ["max_sha", "n_sha_nontriv"], ascending=[False, False]
    ).to_csv(out_csv, index=False)
    print(f"  CSV: {out_csv}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
