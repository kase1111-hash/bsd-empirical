#!/usr/bin/env python3
"""investigate_rank1_sha4.py - the bridge cohort: 18,668 rank-1 curves with |Sha|=2.

This cohort is the bridge between the two specimens already analyzed:
    rank-2 Sha=4 (n=42)     : showed 97.6% 2-torsion saturation
    rank-1 Sha=25 (n=96)    : showed 0% 5-torsion (trivial-torsion saturation)
    rank-1 Sha=4 (n=18,668) : THIS SCRIPT  --  decides the 2-torsion question

If rank-1 Sha=4 shows strong 2-torsion over-representation (e.g. 70%+ when
baseline is 38.1%), the 2-torsion correlation is rank-independent and the
interpretation is "2-torsion makes 2-descent tractable, which detects Sha[2]".
If rank-1 Sha=4 looks baseline (~38%), the rank-2 result was rank-specific
and we need a different explanation.

The script also re-confirms (or refutes) the Sha-Tamagawa anti-correlation
at this larger sample size, where the ~7.5x ratio observed in two smaller
cohorts becomes a much sharper claim.

Cross-cohort comparison rows are added at the end so the three non-trivial
Sha cohorts can be read side-by-side.

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet
Output: lmfdb_ec_out/rank1_sha4_cohort_summary.csv

Author: Kase Branham - Independent Researcher
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, C, debug,
)


DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_spectrum.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")


def cohort(df, rank, sha_int):
    return df[(df["rank"] == rank) & (df["sha_int"] == sha_int)]


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

    banner("Rank-1 |Sha|=2 cohort - bridge investigation", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    step(1, 3, "Loading and filtering...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    corpus = corpus[corpus["sha_an"].notna()].copy()
    corpus["sha_int"] = corpus["sha_an"].round().astype(int)

    r1_base = cohort(corpus, 1, 1)
    r1_4    = cohort(corpus, 1, 4)
    r1_25   = cohort(corpus, 1, 25)
    r2_base = cohort(corpus, 2, 1)
    r2_4    = cohort(corpus, 2, 4)

    print(f"  rank-1 Sha=1 baseline:  {len(r1_base):,}")
    print(f"  rank-1 Sha=4 cohort:    {len(r1_4):,}     <-- primary subject")
    print(f"  rank-1 Sha=25 cohort:   {len(r1_25):,}")
    print(f"  rank-2 Sha=1 baseline:  {len(r2_base):,}")
    print(f"  rank-2 Sha=4 cohort:    {len(r2_4):,}")

    if len(r1_4) < 100:
        print(C.fail("  Primary subject too small."))
        return

    step(2, 3, "Running tests...")

    # =================================================================
    #  1.  TORSION SATURATION TEST
    # =================================================================
    section("1. TORSION ORDER  <<< KEY TEST: rank-1 Sha=4 -> 2-torsion saturation? >>>")

    print(C.info("  Distribution within rank-1 Sha=4 cohort vs rank-1 baseline:"))
    rows = []
    all_tors = sorted(set(r1_4["torsion_order"].unique())
                      | set(r1_base["torsion_order"].unique()))
    for t in all_tors:
        s_n = int((r1_4["torsion_order"] == t).sum())
        b_n = int((r1_base["torsion_order"] == t).sum())
        if s_n == 0 and b_n == 0:
            continue
        s_pct = 100 * s_n / len(r1_4)
        b_pct = 100 * b_n / len(r1_base)
        ratio = s_pct / b_pct if b_pct > 0 else float("inf")
        rows.append((int(t), s_n, f"{s_pct:.2f}%",
                     b_n, f"{b_pct:.2f}%",
                     f"{ratio:.2f}x" if np.isfinite(ratio) else "inf"))
    summary_table(
        rows,
        ["|T|", "Sha=4 n", "Sha=4 %", "Sha=1 n", "Sha=1 %", "over/under"],
        title="rank-1 torsion: Sha=4 vs Sha=1",
        fmt=[">4d", ">10,d", ">10s", ">12,d", ">10s", ">12s"],
    )

    div2_4    = int((r1_4["torsion_order"] % 2 == 0).sum())
    div2_base = int((r1_base["torsion_order"] % 2 == 0).sum())
    print()
    print(C.info("  Curves with 2 | |T(E)|:"))
    print(f"    rank-1 Sha=4:  {div2_4:,}/{len(r1_4):,}  "
          f"({100*div2_4/len(r1_4):.2f}%)")
    print(f"    rank-1 Sha=1:  {div2_base:,}/{len(r1_base):,}  "
          f"({100*div2_base/len(r1_base):.2f}%)")
    ratio = (div2_4/len(r1_4)) / (div2_base/len(r1_base))
    print(f"    Over-representation: {C.ok(f'{ratio:.2f}x')}")
    print()
    print(C.info("  Cross-cohort 2-torsion picture:"))
    for name, c in [("rank-1 Sha=1  baseline", r1_base),
                    ("rank-1 Sha=4  primary",  r1_4),
                    ("rank-1 Sha=25 cohort",   r1_25),
                    ("rank-2 Sha=1  baseline", r2_base),
                    ("rank-2 Sha=4  cohort",   r2_4)]:
        if len(c) == 0:
            continue
        pct = 100 * (c["torsion_order"] % 2 == 0).sum() / len(c)
        bar = "#" * int(pct / 2)
        print(f"    {name:<24}  n={len(c):>9,d}   |T| even: {pct:>5.1f}%  {bar}")

    # =================================================================
    #  2.  TAMAGAWA ANTI-CORRELATION
    # =================================================================
    section("2. SHA-TAMAGAWA ANTI-CORRELATION")
    print(C.info("  Cross-cohort Tamagawa mean and median:"))
    rows = []
    for name, c, base in [
        ("rank-1 Sha=1  baseline", r1_base, None),
        ("rank-1 Sha=4  primary",  r1_4,    r1_base),
        ("rank-1 Sha=25 cohort",   r1_25,   r1_base),
        ("rank-2 Sha=1  baseline", r2_base, None),
        ("rank-2 Sha=4  cohort",   r2_4,    r2_base),
    ]:
        if len(c) == 0:
            continue
        med = float(c["tamagawa_product"].median())
        mn  = float(c["tamagawa_product"].mean())
        if base is not None and len(base) > 0:
            base_mn = float(base["tamagawa_product"].mean())
            ratio = base_mn / mn if mn > 0 else float("nan")
            rows.append((name, int(len(c)), med, mn, f"{ratio:.2f}x"))
        else:
            rows.append((name, int(len(c)), med, mn, "(baseline)"))
    summary_table(
        rows,
        ["cohort", "n", "med c", "mean c", "base_mean / c_mean"],
        title="Tamagawa product across cohorts",
        fmt=["<28s", ">10,d", ">8.1f", ">10.2f", ">22s"],
    )

    # =================================================================
    #  3.  CONDUCTOR DISTRIBUTION + LOWER-BOUND TEST
    # =================================================================
    section("3. CONDUCTOR DISTRIBUTION")
    print(C.info("  Conductor range per cohort:"))
    for name, c in [("rank-1 Sha=1",  r1_base),
                    ("rank-1 Sha=4",  r1_4),
                    ("rank-1 Sha=25", r1_25),
                    ("rank-2 Sha=4",  r2_4)]:
        if len(c) == 0:
            continue
        print(f"    {name:<16}  n={len(c):>9,d}   "
              f"min={int(c['conductor'].min()):>7,d}  "
              f"median={int(c['conductor'].median()):>7,d}  "
              f"max={int(c['conductor'].max()):>7,d}")

    print()
    print(C.info("  Conductor parity in rank-1 Sha=4 vs baseline:"))
    even_4 = int((r1_4["conductor"] % 2 == 0).sum())
    even_b = int((r1_base["conductor"] % 2 == 0).sum())
    print(f"    Sha=4: {even_4:,}/{len(r1_4):,} even  "
          f"({100*even_4/len(r1_4):.2f}%)")
    print(f"    Sha=1: {even_b:,}/{len(r1_base):,} even  "
          f"({100*even_b/len(r1_base):.2f}%)")

    bins = [0, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000]
    s_bins = pd.cut(r1_4["conductor"], bins=bins).value_counts().sort_index()
    b_bins = pd.cut(r1_base["conductor"], bins=bins).value_counts().sort_index()
    rows = []
    for interval in s_bins.index:
        s = int(s_bins.get(interval, 0))
        b = int(b_bins.get(interval, 0))
        rows.append((
            str(interval),
            s, f"{100*s/len(r1_4):.2f}%",
            b, f"{100*b/len(r1_base):.2f}%",
        ))
    summary_table(
        rows,
        ["bin", "Sha=4 n", "Sha=4 %", "Sha=1 n", "Sha=1 %"],
        title="Rank-1 Sha=4 vs baseline conductor bins",
        fmt=["<22s", ">10,d", ">10s", ">12,d", ">10s"],
    )

    # =================================================================
    #  4.  ISOGENY CLASS CLUSTERING
    # =================================================================
    section("4. ISOGENY CLASS CLUSTERING (rank-1 Sha=4 primary)")
    r1_4 = r1_4.copy()
    r1_4["cremona_class"] = (r1_4["conductor"].astype(str) + r1_4["iso"])
    class_counts = r1_4["cremona_class"].value_counts()
    multi = class_counts[class_counts > 1]
    print(f"  Distinct classes: {len(class_counts):,} / {len(r1_4):,} specimens")
    print(f"  Classes with multi-membership (specimen count by class size):")
    size_dist = multi.value_counts().sort_index()
    for size, n_classes in size_dist.items():
        print(f"    {int(size)} curves/class: {int(n_classes):,} classes "
              f"({int(size * n_classes):,} curves)")
    if len(multi) > 0:
        print(f"  Largest cluster: {int(multi.max())} curves "
              f"in class {multi.idxmax()}")
    singletons = (class_counts == 1).sum()
    print(f"  Singleton classes: {int(singletons):,} "
          f"({100*singletons/len(class_counts):.2f}%)")

    # =================================================================
    #  5.  HEIGHT DISTRIBUTIONS
    # =================================================================
    section("5. HEIGHT DISTRIBUTIONS")
    rows = []
    for col, label in [
        ("regulator",            "Canonical height (R)"),
        ("naive_h_max",          "Naive log-height"),
        ("gens_coord_max_log10", "Gen coord max log10"),
    ]:
        s_med = float(r1_4[col].median())
        b_med = float(r1_base[col].median())
        rows.append((
            label,
            float(r1_4[col].min()),
            s_med,
            float(r1_4[col].max()),
            b_med,
            s_med / b_med if b_med > 0 else float("nan"),
        ))
    summary_table(
        rows,
        ["variable", "spec_min", "spec_med", "spec_max", "base_med", "ratio"],
        title="Rank-1 Sha=4 vs baseline heights",
        fmt=["<24s", ">10.4f", ">10.4f", ">12.4f", ">10.4f", ">8.3f"],
    )

    # =================================================================
    #  6.  BSD SELF-CONSISTENCY (sanity)
    # =================================================================
    section("6. BSD SELF-CONSISTENCY  (recompute Sha from L,Omega,R,c,T)")
    r1_4 = r1_4.copy()
    r1_4["sha_bsd"] = (r1_4["L_value"] * r1_4["torsion_order"]**2 /
                      (r1_4["real_period"] * r1_4["regulator"] *
                       r1_4["tamagawa_product"]))
    err = (r1_4["sha_bsd"] - r1_4["sha_an"]).abs()
    rel_err = err / r1_4["sha_an"]
    print(f"  median |sha_bsd - sha_an|:    {err.median():.3e}")
    print(f"  max    |sha_bsd - sha_an|:    {err.max():.3e}")
    print(f"  Within relative err 1e-6:     "
          f"{int((rel_err < 1e-6).sum()):,} / {len(r1_4):,}")

    # =================================================================
    #  7.  HEADLINE CROSS-COHORT COMPARISON
    # =================================================================
    section("7. CROSS-COHORT HEADLINE COMPARISON")

    def stats(c, col):
        if len(c) == 0:
            return None, None
        return float(c[col].median()), float(c[col].mean())

    cohorts = [
        ("rank-1 Sha=1",  r1_base),
        ("rank-1 Sha=4",  r1_4),
        ("rank-1 Sha=25", r1_25),
        ("rank-2 Sha=1",  r2_base),
        ("rank-2 Sha=4",  r2_4),
    ]
    rows = []
    for name, c in cohorts:
        if len(c) == 0:
            continue
        even2 = 100 * (c["torsion_order"] % 2 == 0).sum() / len(c)
        rows.append((
            name, int(len(c)),
            float(c["tamagawa_product"].median()),
            float(c["tamagawa_product"].mean()),
            even2,
            float(c["regulator"].median()),
            float(c["conductor"].median()),
        ))
    summary_table(
        rows,
        ["cohort", "n", "med_c", "mean_c", "%|T| even",
         "med_R", "med_N"],
        title="All cohorts side-by-side",
        fmt=["<18s", ">10,d", ">8.1f", ">10.2f",
             ">11.2f", ">10.4f", ">12.0f"],
    )

    # =================================================================
    #  SAVE
    # =================================================================
    step(3, 3, "Saving cohort summary CSV...")
    csv_rows = []
    for name, c in cohorts:
        if len(c) == 0:
            continue
        csv_rows.append({
            "cohort":              name,
            "n":                   int(len(c)),
            "med_tamagawa":        float(c["tamagawa_product"].median()),
            "mean_tamagawa":       float(c["tamagawa_product"].mean()),
            "pct_torsion_even":    float(100 * (c["torsion_order"] % 2 == 0).sum() / len(c)),
            "pct_torsion_trivial": float(100 * (c["torsion_order"] == 1).sum() / len(c)),
            "med_regulator":       float(c["regulator"].median()),
            "med_naive_h":         float(c["naive_h_max"].median()),
            "med_coord_log10":     float(c["gens_coord_max_log10"].median()),
            "med_conductor":       float(c["conductor"].median()),
            "min_conductor":       float(c["conductor"].min()),
        })
    out_csv = output_dir / "rank1_sha4_cohort_summary.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"  CSV: {out_csv}  ({len(csv_rows)} cohorts)")


if __name__ == "__main__":
    main()
