#!/usr/bin/env python3
"""investigate_rank2_sha4.py - deep dive on the 42 rank-2 curves with |Sha|=2.

In the full Cremona corpus (conductor <= 499998), exactly 42 curves of
rank 2 have Sha_an rounding to 4 (i.e., |Sha|=2). They are the ONLY
specimens with both a non-trivial Mordell-Weil lattice AND a non-trivial
Tate-Shafarevich group in the entire dataset. If they share structural
features -- isogeny-class clustering, conductor patterns, torsion
constraints, parity of conductor, Tamagawa structure -- that's evidence
of an arithmetic constraint producing rare Sha-at-rank-2.

Output:
    1. Per-curve listing of all 42 specimens (key invariants, shape)
    2. Conductor distribution + comparison to baseline (Sha=1)
    3. Tamagawa product distribution
    4. Torsion order distribution
    5. Isogeny class clustering analysis
    6. Conductor parity (even/odd) comparison
    7. Shape median comparison: specimens vs Sha=1 baseline
    8. Regulator comparison: how much does R grow beyond the BSD-required 2x?
    9. CSV: rank2_sha4_specimens.csv

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet

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

    banner("Rank-2 |Sha|=2 specimens - deep dive", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    step(1, 3, "Loading and filtering...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)

    rank2 = corpus[(corpus["rank"] == 2) & corpus["sha_an"].notna()].copy()
    rank2["sha_int"] = rank2["sha_an"].round().astype(int)

    baseline  = rank2[rank2["sha_int"] == 1]
    specimens = rank2[rank2["sha_int"] == 4].copy().sort_values("conductor")
    print(f"  Rank-2 Sha=1 baseline: {len(baseline):,}")
    print(f"  Rank-2 Sha=4 specimens: {len(specimens):,}")

    if len(specimens) == 0:
        print(C.fail("  No specimens; aborting."))
        return

    step(2, 3, "Inspecting specimens...")

    # =================================================================
    #  PER-CURVE LISTING
    # =================================================================
    section("THE SPECIMENS (sorted by conductor)")
    print(f"  {'label':<14} {'N':>7}  {'|T|':>3}  {'c':>4}  "
          f"{'R':>10}  {'lam_min':>8}  {'lam_max':>8}  "
          f"{'gini':>7}  {'aspect':>7}  {'ainvs'}")
    print("  " + "-" * 110)
    for row in specimens.itertuples():
        print(f"  {row.label:<14} {int(row.conductor):>7}  "
              f"{int(row.torsion_order):>3}  {int(row.tamagawa_product):>4}  "
              f"{row.regulator:>10.4f}  "
              f"{row.spec_lambda_min:>8.4f}  {row.spec_lambda_max:>8.4f}  "
              f"{row.spec_gini_norm:>7.4f}  {row.aspect_ratio:>7.4f}  "
              f"{row.ainvs}")

    # =================================================================
    #  STRUCTURAL PATTERNS
    # =================================================================
    section("1. CONDUCTOR DISTRIBUTION")
    print(f"  Specimens:  min={int(specimens['conductor'].min()):,}   "
          f"median={int(specimens['conductor'].median()):,}   "
          f"max={int(specimens['conductor'].max()):,}")
    print(f"  Baseline:   min={int(baseline['conductor'].min()):,}   "
          f"median={int(baseline['conductor'].median()):,}   "
          f"max={int(baseline['conductor'].max()):,}")
    print()
    print(C.info("  Conductor bin distribution:"))
    bins = [0, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000]
    spec_bins = pd.cut(specimens["conductor"], bins=bins).value_counts().sort_index()
    base_bins = pd.cut(baseline["conductor"], bins=bins).value_counts().sort_index()
    base_total = len(baseline)
    spec_total = len(specimens)
    rows = []
    for interval in spec_bins.index:
        s = int(spec_bins.get(interval, 0))
        b = int(base_bins.get(interval, 0))
        rows.append((
            str(interval),
            s, f"{100*s/spec_total:.1f}%",
            b, f"{100*b/base_total:.1f}%",
        ))
    summary_table(
        rows,
        ["bin", "Sha=4 n", "Sha=4 %", "Sha=1 n", "Sha=1 %"],
        title="Conductor distribution",
        fmt=["<22s", ">8d", ">10s", ">10,d", ">10s"],
    )

    section("2. TAMAGAWA PRODUCT DISTRIBUTION")
    tama_counts = specimens["tamagawa_product"].value_counts().sort_index()
    rows = []
    for tc, n in tama_counts.items():
        rows.append((
            int(tc), int(n), f"{100*n/spec_total:.1f}%",
            int((baseline['tamagawa_product'] == tc).sum()),
        ))
    summary_table(
        rows,
        ["Tamagawa c", "Sha=4 n", "Sha=4 %", "Sha=1 n"],
        title="Tamagawa products in specimens",
        fmt=[">10d", ">8d", ">10s", ">10,d"],
    )
    print(f"  Median c (specimens): {int(specimens['tamagawa_product'].median())}")
    print(f"  Median c (baseline):  {int(baseline['tamagawa_product'].median())}")
    print(f"  Mean c   (specimens): {specimens['tamagawa_product'].mean():.2f}")
    print(f"  Mean c   (baseline):  {baseline['tamagawa_product'].mean():.2f}")

    section("3. TORSION ORDER DISTRIBUTION")
    tors_counts = specimens["torsion_order"].value_counts().sort_index()
    for t, n in tors_counts.items():
        base_n = int((baseline["torsion_order"] == t).sum())
        spec_pct = 100 * n / spec_total
        base_pct = 100 * base_n / base_total
        print(f"  |T| = {int(t):<3d}  Sha=4: {int(n):>3} ({spec_pct:5.1f}%)   "
              f"Sha=1: {base_n:>8,d} ({base_pct:5.2f}%)")

    section("4. ISOGENY CLASS CLUSTERING")
    specimens["cremona_class"] = (specimens["conductor"].astype(str)
                                  + specimens["iso"])
    class_counts = specimens["cremona_class"].value_counts()
    multi = class_counts[class_counts > 1]
    print(f"  Distinct isogeny classes in 42 specimens: {len(class_counts)}")
    if len(multi) > 0:
        print(f"  Classes with 2+ specimens:")
        for cls, n in multi.items():
            members = specimens[specimens["cremona_class"] == cls]["label"].tolist()
            print(f"    {cls}: {n} curves -> {members}")
    else:
        print(C.warn("  No isogeny class has more than one Sha=4 specimen."))
        print("  All 42 specimens come from distinct isogeny classes.")

    section("5. CONDUCTOR PARITY")
    even_s = int((specimens["conductor"] % 2 == 0).sum())
    odd_s  = spec_total - even_s
    even_b = int((baseline["conductor"] % 2 == 0).sum())
    odd_b  = base_total - even_b
    print(f"  Sha=4: even {even_s:>3} ({100*even_s/spec_total:5.1f}%)   "
          f"odd {odd_s:>3} ({100*odd_s/spec_total:5.1f}%)")
    print(f"  Sha=1: even {even_b:>9,d} ({100*even_b/base_total:5.2f}%)   "
          f"odd {odd_b:>9,d} ({100*odd_b/base_total:5.2f}%)")

    section("6. SHAPE INVARIANT COMPARISON (medians)")
    rows = []
    for var in ["spec_gini_norm", "spec_entropy_norm", "spec_participation",
                "hermite_quotient", "aspect_ratio",
                "nt_diag_min", "nt_diag_max"]:
        if var not in specimens.columns or var not in baseline.columns:
            continue
        s_med = float(specimens[var].median())
        b_med = float(baseline[var].median())
        rows.append((var, s_med, b_med, s_med - b_med,
                     f"{(s_med/b_med - 1) * 100:+.1f}%"))
    summary_table(
        rows,
        ["variable", "Sha=4 med", "Sha=1 med", "delta", "rel"],
        title="Shape comparison",
        fmt=["<22s", ">10.4f", ">10.4f", ">+10.4f", ">8s"],
    )

    section("7. REGULATOR: BSD-EXPECTED vs ACTUAL")
    s_R = float(specimens["regulator"].median())
    b_R = float(baseline["regulator"].median())
    print(f"  BSD prediction: if Sha=4 alone caused the difference, R should grow by |Sha|=2.")
    print(f"  Sha=4 median R:   {s_R:.4f}")
    print(f"  Sha=1 median R:   {b_R:.4f}")
    print(f"  Actual ratio:     {s_R/b_R:.3f}")
    print(f"  BSD-only ratio:   2.000")
    print(f"  Excess factor:    {s_R / b_R / 2.0:.3f}")
    print()
    print(f"  (Excess > 1 means Sha=4 curves also have systematically bigger")
    print(f"   generators / other arithmetic complexity beyond the BSD bookkeeping.)")

    # =================================================================
    #  SAVE
    # =================================================================
    step(3, 3, "Saving...")
    out_csv = output_dir / "rank2_sha4_specimens.csv"
    save_cols = ["label", "conductor", "iso", "curve_number", "ainvs",
                 "torsion_order", "tamagawa_product",
                 "real_period", "L_value", "regulator", "sha_an",
                 "naive_h_max", "gens_coord_max_log10",
                 "spec_lambda_min", "spec_lambda_max",
                 "spec_gini_norm", "spec_entropy_norm",
                 "spec_participation", "hermite_quotient", "aspect_ratio",
                 "orthogonality_defect", "gram_matrix"]
    save_cols = [c for c in save_cols if c in specimens.columns]
    specimens[save_cols].to_csv(out_csv, index=False)
    print(f"  CSV: {out_csv}  ({len(specimens)} rows, {len(save_cols)} cols)")


if __name__ == "__main__":
    main()
