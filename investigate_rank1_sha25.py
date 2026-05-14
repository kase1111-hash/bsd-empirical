#!/usr/bin/env python3
"""investigate_rank1_sha25.py - deep dive on the 96 rank-1 curves with |Sha|=5.

The 96 specimens are the most extreme tractable Sha cohort in this corpus:
rank 1, Sha_an rounding to 25 (|Sha|=5), conductor <= 499,998. They are
0.006% of the rank-1 population.

THE KEY PREDICTIVE TEST: in rank-2 Sha=4 specimens, 97.6% had 2-torsion
(vs 28.7% baseline). The hypothesis -- |Sha|=p forces curves to carry
p-torsion -- predicts that the rank-1 Sha=25 specimens should show
5-torsion saturation. If they do, that's a real arithmetic pattern. If
they don't, the rank-2 result was incidental and we need a different
explanation.

Tests:
    1. Per-curve listing of all 96
    2. Conductor distribution (lower-bound effect again?)
    3. Torsion order distribution -- THE KEY TEST
    4. Tamagawa product (anti-correlation with Sha continues?)
    5. Isogeny class clustering (Sha invariant under isogeny here?)
    6. Conductor parity + divisibility by 5
    7. Height distributions: regulator + naive height
    8. BSD self-consistency check
    9. Three-way comparison: Sha=1 vs Sha=4 vs Sha=25 trajectory

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet
Output: lmfdb_ec_out/rank1_sha25_specimens.csv

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

    banner("Rank-1 |Sha|=5 specimens - deep dive", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load ---------------------------------------------------
    step(1, 3, "Loading and filtering...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)

    rank1 = corpus[(corpus["rank"] == 1) & corpus["sha_an"].notna()].copy()
    rank1["sha_int"] = rank1["sha_an"].round().astype(int)

    baseline  = rank1[rank1["sha_int"] == 1]
    sha4      = rank1[rank1["sha_int"] == 4]
    specimens = rank1[rank1["sha_int"] == 25].copy().sort_values("conductor")

    print(f"  Rank-1 Sha=1 baseline:    {len(baseline):,}")
    print(f"  Rank-1 Sha=4 cohort:      {len(sha4):,}")
    print(f"  Rank-1 Sha=25 specimens:  {len(specimens):,}")

    if len(specimens) == 0:
        print(C.fail("  No specimens; aborting."))
        return

    step(2, 3, "Inspecting specimens...")

    # ---- Per-curve listing ----------------------------------------
    section("THE 96 SPECIMENS (sorted by conductor)")
    print(f"  {'label':<14} {'N':>7}  {'|T|':>3}  {'c':>4}  "
          f"{'R':>9}  {'h_naive':>8}  {'logH':>8}  {'sha_an':>9}  ainvs")
    print("  " + "-" * 120)
    for row in specimens.itertuples():
        ainvs_print = row.ainvs
        if len(ainvs_print) > 60:
            ainvs_print = ainvs_print[:57] + "..."
        print(f"  {row.label:<14} {int(row.conductor):>7}  "
              f"{int(row.torsion_order):>3}  {int(row.tamagawa_product):>4}  "
              f"{row.regulator:>9.4f}  {row.naive_h_max:>8.4f}  "
              f"{row.gens_coord_max_log10:>8.4f}  "
              f"{row.sha_an:>9.4f}  {ainvs_print}")

    # ---- 1. Conductor distribution --------------------------------
    section("1. CONDUCTOR DISTRIBUTION")
    print(f"  Specimens: min={int(specimens['conductor'].min()):,}   "
          f"median={int(specimens['conductor'].median()):,}   "
          f"max={int(specimens['conductor'].max()):,}")
    print(f"  Baseline:  median={int(baseline['conductor'].median()):,}")
    print()
    bins = [0, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000]
    spec_bins = pd.cut(specimens["conductor"], bins=bins).value_counts().sort_index()
    base_bins = pd.cut(baseline["conductor"], bins=bins).value_counts().sort_index()
    rows = []
    for interval in spec_bins.index:
        s = int(spec_bins.get(interval, 0))
        b = int(base_bins.get(interval, 0))
        rows.append((
            str(interval),
            s, f"{100*s/len(specimens):.1f}%",
            b, f"{100*b/len(baseline):.1f}%",
        ))
    summary_table(
        rows,
        ["bin", "Sha=25 n", "Sha=25 %", "Sha=1 n", "Sha=1 %"],
        title="Conductor distribution",
        fmt=["<22s", ">9d", ">10s", ">11,d", ">10s"],
    )

    # ---- 2. TORSION - THE KEY TEST --------------------------------
    section("2. TORSION ORDER DISTRIBUTION  <<< KEY TEST: 5-torsion saturation? >>>")
    tors_counts = specimens["torsion_order"].value_counts().sort_index()
    base_tors_counts = baseline["torsion_order"].value_counts()
    rows = []
    for t, n in tors_counts.items():
        base_n = int(base_tors_counts.get(t, 0))
        spec_pct = 100 * n / len(specimens)
        base_pct = 100 * base_n / len(baseline)
        ratio = spec_pct / base_pct if base_pct > 0 else float("inf")
        rows.append((
            int(t), int(n), f"{spec_pct:.1f}%",
            int(base_n), f"{base_pct:.2f}%",
            f"{ratio:.2f}x" if np.isfinite(ratio) else "inf",
        ))
    summary_table(
        rows,
        ["|T|", "Sha=25 n", "Sha=25 %", "Sha=1 n", "Sha=1 %", "over/under"],
        title="Torsion order: Sha=25 specimens vs Sha=1 baseline",
        fmt=[">4d", ">9d", ">10s", ">11,d", ">10s", ">12s"],
    )

    # Direct test: divisibility by 5
    print()
    div5_spec = int((specimens["torsion_order"] % 5 == 0).sum())
    div5_base = int((baseline["torsion_order"] % 5 == 0).sum())
    print(C.info(f"  >>> Curves with 5 | |T(E)|:"))
    print(f"      Specimens: {div5_spec}/{len(specimens)}  "
          f"({100*div5_spec/len(specimens):.1f}%)")
    print(f"      Baseline:  {div5_base:,}/{len(baseline):,}  "
          f"({100*div5_base/len(baseline):.3f}%)")
    if div5_spec > 0 and div5_base > 0:
        ratio = (div5_spec/len(specimens)) / (div5_base/len(baseline))
        print(f"      Over-representation: {ratio:.1f}x")
    elif div5_spec == 0:
        print(C.warn(f"      NO specimen has 5-torsion. Hypothesis FAILS."))
    print()
    print(C.info(f"  Comparison: rank-2 Sha=4 had 97.6% 2-torsion vs 28.7% baseline (3.4x)"))

    # ---- 3. Tamagawa ----------------------------------------------
    section("3. TAMAGAWA PRODUCT DISTRIBUTION")
    print(f"  Median c (specimens): {int(specimens['tamagawa_product'].median())}")
    print(f"  Median c (baseline):  {int(baseline['tamagawa_product'].median())}")
    print(f"  Mean c   (specimens): {specimens['tamagawa_product'].mean():.2f}")
    print(f"  Mean c   (baseline):  {baseline['tamagawa_product'].mean():.2f}")
    print(f"  Ratio (baseline/specimens): "
          f"{baseline['tamagawa_product'].mean() / specimens['tamagawa_product'].mean():.2f}x")
    print()
    print(C.info(f"  Comparison: rank-2 Sha=4 had baseline/specimen mean c ratio = 7.5x"))

    # ---- 4. Isogeny class clustering ------------------------------
    section("4. ISOGENY CLASS CLUSTERING")
    specimens["cremona_class"] = (specimens["conductor"].astype(str)
                                  + specimens["iso"])
    class_counts = specimens["cremona_class"].value_counts()
    multi = class_counts[class_counts > 1]
    print(f"  Distinct isogeny classes: {len(class_counts)} / {len(specimens)} specimens")
    if len(multi) > 0:
        print(f"  Classes with 2+ specimens:")
        for cls, n in multi.items():
            members = specimens[specimens["cremona_class"] == cls]["label"].tolist()
            print(f"    {cls}: {n} -> {members}")
        print(C.info(f"  (Note: for rank 1, Sha-non-triviality typically IS preserved"))
        print(C.info(f"   under isogeny up to factors of 2, so clustering is expected here)"))
    else:
        print(f"  All specimens come from distinct isogeny classes.")

    # ---- 5. Conductor parity + divisibility -----------------------
    section("5. CONDUCTOR PROPERTIES")
    even_s = int((specimens["conductor"] % 2 == 0).sum())
    even_b = int((baseline["conductor"] % 2 == 0).sum())
    print(f"  Even conductor:")
    print(f"    Specimens: {even_s}/{len(specimens)}  "
          f"({100*even_s/len(specimens):.1f}%)")
    print(f"    Baseline:  {100*even_b/len(baseline):.2f}%")

    div5_n = int((specimens["conductor"] % 5 == 0).sum())
    div5_n_b = int((baseline["conductor"] % 5 == 0).sum())
    print(f"  Conductor divisible by 5:")
    print(f"    Specimens: {div5_n}/{len(specimens)}  "
          f"({100*div5_n/len(specimens):.1f}%)")
    print(f"    Baseline:  {100*div5_n_b/len(baseline):.2f}%")

    div25_n = int((specimens["conductor"] % 25 == 0).sum())
    div25_n_b = int((baseline["conductor"] % 25 == 0).sum())
    print(f"  Conductor divisible by 25:")
    print(f"    Specimens: {div25_n}/{len(specimens)}  "
          f"({100*div25_n/len(specimens):.1f}%)")
    print(f"    Baseline:  {100*div25_n_b/len(baseline):.2f}%")

    # ---- 6. Heights -----------------------------------------------
    section("6. HEIGHT DISTRIBUTIONS")
    rows = []
    for col, label in [
        ("regulator",            "Canonical height (= R)"),
        ("naive_h_max",          "Naive log-height"),
        ("gens_coord_max_log10", "Gen coord max log10"),
    ]:
        rows.append((
            label,
            float(specimens[col].min()),
            float(specimens[col].median()),
            float(specimens[col].max()),
            float(baseline[col].median()),
            float(specimens[col].median()) / float(baseline[col].median()),
        ))
    summary_table(
        rows,
        ["variable", "spec_min", "spec_med", "spec_max", "base_med", "ratio"],
        title="Specimens vs Baseline (height distributions)",
        fmt=["<24s", ">10.4f", ">10.4f", ">10.4f", ">10.4f", ">8.2f"],
    )

    # ---- 7. BSD self-consistency check ----------------------------
    section("7. BSD SELF-CONSISTENCY  (L = Omega * R * |Sha| * prod c / |T|^2)")
    specimens["sha_bsd"] = (specimens["L_value"] * specimens["torsion_order"]**2 /
                            (specimens["real_period"] * specimens["regulator"] *
                             specimens["tamagawa_product"]))
    err = (specimens["sha_bsd"] - specimens["sha_an"]).abs()
    rel_err = err / specimens["sha_an"]
    print(f"  Recomputed sha_bsd from BSD vs stored sha_an:")
    print(f"    median |delta|: {err.median():.2e}")
    print(f"    max    |delta|: {err.max():.2e}")
    print(f"    median rel err: {rel_err.median():.2e}")
    print(f"  Within rel err 1e-6:  {(rel_err < 1e-6).sum()} / {len(specimens)}")
    print(f"  Within rel err 1e-10: {(rel_err < 1e-10).sum()} / {len(specimens)}")

    # ---- 8. Three-way trajectory ----------------------------------
    section("8. THREE-WAY TRAJECTORY  Sha=1 -> Sha=4 -> Sha=25")
    def med(df, c): return float(df[c].median())
    rows = []
    for col, label in [
        ("regulator",            "regulator (R)"),
        ("naive_h_max",          "naive log-height"),
        ("gens_coord_max_log10", "gen coord log10"),
        ("conductor",            "conductor"),
        ("tamagawa_product",     "Tamagawa product"),
        ("torsion_order",        "|torsion|"),
    ]:
        b = med(baseline, col)
        m = med(sha4, col)
        s = med(specimens, col)
        rows.append((label, b, m, s, s - b, s / b if b != 0 else float("nan")))
    summary_table(
        rows,
        ["variable", "Sha=1", "Sha=4", "Sha=25", "delta(25-1)", "ratio(25/1)"],
        title="Median trajectory across Sha cohorts",
        fmt=["<22s", ">12.4f", ">12.4f", ">12.4f", ">+14.4f", ">12.4f"],
    )

    # ---- Save -----------------------------------------------------
    step(3, 3, "Saving...")
    out_csv = output_dir / "rank1_sha25_specimens.csv"
    save_cols = ["label", "conductor", "iso", "curve_number", "ainvs",
                 "torsion_order", "tamagawa_product",
                 "real_period", "L_value", "regulator", "sha_an",
                 "naive_h_max", "gens_coord_max_log10",
                 "generators_str"]
    save_cols = [c for c in save_cols if c in specimens.columns]
    specimens[save_cols].to_csv(out_csv, index=False)
    print(f"  CSV: {out_csv}  ({len(specimens)} rows, {len(save_cols)} cols)")


if __name__ == "__main__":
    main()
