#!/usr/bin/env python3
"""analyze_rank1_height_vs_sha.py - canonical height vs Sha for rank-1 curves.

Rank-1 curves have a single Z-generator P with canonical height ^h(P) equal
to the regulator R_E. BSD: L'(E,1) = Omega * R * |Sha| * prod(c_p) / |tors|^2,
so R grows with Sha by simple bookkeeping. The non-trivial question is
whether the DISTRIBUTION of R (and related quantities) carries extra
structure -- beyond what BSD bookkeeping requires.

We test, across Sha cohorts (1, 4, 9, 16, 25, ...):
    - regulator (= NT canonical height of the generator)
    - log(regulator)             closer to normal, often more informative
    - naive log-height of the generator (max(|X|, Z^2))
    - generator coord max (log10) -- raw arithmetic complexity
    - conductor                  context covariate
    - log(conductor)

For each, runs K-S 2-sample tests against the Sha=1 baseline and reports
the median shift. Rank-1 has ~1.5M curves with substantial Sha variation
(up to Sha=5625 in the corpus), so the comparisons have real statistical
power -- unlike the rank-2 Sha=4 case with only 42 specimens.

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet
Output: lmfdb_ec_out/rank1_height_by_sha.csv

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

# Perfect-square Sha values worth inspecting (Cassels-Tate forces square)
COHORTS = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225,
           256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 729, 841, 1024]

MIN_N_FOR_TABLE = 10
MIN_N_FOR_KS    = 30

QUANTITIES = [
    ("regulator",            "Regulator (= NT canonical height)"),
    ("log_regulator",        "log(Regulator)"),
    ("naive_h_max",          "Naive log-height of generator"),
    ("gens_coord_max_log10", "Generator coord max (log10)"),
    ("conductor",            "Conductor"),
    ("log_conductor",        "log(Conductor)"),
    ("tamagawa_product",     "Tamagawa product"),
]


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

    banner("Rank-1 canonical height vs Sha", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load --------------------------------------------------
    step(1, 4, "Loading and filtering...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    print(f"  Loaded {len(corpus):,} rows")

    rank1 = corpus[
        (corpus["rank"] == 1)
        & corpus["regulator"].notna()
        & corpus["sha_an"].notna()
    ].copy()
    print(f"  Rank-1 with regulator + sha_an: {len(rank1):,}")

    rank1["sha_int"] = rank1["sha_an"].round().astype(int)
    rank1["log_regulator"] = np.log(rank1["regulator"].clip(lower=1e-300))
    rank1["log_conductor"] = np.log(rank1["conductor"].astype(float))

    # ---- 2. Sha distribution --------------------------------------
    step(2, 4, "Sha distribution...")
    section("SHA DISTRIBUTION FOR RANK-1")
    sha_counts = rank1["sha_int"].value_counts().sort_index()
    rows = []
    shown = 0
    for sha_val, count in sha_counts.items():
        if count < MIN_N_FOR_TABLE:
            continue
        sq = int(round(np.sqrt(sha_val)))
        is_sq = (sq * sq == sha_val)
        tag = f"|Sha|={sq}" if is_sq else "NON-SQUARE"
        rows.append((int(sha_val), int(count),
                     f"{100*count/len(rank1):.3f}%", tag))
        shown += 1
        if shown >= 25:
            break
    summary_table(
        rows,
        ["Sha_an", "count", "pct", "interp"],
        title=f"Top cohorts with n >= {MIN_N_FOR_TABLE}",
        fmt=[">6d", ">12,d", ">10s", "<14s"],
    )
    non_trivial = int((rank1["sha_int"] >= 4).sum())
    print(f"\n  Non-trivial Sha (>= 4): {non_trivial:,} "
          f"({100*non_trivial/len(rank1):.2f}%)")
    print(f"  Max Sha_an observed:    {int(rank1['sha_int'].max())}")

    # ---- 3. Medians by cohort -------------------------------------
    step(3, 4, "Medians by Sha cohort...")
    section("MEDIAN QUANTITIES BY SHA COHORT")
    rows = []
    for sha in COHORTS:
        cohort = rank1[rank1["sha_int"] == sha]
        if len(cohort) < MIN_N_FOR_TABLE:
            continue
        rows.append((
            int(sha), int(len(cohort)),
            float(cohort["regulator"].median()),
            float(cohort["log_regulator"].median()),
            float(cohort["naive_h_max"].median()),
            float(cohort["gens_coord_max_log10"].median()),
            float(cohort["conductor"].median()),
            float(cohort["tamagawa_product"].median()),
        ))
    summary_table(
        rows,
        ["Sha", "n", "med_R", "med_logR", "med_h_naive",
         "med_coord", "med_N", "med_c"],
        title="Rank-1 medians by Sha cohort",
        fmt=[">5d", ">10,d", ">10.4f", ">10.4f",
             ">12.4f", ">10.4f", ">10.0f", ">8.1f"],
    )

    # ---- K-S tests -------------------------------------------------
    section("K-S TESTS  vs  Sha=1 BASELINE")
    baseline = rank1[rank1["sha_int"] == 1]
    if len(baseline) < MIN_N_FOR_KS:
        print(C.fail(f"  Baseline too small (n={len(baseline)})"))
        return
    print(f"  Baseline n = {len(baseline):,}")
    print()
    print(f"  *** p<0.001    ** p<0.01    * p<0.05    n.s. otherwise")
    print(f"  delta_med = median(cohort) - median(baseline)")
    print(f"  BSD predicts log(R) shifts by ~log(|Sha|) -- check delta_logR")
    print()

    for col, label in QUANTITIES:
        if col not in rank1.columns:
            continue
        baseline_vals = baseline[col].dropna().values
        if len(baseline_vals) < MIN_N_FOR_KS:
            continue
        baseline_med = float(np.median(baseline_vals))

        rows = []
        for sha in COHORTS[1:]:
            cohort = rank1[rank1["sha_int"] == sha]
            if len(cohort) < MIN_N_FOR_KS:
                continue
            vals = cohort[col].dropna().values
            if len(vals) < MIN_N_FOR_KS:
                continue
            ks_stat, p_val = stats.ks_2samp(baseline_vals, vals)
            delta = float(np.median(vals)) - baseline_med
            if p_val < 0.001: sig = "***"
            elif p_val < 0.01: sig = " **"
            elif p_val < 0.05: sig = " * "
            else:              sig = "n.s."
            rows.append((int(sha), int(len(vals)),
                         float(ks_stat), p_val, delta, sig))
        if rows:
            summary_table(
                rows,
                ["Sha", "n", "KS_stat", "p_value", "delta_med", "sig"],
                title=f"K-S: {label}",
                fmt=[">5d", ">10,d", ">10.4f", ">12.3e", ">+14.5f", ">6s"],
            )
            print()

    # ---- BSD bookkeeping check ------------------------------------
    section("BSD BOOKKEEPING vs OBSERVED log(R) SHIFTS")
    print("  BSD says: log(R) should shift by log(|Sha|) across Sha cohorts")
    print("  if all other invariants were equal. Excess shift = extra structure.\n")
    rows_bsd = []
    baseline_log_R = float(baseline["log_regulator"].median())
    for sha in COHORTS[1:]:
        cohort = rank1[rank1["sha_int"] == sha]
        if len(cohort) < MIN_N_FOR_KS:
            continue
        obs_shift = float(cohort["log_regulator"].median()) - baseline_log_R
        sq = int(round(np.sqrt(sha)))
        bsd_expected = np.log(sq)   # log(|Sha|) since R ~ |Sha| in BSD
        excess = obs_shift - bsd_expected
        rows_bsd.append((
            int(sha), sq, int(len(cohort)),
            obs_shift, bsd_expected, excess,
        ))
    summary_table(
        rows_bsd,
        ["Sha", "|Sha|", "n", "obs_logR_shift",
         "BSD_predicted (log|Sha|)", "excess"],
        title="Observed log(R) shift vs BSD-required log|Sha|",
        fmt=[">5d", ">5d", ">10,d", ">+16.4f", ">25.4f", ">+10.4f"],
    )

    # ---- 4. Save CSV ----------------------------------------------
    step(4, 4, "Saving...")
    csv_rows = []
    for sha in sorted(rank1["sha_int"].unique()):
        cohort = rank1[rank1["sha_int"] == sha]
        if len(cohort) < 5:
            continue
        csv_rows.append({
            "sha_an_int":         int(sha),
            "n":                  int(len(cohort)),
            "med_regulator":      float(cohort["regulator"].median()),
            "p25_regulator":      float(np.percentile(cohort["regulator"], 25)),
            "p75_regulator":      float(np.percentile(cohort["regulator"], 75)),
            "med_log_regulator":  float(cohort["log_regulator"].median()),
            "med_naive_h":        float(cohort["naive_h_max"].median()),
            "med_coord_log10":    float(cohort["gens_coord_max_log10"].median()),
            "med_conductor":      float(cohort["conductor"].median()),
            "med_log_conductor":  float(cohort["log_conductor"].median()),
            "med_tamagawa":       float(cohort["tamagawa_product"].median()),
        })
    out_csv = output_dir / "rank1_height_by_sha.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"  CSV: {out_csv}  ({len(csv_rows)} cohorts)")


if __name__ == "__main__":
    main()
