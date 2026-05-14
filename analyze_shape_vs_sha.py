#!/usr/bin/env python3
"""analyze_shape_vs_sha.py - MW lattice shape distributions stratified by Sha.

For each rank cohort (2, 3, 4), groups curves by the analytic order of the
Tate-Shafarevich group (rounded to nearest integer, conjecturally a perfect
square) and compares the SHAPE of the Mordell-Weil lattice across cohorts.

The point: BSD already forces a relationship between Sha and the regulator
(det of the height pairing matrix). So R ~ Sha is built in. What is NOT
forced by BSD is whether the shape of the lattice -- spec_gini_norm,
spec_entropy_norm, hermite_quotient, etc. -- carries information about
Sha. These are rotation invariants of the Gram matrix; they're independent
of the scale that BSD fixes.

If shape distributions differ by Sha cohort, that's structural information
about the Tate-Shafarevich group not captured by any existing formula.

For each (rank, Sha_an) cohort with n>=30, runs Kolmogorov-Smirnov tests
of the shape distribution against the Sha=1 baseline. Significance levels
indicate whether the cohort distribution is genuinely different.

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet
Output: per-rank cohort CSVs in lmfdb_ec_out/

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


# =====================================================================
#  CONFIG
# =====================================================================

DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_spectrum.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

# Sha_an cohorts to inspect. Conjecturally Sha is a perfect square,
# so canonical values are 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, ...
COHORTS = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]

MIN_N_FOR_TABLE = 10
MIN_N_FOR_KS    = 30

# Shape invariants to test
SHAPE_VARS = [
    ("spec_gini_norm",     "Gini (normalized)"),
    ("spec_entropy_norm",  "Entropy (normalized)"),
    ("spec_participation", "Participation ratio"),
    ("hermite_quotient",   "Hermite quotient"),
]


# =====================================================================
#  PER-RANK ANALYSIS
# =====================================================================

def analyze_rank(eligible, rank, output_dir):
    sub = eligible[eligible["rank"] == rank].copy()
    if len(sub) < 100:
        print(C.warn(f"  rank {rank}: only {len(sub)} curves, skipping"))
        return

    section(f"RANK {rank}  (n = {len(sub):,})")

    # Round sha_an to nearest integer for cohort assignment.
    # Cassels-Tate forces |Sha| to be a perfect square; values like 3.99999
    # round to 4, which corresponds to |Sha|=2.
    sub["sha_int"] = sub["sha_an"].round().astype(int)

    # ── Sha value distribution ─────────────────────────────────────
    print("  Sha_an distribution (rounded; conjecturally a perfect square):")
    sha_counts = sub["sha_int"].value_counts().sort_index()
    shown = 0
    rows = []
    for sha_val, count in sha_counts.items():
        if count >= MIN_N_FOR_TABLE:
            sqrt_sha = int(round(np.sqrt(sha_val)))
            is_square = (sqrt_sha * sqrt_sha == sha_val)
            tag = f"|Sha|={sqrt_sha}" if is_square else "NON-SQUARE"
            rows.append((int(sha_val), int(count),
                         f"{100*count/len(sub):.3f}%", tag))
            shown += 1
            if shown >= 12:
                break
    summary_table(
        rows, ["Sha_an", "count", "pct", "interp"],
        title=f"Sha_an cohorts with n >= {MIN_N_FOR_TABLE}",
        fmt=[">6d", ">10,d", ">10s", "<14s"],
    )
    n_outside = len(sub) - sum(r[1] for r in rows)
    if n_outside:
        print(f"  ({n_outside:,} curves in cohorts with n<{MIN_N_FOR_TABLE}, not shown)")

    # ── Shape stats by Sha cohort ──────────────────────────────────
    section_rows = []
    for sha_val in COHORTS:
        cohort = sub[sub["sha_int"] == sha_val]
        if len(cohort) < MIN_N_FOR_TABLE:
            continue
        section_rows.append((
            int(sha_val), int(len(cohort)),
            float(cohort["spec_gini_norm"].median()),
            float(cohort["spec_entropy_norm"].median()),
            float(cohort["spec_participation"].median()),
            float(cohort["hermite_quotient"].median()),
            float(cohort["regulator"].median()),
        ))
    summary_table(
        section_rows,
        ["Sha", "n", "med_gini", "med_H_norm", "med_PR",
         "med_hermite", "med_reg"],
        title=f"Rank {rank}: shape medians by Sha cohort",
        fmt=[">5d", ">10,d", ">10.4f", ">12.4f", ">10.4f",
             ">12.4f", ">14.4f"],
    )

    # ── K-S tests per shape variable ───────────────────────────────
    baseline = sub[sub["sha_int"] == 1]
    if len(baseline) < MIN_N_FOR_KS:
        print(C.warn(f"  rank {rank}: Sha=1 baseline n={len(baseline)} too small"))
        return

    print()
    print(C.info(f"  K-S TESTS  vs  Sha=1 baseline  (n_baseline = {len(baseline):,})"))
    print(C.info(f"  *** p<0.001    ** p<0.01    * p<0.05    (n.s. otherwise)"))
    print(C.info(f"  delta_med = median(cohort) - median(baseline)"))
    print()

    for var, var_label in SHAPE_VARS:
        baseline_vals = baseline[var].dropna().values
        if len(baseline_vals) < MIN_N_FOR_KS:
            continue
        baseline_med = float(np.median(baseline_vals))
        rows_ks = []
        for sha_val in COHORTS[1:]:  # skip Sha=1 itself
            cohort = sub[sub["sha_int"] == sha_val]
            if len(cohort) < MIN_N_FOR_KS:
                continue
            cohort_vals = cohort[var].dropna().values
            if len(cohort_vals) < MIN_N_FOR_KS:
                continue
            ks_stat, p_value = stats.ks_2samp(baseline_vals, cohort_vals)
            delta = float(np.median(cohort_vals)) - baseline_med
            if p_value < 0.001: flag = "***"
            elif p_value < 0.01: flag = " **"
            elif p_value < 0.05: flag = " * "
            else:                flag = "n.s."
            rows_ks.append((
                int(sha_val), int(len(cohort_vals)),
                float(ks_stat), p_value, delta, flag,
            ))
        if rows_ks:
            summary_table(
                rows_ks,
                ["Sha", "n", "KS_stat", "p_value", "delta_med", "sig"],
                title=f"K-S: {var_label}",
                fmt=[">5d", ">10,d", ">10.4f", ">12.3e", ">+12.5f", ">6s"],
            )
            print()

    # ── Save full cohort summary CSV ───────────────────────────────
    csv_rows = []
    for sha_val in sorted(sub["sha_int"].unique()):
        cohort = sub[sub["sha_int"] == sha_val]
        if len(cohort) < 5:
            continue
        csv_rows.append({
            "rank":             int(rank),
            "sha_an_int":       int(sha_val),
            "n":                int(len(cohort)),
            "med_gini_norm":    float(cohort["spec_gini_norm"].median()),
            "p25_gini_norm":    float(np.percentile(cohort["spec_gini_norm"], 25)),
            "p75_gini_norm":    float(np.percentile(cohort["spec_gini_norm"], 75)),
            "med_entropy_norm": float(cohort["spec_entropy_norm"].median()),
            "med_participation": float(cohort["spec_participation"].median()),
            "med_hermite":      float(cohort["hermite_quotient"].median()),
            "med_aspect":       float(cohort["aspect_ratio"].median()),
            "med_regulator":    float(cohort["regulator"].median()),
            "med_lambda_min":   float(cohort["spec_lambda_min"].median()),
            "med_lambda_max":   float(cohort["spec_lambda_max"].median()),
        })
    df_out = pd.DataFrame(csv_rows)
    out_csv = output_dir / f"shape_by_sha_rank{rank}.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"  {C.info('CSV:')} {out_csv}  ({len(df_out)} cohorts)")


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
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("MW Lattice Shape vs Sha (Tate-Shafarevich)", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
        "Preset":     args.preset,
    })

    step(1, 3, "Loading corpus...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    print(f"  Loaded {len(corpus):,} rows")

    eligible = corpus[
        corpus["spec_gini_norm"].notna()
        & corpus["sha_an"].notna()
        & (corpus["rank"] >= 2)
    ]
    print(f"  Eligible (rank>=2, shape + Sha present): {len(eligible):,}")

    step(2, 3, "Analyzing by rank...")
    with Timer("Analysis"):
        for r in sorted(eligible["rank"].unique()):
            analyze_rank(eligible, int(r), output_dir)

    step(3, 3, "Done")
    print(f"  See per-rank CSVs in {output_dir}/")


if __name__ == "__main__":
    main()
