#!/usr/bin/env python3
"""check_modular_degree_baseline.py - verify §4.1 claim that 177450dr1's
modular degree of 37,158,912 is an outlier at conductor 177450.

Cremona stores modular degree only for the optimal curve of each isogeny
class (curve_number = 1). For each conductor N, the natural reference is
the distribution of deg_phi across all optimal curves at that N -- i.e.,
across all isogeny classes at that conductor.

Three reference windows are reported:
    1. At N = 177450 exactly (the §4.1 claim's reference)
    2. At N within 10% of 177450
    3. At N within 25% of 177450

Plus a corpus-wide log-log scaling fit deg_phi ~ N^alpha, which gives a
predicted "typical" deg_phi at any N and lets us check how 37,158,912
compares to the corpus-wide expectation at N = 177450.

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet
Output: lmfdb_ec_out/deg_phi_at_177450.csv  (distribution at target N)

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
TARGET_N           = 177450
TARGET_DEG_PHI     = 37_158_912        # 177450dr1


def percentile_rank(value, distribution):
    """Percentile rank of `value` in `distribution`."""
    arr = np.asarray(distribution, dtype=float)
    if len(arr) == 0:
        return float("nan")
    return 100.0 * np.sum(arr <= value) / len(arr)


def report_distribution(values, target, label):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        print(f"  {label}: no data")
        return
    n      = len(arr)
    median = float(np.median(arr))
    mean   = float(np.mean(arr))
    p25    = float(np.percentile(arr, 25))
    p75    = float(np.percentile(arr, 75))
    p90    = float(np.percentile(arr, 90))
    p95    = float(np.percentile(arr, 95))
    p99    = float(np.percentile(arr, 99))
    maxv   = float(np.max(arr))
    rank   = percentile_rank(target, arr)
    ratio_med = target / median if median > 0 else float("inf")
    ratio_p95 = target / p95    if p95    > 0 else float("inf")
    print(f"  {label}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"    n optimal curves:    {n:>14,}")
    print(f"    median deg_phi:      {median:>14,.0f}")
    print(f"    mean deg_phi:        {mean:>14,.0f}")
    print(f"    p25:                 {p25:>14,.0f}")
    print(f"    p75:                 {p75:>14,.0f}")
    print(f"    p90:                 {p90:>14,.0f}")
    print(f"    p95:                 {p95:>14,.0f}")
    print(f"    p99:                 {p99:>14,.0f}")
    print(f"    max:                 {maxv:>14,.0f}")
    print()
    print(f"    target deg_phi:      {target:>14,}")
    print(f"      percentile rank:   {rank:>14.2f}%")
    print(f"      ratio to median:   {ratio_med:>14.2f}x")
    print(f"      ratio to p95:      {ratio_p95:>14.2f}x")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(p)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--target-N",   type=int, default=TARGET_N)
    p.add_argument("--target-deg", type=int, default=TARGET_DEG_PHI)
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    banner(f"Modular degree baseline check at N = {args.target_N}", {
        "Corpus":         str(args.corpus),
        "Target N":       args.target_N,
        "Target deg_phi": f"{args.target_deg:,}",
        "Output dir":     str(output_dir),
    })

    # ---- 1. Load and restrict to optimal curves -------------------
    step(1, 3, "Loading optimal-curve subset...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    corpus = corpus[corpus["deg_phi"].notna()].copy()
    corpus["deg_phi"]   = corpus["deg_phi"].astype(float)
    corpus["conductor"] = corpus["conductor"].astype(int)
    print(f"  Optimal curves with deg_phi: {len(corpus):,}")
    print(f"  (one per isogeny class; deg_phi is recorded only for")
    print(f"   the optimal curve, curve_number=1, in each class)")

    # ---- 2. Reference windows -------------------------------------
    step(2, 3, "Reference distributions...")

    section(f"BASELINE 1 -- At N = {args.target_N} exactly")
    sub = corpus[corpus["conductor"] == args.target_N]
    report_distribution(sub["deg_phi"].values, args.target_deg,
                        f"deg_phi distribution at N = {args.target_N}")

    section(f"BASELINE 2 -- At N within 10% of {args.target_N}")
    lo, hi = int(0.9 * args.target_N), int(1.1 * args.target_N)
    sub = corpus[(corpus["conductor"] >= lo) & (corpus["conductor"] <= hi)]
    report_distribution(sub["deg_phi"].values, args.target_deg,
                        f"deg_phi for N in [{lo:,}, {hi:,}]")

    section(f"BASELINE 3 -- At N within 25% of {args.target_N}")
    lo, hi = int(0.75 * args.target_N), int(1.25 * args.target_N)
    sub = corpus[(corpus["conductor"] >= lo) & (corpus["conductor"] <= hi)]
    report_distribution(sub["deg_phi"].values, args.target_deg,
                        f"deg_phi for N in [{lo:,}, {hi:,}]")

    # ---- 3. Corpus-wide scaling deg_phi ~ N^alpha ----------------
    section("BASELINE 4 -- Corpus-wide log-log scaling")
    mask = corpus["deg_phi"] > 0
    log_N  = np.log10(corpus.loc[mask, "conductor"].values.astype(float))
    log_dp = np.log10(corpus.loc[mask, "deg_phi"].values)
    slope, intercept = np.polyfit(log_N, log_dp, 1)
    residuals = log_dp - (intercept + slope * log_N)
    sigma = float(np.std(residuals))
    print(f"  Log-log fit: log10(deg_phi) ≈ {intercept:+.3f} + {slope:.3f} · log10(N)")
    print(f"  Scaling exponent alpha ≈ {slope:.3f}")
    print(f"  Residual sigma (in log10 units): {sigma:.3f}")
    print()

    pred_log_at_N = intercept + slope * np.log10(args.target_N)
    pred_dp       = 10 ** pred_log_at_N
    target_log    = np.log10(args.target_deg)
    z_log         = (target_log - pred_log_at_N) / sigma
    print(f"  At N = {args.target_N}:")
    print(f"    Predicted typical deg_phi:        {pred_dp:>14,.0f}")
    print(f"    Target deg_phi:                   {args.target_deg:>14,}")
    print(f"    Ratio target/predicted:           {args.target_deg/pred_dp:>14.2f}x")
    print(f"    Z-score in log-deg_phi residual:  {z_log:>14.2f} sigma")

    # ---- 4. Save the N=target distribution -----------------------
    step(3, 3, "Saving distribution at target N...")
    cols = ["label", "conductor", "iso", "rank", "torsion_order",
            "tamagawa_product", "sha_an", "deg_phi"]
    cols = [c for c in cols if c in corpus.columns]
    out = (corpus[corpus["conductor"] == args.target_N][cols]
           .sort_values("deg_phi", ascending=False))
    out_csv = output_dir / f"deg_phi_at_{args.target_N}.csv"
    out.to_csv(out_csv, index=False)
    print(f"  CSV: {out_csv}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
