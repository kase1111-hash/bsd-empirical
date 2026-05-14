#!/usr/bin/env python3
"""sato_tate_empirics.py - test Sato-Tate conjecture across the Cremona corpus.

THEORETICAL BACKGROUND

For an elliptic curve E/Q without CM, the Sato-Tate conjecture (proven by
Taylor, Clozel, Harris, and Shepherd-Barron for non-CM elliptic curves
modulo Galois rep modularity, then unconditionally for E/Q) predicts:

    The Frobenius angles theta_p(E) = arccos(a_p / (2*sqrt(p))) become
    equidistributed with respect to the Sato-Tate measure
    (2/pi) sin^2(theta) d(theta) on [0, pi] as p varies over good primes.

Equivalently, the variable x = a_p / sqrt(p) has limit density
(1/(2*pi)) sqrt(4 - x^2) on [-2, 2].

For CM curves, the limit measure is different: a point mass of weight 1/2
at theta = pi/2 (inert primes; a_p = 0 by CM theory) plus uniform on
[0, pi] of weight 1/2 (split primes).

MOMENTS OF a_p / sqrt(p) UNDER THESE MEASURES

    Non-CM (Sato-Tate, Catalan numbers):
        E[x^2] = 1,    E[x^4] = 2,    E[x^6] = 5

    CM (mixed split + inert):
        E[x^2] = 1,    E[x^4] = 3,    E[x^6] = 10

m_2 coincides for both; m_4 and m_6 are the discriminating moments. With
3M curves and 25 primes per curve, we have ~75M (curve, prime) data points
for vertical Sato-Tate testing.

WHAT THIS SCRIPT TESTS

This is the "vertical" Sato-Tate test: at each fixed prime p, the a_p values
across curves should follow the ST distribution. Distinct from the original
"horizontal" Sato-Tate (single curve, varying p), but both are predicted
by Sato-Tate measure theory.

OUTPUT

    1. Per-prime moments table (non-CM and CM separately)
    2. Aggregate moments across all primes
    3. Text histogram of theta_p at a representative large prime
    4. Per-CM-discriminant sample sizes
    5. CSV of per-prime empirical vs predicted moments

Input:  lmfdb_ec_out/ec_corpus_with_isog.parquet
Output: lmfdb_ec_out/sato_tate_moments.csv

Author: Kase Branham - Independent Researcher
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, Timer, C, debug,
)


DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

# First 25 primes (matches Cremona's a_p list order)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
          53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# Predicted moments of a_p / sqrt(p)
ST_MOMENTS_NON_CM = {2: 1.0,   4: 2.0,   6: 5.0}   # Catalan numbers
ST_MOMENTS_CM     = {2: 1.0,   4: 3.0,   6: 10.0}  # (1/2) * (2k choose k)


# =====================================================================
#  HELPERS
# =====================================================================

def text_histogram(values, n_bins=24, width=50, lo=0.0, hi=np.pi):
    """Print a text-mode histogram of theta values on [0, pi]."""
    counts, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
    max_count = counts.max() if counts.max() > 0 else 1
    total = counts.sum()
    for i, c in enumerate(counts):
        bar = "█" * int(width * c / max_count)
        pct = 100.0 * c / total if total > 0 else 0
        print(f"  θ ∈ [{edges[i]:.2f}, {edges[i+1]:.2f}]  "
              f"{c:>10,d} ({pct:5.2f}%)  {bar}")


def st_density_predicted(theta):
    """Sato-Tate density (2/π) sin²(θ) on [0, π]."""
    return (2.0 / np.pi) * np.sin(theta) ** 2


def compute_moments(ap_values, prime):
    """Return (m_2, m_4, m_6) of a_p / sqrt(p) for given ap_values."""
    x = ap_values.astype(np.float64) / np.sqrt(prime)
    return (
        float((x ** 2).mean()) if len(x) > 0 else 0.0,
        float((x ** 4).mean()) if len(x) > 0 else 0.0,
        float((x ** 6).mean()) if len(x) > 0 else 0.0,
    )


# =====================================================================
#  MAIN
# =====================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(p)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--hist-prime", type=int, default=97,
                   help="Prime to use for theta histogram visualization")
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("Sato-Tate empirical test across the Cremona corpus", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
        "Hist prime": args.hist_prime,
    })

    # ---- 1. Load corpus ----------------------------------------
    step(1, 4, "Loading corpus...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")

    n_cm = int(corpus["is_cm"].sum())
    n_non_cm = len(corpus) - n_cm
    print(f"  CM curves:     {n_cm:,}")
    print(f"  Non-CM curves: {n_non_cm:,}")

    # Pre-extract arrays we'll need
    print("  Extracting a_p arrays from aps_list...")
    with Timer("Array extraction"):
        # Pre-convert aps_list to a 2D array of shape (n_curves, 25)
        aps_array = np.stack([np.asarray(x, dtype=np.int32)
                              for x in corpus["aps_list"]])
        N_arr     = corpus["conductor"].to_numpy(dtype=np.int64)
        is_cm_arr = corpus["is_cm"].to_numpy(dtype=bool)
    print(f"  a_p array shape: {aps_array.shape}")

    # ---- 2. Per-prime moments -----------------------------------
    step(2, 4, "Computing per-prime Sato-Tate moments...")
    section("PER-PRIME MOMENTS OF a_p / √p")
    print("  Predictions:")
    print(f"    Non-CM: m_2={ST_MOMENTS_NON_CM[2]:.1f}  "
          f"m_4={ST_MOMENTS_NON_CM[4]:.1f}  "
          f"m_6={ST_MOMENTS_NON_CM[6]:.1f}    (Catalan)")
    print(f"    CM:     m_2={ST_MOMENTS_CM[2]:.1f}  "
          f"m_4={ST_MOMENTS_CM[4]:.1f}  "
          f"m_6={ST_MOMENTS_CM[6]:.1f}    (split+inert mix)")
    print()

    rows_non_cm = []
    rows_cm     = []
    csv_rows    = []

    for idx, prime in enumerate(PRIMES):
        ap_full = aps_array[:, idx]
        # Good-reduction filter: p does not divide N
        good = (N_arr % prime != 0)
        ap_good = ap_full[good]
        cm_good = is_cm_arr[good]

        ap_non_cm = ap_good[~cm_good]
        ap_cm     = ap_good[ cm_good]

        m2_n, m4_n, m6_n = compute_moments(ap_non_cm, prime)
        m2_c, m4_c, m6_c = compute_moments(ap_cm,     prime)

        rows_non_cm.append((
            prime, int(len(ap_non_cm)),
            m2_n, m2_n - ST_MOMENTS_NON_CM[2],
            m4_n, m4_n - ST_MOMENTS_NON_CM[4],
            m6_n, m6_n - ST_MOMENTS_NON_CM[6],
        ))
        rows_cm.append((
            prime, int(len(ap_cm)),
            m2_c, m2_c - ST_MOMENTS_CM[2],
            m4_c, m4_c - ST_MOMENTS_CM[4],
            m6_c, m6_c - ST_MOMENTS_CM[6],
        ))
        csv_rows.append({
            "prime":     prime,
            "n_non_cm":  int(len(ap_non_cm)),
            "n_cm":      int(len(ap_cm)),
            "m2_non_cm": m2_n,
            "m4_non_cm": m4_n,
            "m6_non_cm": m6_n,
            "m2_cm":     m2_c,
            "m4_cm":     m4_c,
            "m6_cm":     m6_c,
        })

    summary_table(
        rows_non_cm,
        ["prime", "n", "m_2", "Δ from 1.0", "m_4", "Δ from 2.0",
         "m_6", "Δ from 5.0"],
        title="Non-CM moments (Sato-Tate prediction)",
        fmt=[">7d", ">12,d", ">8.4f", ">12.4f", ">9.4f", ">12.4f",
             ">9.4f", ">12.4f"],
    )

    summary_table(
        rows_cm,
        ["prime", "n", "m_2", "Δ from 1.0", "m_4", "Δ from 3.0",
         "m_6", "Δ from 10.0"],
        title="CM moments (mixed split-inert prediction)",
        fmt=[">7d", ">12,d", ">8.4f", ">12.4f", ">9.4f", ">12.4f",
             ">9.4f", ">12.4f"],
    )

    # ---- 3. Aggregate moments across primes ---------------------
    step(3, 4, "Aggregate moments across all primes...")
    section("AGGREGATE MOMENTS (all primes, all curves)")

    # Pool a_p / sqrt(p) values across primes (good reduction only)
    print("  Pooling a_p/√p across all 25 primes and all good-reduction (curve, prime) pairs...")
    with Timer("Pooling"):
        non_cm_x = []
        cm_x     = []
        for idx, prime in enumerate(PRIMES):
            ap_full = aps_array[:, idx]
            good = (N_arr % prime != 0)
            ap_good = ap_full[good]
            cm_good = is_cm_arr[good]
            x = ap_good.astype(np.float64) / np.sqrt(prime)
            non_cm_x.append(x[~cm_good])
            cm_x.append(x[cm_good])
        non_cm_pool = np.concatenate(non_cm_x)
        cm_pool     = np.concatenate(cm_x)

    print(f"  Non-CM pooled samples: {len(non_cm_pool):,}")
    print(f"  CM     pooled samples: {len(cm_pool):,}")
    print()

    rows = []
    for label, pool, predicted in [
        ("non-CM", non_cm_pool, ST_MOMENTS_NON_CM),
        ("CM",     cm_pool,     ST_MOMENTS_CM),
    ]:
        if len(pool) == 0:
            continue
        m2 = float((pool ** 2).mean())
        m4 = float((pool ** 4).mean())
        m6 = float((pool ** 6).mean())
        rows.append((
            label, int(len(pool)),
            m2, predicted[2], m2 - predicted[2],
            m4, predicted[4], m4 - predicted[4],
            m6, predicted[6], m6 - predicted[6],
        ))
    summary_table(
        rows,
        ["subset", "n", "m_2", "pred", "Δ", "m_4", "pred", "Δ",
         "m_6", "pred", "Δ"],
        title="Aggregate empirical vs Sato-Tate predicted moments",
        fmt=["<10s", ">14,d", ">8.4f", ">7.1f", ">9.4f",
             ">8.4f", ">7.1f", ">9.4f",
             ">9.4f", ">8.1f", ">9.4f"],
    )

    # ---- 4. Theta histogram at large prime ---------------------
    step(4, 4, "Building theta_p histogram for visualization...")
    section(f"θ_p HISTOGRAM AT p = {args.hist_prime} (NON-CM)")

    if args.hist_prime not in PRIMES:
        print(C.warn(f"  Prime {args.hist_prime} not in PRIMES list; using 97."))
        args.hist_prime = 97
    hist_idx = PRIMES.index(args.hist_prime)

    ap_full = aps_array[:, hist_idx]
    good = (N_arr % args.hist_prime != 0)
    ap_good_non_cm = ap_full[good & ~is_cm_arr]
    # theta = arccos(a_p / (2*sqrt(p)))
    theta = np.arccos(ap_good_non_cm / (2.0 * np.sqrt(args.hist_prime)))

    print(f"  Distribution of θ_p across {len(theta):,} non-CM curves "
          f"with good reduction at p = {args.hist_prime}:")
    print(f"  Predicted: (2/π) sin²(θ) on [0, π] -- peaks at π/2 ≈ 1.57, "
          f"vanishes at 0 and π.")
    print()
    text_histogram(theta, n_bins=24, width=45)

    # CM-specific histogram
    if int((good & is_cm_arr).sum()) > 0:
        section(f"θ_p HISTOGRAM AT p = {args.hist_prime} (CM)")
        ap_good_cm = ap_full[good & is_cm_arr]
        theta_cm = np.arccos(ap_good_cm / (2.0 * np.sqrt(args.hist_prime)))
        print(f"  Distribution of θ_p across {len(theta_cm):,} CM curves "
              f"with good reduction at p = {args.hist_prime}:")
        print(f"  Predicted: half mass uniform on [0, π], half mass at π/2 "
              f"(inert primes give a_p = 0).")
        print()
        text_histogram(theta_cm, n_bins=24, width=45)

    # ---- Save CSV -----------------------------------------------
    out_csv = output_dir / "sato_tate_moments.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"\n  Per-prime moments CSV: {out_csv}")

    # ---- Verdict ------------------------------------------------
    section("VERDICT")
    non_cm_m4_dev = float((non_cm_pool ** 4).mean()) - ST_MOMENTS_NON_CM[4]
    cm_m4_dev     = float((cm_pool ** 4).mean()) - ST_MOMENTS_CM[4]
    print(f"  Aggregate non-CM m_4 deviation: {non_cm_m4_dev:+.4f}  "
          f"(prediction: 2.0)")
    print(f"  Aggregate CM     m_4 deviation: {cm_m4_dev:+.4f}  "
          f"(prediction: 3.0)")
    print()
    if abs(non_cm_m4_dev) < 0.05:
        print(C.ok("  Non-CM moments match Sato-Tate predictions to within 0.05."))
    else:
        print(C.warn(f"  Non-CM m_4 deviates by {non_cm_m4_dev:+.4f} from prediction."))
    if abs(cm_m4_dev) < 0.2:
        print(C.ok("  CM moments match split-inert prediction to within 0.20."))
    else:
        print(C.warn(f"  CM m_4 deviates by {cm_m4_dev:+.4f} from prediction."))


if __name__ == "__main__":
    main()
