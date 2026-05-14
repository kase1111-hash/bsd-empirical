#!/usr/bin/env python3
"""regression_v2_deg_phi.py - empirical refinement of Watkins's conjecture.

After verifying Watkins's conjecture (2^rank | deg_phi) at 100% across the
Cremona corpus via watkins_conjecture_test.py, the same output revealed
two surprises:

    1. The Watkins bound is dramatically LOOSE: typical v_2(deg_phi) sits
       between 8 and 9 regardless of rank, while Watkins only requires
       v_2 >= rank.
    2. v_2(deg_phi) is essentially rank-INDEPENDENT in its baseline level:
       mean v_2 = 8.85 at rank 0, 8.92 at rank 1, 8.94 at rank 2, 8.28 at
       rank 3 (all within 1.0 of each other).

This raises the question: what DOES predict v_2(deg_phi) across the corpus?
This script fits an OLS regression of v_2(deg_phi) on:

    rank                    (the Watkins predictor)
    v_2(conductor)          (2-part of N from local arithmetic at 2)
    v_2(Tamagawa product)   (2-adic content of local component sizes)
    v_2(torsion order)      (2-adic content of |T|)
    v_2(|Sha|_an)           (2-adic content of analytic Sha)
    log10(conductor)        (magnitude control)
    omega(N)                (number of distinct prime factors)

If the coefficient on rank is close to 1 (the Watkins-tight value) and the
2-adic predictors carry their own significant coefficients, we have an
empirical refinement of Watkins. If rank's coefficient is much smaller
than 1, then v_2(deg_phi)'s rank-dependence is weaker than Watkins's
exact bound suggests, and the structure is genuinely elsewhere.

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet
Output: lmfdb_ec_out/regression_v2_deg_phi.csv

Author: Kase Branham - Independent Researcher
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, Timer, C, debug,
)


DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_spectrum.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")


def vectorized_v_2(arr):
    """Vectorized 2-adic valuation. For n > 0: v_2(n) = log2(n & -n)."""
    arr = np.asarray(arr, dtype=np.int64)
    out = np.zeros(len(arr), dtype=np.int64)
    pos = arr > 0
    lowest_bit = arr[pos] & -arr[pos]
    out[pos] = np.log2(lowest_bit).astype(np.int64)
    return out


def precompute_omega(max_N):
    """omega[n] = number of distinct prime factors of n, for n in 0..max_N.

    Uses a linear sieve: for each prime p, increment omega at every multiple of p.
    """
    is_prime = np.ones(max_N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(np.sqrt(max_N)) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False
    primes = np.flatnonzero(is_prime)

    omega = np.zeros(max_N + 1, dtype=np.int32)
    for p in primes:
        omega[p :: p] += 1
    return omega


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

    banner("Regression: v_2(deg_phi) on structural predictors", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load corpus and compute features ----------------------
    step(1, 3, "Loading and computing predictors...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    corpus = corpus[corpus["deg_phi"].notna()].copy()

    # Casts
    corpus["deg_phi"]          = corpus["deg_phi"].astype(np.int64)
    corpus["conductor"]        = corpus["conductor"].astype(np.int64)
    corpus["tamagawa_product"] = corpus["tamagawa_product"].astype(np.int64)
    corpus["torsion_order"]    = corpus["torsion_order"].astype(np.int64)
    corpus["sha_int"]          = corpus["sha_an"].round().astype(np.int64)
    corpus["rank_int"]         = corpus["rank"].astype(np.int64)
    print(f"  Optimal curves: {len(corpus):,}")

    # v_2 features
    print("  Computing v_2 features...")
    with Timer("v_2 features"):
        corpus["v2_deg_phi"]   = vectorized_v_2(corpus["deg_phi"].values)
        corpus["v2_conductor"] = vectorized_v_2(corpus["conductor"].values)
        corpus["v2_tamagawa"]  = vectorized_v_2(corpus["tamagawa_product"].values)
        corpus["v2_torsion"]   = vectorized_v_2(corpus["torsion_order"].values)
        corpus["v2_sha"]       = vectorized_v_2(corpus["sha_int"].values)

    # omega(N) via sieve
    print("  Computing omega(N) via sieve...")
    with Timer("omega sieve"):
        max_N = int(corpus["conductor"].max())
        omega_table = precompute_omega(max_N)
        corpus["omega_N"] = omega_table[corpus["conductor"].values]

    corpus["log10_N"] = np.log10(corpus["conductor"].astype(float))

    # ---- 2. Predictor summaries ---------------------------------
    step(2, 3, "Predictor summary statistics...")
    section("PREDICTOR AND TARGET SUMMARIES")

    summary_cols = ["v2_deg_phi", "rank_int", "v2_conductor",
                    "v2_tamagawa", "v2_torsion", "v2_sha",
                    "omega_N", "log10_N"]
    rows = []
    for col in summary_cols:
        rows.append((
            col,
            float(corpus[col].mean()),
            float(corpus[col].median()),
            float(corpus[col].std()),
            float(corpus[col].min()),
            float(corpus[col].max()),
        ))
    summary_table(
        rows,
        ["variable", "mean", "median", "std", "min", "max"],
        title="Variables in the regression",
        fmt=["<16s", ">12.3f", ">12.3f", ">10.3f", ">10.2f", ">10.2f"],
    )

    # ---- 3. OLS regression ---------------------------------------
    step(3, 3, "Fitting OLS regression...")
    section("OLS: v_2(deg_phi) ~ predictors")

    predictors = ["rank_int", "v2_conductor", "v2_tamagawa",
                  "v2_torsion", "v2_sha", "omega_N", "log10_N"]
    y = corpus["v2_deg_phi"].astype(float).values
    X = corpus[predictors].astype(float).values
    X = sm.add_constant(X, has_constant="add")

    print(f"  Observations: {len(y):,}")
    print(f"  Predictors:   {len(predictors)}")
    print()

    with Timer("OLS fit"):
        fit = sm.OLS(y, X).fit()

    print(f"  R²:               {fit.rsquared:.4f}")
    print(f"  Adjusted R²:      {fit.rsquared_adj:.4f}")
    print(f"  F-statistic:      {fit.fvalue:,.1f}  (p < {fit.f_pvalue:.0e})")
    print(f"  Residual SE:      {np.sqrt(fit.mse_resid):.3f}")
    print(f"  Mean v_2(deg_phi): {float(corpus['v2_deg_phi'].mean()):.3f}")
    print()

    coef   = fit.params
    stderr = fit.bse
    tvals  = fit.tvalues
    pvals  = fit.pvalues
    ci     = fit.conf_int(alpha=0.05)
    names  = ["intercept"] + predictors

    coef_rows = []
    for i, name in enumerate(names):
        coef_rows.append((
            name,
            float(coef[i]),
            float(stderr[i]),
            float(ci[i, 0]),
            float(ci[i, 1]),
            float(tvals[i]),
            float(pvals[i]),
        ))
    summary_table(
        coef_rows,
        ["predictor", "coefficient", "std err", "CI lo", "CI hi",
         "t", "p-value"],
        title="OLS coefficient estimates",
        fmt=["<16s", ">13.4f", ">10.4f", ">10.4f", ">10.4f",
             ">10.2f", ">11.2e"],
    )

    print()
    print("  INTERPRETATION:")
    print("  ──────────────")
    rank_coef = float(coef[1])  # rank_int is first non-intercept predictor
    rank_ci_lo = float(ci[1, 0])
    rank_ci_hi = float(ci[1, 1])
    print(f"  Rank coefficient:  {rank_coef:.3f}  (95% CI: {rank_ci_lo:.3f} to {rank_ci_hi:.3f})")
    if abs(rank_coef - 1.0) < 0.1:
        print(f"  → Close to 1.0, consistent with Watkins's bound being tight on average")
    elif rank_coef < 0.5:
        print(f"  → Substantially below 1.0; rank contributes less than Watkins suggests")
    else:
        print(f"  → Below 1.0 but non-trivial; partial Watkins-tightness")

    # ---- By-rank fit quality -------------------------------------
    section("BY-RANK FIT QUALITY")
    print("  Does the regression fit comparably across ranks?")
    print("  Big residual mean at a rank = systematic miss; small = good fit.")
    print()

    corpus["v2_predicted"] = fit.predict(X)
    corpus["v2_residual"]  = corpus["v2_deg_phi"] - corpus["v2_predicted"]

    rows = []
    for r in sorted(corpus["rank_int"].unique()):
        sub = corpus[corpus["rank_int"] == r]
        n = int(len(sub))
        if n < 1:
            continue
        rows.append((
            int(r),
            n,
            float(sub["v2_deg_phi"].mean()),
            float(sub["v2_predicted"].mean()),
            float(sub["v2_residual"].mean()),
            float(sub["v2_residual"].std()) if n > 1 else 0.0,
        ))
    summary_table(
        rows,
        ["rank", "n", "actual mean", "predicted mean",
         "residual mean", "residual std"],
        title="Model fit by rank",
        fmt=[">5d", ">12,d", ">14.3f", ">16.3f", ">15.4f", ">14.3f"],
    )

    # ---- Variable importance (partial R²) ------------------------
    section("VARIABLE IMPORTANCE: COEFFICIENT × STD(PREDICTOR)")
    print("  Approximate contribution to v_2(deg_phi) = β · σ(x):")
    print("  This shows the typical effect on v_2(deg_phi) of a 1-σ change")
    print("  in each predictor, useful for comparing across predictors with")
    print("  different scales.")
    print()
    rows = []
    for i, name in enumerate(predictors, start=1):
        sigma = float(corpus[name].std())
        beta  = float(coef[i])
        impact = beta * sigma
        rows.append((name, beta, sigma, impact))
    rows.sort(key=lambda r: abs(r[3]), reverse=True)
    summary_table(
        rows,
        ["predictor", "coefficient (β)", "σ(predictor)", "β·σ"],
        title="Standardized impact (sorted by |β·σ|)",
        fmt=["<16s", ">15.4f", ">15.4f", ">15.4f"],
    )

    # ---- Save ----------------------------------------------------
    out_csv = output_dir / "regression_v2_deg_phi.csv"
    pd.DataFrame([{
        "predictor": names[i],
        "coefficient": float(coef[i]),
        "std_err":     float(stderr[i]),
        "ci_lo":       float(ci[i, 0]),
        "ci_hi":       float(ci[i, 1]),
        "t":           float(tvals[i]),
        "p_value":     float(pvals[i]),
    } for i in range(len(names))]).to_csv(out_csv, index=False)
    print(f"\n  Coefficients CSV: {out_csv}")


if __name__ == "__main__":
    main()
