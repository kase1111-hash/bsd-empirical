#!/usr/bin/env python3
"""regression_isog_vs_sha_exact.py - re-run Paper 1's regression with EXACT
has_p_isog data and extend to the Mazur exceptional primes.

Paper 1 v5 used a torsion-witness proxy for has_p_isog at p in {2, 3, 5, 7},
because the rational isogeny matrix data wasn't yet parsed into the corpus.
With Tier 1C complete (add_allisog.py landed), each isogeny class now has
EXACT has_p_isog flags for all 12 Mazur primes:

    {2, 3, 5, 7, 11, 13, 17, 19, 37, 43, 67, 163}

This script:
    1. Aggregates the curve-level corpus to one row per isogeny class
    2. For each Mazur prime p, computes:
       - Contingency table for has_p_isog vs (p^2 | |Sha|_an in some curve)
       - Unconditional odds ratio with Wald CI
       - Fisher's exact test
       - Conditional logistic regression with controls (log10 N, rank,
         log Tamagawa product, log class size)
    3. Reports per-prime results in three tables

Two outputs:
    a. v6 numbers for Paper 1 at p in {2, 3, 5, 7} — proxy bias removed
    b. NEW results for Mazur exceptional p in {11, 13, 17, 19, 37, 43, 67, 163}
       — territory Paper 1 v5 couldn't probe

For small samples (few classes with has_p_isog=True), the script falls back
to Fisher's exact test rather than logistic regression.

Input:  lmfdb_ec_out/ec_corpus_with_isog.parquet
Output: lmfdb_ec_out/regression_isog_vs_sha_exact.csv

Author: Kase Branham - Independent Researcher
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, Timer, C, debug,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

MAZUR_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 37, 43, 67, 163]

# Minimum sample sizes to attempt logistic regression
LOGISTIC_MIN_ISOG = 20
LOGISTIC_MIN_BOTH = 1


# =====================================================================
#  STATISTICAL HELPERS
# =====================================================================

def unconditional_odds_ratio(n00, n01, n10, n11):
    """Compute unconditional odds ratio with 95% Wald CI.

    Returns (OR, ci_lo, ci_hi) or (nan, nan, nan) for empty cells.
    """
    if n00 == 0 or n01 == 0 or n10 == 0 or n11 == 0:
        # Apply Haldane correction (+0.5 to each cell) for stable estimate
        a, b, c, d = n00 + 0.5, n01 + 0.5, n10 + 0.5, n11 + 0.5
        OR = (d * a) / (c * b)
        log_OR = np.log(OR)
        se_log_OR = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_lo = np.exp(log_OR - 1.96 * se_log_OR)
        ci_hi = np.exp(log_OR + 1.96 * se_log_OR)
        return OR, ci_lo, ci_hi
    OR = (n11 * n00) / (n10 * n01)
    log_OR = np.log(OR)
    se_log_OR = np.sqrt(1/n11 + 1/n10 + 1/n01 + 1/n00)
    ci_lo = np.exp(log_OR - 1.96 * se_log_OR)
    ci_hi = np.exp(log_OR + 1.96 * se_log_OR)
    return OR, ci_lo, ci_hi


def conditional_logistic(df, feature, outcome, controls):
    """Fit logistic regression with controls. Returns (OR, ci_lo, ci_hi, status)."""
    try:
        X = df[[feature] + controls].astype(float)
        X = sm.add_constant(X, has_constant="add")
        y = df[outcome].astype(int)
        model = sm.Logit(y, X).fit(disp=0, method="lbfgs", maxiter=500)
        coef = float(model.params[feature])
        se = float(model.bse[feature])
        if not np.isfinite(coef) or not np.isfinite(se):
            return np.nan, np.nan, np.nan, "non-finite"
        OR = np.exp(coef)
        ci_lo = np.exp(coef - 1.96 * se)
        ci_hi = np.exp(coef + 1.96 * se)
        return OR, ci_lo, ci_hi, "logistic"
    except Exception as e:
        debug(f"Logistic failed for {feature}: {e}")
        return np.nan, np.nan, np.nan, "logistic_failed"


# =====================================================================
#  MAIN
# =====================================================================

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

    banner("Paper 1 re-run with EXACT has_p_isog data + Mazur exceptional primes", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load and aggregate to class level ------------------
    step(1, 4, "Loading corpus and aggregating to class level...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")

    # Round sha_an to int
    corpus["sha_int"] = corpus["sha_an"].round().astype(np.int64)

    # Class-level aggregation: one row per (conductor, iso)
    print("  Building class-level table...")
    agg_dict = {"sha_int": list, "rank": "first", "tamagawa_product": "max"}
    for p_ in MAZUR_PRIMES:
        agg_dict[f"has_{p_}_isog"] = "first"

    with Timer("Class aggregation"):
        classes = (corpus.groupby(["conductor", "iso"], as_index=False)
                          .agg(agg_dict))
        class_sizes = (corpus.groupby(["conductor", "iso"]).size()
                              .rename("class_size").reset_index())
        classes = classes.merge(class_sizes, on=["conductor", "iso"])

    print(f"  Classes: {len(classes):,}")

    # Control variables (log-transformed where useful)
    classes["log10_N"]        = np.log10(classes["conductor"].astype(float))
    classes["log_tam"]        = np.log(classes["tamagawa_product"].astype(float) + 1.0)
    classes["log_class_size"] = np.log(classes["class_size"].astype(float))

    controls = ["log10_N", "rank", "log_tam", "log_class_size"]

    # ---- 2. Per-prime analysis ---------------------------------
    step(2, 4, "Per-prime contingency, unconditional OR, conditional logistic...")
    results = []

    for p_ in MAZUR_PRIMES:
        p2 = p_ * p_
        feature  = f"has_{p_}_isog"
        outcome  = f"has_{p2}_sha"

        # Build outcome column: any curve in class has p^2 | |Sha|_an
        classes[outcome] = classes["sha_int"].apply(
            lambda lst, p2=p2: any(int(s) > 0 and int(s) % p2 == 0 for s in lst)
        )

        # Build 2x2 contingency
        feat = classes[feature].astype(bool).to_numpy()
        out  = classes[outcome].astype(bool).to_numpy()
        n00 = int(((~feat) & (~out)).sum())
        n01 = int(((~feat) &   out ).sum())
        n10 = int((  feat  & (~out)).sum())
        n11 = int((  feat  &   out ).sum())

        # Unconditional OR
        uncond_OR, ci_lo_u, ci_hi_u = unconditional_odds_ratio(n00, n01, n10, n11)

        # Fisher's exact (always)
        try:
            fisher_OR, fisher_p = fisher_exact([[n00, n01], [n10, n11]])
        except Exception:
            fisher_OR = fisher_p = np.nan

        # Conditional logistic (where feasible)
        n_isog = n10 + n11
        if n_isog >= LOGISTIC_MIN_ISOG and n11 >= LOGISTIC_MIN_BOTH:
            cond_OR, cond_ci_lo, cond_ci_hi, method = conditional_logistic(
                classes, feature, outcome, controls
            )
        else:
            cond_OR = cond_ci_lo = cond_ci_hi = np.nan
            method = f"skip (n_isog={n_isog}, n11={n11})"

        results.append({
            "prime":         p_,
            "p2":            p2,
            "n_classes":     len(classes),
            "n_isog":        n_isog,
            "n_psha":        n01 + n11,
            "n_both":        n11,
            "n_neither":     n00,
            "uncond_OR":     uncond_OR,
            "uncond_ci_lo":  ci_lo_u,
            "uncond_ci_hi":  ci_hi_u,
            "fisher_OR":     fisher_OR,
            "fisher_p":      fisher_p,
            "cond_OR":       cond_OR,
            "cond_ci_lo":    cond_ci_lo,
            "cond_ci_hi":    cond_ci_hi,
            "method":        method,
        })

    # ---- 3. Print results -------------------------------------
    step(3, 4, "Formatting tables...")

    section("SAMPLE SIZES BY MAZUR PRIME")
    rows = [(r["prime"], r["p2"], r["n_isog"], r["n_psha"], r["n_both"])
            for r in results]
    summary_table(
        rows,
        ["p", "p²", "classes with p-isog", "classes with p²|Sha",
         "classes with both"],
        title="Per-prime sample sizes (class-level, n=2,164,260 total)",
        fmt=[">5d", ">8d", ">22,d", ">22,d", ">20,d"],
    )

    section("UNCONDITIONAL ODDS RATIOS")
    rows = []
    for r in results:
        rows.append((
            r["prime"],
            r["uncond_OR"],
            r["uncond_ci_lo"],
            r["uncond_ci_hi"],
            r["fisher_OR"],
            r["fisher_p"],
        ))
    summary_table(
        rows,
        ["p", "uncond OR", "95% CI lo", "95% CI hi", "Fisher OR", "Fisher p"],
        title="Unconditional p-isogeny ↔ p²|Sha association",
        fmt=[">5d", ">14.3f", ">12.3f", ">14.3f", ">14.3f", ">12.2e"],
    )

    section("CONDITIONAL ODDS RATIOS (controlling for log N, rank, log c, log class size)")
    rows = []
    for r in results:
        rows.append((
            r["prime"],
            r["cond_OR"] if np.isfinite(r["cond_OR"]) else float("nan"),
            r["cond_ci_lo"] if np.isfinite(r["cond_ci_lo"]) else float("nan"),
            r["cond_ci_hi"] if np.isfinite(r["cond_ci_hi"]) else float("nan"),
            r["method"],
        ))
    summary_table(
        rows,
        ["p", "cond OR", "95% CI lo", "95% CI hi", "method"],
        title="Conditional odds ratios from logistic regression",
        fmt=[">5d", ">14.3f", ">14.3f", ">14.3f", "<32s"],
    )

    # ---- 4. Save CSV ------------------------------------------
    step(4, 4, "Saving results CSV...")
    out_csv = output_dir / "regression_isog_vs_sha_exact.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"  CSV: {out_csv}")

    # ---- Verdict ----------------------------------------------
    section("INTERPRETATION GUIDE")
    print("  Standard primes p in {2, 3, 5, 7}:")
    print("    These were in Paper 1 v5 via the torsion-witness proxy.")
    print("    The conditional OR here should be the proxy-free value.")
    print()
    print("  Mazur exceptional primes p in {11, 13, 17, 19, 37, 43, 67, 163}:")
    print("    Paper 1 v5 did not analyze these. NEW results.")
    print("    For very small n_isog, conditional logistic is not run; rely on")
    print("    Fisher's exact OR and p-value instead.")
    print()
    print("  A conditional OR substantially > 1 means: even after controlling")
    print("  for conductor, rank, Tamagawa, and class size, classes with a")
    print("  rational p-isogeny have higher odds of hosting a curve with")
    print("  p² | |Sha|_an. This is the Cassels-Fisher prediction made")
    print("  empirical at corpus scale.")


if __name__ == "__main__":
    main()
