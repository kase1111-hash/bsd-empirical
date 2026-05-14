#!/usr/bin/env python3
"""regression_isog_vs_sha.py - logistic regression controlling for confounders.

The headline test of catalog_isogeny_classes_sha.py reports unconditional rate
ratios for p-isogeny witness vs p^2 | max_sha across the corpus. The natural
referee objection is that conductor size, rank, Tamagawa magnitude, or class
size could be confounding the effect. This script tests that.

For each prime p in {2, 3, 5, 7}, fit:

    logit P(p^2 | max_sha_class)
      = beta_0
      + beta_iso  * has_p_isogeny_witness
      + beta_N    * log10(conductor)
      + beta_R    * max_rank
      + beta_C    * log10(max_tamagawa)
      + beta_S    * class_size

Compare the resulting conditional odds ratio for has_p_isog against the
unconditional rate ratio reported in the headline test. If the conditional
OR is still much greater than 1 after controls, the contingency-table
result is robust.

The p=7 model has only ~1655 events in 2.16M observations -- a rare-events
regime where logistic-regression OR estimates can be biased upward.
Interpret cautiously; the qualitative size of the effect is the robust claim.

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet
Output: lmfdb_ec_out/regression_isog_vs_sha.csv

Author: Kase Branham - Independent Researcher
"""

import argparse
import warnings
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

ISOGENY_PRIMES = [2, 3, 5, 7]


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

    banner("Logistic regression: p-isogeny witness vs |Sha[p]|", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
        "Primes":     str(ISOGENY_PRIMES),
    })

    # ---- 1. Load corpus and aggregate to class level --------------
    step(1, 3, "Loading corpus and aggregating to class level...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    corpus = corpus[corpus["sha_an"].notna()].copy()
    corpus["sha_int"]     = corpus["sha_an"].round().astype(int)
    corpus["torsion_int"] = corpus["torsion_order"].astype(int)
    corpus["rank_int"]    = corpus["rank"].astype(int)
    corpus["c_int"]       = corpus["tamagawa_product"].astype(int)
    print(f"  Loaded {len(corpus):,} curves")

    # Per-curve p-torsion indicator
    for prime in ISOGENY_PRIMES:
        corpus[f"has_{prime}_tors"] = (corpus["torsion_int"] % prime == 0)

    # Aggregate to class level
    print("  Aggregating per isogeny class...")
    with Timer("Class aggregation"):
        agg_dict = {
            "n_curves":     ("sha_int",   "count"),
            "max_rank":     ("rank_int",  "max"),
            "max_sha":      ("sha_int",   "max"),
            "max_c":        ("c_int",     "max"),
        }
        for prime in ISOGENY_PRIMES:
            agg_dict[f"has_{prime}_isog"] = (f"has_{prime}_tors", "any")
        df = (corpus.groupby(["conductor", "iso"])
              .agg(**agg_dict)
              .reset_index())
    print(f"  {len(df):,} classes aggregated")

    # Free per-curve memory
    del corpus

    # Derived features
    df["log10_conductor"] = np.log10(df["conductor"].astype(float))
    df["log10_max_c"]     = np.log10(df["max_c"].astype(float).clip(lower=1))
    df["class_size"]      = df["n_curves"].astype(float)
    df["max_rank_f"]      = df["max_rank"].astype(float)

    # Binary target and integer p-isog flag for each prime
    for prime in ISOGENY_PRIMES:
        df[f"sha_has_{prime}sq"]   = (df["max_sha"] % (prime ** 2) == 0).astype(int)
        df[f"has_{prime}_isog_i"]  = df[f"has_{prime}_isog"].astype(int)

    # ---- 2. Fit logistic regressions per prime --------------------
    step(2, 3, "Fitting logistic regressions...")

    results_summary = []
    all_coef_rows  = []

    for prime in ISOGENY_PRIMES:
        section(f"PRIME p = {prime}")

        y_col = f"sha_has_{prime}sq"
        x_cols = [
            f"has_{prime}_isog_i",
            "log10_conductor",
            "max_rank_f",
            "log10_max_c",
            "class_size",
        ]

        y = df[y_col].astype(float).values
        X = df[x_cols].astype(float).values
        X = sm.add_constant(X, has_constant="add")

        n_obs    = int(len(y))
        n_events = int(y.sum())
        baseline = 100 * n_events / n_obs

        print(f"  observations:           {n_obs:,}")
        print(f"  events (p^2 | max_sha): {n_events:,}  "
              f"({baseline:.3f}%)")

        # Unconditional contingency-table rate ratio (for comparison)
        iso_col = f"has_{prime}_isog_i"
        with_iso    = df[df[iso_col] == 1]
        without_iso = df[df[iso_col] == 0]
        r_w  = float(with_iso[y_col].mean())
        r_wo = float(without_iso[y_col].mean())
        uncond_ratio = (r_w / r_wo) if r_wo > 0 else float("inf")
        # Unconditional OR is more directly comparable to logistic OR:
        # OR = [r_w/(1-r_w)] / [r_wo/(1-r_wo)]
        uncond_or = (r_w / max(1e-12, 1 - r_w)) / (r_wo / max(1e-12, 1 - r_wo))
        print(f"  unconditional rate ratio (with/without iso):  {uncond_ratio:7.2f}x")
        print(f"  unconditional odds ratio (with/without iso):  {uncond_or:7.2f}x")
        print()

        # Fit
        try:
            with Timer(f"Fit p={prime}"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = sm.Logit(y, X)
                    fit = model.fit(disp=False, maxiter=200, method="newton")

            coef   = fit.params
            stderr = fit.bse
            zvals  = fit.tvalues
            pvals  = fit.pvalues
            ci     = fit.conf_int(alpha=0.05)

            names = ["const"] + x_cols
            display = {
                "const":            "(intercept)",
                f"has_{prime}_isog_i": f"has_{prime}_isog",
                "log10_conductor":  "log10(conductor)",
                "max_rank_f":       "max_rank",
                "log10_max_c":      "log10(max_c)",
                "class_size":       "class_size",
            }
            rows = []
            for i, name in enumerate(names):
                or_val = float(np.exp(coef[i]))
                or_lo  = float(np.exp(ci[i, 0]))
                or_hi  = float(np.exp(ci[i, 1]))
                rows.append((
                    display[name],
                    float(coef[i]),
                    float(stderr[i]),
                    or_val,
                    or_lo,
                    or_hi,
                    float(zvals[i]),
                    float(pvals[i]),
                ))
                all_coef_rows.append({
                    "prime":   prime,
                    "term":    display[name],
                    "beta":    float(coef[i]),
                    "stderr":  float(stderr[i]),
                    "OR":      or_val,
                    "OR_lo":   or_lo,
                    "OR_hi":   or_hi,
                    "z":       float(zvals[i]),
                    "p_value": float(pvals[i]),
                })
            summary_table(
                rows,
                ["predictor", "beta", "stderr", "OR",
                 "OR 95% lo", "OR 95% hi", "z", "p-value"],
                title=f"Logistic regression coefficients at p={prime}",
                fmt=["<18s", ">11.4f", ">10.4f", ">14.3f",
                     ">14.3f", ">14.3f", ">10.2f", ">11.2e"],
            )

            print(f"\n  log-likelihood:        {fit.llf:.2f}")
            print(f"  null log-likelihood:   {fit.llnull:.2f}")
            print(f"  McFadden pseudo-R^2:   {fit.prsquared:.4f}")
            print(f"  converged:             {fit.mle_retvals.get('converged', 'unknown')}")

            # Save summary for cross-prime table
            iso_idx = 1  # has_p_isog_i is the first non-intercept predictor
            iso_or    = float(np.exp(coef[iso_idx]))
            iso_or_lo = float(np.exp(ci[iso_idx, 0]))
            iso_or_hi = float(np.exp(ci[iso_idx, 1]))
            iso_z     = float(zvals[iso_idx])
            iso_p     = float(pvals[iso_idx])

            results_summary.append({
                "prime":            prime,
                "n_obs":            n_obs,
                "n_events":         n_events,
                "uncond_OR":        uncond_or,
                "uncond_rate_ratio": uncond_ratio,
                "cond_iso_OR":      iso_or,
                "cond_OR_lo":       iso_or_lo,
                "cond_OR_hi":       iso_or_hi,
                "cond_iso_z":       iso_z,
                "cond_iso_p":       iso_p,
                "pseudo_R2":        float(fit.prsquared),
                "converged":        bool(fit.mle_retvals.get("converged", False)),
            })

        except Exception as e:
            print(C.fail(f"  Fit failed for p={prime}: {e}"))
            results_summary.append({
                "prime":            prime,
                "n_obs":            n_obs,
                "n_events":         n_events,
                "uncond_OR":        uncond_or,
                "uncond_rate_ratio": uncond_ratio,
                "cond_iso_OR":      float("nan"),
                "cond_OR_lo":       float("nan"),
                "cond_OR_hi":       float("nan"),
                "cond_iso_z":       float("nan"),
                "cond_iso_p":       float("nan"),
                "pseudo_R2":        float("nan"),
                "converged":        False,
            })

    # ---- 3. Cross-prime summary -----------------------------------
    step(3, 3, "Cross-prime summary...")
    section("HEADLINE: CONDITIONAL vs UNCONDITIONAL OR FOR p-ISOG WITNESS")
    print("  Compare the unconditional odds ratio (from contingency table) to the")
    print("  conditional odds ratio (from logistic regression, after controlling for")
    print("  log10(conductor), max_rank, log10(max_c), and class_size).")
    print("  If conditional OR remains substantially > 1, the headline effect")
    print("  is robust to the named confounders.")
    print()

    rows = []
    for r in results_summary:
        rows.append((
            int(r["prime"]),
            r["uncond_OR"],
            r["cond_iso_OR"],
            r["cond_OR_lo"],
            r["cond_OR_hi"],
            r["cond_iso_z"],
            r["pseudo_R2"],
            int(r["n_events"]),
        ))
    summary_table(
        rows,
        ["p", "uncond OR", "cond OR", "cond 95% lo", "cond 95% hi",
         "z", "McFadden R^2", "n events"],
        title="Conditional vs unconditional effect of p-isogeny witness",
        fmt=[">3d", ">12.3f", ">12.3f", ">13.3f", ">13.3f",
             ">10.2f", ">15.4f", ">12,d"],
    )

    # ---- Save ----------------------------------------------------
    out_csv = output_dir / "regression_isog_vs_sha.csv"
    pd.DataFrame(results_summary).to_csv(out_csv, index=False)
    print(f"\n  Summary CSV: {out_csv}")

    out_coef_csv = output_dir / "regression_isog_vs_sha_coefs.csv"
    pd.DataFrame(all_coef_rows).to_csv(out_coef_csv, index=False)
    print(f"  Coefficient CSV: {out_coef_csv}")


if __name__ == "__main__":
    main()
