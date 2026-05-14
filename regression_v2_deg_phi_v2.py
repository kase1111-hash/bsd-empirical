#!/usr/bin/env python3
"""regression_v2_deg_phi_v2.py — diagnostic upgrade to v1.

Drop-in upgrade for regression_v2_deg_phi.py. Same input (ec_corpus parquet
with v_2(deg φ), v_2(c), etc.) and same conceptual model, but adds:

  (a) HC3 robust standard errors from statsmodels OLS
  (b) Variance inflation factors (VIF) on the predictor matrix
  (c) Standardized beta coefficients (each predictor scaled to unit variance)
  (d) Partial R² per predictor (incremental R² when each is added last)
  (e) Pearson correlation matrix among predictors
  (f) Interaction-model variants:
       (i)   rank × v_2(c)
       (ii)  rank × ω(N)
       (iii) v_2(c)² (quadratic Tamagawa)
       (iv)  spline on log_10(N) (with 3 knots)
  (g) Output CSVs plug-in-ready for Paper 4 v3:
       - regression_v2_deg_phi_main.csv       (coefficients with HC3 SEs)
       - regression_v2_deg_phi_vif.csv        (variance inflation factors)
       - regression_v2_deg_phi_corr.csv       (predictor correlation matrix)
       - regression_v2_deg_phi_partial_r2.csv (partial R² per predictor)
       - regression_v2_deg_phi_interactions.csv (interaction model R² and selected coefs)

This script is parallel in spirit to regression_isog_vs_sha_exact_v2.py
(Paper 1 v8 diagnostic upgrade).

Author: Kase Branham — Independent Researcher
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    Timer, debug,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

# Predictors used in the baseline 7-predictor regression
BASE_PREDICTORS = [
    "rank",
    "v2_c",            # v_2(Tamagawa product)
    "v2_N",            # v_2(conductor)
    "omega_N",         # number of distinct prime divisors of N
    "log10_N",         # log_10(conductor)
    "v2_sha_an",       # v_2(|Sha|_an) at max-Sha curve in class
    "v2_torsion",      # v_2(|E(Q)_tors|)
]


def compute_vif(X_df, predictors):
    """Variance inflation factor for each predictor."""
    vif = {}
    for c in predictors:
        X_other = X_df[[p for p in predictors if p != c]].values
        y_this = X_df[c].values
        if X_other.shape[1] == 0:
            vif[c] = 1.0
            continue
        lr = LinearRegression().fit(X_other, y_this)
        r2 = lr.score(X_other, y_this)
        vif[c] = 1.0 / max(1.0 - r2, 1e-12)
    return vif


def standardized_betas(model, X_df, y, predictors):
    """β_standardized = β · (SD(x) / SD(y))."""
    sd_y = float(np.std(y, ddof=0))
    out = {}
    for c in predictors:
        sd_x = float(np.std(X_df[c].values, ddof=0))
        beta = float(model.params[c]) if c in model.params else np.nan
        if sd_y == 0 or sd_x == 0:
            out[c] = np.nan
        else:
            out[c] = beta * sd_x / sd_y
    return out


def partial_r2(X_df, y, predictors):
    """Incremental R² for each predictor when added LAST to the model."""
    out = {}
    # Full model R²
    X_full = sm.add_constant(X_df[predictors].astype(float))
    full = sm.OLS(y, X_full).fit()
    R2_full = full.rsquared
    for c in predictors:
        rest = [p for p in predictors if p != c]
        X_rest = sm.add_constant(X_df[rest].astype(float)) if rest else sm.add_constant(np.ones((len(y), 0)))
        m_rest = sm.OLS(y, X_rest).fit()
        R2_rest = m_rest.rsquared
        out[c] = R2_full - R2_rest
    return out


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

    banner("Paper 4 v3: v_2(deg φ) regression with HC3 SEs and interactions", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load corpus and aggregate to class level ----
    step(1, 6, "Loading corpus and aggregating to class level...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")
    print(f"  Columns ({len(corpus.columns)} total): {list(corpus.columns)}")

    # Resolve column names — the parquet may use any of several conventions
    def find_col(corpus, candidates, label):
        for c in candidates:
            if c in corpus.columns:
                return c
        raise KeyError(f"Could not find column for '{label}'. Tried: {candidates}. "
                       f"Available columns: {list(corpus.columns)}")

    col_degphi   = find_col(corpus, ["degphi", "deg_phi", "modular_degree", "mod_degree", "deg"], "modular degree")
    col_tam      = find_col(corpus, ["tamagawa_product", "tam_prod", "tamagawa", "tam", "prod_tam"], "Tamagawa product")
    col_sha      = find_col(corpus, ["sha_an", "sha", "analytic_sha", "sha_an_order", "analytic_sha_order"], "analytic Sha")
    col_torsion  = find_col(corpus, ["torsion_order", "torsion", "tors", "num_torsion", "torsion_size"], "torsion order")
    col_rank     = find_col(corpus, ["rank", "mw_rank", "rank_mw", "analytic_rank"], "rank")
    col_iso      = find_col(corpus, ["iso", "iso_label", "isogeny_class"], "iso class label")
    col_cond     = find_col(corpus, ["conductor", "N"], "conductor")

    print(f"\n  Resolved columns:")
    print(f"    modular degree → {col_degphi}")
    print(f"    Tamagawa product → {col_tam}")
    print(f"    analytic Sha → {col_sha}")
    print(f"    torsion order → {col_torsion}")
    print(f"    rank → {col_rank}")
    print(f"    iso class → {col_iso}")
    print(f"    conductor → {col_cond}")

    # Filter rows where any required column is NaN/inf (Cremona's degphi file does not cover every curve)
    required_cols = [col_degphi, col_tam, col_sha, col_torsion, col_rank]
    n_before = len(corpus)
    finite_mask = np.ones(n_before, dtype=bool)
    for c in required_cols:
        finite_mask &= np.isfinite(pd.to_numeric(corpus[c], errors="coerce").values)
    n_filtered = int(finite_mask.sum())
    n_dropped = n_before - n_filtered
    print(f"\n  Filtering: {n_filtered:,} of {n_before:,} rows have finite values in all required columns "
          f"({n_dropped:,} dropped, {100*n_dropped/n_before:.2f}%)")
    if n_dropped > 0:
        per_col_missing = {c: int(corpus[c].isna().sum()) for c in required_cols}
        print(f"  Per-column missing counts: {per_col_missing}")
    corpus = corpus.loc[finite_mask].copy()

    # Compute the class-level 2-adic invariants from the resolved columns
    def v2_vec(arr):
        """Vectorized 2-adic valuation. v_2(0) := 0."""
        arr = np.asarray(arr, dtype=np.int64)
        out = np.zeros(arr.shape, dtype=np.int32)
        mask = arr > 0
        if mask.any():
            lowbit = arr[mask] & -arr[mask]
            out[mask] = np.log2(lowbit).astype(np.int32)
        return out

    corpus["v2_degphi"]  = v2_vec(corpus[col_degphi].astype(np.int64).values)
    corpus["v2_c"]       = v2_vec(corpus[col_tam].astype(np.int64).values)
    corpus["sha_int"]    = corpus[col_sha].round().astype(np.int64)
    corpus["v2_sha_an"]  = v2_vec(corpus["sha_int"].values)
    corpus["v2_torsion"] = v2_vec(corpus[col_torsion].astype(np.int64).values)
    corpus["rank"]       = corpus[col_rank].astype(np.int64)
    corpus["conductor"]  = corpus[col_cond]
    corpus["iso"]        = corpus[col_iso]

    # Conductor structure
    def factorize_omega(n):
        n = int(n)
        omega = 0
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if n % p == 0:
                omega += 1
                while n % p == 0:
                    n //= p
        if n > 1:
            omega += 1  # remaining large prime
        return omega

    print("  Computing conductor structure (v_2(N), omega(N), log_10(N))...")
    unique_N = corpus["conductor"].unique()
    N_v2 = {n: int(np.log2(max(n & -n, 1))) if n > 0 else 0 for n in unique_N}
    N_omega = {n: factorize_omega(n) for n in unique_N}
    corpus["v2_N"]    = corpus["conductor"].map(N_v2)
    corpus["omega_N"] = corpus["conductor"].map(N_omega)
    corpus["log10_N"] = np.log10(corpus["conductor"].astype(float))

    # Pick the column that identifies curve order within an isogeny class (optimal = lowest)
    number_col = None
    for c in ["curve_number", "number", "curve_num", "num"]:
        if c in corpus.columns:
            number_col = c
            break
    print(f"\n  Within-class ordering column: {number_col if number_col else '(none — using DataFrame order)'}")

    # Aggregate to class level — use optimal curve (lowest curve_number) for class-level statistics
    print("  Aggregating to class level...")
    with Timer("Class aggregation"):
        sort_cols = ["conductor", "iso"] + ([number_col] if number_col else [])
        classes = (corpus.sort_values(sort_cols).groupby(
                ["conductor", "iso"], as_index=False).agg({
            "v2_degphi": "first",  # optimal-curve modular degree
            "rank": "first",
            "v2_c": "max",          # max over class (Tamagawa is curve-level but max is robust)
            "v2_N": "first",
            "omega_N": "first",
            "log10_N": "first",
            "v2_sha_an": "max",     # max over class
            "v2_torsion": "first",  # optimal-curve torsion
        }))
    print(f"  Classes: {len(classes):,}")

    # ---- 2. Baseline regression with HC3 SEs ----
    step(2, 6, "Running baseline OLS with HC3 robust SEs...")
    y = classes["v2_degphi"].astype(float)
    X = sm.add_constant(classes[BASE_PREDICTORS].astype(float))
    model = sm.OLS(y, X).fit(cov_type="HC3")
    print(f"\n  R²: {model.rsquared:.4f}")
    print(f"  Adjusted R²: {model.rsquared_adj:.4f}")
    print(f"  N: {model.nobs:,.0f}")
    print(f"\n  {'Predictor':<15s} {'coef':>10s} {'HC3 SE':>10s} {'t':>8s} {'p-value':>12s} {'95% CI':>22s}")
    main_rows = []
    for c in ["const"] + BASE_PREDICTORS:
        b  = float(model.params[c])
        se = float(model.bse[c])
        t  = float(model.tvalues[c])
        pv = float(model.pvalues[c])
        lo = b - 1.96 * se
        hi = b + 1.96 * se
        print(f"  {c:<15s} {b:>+10.4f} {se:>10.4f} {t:>8.1f} {pv:>12.2e} "
              f"({lo:+.4f},{hi:+.4f})")
        main_rows.append({
            "predictor": c, "coef": b, "se_hc3": se, "t": t, "p_value": pv,
            "ci_lo": lo, "ci_hi": hi
        })

    # ---- 3. VIFs, standardized betas, partial R², correlation matrix ----
    step(3, 6, "Computing VIFs, standardized betas, partial R², correlations...")

    vif_table = compute_vif(classes, BASE_PREDICTORS)
    print("\n  VIFs (★ flags > 5):")
    for c, v in vif_table.items():
        flag = " ★" if v > 5 else ""
        print(f"    {c:<15s} VIF = {v:>7.2f}{flag}")

    stdbeta = standardized_betas(model, classes, y, BASE_PREDICTORS)
    print("\n  Standardized betas:")
    for c, b in stdbeta.items():
        print(f"    {c:<15s} β_std = {b:>+.4f}")

    partR2 = partial_r2(classes, y, BASE_PREDICTORS)
    print(f"\n  Partial R² (incremental, when added last):")
    for c, r2 in sorted(partR2.items(), key=lambda x: -x[1]):
        print(f"    {c:<15s} partial R² = {r2:>.4f}")

    print("\n  Pearson correlations among predictors:")
    corr = classes[BASE_PREDICTORS].corr(method="pearson")
    print(corr.round(3).to_string())

    # Save diagnostic CSVs
    pd.DataFrame(main_rows).to_csv(output_dir / "regression_v2_deg_phi_main.csv", index=False)
    pd.DataFrame([{"predictor": c, "vif": v, "std_beta": stdbeta[c], "partial_r2": partR2[c]}
                  for c, v in vif_table.items()]).to_csv(
        output_dir / "regression_v2_deg_phi_vif.csv", index=False)
    corr.to_csv(output_dir / "regression_v2_deg_phi_corr.csv")

    pd.DataFrame([{"predictor": c, "partial_r2": partR2[c]} for c in BASE_PREDICTORS]).to_csv(
        output_dir / "regression_v2_deg_phi_partial_r2.csv", index=False)

    # ---- 4. Interaction models ----
    step(4, 6, "Fitting interaction-model variants...")

    classes["rank_x_v2c"]    = classes["rank"] * classes["v2_c"]
    classes["rank_x_omega"]  = classes["rank"] * classes["omega_N"]
    classes["v2c_sq"]        = classes["v2_c"] ** 2

    interaction_results = []

    # Model M1: baseline (reference)
    interaction_results.append({
        "model": "M0 baseline",
        "R²": model.rsquared,
        "rank_coef": model.params["rank"],
        "v2c_coef":  model.params["v2_c"],
        "added_term_coef": np.nan,
        "added_term_se":   np.nan,
    })

    # Model M1: + rank × v_2(c)
    X1 = sm.add_constant(classes[BASE_PREDICTORS + ["rank_x_v2c"]].astype(float))
    m1 = sm.OLS(y, X1).fit(cov_type="HC3")
    interaction_results.append({
        "model": "M1 + rank × v_2(c)",
        "R²": m1.rsquared,
        "rank_coef": m1.params["rank"],
        "v2c_coef":  m1.params["v2_c"],
        "added_term_coef": m1.params["rank_x_v2c"],
        "added_term_se":   m1.bse["rank_x_v2c"],
    })

    # Model M2: + rank × ω(N)
    X2 = sm.add_constant(classes[BASE_PREDICTORS + ["rank_x_omega"]].astype(float))
    m2 = sm.OLS(y, X2).fit(cov_type="HC3")
    interaction_results.append({
        "model": "M2 + rank × ω(N)",
        "R²": m2.rsquared,
        "rank_coef": m2.params["rank"],
        "v2c_coef":  m2.params["v2_c"],
        "added_term_coef": m2.params["rank_x_omega"],
        "added_term_se":   m2.bse["rank_x_omega"],
    })

    # Model M3: + v_2(c)²
    X3 = sm.add_constant(classes[BASE_PREDICTORS + ["v2c_sq"]].astype(float))
    m3 = sm.OLS(y, X3).fit(cov_type="HC3")
    interaction_results.append({
        "model": "M3 + v_2(c)²",
        "R²": m3.rsquared,
        "rank_coef": m3.params["rank"],
        "v2c_coef":  m3.params["v2_c"],
        "added_term_coef": m3.params["v2c_sq"],
        "added_term_se":   m3.bse["v2c_sq"],
    })

    # Model M4: spline on log_10(N) with 3 internal knots (at conductor quartiles)
    log10_knots = np.quantile(classes["log10_N"].values, [0.25, 0.5, 0.75])
    for i, k in enumerate(log10_knots):
        classes[f"log10N_spline_{i}"] = np.maximum(classes["log10_N"] - k, 0)
    spline_cols = [f"log10N_spline_{i}" for i in range(len(log10_knots))]
    X4 = sm.add_constant(classes[BASE_PREDICTORS + spline_cols].astype(float))
    m4 = sm.OLS(y, X4).fit(cov_type="HC3")
    interaction_results.append({
        "model": "M4 + log_10(N) spline (3 knots)",
        "R²": m4.rsquared,
        "rank_coef": m4.params["rank"],
        "v2c_coef":  m4.params["v2_c"],
        "added_term_coef": np.nan,  # spline has 3 terms
        "added_term_se":   np.nan,
    })

    section("INTERACTION MODEL COMPARISON")
    print(f"  {'Model':<40s} {'R²':>8s} {'rank coef':>12s} {'v_2(c) coef':>14s} {'added term':>18s}")
    for r in interaction_results:
        added = f"{r['added_term_coef']:+.4f} ± {r['added_term_se']:.4f}" if np.isfinite(r['added_term_se']) else "(see CSV)"
        print(f"  {r['model']:<40s} {r['R²']:>8.4f} {r['rank_coef']:>+12.4f} {r['v2c_coef']:>+14.4f} {added:>18s}")

    pd.DataFrame(interaction_results).to_csv(
        output_dir / "regression_v2_deg_phi_interactions.csv", index=False)

    # ---- 5. Watkins verification re-check ----
    step(5, 6, "Verifying Watkins's conjecture at the class level...")
    classes["watkins_pass"] = classes["v2_degphi"] >= classes["rank"]
    pass_rate = classes["watkins_pass"].mean() * 100
    tight_rate_total = (classes["v2_degphi"] == classes["rank"]).mean() * 100
    print(f"  Pass rate: {pass_rate:.6f}%  ({classes['watkins_pass'].sum():,} of {len(classes):,})")
    print(f"  Tight (v_2 = r): {tight_rate_total:.4f}%  ({(classes['v2_degphi'] == classes['rank']).sum():,})")

    print("\n  Mean v_2(deg φ) by rank:")
    for r in sorted(classes["rank"].unique()):
        sub = classes[classes["rank"] == r]
        n = len(sub)
        mean_v2 = sub["v2_degphi"].mean()
        med_v2 = sub["v2_degphi"].median()
        tight = (sub["v2_degphi"] == r).sum()
        tight_pct = 100 * tight / n
        print(f"    rank {r}:  n = {n:>10,d}   mean v_2 = {mean_v2:>5.2f}   median = {med_v2:>4.1f}   tight = {tight:>5,d} ({tight_pct:>5.3f}%)")

    # ---- 6. Save and verdict ----
    step(6, 6, "Done. CSVs written to:")
    for f in ["regression_v2_deg_phi_main.csv",
              "regression_v2_deg_phi_vif.csv",
              "regression_v2_deg_phi_corr.csv",
              "regression_v2_deg_phi_partial_r2.csv",
              "regression_v2_deg_phi_interactions.csv"]:
        print(f"    {output_dir / f}")

    section("WHAT TO DROP INTO PAPER 4 v3")
    print("""
The main coefficients with HC3 SEs (regression_v2_deg_phi_main.csv) replace
Paper 4 v2 Table 4's single-column "Coefficient" with a full row per predictor
of {coef, HC3 SE, t, p-value, 95% CI}.

The VIF table (regression_v2_deg_phi_vif.csv) populates a new diagnostic
table in §5.3, flagging predictors with VIF > 5 as moderately collinear.

The standardized betas and partial R² (in regression_v2_deg_phi_vif.csv)
populate the v3 §5.1 paragraph comparing the relative magnitudes of the
predictors on a common scale.

The correlation matrix (regression_v2_deg_phi_corr.csv) goes into a small
table in §5.3 supporting the multicollinearity discussion.

The interaction-model comparison (regression_v2_deg_phi_interactions.csv)
populates a new §5.4 "Robustness to specification" subsection showing how
the rank coefficient shifts (or not) under interaction models. The headline
sentence is whether M1, M2, M3 substantively change the rank coefficient
relative to M0 — the reviewer of v1 specifically asked for this check.

The Watkins verification numbers reported in step 5 should match v2 exactly;
they're regenerated here as a sanity check.
""")


if __name__ == "__main__":
    main()
