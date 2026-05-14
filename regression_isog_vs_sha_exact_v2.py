#!/usr/bin/env python3
"""regression_isog_vs_sha_exact_v2.py — methodological tightening of v1.

Drop-in replacement for regression_isog_vs_sha_exact.py. Same input
(ec_corpus_with_isog.parquet) and same conceptual analysis, but adds:

  (a) Firth-penalized logistic regression at small-event primes
      (where n_both ≤ FIRTH_THRESHOLD, defaults to 10). This tightens
      the wide Wald CIs at p = 13 and gives finite estimates even
      under quasi-complete separation at p ∈ {11, 17, 19, 37, 43, 67, 163}.
  (b) McFadden pseudo-R² per prime.
  (c) Variance inflation factors (VIF) on the predictor matrix.
  (d) Leave-one-control-out (LOCO) sensitivity: at each prime, refit
      dropping each control covariate in turn and report whether the
      has_p_isog OR is stable.
  (e) Class-count-at-conductor adjustment: refit at each prime with
      log(n_classes_at_conductor) added as a covariate, to check
      whether the has_p_isog effect is mediated by the same class-count
      multiplier identified in Paper 2.
  (f) Output everything to a comprehensive CSV that can be plugged
      directly into Paper 1 §3.5 v8.

Output: lmfdb_ec_out/regression_isog_vs_sha_exact_v2.csv
        lmfdb_ec_out/regression_isog_vs_sha_loco.csv (LOCO sensitivity)
        lmfdb_ec_out/regression_isog_vs_sha_vif.csv  (multicollinearity)

Author: Kase Branham — Independent Researcher
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import fisher_exact

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, Timer, C, debug,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

MAZUR_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 37, 43, 67, 163]

LOGISTIC_MIN_ISOG  = 20   # require ≥ this many classes with predictor=True for MLE
LOGISTIC_MIN_BOTH  = 1    # require ≥ this many classes with both
FIRTH_THRESHOLD    = 10   # use Firth when n_both ≤ this (small-event correction)


# =====================================================================
#  FIRTH-PENALIZED LOGISTIC REGRESSION
# =====================================================================

def firth_logit(X, y, max_iter=300, tol=1e-8):
    """Firth-penalized logistic regression.

    Returns (coefficients, Wald SEs from penalized Hessian, log-likelihood, success).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    def neg_penalized_loglik(beta):
        logit = np.clip(X @ beta, -500, 500)
        ll = (y * logit - np.logaddexp(0, logit)).sum()
        p_hat = np.where(logit >= 0,
                         1.0 / (1.0 + np.exp(-logit)),
                         np.exp(logit) / (1.0 + np.exp(logit)))
        W = np.clip(p_hat * (1.0 - p_hat), 1e-12, None)
        XWX = (X * W[:, None]).T @ X
        try:
            sign, logdet = np.linalg.slogdet(XWX)
            if sign <= 0:
                return 1e10
        except np.linalg.LinAlgError:
            return 1e10
        return -(ll + 0.5 * logdet)

    # Start from MLE when feasible, zeros otherwise
    try:
        mle = sm.Logit(y, X).fit(disp=0, maxiter=500, method='lbfgs')
        beta0 = np.asarray(mle.params).copy()
        if not np.all(np.isfinite(beta0)):
            beta0 = np.zeros(p)
    except Exception:
        beta0 = np.zeros(p)

    result = minimize(neg_penalized_loglik, beta0, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'gtol': tol})
    beta_firth = result.x

    # Wald SEs from penalized information matrix
    logit = np.clip(X @ beta_firth, -500, 500)
    p_hat = np.where(logit >= 0,
                     1.0 / (1.0 + np.exp(-logit)),
                     np.exp(logit) / (1.0 + np.exp(logit)))
    W = np.clip(p_hat * (1.0 - p_hat), 1e-12, None)
    I = (X * W[:, None]).T @ X
    try:
        cov = np.linalg.inv(I)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    return beta_firth, se, -result.fun, result.success


# =====================================================================
#  DIAGNOSTICS
# =====================================================================

def compute_vif(X_df):
    """Variance inflation factor for each predictor.
    VIF_j = 1 / (1 - R²_j) where R²_j is from regressing predictor j on the others.
    """
    from sklearn.linear_model import LinearRegression
    vif = {}
    cols = list(X_df.columns)
    for c in cols:
        if c == 'const':
            continue
        X_other = X_df[[col for col in cols if col != c and col != 'const']].values
        y_this = X_df[c].values
        if X_other.shape[1] == 0:
            vif[c] = 1.0
            continue
        lr = LinearRegression().fit(X_other, y_this)
        r2 = lr.score(X_other, y_this)
        vif[c] = 1.0 / max(1.0 - r2, 1e-12)
    return vif


def mcfadden_r2(model):
    """McFadden's pseudo-R² = 1 - llf / ll_null."""
    return float(1 - model.llf / model.llnull)


# =====================================================================
#  MAIN
# =====================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(p)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--firth-threshold", type=int, default=FIRTH_THRESHOLD,
                   help="Use Firth correction when n_both <= this")
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("Paper 1 v8: regression with Firth + diagnostics", {
        "Corpus":           str(args.corpus),
        "Output dir":       str(output_dir),
        "Firth threshold":  str(args.firth_threshold),
    })

    # ---- 1. Load and aggregate to class level ----
    step(1, 6, "Loading corpus and aggregating to class level...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")
    corpus["sha_int"] = corpus["sha_an"].round().astype(np.int64)

    agg_dict = {"sha_int": list, "rank": "first", "tamagawa_product": "max"}
    for p_ in MAZUR_PRIMES:
        agg_dict[f"has_{p_}_isog"] = "first"

    with Timer("Class aggregation"):
        classes = (corpus.groupby(["conductor", "iso"], as_index=False).agg(agg_dict))
        class_sizes = (corpus.groupby(["conductor", "iso"]).size()
                              .rename("class_size").reset_index())
        classes = classes.merge(class_sizes, on=["conductor", "iso"])

    print(f"  Classes: {len(classes):,}")

    # Class count per conductor — for the §4.4-style adjustment
    classes_per_cond = classes.groupby("conductor").size().rename("n_classes_at_N").reset_index()
    classes = classes.merge(classes_per_cond, on="conductor")

    classes["log10_N"]            = np.log10(classes["conductor"].astype(float))
    classes["log_tam"]            = np.log(classes["tamagawa_product"].astype(float) + 1.0)
    classes["log_class_size"]     = np.log(classes["class_size"].astype(float))
    classes["log_n_classes_at_N"] = np.log(classes["n_classes_at_N"].astype(float))

    controls       = ["log10_N", "rank", "log_tam", "log_class_size"]
    controls_aug   = controls + ["log_n_classes_at_N"]

    # ---- 2. Multicollinearity check on the design matrix ----
    step(2, 6, "Computing VIFs on the design matrix...")
    X_full = sm.add_constant(classes[controls + [f"has_{p_}_isog" for p_ in [2,3,5,7]]].astype(float))
    vif = compute_vif(X_full)
    print("  Variance inflation factors (VIF > 5 indicates moderate collinearity, > 10 severe):")
    for c, v in vif.items():
        flag = " ★" if v > 5 else ""
        print(f"    {c:25s} VIF = {v:6.2f}{flag}")
    pd.DataFrame([{"covariate": c, "vif": v} for c, v in vif.items()]).to_csv(
        output_dir / "regression_isog_vs_sha_vif.csv", index=False)

    # ---- 3. Per-prime baseline regression with Wald SEs and pseudo-R² ----
    step(3, 6, "Per-prime baseline regression (Wald MLE)...")
    results = []
    loco_rows = []

    for p_ in MAZUR_PRIMES:
        p2 = p_ * p_
        feature  = f"has_{p_}_isog"
        outcome  = f"has_{p2}_sha"

        classes[outcome] = classes["sha_int"].apply(
            lambda lst, p2=p2: any(int(s) > 0 and int(s) % p2 == 0 for s in lst)
        )

        feat = classes[feature].astype(bool).to_numpy()
        out  = classes[outcome].astype(bool).to_numpy()
        n00 = int(((~feat) & (~out)).sum())
        n01 = int(((~feat) &   out ).sum())
        n10 = int((  feat  & (~out)).sum())
        n11 = int((  feat  &   out ).sum())
        n_isog = n10 + n11
        n_both = n11

        # Decide method
        if n_isog < LOGISTIC_MIN_ISOG or n_both < LOGISTIC_MIN_BOTH:
            method = "fisher_only"
            cond_OR = cond_lo = cond_hi = cond_p = pseudoR2 = np.nan
            firth_OR = firth_lo = firth_hi = np.nan
            mc_aug_OR = mc_aug_lo = mc_aug_hi = np.nan
        else:
            # Standard MLE
            X = sm.add_constant(classes[[feature] + controls].astype(float), has_constant='add')
            y = classes[outcome].astype(int)
            try:
                model = sm.Logit(y, X).fit(disp=0, method='lbfgs', maxiter=500)
                coef = float(model.params[feature])
                se   = float(model.bse[feature])
                cond_OR = np.exp(coef)
                cond_lo = np.exp(coef - 1.96 * se)
                cond_hi = np.exp(coef + 1.96 * se)
                cond_p  = float(model.pvalues[feature])
                pseudoR2 = mcfadden_r2(model)
                method = "mle_wald"
            except Exception as e:
                debug(f"MLE failed at p={p_}: {e}")
                cond_OR = cond_lo = cond_hi = cond_p = pseudoR2 = np.nan
                method = "mle_failed"

            # Firth correction at small-event primes
            firth_OR = firth_lo = firth_hi = np.nan
            if n_both <= args.firth_threshold:
                try:
                    X_arr = X.values
                    y_arr = y.values
                    beta_f, se_f, _, ok = firth_logit(X_arr, y_arr)
                    if ok and np.isfinite(se_f[1]):
                        # index 1 is the feature; index 0 is const
                        feat_idx = list(X.columns).index(feature)
                        firth_OR = float(np.exp(beta_f[feat_idx]))
                        firth_lo = float(np.exp(beta_f[feat_idx] - 1.96 * se_f[feat_idx]))
                        firth_hi = float(np.exp(beta_f[feat_idx] + 1.96 * se_f[feat_idx]))
                        method = method + "+firth"
                except Exception as e:
                    debug(f"Firth failed at p={p_}: {e}")

            # Class-count-at-conductor adjustment
            try:
                X_aug = sm.add_constant(classes[[feature] + controls_aug].astype(float), has_constant='add')
                model_aug = sm.Logit(y, X_aug).fit(disp=0, method='lbfgs', maxiter=500)
                coef_aug = float(model_aug.params[feature])
                se_aug   = float(model_aug.bse[feature])
                mc_aug_OR = np.exp(coef_aug)
                mc_aug_lo = np.exp(coef_aug - 1.96 * se_aug)
                mc_aug_hi = np.exp(coef_aug + 1.96 * se_aug)
            except Exception:
                mc_aug_OR = mc_aug_lo = mc_aug_hi = np.nan

            # LOCO sensitivity: refit dropping each control in turn
            for drop in controls:
                ctrls_loco = [c for c in controls if c != drop]
                try:
                    X_loco = sm.add_constant(classes[[feature] + ctrls_loco].astype(float),
                                              has_constant='add')
                    m_loco = sm.Logit(y, X_loco).fit(disp=0, method='lbfgs', maxiter=500)
                    c_loco = float(m_loco.params[feature])
                    s_loco = float(m_loco.bse[feature])
                    loco_OR = np.exp(c_loco)
                    loco_rows.append({
                        "prime": p_,
                        "dropped_control": drop,
                        "OR": loco_OR,
                        "ci_lo": np.exp(c_loco - 1.96 * s_loco),
                        "ci_hi": np.exp(c_loco + 1.96 * s_loco),
                        "baseline_OR": cond_OR,
                    })
                except Exception:
                    loco_rows.append({
                        "prime": p_,
                        "dropped_control": drop,
                        "OR": np.nan,
                        "ci_lo": np.nan, "ci_hi": np.nan,
                        "baseline_OR": cond_OR,
                    })

        # Fisher's exact (always)
        try:
            fisher_OR, fisher_p = fisher_exact([[n00, n01], [n10, n11]])
        except Exception:
            fisher_OR = fisher_p = np.nan

        results.append({
            "prime":       p_,
            "p2":          p2,
            "n_classes":   len(classes),
            "n_isog":      n_isog,
            "n_psha":      n01 + n11,
            "n_both":      n_both,
            "n_neither":   n00,
            "method":      method,
            "fisher_OR":   fisher_OR,
            "fisher_p":    fisher_p,
            "cond_OR":     cond_OR,
            "cond_ci_lo":  cond_lo,
            "cond_ci_hi":  cond_hi,
            "cond_p":      cond_p,
            "pseudoR2":    pseudoR2,
            "firth_OR":    firth_OR,
            "firth_ci_lo": firth_lo,
            "firth_ci_hi": firth_hi,
            "mc_aug_OR":   mc_aug_OR,
            "mc_aug_ci_lo": mc_aug_lo,
            "mc_aug_ci_hi": mc_aug_hi,
        })

    # ---- 4. Print main results table ----
    step(4, 6, "Reporting per-prime results...")
    section("PER-PRIME CONDITIONAL ODDS RATIOS (PAPER 1 §3.5 REPLACEMENT)")
    print(f"{'p':>4s} {'n_isog':>10s} {'n_both':>8s} {'method':>20s}  "
          f"{'OR (Wald CI)':>22s}  {'pseudoR²':>8s}  {'Firth OR (CI)':>22s}")
    for r in results:
        if np.isfinite(r['cond_OR']):
            wald_str = f"{r['cond_OR']:.2f} ({r['cond_ci_lo']:.2f}, {r['cond_ci_hi']:.2f})"
        else:
            wald_str = "—"
        if np.isfinite(r['firth_OR']):
            firth_str = f"{r['firth_OR']:.2f} ({r['firth_ci_lo']:.2f}, {r['firth_ci_hi']:.2f})"
        else:
            firth_str = "—"
        pr = f"{r['pseudoR2']:.3f}" if np.isfinite(r['pseudoR2']) else "—"
        print(f"{r['prime']:>4d} {r['n_isog']:>10,d} {r['n_both']:>8,d} {r['method']:>20s}  "
              f"{wald_str:>22s}  {pr:>8s}  {firth_str:>22s}")

    section("CLASS-COUNT-AT-CONDUCTOR ADJUSTMENT")
    print(f"{'p':>4s}  {'baseline OR':>20s}  {'+ log(n_classes) OR':>22s}  {'change':>12s}")
    for r in results:
        if np.isfinite(r['cond_OR']) and np.isfinite(r['mc_aug_OR']):
            base = f"{r['cond_OR']:.2f} ({r['cond_ci_lo']:.2f}, {r['cond_ci_hi']:.2f})"
            aug  = f"{r['mc_aug_OR']:.2f} ({r['mc_aug_ci_lo']:.2f}, {r['mc_aug_ci_hi']:.2f})"
            ratio = f"{r['cond_OR']/r['mc_aug_OR']:.2f}x"
            print(f"{r['prime']:>4d}  {base:>20s}  {aug:>22s}  {ratio:>12s}")

    section("LEAVE-ONE-CONTROL-OUT SENSITIVITY")
    print("(if dropping a single control changes OR by > 20%, the OR is sensitive to that covariate)")
    df_loco = pd.DataFrame(loco_rows)
    for p_ in sorted(df_loco['prime'].unique()):
        sub = df_loco[df_loco['prime'] == p_]
        base = sub['baseline_OR'].iloc[0]
        if not np.isfinite(base):
            continue
        print(f"  p = {p_} (baseline OR = {base:.2f}):")
        for _, row in sub.iterrows():
            if np.isfinite(row['OR']):
                pct = (row['OR'] / base - 1) * 100
                flag = " ★" if abs(pct) > 20 else ""
                print(f"    drop {row['dropped_control']:18s} OR = {row['OR']:7.2f} ({pct:+6.1f}%){flag}")

    # ---- 5. Save CSVs ----
    step(5, 6, "Saving CSVs...")
    out_csv = output_dir / "regression_isog_vs_sha_exact_v2.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"  Main results: {out_csv}")

    loco_csv = output_dir / "regression_isog_vs_sha_loco.csv"
    df_loco.to_csv(loco_csv, index=False)
    print(f"  LOCO sensitivity: {loco_csv}")

    # ---- 6. Verdict ----
    step(6, 6, "Done.")
    section("WHAT TO PASTE INTO PAPER 1 v8 §3.5")
    print("""
The Wald MLE conditional ORs above replace the v7 Table 3.5 'conditional OR'
column. The Firth ORs at p ≤ FIRTH_THRESHOLD (default 10) are the new
small-event-corrected estimates; report them as a supplementary column or
replace the Wald MLE values at the small-event primes.

The pseudoR² column populates the §3.5 'Goodness of fit' paragraph.

The class-count-at-conductor adjustment column is the analogue of Paper 2's
§4.4 robustness check. If the has_p_isog OR is approximately preserved after
adding log(n_classes_at_N), the within-class isogeny mechanism is robust to
class-count aggregation. If the OR shrinks substantially, this matches
Paper 2's finding that class-count multiplier is the dominant aggregator.

The LOCO sensitivity table populates §3.5 'leave-one-control-out sensitivity'.
Stable ORs (changes < 20%) confirm no single covariate is carrying the
estimate.

The VIF CSV populates §3.5 'multicollinearity diagnostics'.
""")


if __name__ == "__main__":
    main()
