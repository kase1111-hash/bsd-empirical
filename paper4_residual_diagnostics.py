#!/usr/bin/env python3
"""paper4_residual_diagnostics.py — residual diagnostics + visualization for v5.

Companion to regression_v2_deg_phi_v2.py. Re-fits the baseline OLS model and
produces:

  1.  Residual histogram (PNG)
  2.  Residual QQ plot (PNG)
  3.  Residual vs fitted scatter (hexbin) (PNG)
  4.  Hexbin of v_2(deg φ) vs v_2(c) colored by rank (PNG)
  5.  Breusch-Pagan heteroscedasticity test
  6.  Residual SD, quantiles, tail behavior
  7.  Poisson + Negative Binomial robustness comparison
  8.  CSV summary of all diagnostic numbers

The figures address point 7 of the v3 review (need at least one visualization)
and the residual stats address point 2 (need residual diagnostics).

Author: Kase Branham — Independent Researcher
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    Timer, debug,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

BASE_PREDICTORS = ["rank", "v2_c", "v2_N", "omega_N", "log10_N", "v2_sha_an", "v2_torsion"]


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

    banner("Paper 4 v5: residual diagnostics + visualizations", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load corpus and rebuild class-level data ----
    step(1, 7, "Loading corpus and rebuilding class-level data...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")

    def find_col(corpus, candidates):
        for c in candidates:
            if c in corpus.columns:
                return c
        raise KeyError(f"Could not find column. Tried: {candidates}")

    col_degphi  = find_col(corpus, ["deg_phi", "degphi", "modular_degree"])
    col_tam     = find_col(corpus, ["tamagawa_product", "tam_prod"])
    col_sha     = find_col(corpus, ["sha_an", "sha"])
    col_torsion = find_col(corpus, ["torsion_order", "torsion"])
    col_rank    = find_col(corpus, ["rank"])
    col_iso     = find_col(corpus, ["iso"])
    col_cond    = find_col(corpus, ["conductor", "N"])
    number_col = "curve_number" if "curve_number" in corpus.columns else "number"

    # Filter rows with missing required values
    required = [col_degphi, col_tam, col_sha, col_torsion, col_rank]
    fmask = np.ones(len(corpus), dtype=bool)
    for c in required:
        fmask &= np.isfinite(pd.to_numeric(corpus[c], errors="coerce").values)
    corpus = corpus.loc[fmask].copy()
    print(f"  Surviving rows: {len(corpus):,}")

    def v2_vec(arr):
        arr = np.asarray(arr, dtype=np.int64)
        out = np.zeros(arr.shape, dtype=np.int32)
        mask = arr > 0
        if mask.any():
            lowbit = arr[mask] & -arr[mask]
            out[mask] = np.log2(lowbit).astype(np.int32)
        return out

    corpus["v2_degphi"]  = v2_vec(corpus[col_degphi].astype(np.int64).values)
    corpus["v2_c"]       = v2_vec(corpus[col_tam].astype(np.int64).values)
    corpus["v2_sha_an"]  = v2_vec(corpus[col_sha].round().astype(np.int64).values)
    corpus["v2_torsion"] = v2_vec(corpus[col_torsion].astype(np.int64).values)
    corpus["rank"]       = corpus[col_rank].astype(np.int64)

    # Conductor structure
    print("  Computing conductor structure...")
    unique_N = corpus[col_cond].unique()
    N_v2 = {n: int(np.log2(n & -n)) if n > 0 else 0 for n in unique_N}
    def omega(n):
        n = int(n); k = 0
        for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]:
            if n % p == 0:
                k += 1
                while n % p == 0: n //= p
        if n > 1: k += 1
        return k
    N_omega = {n: omega(n) for n in unique_N}
    corpus["v2_N"]    = corpus[col_cond].map(N_v2)
    corpus["omega_N"] = corpus[col_cond].map(N_omega)
    corpus["log10_N"] = np.log10(corpus[col_cond].astype(float))
    corpus["conductor"] = corpus[col_cond]
    corpus["iso"] = corpus[col_iso]

    # Class-level aggregation (optimal curve)
    sort_cols = ["conductor", "iso"] + ([number_col] if number_col in corpus.columns else [])
    classes = corpus.sort_values(sort_cols).groupby(["conductor", "iso"], as_index=False).agg({
        "v2_degphi": "first", "rank": "first", "v2_c": "max", "v2_N": "first",
        "omega_N": "first", "log10_N": "first", "v2_sha_an": "max", "v2_torsion": "first",
    })
    print(f"  Classes: {len(classes):,}")

    # ---- 2. Re-fit baseline OLS, extract residuals and fitted values ----
    step(2, 7, "Re-fitting baseline OLS with HC3 SEs...")
    y = classes["v2_degphi"].astype(float).values
    X = sm.add_constant(classes[BASE_PREDICTORS].astype(float))
    model = sm.OLS(y, X).fit(cov_type="HC3")
    fitted = model.fittedvalues
    resid  = y - fitted
    print(f"  R² = {model.rsquared:.4f}")
    print(f"  N = {len(y):,}")

    # ---- 3. Residual statistics ----
    step(3, 7, "Computing residual statistics...")
    resid_sd = np.std(resid, ddof=X.shape[1])
    quantiles = np.quantile(resid, [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
    print(f"  Residual SD: {resid_sd:.4f}")
    print(f"  Residual quantiles:")
    print(f"    q0.1%:  {quantiles[0]:+.3f}")
    print(f"    q1%:    {quantiles[1]:+.3f}")
    print(f"    q5%:    {quantiles[2]:+.3f}")
    print(f"    q25%:   {quantiles[3]:+.3f}")
    print(f"    q50%:   {quantiles[4]:+.3f}")
    print(f"    q75%:   {quantiles[5]:+.3f}")
    print(f"    q95%:   {quantiles[6]:+.3f}")
    print(f"    q99%:   {quantiles[7]:+.3f}")
    print(f"    q99.9%: {quantiles[8]:+.3f}")
    print(f"  Min: {resid.min():+.3f}, Max: {resid.max():+.3f}")
    skew = float(((resid - resid.mean())**3).mean() / (resid.std()**3))
    kurt = float(((resid - resid.mean())**4).mean() / (resid.std()**4)) - 3
    print(f"  Skewness: {skew:+.3f}, Excess kurtosis: {kurt:+.3f}")

    # ---- 4. Breusch-Pagan heteroscedasticity test ----
    step(4, 7, "Running Breusch-Pagan test for heteroscedasticity...")
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(resid, X)
    print(f"  Breusch-Pagan LM statistic: {bp_lm:.2f}")
    print(f"  Breusch-Pagan LM p-value: {bp_lm_p:.2e}")
    print(f"  Breusch-Pagan F p-value: {bp_f_p:.2e}")
    print(f"  (Significant heteroscedasticity expected at large N; HC3 SEs already handle this)")

    # ---- 5. Poisson + NB robustness ----
    step(5, 7, "Fitting Poisson and Negative Binomial robustness models...")
    try:
        poisson = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        print(f"  Poisson R²-pseudo: 1 - LL/LL_null = {1 - poisson.llf/poisson.llnull:.4f}")
        print(f"  Poisson rank coef: {poisson.params['rank']:+.4f}")
        print(f"  Poisson v_2(c) coef: {poisson.params['v2_c']:+.4f}")
    except Exception as e:
        print(f"  Poisson fit failed: {e}")

    try:
        nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        print(f"  NB R²-pseudo: 1 - LL/LL_null = {1 - nb.llf/nb.llnull:.4f}")
        print(f"  NB rank coef: {nb.params['rank']:+.4f}")
        print(f"  NB v_2(c) coef: {nb.params['v2_c']:+.4f}")
    except Exception as e:
        print(f"  NB fit failed: {e}")

    # ---- 6. Generate figures ----
    step(6, 7, "Generating residual diagnostic figures...")

    # Figure 2: Residual histogram + QQ plot side by side
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    axL.hist(resid, bins=80, color='#5a8fd0', edgecolor='#1f4e79', linewidth=0.5)
    axL.axvline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
    axL.set_xlabel('OLS residual: v_2(deg φ) − fitted')
    axL.set_ylabel('Count')
    axL.set_title(f'Residual histogram\n(N = {len(resid):,}, SD = {resid_sd:.3f}, skew = {skew:+.2f})')
    axL.spines['top'].set_visible(False)
    axL.spines['right'].set_visible(False)

    # QQ plot — sample residuals against normal quantiles
    sorted_resid = np.sort(resid)
    n = len(sorted_resid)
    # Use a sample for plotting if too many points
    if n > 50000:
        idx = np.linspace(0, n-1, 50000).astype(int)
        sorted_resid_plot = sorted_resid[idx]
    else:
        sorted_resid_plot = sorted_resid
    from scipy.stats import norm as scipy_norm
    qq_x = scipy_norm.ppf((np.arange(1, len(sorted_resid_plot) + 1) - 0.5) / len(sorted_resid_plot))
    axR.scatter(qq_x, sorted_resid_plot, s=2, alpha=0.4, color='#1f4e79')
    lo, hi = min(qq_x.min(), sorted_resid_plot.min()), max(qq_x.max(), sorted_resid_plot.max())
    axR.plot([lo, hi], [lo*resid_sd, hi*resid_sd], color='red', linewidth=1, linestyle='--')
    axR.set_xlabel('Theoretical normal quantile')
    axR.set_ylabel('Empirical residual quantile')
    axR.set_title(f'Normal QQ plot\n(excess kurtosis = {kurt:+.2f}; OLS valid asymptotically)')
    axR.spines['top'].set_visible(False)
    axR.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'paper4_fig2_residual_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'paper4_fig2_residual_diagnostics.png'}")
    plt.close()

    # Figure 3: Residual vs fitted (hexbin)
    fig, ax = plt.subplots(figsize=(9, 5))
    hb = ax.hexbin(fitted, resid, gridsize=50, cmap='Blues', mincnt=1, bins='log')
    ax.axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_xlabel('Fitted v_2(deg φ)')
    ax.set_ylabel('Residual')
    ax.set_title('Residual vs fitted (hexbin, log color scale)')
    plt.colorbar(hb, ax=ax, label='Count (log)')
    plt.tight_layout()
    plt.savefig(output_dir / 'paper4_fig3_residual_vs_fitted.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'paper4_fig3_residual_vs_fitted.png'}")
    plt.close()

    # Figure 4: hexbin of v_2(deg φ) vs v_2(c), colored by rank stratification
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    rank_strata = [0, 1, 2, 3]
    for ax, r in zip(axes.flat, rank_strata):
        sub = classes[classes['rank'] == r]
        if len(sub) == 0:
            ax.set_title(f'rank {r} (n = 0)')
            continue
        hb = ax.hexbin(sub['v2_c'], sub['v2_degphi'], gridsize=25, cmap='YlOrRd',
                       mincnt=1, bins='log')
        ax.plot([0, 10], [0, 10], color='red', linewidth=0.5, linestyle='--', alpha=0.5, label='y = v_2(c)')
        ax.axhline(r, color='green', linewidth=0.8, linestyle=':', alpha=0.7, label=f'Watkins: y ≥ {r}')
        ax.set_title(f'rank {r} (n = {len(sub):,})')
        ax.set_xlabel('v_2(c) (2-adic Tamagawa)')
        ax.set_ylabel('v_2(deg φ)')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_xlim(-0.5, 10)
        ax.set_ylim(-0.5, 18)
    plt.suptitle('Figure 4. Joint distribution of v_2(deg φ) and v_2(c) by rank stratum', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'paper4_fig4_hexbin_by_rank.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'paper4_fig4_hexbin_by_rank.png'}")
    plt.close()

    # ---- 7. Save residual stats CSV ----
    step(7, 7, "Saving residual statistics CSV...")
    stats = {
        "n":               len(y),
        "rsquared":        float(model.rsquared),
        "rsquared_adj":    float(model.rsquared_adj),
        "residual_sd":     float(resid_sd),
        "residual_skew":   skew,
        "residual_kurt":   kurt,
        "residual_min":    float(resid.min()),
        "residual_max":    float(resid.max()),
        "q0.001":          float(quantiles[0]),
        "q0.01":           float(quantiles[1]),
        "q0.05":           float(quantiles[2]),
        "q0.25":           float(quantiles[3]),
        "q0.50":           float(quantiles[4]),
        "q0.75":           float(quantiles[5]),
        "q0.95":           float(quantiles[6]),
        "q0.99":           float(quantiles[7]),
        "q0.999":          float(quantiles[8]),
        "bp_lm":           float(bp_lm),
        "bp_lm_p":         float(bp_lm_p),
        "bp_f_p":          float(bp_f_p),
    }
    pd.DataFrame([stats]).to_csv(output_dir / "paper4_residual_stats.csv", index=False)
    print(f"  Saved: {output_dir / 'paper4_residual_stats.csv'}")

    section("FILES PRODUCED")
    print("""
The following files are now available for Paper 4 v5:

  paper4_fig2_residual_diagnostics.png  — histogram + QQ plot
  paper4_fig3_residual_vs_fitted.png    — residual vs fitted hexbin
  paper4_fig4_hexbin_by_rank.png        — v_2(deg φ) vs v_2(c) hexbin by rank
  paper4_residual_stats.csv             — residual statistics

These address review point 7 (visualizations) and review point 2 (residual
diagnostics) of the v3 review of Paper 4. The Breusch-Pagan test will likely
reject the null of homoscedasticity at this sample size; this is expected for
a discrete heavy-tailed dependent variable, and HC3 SEs already account for it.
""")


if __name__ == "__main__":
    main()
