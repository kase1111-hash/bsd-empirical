#!/usr/bin/env python3
"""paper3_lattice_spectrum_v1.py — initial corpus analysis for Paper 3.

Aggregates the MW lattice spectral statistics already computed in
ec_corpus_with_isog.parquet for rank-≥-2 classes (where the Gram matrix
is multidimensional and the spectrum is non-trivial).

Outputs:
  - paper3_spectrum_by_rank.csv          — per-rank quantile distributions
  - paper3_three_axis_concentration.csv  — joint distribution on (scale, shape, dim)
  - paper3_extremes.csv                  — top-N classes at extreme spectral values
  - paper3_regulator_regression.csv      — OLS of log(regulator) on spectral predictors
  - paper3_concentration_2d.png          — 2D density: (spec_gini, log_spec_sum) by rank

The analysis treats each isogeny class once, using the optimal-curve row.
For each (rank, statistic) pair, the CSV reports n, mean, SD, and the
quantiles q5, q25, q50, q75, q95, q99. This produces the atlas of empirical
distributions Paper 3 §3 will reference.

Author: Kase Branham — Independent Researcher
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
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

# Spectral observables (must exist as columns in the parquet)
SPECTRAL_COLS = [
    "spec_sum",           # sum of eigenvalues = trace(Gram) = scale
    "spec_lambda_min",    # smallest eigenvalue
    "spec_lambda_max",    # largest eigenvalue
    "spec_lambda_mean",   # arithmetic mean
    "spec_lambda_mid",    # middle eigenvalue
    "spec_gini",          # Gini coefficient of eigenvalues — inequality / shape
    "spec_gini_norm",
    "spec_entropy",       # Shannon entropy of normalized eigenvalues
    "spec_entropy_norm",
    "spec_participation", # inverse participation ratio — effective dimensionality
    "hermite_quotient",
    "aspect_ratio",       # lambda_max / lambda_min
    "orthogonality_defect",
    "regulator_pari",
]

# Quantile points reported per (rank, statistic)
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


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

    banner("Paper 3 v1: MW lattice spectrum atlas", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load and filter to rank ≥ 2 ----
    step(1, 6, "Loading corpus and filtering to rank ≥ 2...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")

    # Resolve column names
    def find_col(corpus, candidates):
        for c in candidates:
            if c in corpus.columns:
                return c
        raise KeyError(f"Could not find column. Tried: {candidates}")
    col_rank = find_col(corpus, ["rank"])
    col_iso  = find_col(corpus, ["iso"])
    col_cond = find_col(corpus, ["conductor", "N"])
    col_label = find_col(corpus, ["label", "cremona_label"]) if any(c in corpus.columns for c in ["label", "cremona_label"]) else None
    number_col = "curve_number" if "curve_number" in corpus.columns else ("number" if "number" in corpus.columns else None)

    # Sanity-check spectral columns exist
    missing = [c for c in SPECTRAL_COLS if c not in corpus.columns]
    if missing:
        print(f"\n  WARNING: missing columns: {missing}")
        print(f"  Available columns matching 'spec' or 'lattice':")
        for c in corpus.columns:
            if "spec" in c.lower() or "lattice" in c.lower() or "lambda" in c.lower():
                print(f"    {c}")
        spectral_cols = [c for c in SPECTRAL_COLS if c in corpus.columns]
        print(f"  Using available {len(spectral_cols)} of {len(SPECTRAL_COLS)} columns.")
    else:
        spectral_cols = SPECTRAL_COLS

    corpus = corpus[corpus[col_rank] >= 2].copy()
    print(f"  Rank ≥ 2 rows: {len(corpus):,}")

    # Aggregate to class level — use optimal curve (lowest curve_number) per class
    print("  Aggregating to class level (optimal curve)...")
    sort_cols = [col_cond, col_iso] + ([number_col] if number_col else [])
    agg_dict = {col_rank: "first"}
    for c in spectral_cols:
        agg_dict[c] = "first"
    if col_label:
        agg_dict[col_label] = "first"
    with Timer("Class aggregation"):
        classes = corpus.sort_values(sort_cols).groupby([col_cond, col_iso], as_index=False).agg(agg_dict)
    classes = classes.rename(columns={col_rank: "rank", col_cond: "conductor", col_iso: "iso"})
    print(f"  Classes: {len(classes):,}")
    print(f"  By rank:")
    for r in sorted(classes["rank"].unique()):
        n = (classes["rank"] == r).sum()
        print(f"    rank {r}: {n:>8,d}")

    # Filter rows with non-finite spectral values
    n_before = len(classes)
    for c in spectral_cols:
        classes = classes[np.isfinite(pd.to_numeric(classes[c], errors="coerce"))]
    print(f"  After filtering non-finite spectral values: {len(classes):,} (dropped {n_before-len(classes):,})")

    # ---- 2. Per-rank quantile distributions ----
    step(2, 6, "Computing per-rank quantile distributions...")
    spectrum_rows = []
    for r in sorted(classes["rank"].unique()):
        sub = classes[classes["rank"] == r]
        for c in spectral_cols:
            vals = pd.to_numeric(sub[c], errors="coerce").dropna().values
            if len(vals) == 0: continue
            row = {
                "rank": int(r), "statistic": c, "n": len(vals),
                "mean": float(np.mean(vals)),
                "sd":   float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            }
            qs = np.quantile(vals, QUANTILES)
            for q, qv in zip(QUANTILES, qs):
                row[f"q{int(q*1000)/1000}"] = float(qv)
            row["min"] = float(np.min(vals))
            row["max"] = float(np.max(vals))
            spectrum_rows.append(row)
    df_spec = pd.DataFrame(spectrum_rows)
    df_spec.to_csv(output_dir / "paper3_spectrum_by_rank.csv", index=False)
    print(f"  Saved: {output_dir / 'paper3_spectrum_by_rank.csv'}")

    # Print headline statistics
    section("HEADLINE: SPECTRAL STATISTICS BY RANK")
    for c in ["spec_gini", "spec_entropy", "spec_participation", "aspect_ratio",
              "orthogonality_defect", "hermite_quotient"]:
        if c not in spectral_cols: continue
        print(f"\n  {c}:")
        print(f"    rank   n    q5    q25   q50   q75   q95   q99   max")
        for r in sorted(classes["rank"].unique()):
            sub = classes[(classes["rank"]==r)][c].dropna()
            if len(sub) == 0: continue
            qs = sub.quantile(QUANTILES + [1.0]).values
            print(f"    {r:>4d} {len(sub):>6,d} " +
                  " ".join(f"{v:>5.3f}" for v in qs[:6]) +
                  f" {qs[6]:>6.2f}")

    # ---- 3. Three-axis decomposition: scale × shape × dimensionality ----
    step(3, 6, "Three-axis (scale × shape × dimensionality) decomposition...")
    # Scale: log(spec_sum)        — total lattice size
    # Shape: spec_gini             — eigenvalue inequality
    # Dim:   spec_participation    — effective number of eigenvalues
    if "spec_sum" in spectral_cols:
        classes["log_spec_sum"] = np.log(np.maximum(classes["spec_sum"].astype(float), 1e-30))

    three_axis_rows = []
    for r in sorted(classes["rank"].unique()):
        sub = classes[classes["rank"] == r]
        if "spec_sum" in spectral_cols:
            ls = sub["log_spec_sum"].dropna().values
        else:
            ls = np.array([])
        gn = pd.to_numeric(sub["spec_gini"], errors="coerce").dropna().values if "spec_gini" in spectral_cols else np.array([])
        pt = pd.to_numeric(sub["spec_participation"], errors="coerce").dropna().values if "spec_participation" in spectral_cols else np.array([])
        if len(ls) and len(gn):
            corr_ls_gn = float(np.corrcoef(ls[:min(len(ls),len(gn))], gn[:min(len(ls),len(gn))])[0,1])
        else:
            corr_ls_gn = np.nan
        if len(ls) and len(pt):
            corr_ls_pt = float(np.corrcoef(ls[:min(len(ls),len(pt))], pt[:min(len(ls),len(pt))])[0,1])
        else:
            corr_ls_pt = np.nan
        if len(gn) and len(pt):
            corr_gn_pt = float(np.corrcoef(gn[:min(len(gn),len(pt))], pt[:min(len(gn),len(pt))])[0,1])
        else:
            corr_gn_pt = np.nan
        three_axis_rows.append({
            "rank": int(r),
            "n": len(sub),
            "mean_log_spec_sum": float(np.mean(ls)) if len(ls) else np.nan,
            "mean_spec_gini":    float(np.mean(gn)) if len(gn) else np.nan,
            "mean_spec_participation": float(np.mean(pt)) if len(pt) else np.nan,
            "corr(log_spec_sum, spec_gini)": corr_ls_gn,
            "corr(log_spec_sum, spec_participation)": corr_ls_pt,
            "corr(spec_gini, spec_participation)": corr_gn_pt,
        })
    df_three = pd.DataFrame(three_axis_rows)
    df_three.to_csv(output_dir / "paper3_three_axis_concentration.csv", index=False)
    print(df_three.round(4).to_string(index=False))

    # ---- 4. Identify extreme specimens ----
    step(4, 6, "Identifying extreme spectral specimens...")
    extremes = []
    for c in ["spec_gini", "spec_entropy", "aspect_ratio", "orthogonality_defect",
              "hermite_quotient"]:
        if c not in spectral_cols: continue
        for direction in ["max", "min"]:
            if direction == "max":
                top = classes.nlargest(5, c)
            else:
                top = classes.nsmallest(5, c)
            for _, row in top.iterrows():
                extremes.append({
                    "statistic": c,
                    "direction": direction,
                    "value": float(row[c]),
                    "rank": int(row["rank"]),
                    "conductor": int(row["conductor"]),
                    "iso": str(row["iso"]),
                    "label": str(row[col_label]) if col_label else f"{int(row['conductor'])}{row['iso']}",
                })
    df_ext = pd.DataFrame(extremes)
    df_ext.to_csv(output_dir / "paper3_extremes.csv", index=False)
    print(df_ext.head(20).to_string(index=False))

    # ---- 5. log(regulator) regression on spectral predictors ----
    step(5, 6, "Regression: log(regulator) on spectral predictors...")
    if "regulator_pari" in spectral_cols:
        classes["log_regulator"] = np.log(np.maximum(classes["regulator_pari"].astype(float), 1e-30))
        predictors = []
        for c in ["spec_gini", "spec_entropy", "spec_participation", "aspect_ratio",
                  "orthogonality_defect", "hermite_quotient", "log_spec_sum"]:
            if c in classes.columns and classes[c].notna().any():
                predictors.append(c)

        # Run regression per rank stratum + pooled
        reg_rows = []
        for r in [None] + sorted(classes["rank"].unique().tolist()):
            sub = classes if r is None else classes[classes["rank"] == r]
            sub = sub.dropna(subset=predictors + ["log_regulator"])
            if len(sub) < 50: continue
            X = sm.add_constant(sub[predictors].astype(float))
            y = sub["log_regulator"].astype(float)
            try:
                model = sm.OLS(y, X).fit(cov_type="HC3")
                for v in ["const"] + predictors:
                    reg_rows.append({
                        "stratum": "pooled" if r is None else f"rank_{r}",
                        "predictor": v,
                        "coef": float(model.params[v]),
                        "hc3_se": float(model.bse[v]),
                        "t": float(model.tvalues[v]),
                        "p": float(model.pvalues[v]),
                        "R2": float(model.rsquared),
                        "n": int(model.nobs),
                    })
                print(f"  stratum {'pooled' if r is None else f'rank={r}'}: R² = {model.rsquared:.4f}, n = {len(sub):,}")
            except Exception as e:
                print(f"  Regression failed at stratum {r}: {e}")
        pd.DataFrame(reg_rows).to_csv(output_dir / "paper3_regulator_regression.csv", index=False)
        print(f"  Saved: {output_dir / 'paper3_regulator_regression.csv'}")

    # ---- 6. Visualization: 2D density on (gini, log_spec_sum) by rank ----
    step(6, 6, "Generating Figure 1 — 2D density on (gini, log_spec_sum) by rank...")
    if "spec_gini" in spectral_cols and "log_spec_sum" in classes.columns:
        rank_strata = sorted([r for r in classes["rank"].unique() if r >= 2])
        n_plots = len(rank_strata)
        ncols = 2
        nrows = (n_plots + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows), squeeze=False)
        for i, r in enumerate(rank_strata):
            ax = axes[i // ncols, i % ncols]
            sub = classes[classes["rank"] == r].dropna(subset=["spec_gini", "log_spec_sum"])
            if len(sub) == 0:
                ax.set_title(f"rank {r} (n = 0)")
                continue
            hb = ax.hexbin(sub["spec_gini"], sub["log_spec_sum"], gridsize=40,
                          cmap="YlOrRd", mincnt=1, bins="log")
            ax.set_xlabel("spec_gini (eigenvalue inequality)")
            ax.set_ylabel("log(spec_sum) (lattice scale)")
            ax.set_title(f"rank {r} (n = {len(sub):,})")
            plt.colorbar(hb, ax=ax, label="count (log)")
        # Hide unused subplot
        if len(rank_strata) < nrows * ncols:
            for j in range(len(rank_strata), nrows * ncols):
                axes[j // ncols, j % ncols].axis("off")
        plt.suptitle("Figure 1. Joint distribution of (spec_gini, log_spec_sum) by rank stratum")
        plt.tight_layout()
        plt.savefig(output_dir / "paper3_concentration_2d.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_dir / 'paper3_concentration_2d.png'}")
        plt.close()

    section("PAPER 3 v1 INITIAL OUTPUTS COMPLETE")
    print(f"""
Five outputs in {output_dir}:

  paper3_spectrum_by_rank.csv          — per-rank quantile atlas
  paper3_three_axis_concentration.csv  — scale × shape × dim correlations
  paper3_extremes.csv                  — top/bottom 5 classes per statistic
  paper3_regulator_regression.csv      — OLS of log(regulator) on spectral predictors
  paper3_concentration_2d.png          — 2D density on (gini, log_spec_sum) by rank

Send these back to Claude. Paper 3 v1 draft will be produced from these
together with the rank-3 single-specimen story (sole rank-4 class at
234446a1, single rank-3 sub-corpus of 8,899 classes) and any structural
concentration points visible in the 2D density.

If any new spectral observables would be useful (e.g. condition number,
specific eigenvalue ratios, Voronoi-cell sphericity from the Gram matrix),
flag them and the script can be extended for v2.
""")


if __name__ == "__main__":
    main()
