#!/usr/bin/env python3
"""paper3_lattice_spectrum_v2.py — rank-aware diagnostic upgrade.

v1 dropped 279,056 rank-2 classes because at least one spectral column was
non-finite per row — most likely spec_lambda_mid, which has no meaning for
2-dimensional Gram matrices. This v2 first reports per-column NaN counts
stratified by rank, then applies a stratum-appropriate filter that retains
rank-2 classes.

Outputs:
  - paper3_nan_audit.csv               — per-rank per-column NaN counts
  - paper3_spectrum_by_rank.csv        — per-rank quantile distributions
  - paper3_three_axis.csv              — scale × shape correlations
  - paper3_extremes.csv                — top/bottom 5 classes per statistic
  - paper3_regulator_regression.csv    — OLS of log(regulator) on spectral
  - paper3_concentration_2d.png        — 2D density (gini, log_spec_sum) by rank

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

from kase_utils import add_standard_args, resolve_args, banner, section, step, Timer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

# Spectral columns are split by whether they require r ≥ 3
# (spec_lambda_mid has no meaning for r = 2)
CORE_SPECTRAL = [
    "spec_sum",
    "spec_lambda_min",
    "spec_lambda_max",
    "spec_lambda_mean",
    "spec_gini",
    "spec_gini_norm",
    "spec_entropy",
    "spec_entropy_norm",
    "spec_participation",
    "hermite_quotient",
    "aspect_ratio",
    "orthogonality_defect",
    "regulator_pari",
]
RANK_3_PLUS_SPECTRAL = ["spec_lambda_mid"]

ALL_SPECTRAL = CORE_SPECTRAL + RANK_3_PLUS_SPECTRAL
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

    banner("Paper 3 v2: lattice spectrum atlas with rank-aware filter", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load and filter to rank ≥ 2 ----
    step(1, 7, "Loading corpus and filtering to rank ≥ 2...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")

    def find_col(corpus, candidates):
        for c in candidates:
            if c in corpus.columns:
                return c
        raise KeyError(f"Could not find column. Tried: {candidates}")
    col_rank = find_col(corpus, ["rank"])
    col_iso  = find_col(corpus, ["iso"])
    col_cond = find_col(corpus, ["conductor", "N"])
    col_label = "label" if "label" in corpus.columns else None
    number_col = "curve_number" if "curve_number" in corpus.columns else None

    present_spectral = [c for c in ALL_SPECTRAL if c in corpus.columns]
    missing_spectral = [c for c in ALL_SPECTRAL if c not in corpus.columns]
    if missing_spectral:
        print(f"  Missing spectral columns: {missing_spectral}")

    corpus = corpus[corpus[col_rank] >= 2].copy()
    print(f"  Rank ≥ 2 rows: {len(corpus):,}")

    # Aggregate to class level — optimal curve only
    sort_cols = [col_cond, col_iso] + ([number_col] if number_col else [])
    agg_dict = {col_rank: "first"}
    for c in present_spectral:
        agg_dict[c] = "first"
    if col_label:
        agg_dict[col_label] = "first"
    with Timer("Class aggregation"):
        classes = corpus.sort_values(sort_cols).groupby(
            [col_cond, col_iso], as_index=False).agg(agg_dict)
    classes = classes.rename(columns={col_rank: "rank", col_cond: "conductor", col_iso: "iso"})
    print(f"  Classes: {len(classes):,}")

    # ---- 2. NaN audit per (rank, spectral column) ----
    step(2, 7, "Auditing NaN counts per (rank, column)...")
    audit_rows = []
    print(f"\n  {'rank':>4s} {'n':>8s}  " + " ".join(f"{c[:16]:>16s}" for c in present_spectral))
    for r in sorted(classes["rank"].unique()):
        sub = classes[classes["rank"] == r]
        nans_per_col = {}
        for c in present_spectral:
            n_nan = int(sub[c].isna().sum())
            nans_per_col[c] = n_nan
            audit_rows.append({
                "rank": int(r),
                "column": c,
                "n_total": len(sub),
                "n_nan":   n_nan,
                "frac_nan": n_nan / len(sub) if len(sub) else np.nan,
            })
        print(f"  {r:>4d} {len(sub):>8,d}  " +
              " ".join(f"{nans_per_col[c]:>16,d}" for c in present_spectral))
    pd.DataFrame(audit_rows).to_csv(output_dir / "paper3_nan_audit.csv", index=False)
    print(f"\n  Saved: {output_dir / 'paper3_nan_audit.csv'}")

    # ---- 3. Apply rank-aware filter ----
    step(3, 7, "Applying rank-aware filter...")
    # For each row, the required spectral columns depend on rank.
    # rank 2: drop only rows where core spectral cols are NaN (skip spec_lambda_mid)
    # rank ≥ 3: drop rows where ANY spectral col is NaN (including spec_lambda_mid)
    n_before = len(classes)

    def row_is_finite(row):
        r = row["rank"]
        required = [c for c in CORE_SPECTRAL if c in present_spectral]
        if r >= 3:
            required = required + [c for c in RANK_3_PLUS_SPECTRAL if c in present_spectral]
        return all(np.isfinite(pd.to_numeric(row[c], errors="coerce")) if c in row else False
                   for c in required)

    # Vectorized version (much faster than apply):
    core_present = [c for c in CORE_SPECTRAL if c in present_spectral]
    rank_3_present = [c for c in RANK_3_PLUS_SPECTRAL if c in present_spectral]

    mask_core = np.ones(len(classes), dtype=bool)
    for c in core_present:
        mask_core &= np.isfinite(pd.to_numeric(classes[c], errors="coerce").values)
    mask_rank2 = (classes["rank"].values == 2) & mask_core
    if rank_3_present:
        mask_rank_3plus_core = mask_core
        mask_rank_3plus_extra = np.ones(len(classes), dtype=bool)
        for c in rank_3_present:
            mask_rank_3plus_extra &= np.isfinite(pd.to_numeric(classes[c], errors="coerce").values)
        mask_rank_3plus = (classes["rank"].values >= 3) & mask_rank_3plus_core & mask_rank_3plus_extra
    else:
        mask_rank_3plus = (classes["rank"].values >= 3) & mask_core

    final_mask = mask_rank2 | mask_rank_3plus
    classes = classes.loc[final_mask].copy()
    print(f"  Before: {n_before:,}   After rank-aware filter: {len(classes):,}")
    print(f"  By rank now:")
    for r in sorted(classes["rank"].unique()):
        n = (classes["rank"] == r).sum()
        print(f"    rank {r}: {n:>8,d}")

    if "spec_sum" in present_spectral:
        classes["log_spec_sum"] = np.log(np.maximum(classes["spec_sum"].astype(float), 1e-30))

    # If we didn't recover rank-2, stop and report
    if (classes["rank"] == 2).sum() == 0:
        print("\n  STILL ZERO rank-2 classes after rank-aware filter.")
        print("  This means rank-2 classes have NaN in one or more CORE spectral columns.")
        print("  Inspect paper3_nan_audit.csv to see which columns are NaN for rank-2 rows.")
        print("  Edit the script's CORE_SPECTRAL list to remove non-essential columns, or recompute.")
        return

    # ---- 4. Per-rank quantile distributions ----
    step(4, 7, "Computing per-rank quantile distributions...")
    quantile_stats = [c for c in present_spectral if c not in ("regulator_pari",)]
    rows = []
    for r in sorted(classes["rank"].unique()):
        sub = classes[classes["rank"] == r]
        for c in quantile_stats:
            vals = pd.to_numeric(sub[c], errors="coerce").dropna().values
            if len(vals) == 0: continue
            row = {"rank": int(r), "statistic": c, "n": len(vals),
                   "mean": float(np.mean(vals)),
                   "sd":   float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan}
            qs = np.quantile(vals, QUANTILES)
            for q, qv in zip(QUANTILES, qs):
                row[f"q{q:.3f}"] = float(qv)
            row["min"] = float(np.min(vals)); row["max"] = float(np.max(vals))
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "paper3_spectrum_by_rank.csv", index=False)

    section("HEADLINE: KEY SPECTRAL STATISTICS BY RANK")
    for c in ["spec_gini", "spec_entropy", "spec_participation", "aspect_ratio",
              "orthogonality_defect", "hermite_quotient"]:
        if c not in present_spectral: continue
        print(f"\n  {c}:")
        print(f"    rank      n      q5     q25     q50     q75     q95     q99       max")
        for r in sorted(classes["rank"].unique()):
            sub = pd.to_numeric(classes[classes["rank"]==r][c], errors="coerce").dropna()
            if len(sub) == 0: continue
            qs = sub.quantile(QUANTILES + [1.0]).values
            print(f"    {r:>4d} {len(sub):>8,d} " +
                  " ".join(f"{v:>7.3f}" for v in qs[:6]) +
                  f" {qs[6]:>9.2f}")

    # ---- 5. Three-axis correlations ----
    step(5, 7, "Computing three-axis correlations...")
    three_axis = []
    for r in sorted(classes["rank"].unique()):
        sub = classes[classes["rank"] == r]
        ls = sub["log_spec_sum"].dropna().values if "log_spec_sum" in classes.columns else np.array([])
        gn = pd.to_numeric(sub["spec_gini"], errors="coerce").dropna().values if "spec_gini" in classes.columns else np.array([])
        pt = pd.to_numeric(sub["spec_participation"], errors="coerce").dropna().values if "spec_participation" in classes.columns else np.array([])
        def corr(a, b):
            if len(a) < 2 or len(b) < 2: return np.nan
            n = min(len(a), len(b))
            return float(np.corrcoef(a[:n], b[:n])[0,1])
        three_axis.append({
            "rank": int(r), "n": len(sub),
            "mean_log_spec_sum":       float(np.mean(ls)) if len(ls) else np.nan,
            "mean_spec_gini":          float(np.mean(gn)) if len(gn) else np.nan,
            "mean_spec_participation": float(np.mean(pt)) if len(pt) else np.nan,
            "corr_logsum_gini":        corr(ls, gn),
            "corr_logsum_part":        corr(ls, pt),
            "corr_gini_part":          corr(gn, pt),
        })
    df_three = pd.DataFrame(three_axis)
    df_three.to_csv(output_dir / "paper3_three_axis.csv", index=False)
    section("THREE-AXIS CORRELATIONS")
    print(df_three.round(4).to_string(index=False))

    # ---- 6. Extreme specimens + regulator regression ----
    step(6, 7, "Extreme specimens and regulator regression...")
    extremes = []
    for c in ["spec_gini", "spec_entropy", "aspect_ratio", "orthogonality_defect",
              "hermite_quotient"]:
        if c not in present_spectral: continue
        for direction in ["max", "min"]:
            top = classes.nlargest(5, c) if direction == "max" else classes.nsmallest(5, c)
            for _, row in top.iterrows():
                extremes.append({
                    "statistic": c, "direction": direction,
                    "value": float(row[c]), "rank": int(row["rank"]),
                    "conductor": int(row["conductor"]), "iso": str(row["iso"]),
                    "label": str(row.get(col_label, f"{int(row['conductor'])}{row['iso']}")),
                })
    pd.DataFrame(extremes).to_csv(output_dir / "paper3_extremes.csv", index=False)

    # Regulator regression (skip if regulator isn't there)
    if "regulator_pari" in present_spectral:
        classes["log_regulator"] = np.log(np.maximum(classes["regulator_pari"].astype(float), 1e-30))
        # Predictors that aren't trivially deterministic of the regulator. We exclude
        # spec_sum and lambda_min/max/mean because they are direct functions of the
        # eigenvalues whose product is the regulator. The point of this regression is
        # to ask how much of log(reg) is explained by SHAPE statistics that are
        # invariant under uniform scaling: gini, entropy, participation, aspect,
        # orthogonality_defect, hermite_quotient.
        predictors = [c for c in ["spec_gini", "spec_entropy", "spec_participation",
                                  "aspect_ratio", "orthogonality_defect", "hermite_quotient"]
                      if c in present_spectral]
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
                        "coef":   float(model.params[v]),
                        "hc3_se": float(model.bse[v]),
                        "t":      float(model.tvalues[v]),
                        "p":      float(model.pvalues[v]),
                        "R2":     float(model.rsquared),
                        "n":      int(model.nobs),
                    })
                print(f"  stratum {'pooled' if r is None else f'rank={r}'}: "
                      f"R² = {model.rsquared:.4f}, n = {len(sub):,}")
            except Exception as e:
                print(f"  Regression failed at stratum {r}: {e}")
        pd.DataFrame(reg_rows).to_csv(output_dir / "paper3_regulator_regression.csv", index=False)

    # ---- 7. Visualization ----
    step(7, 7, "Generating Figure 1...")
    if "spec_gini" in present_spectral and "log_spec_sum" in classes.columns:
        rank_strata = sorted([r for r in classes["rank"].unique() if r >= 2])
        n_plots = len(rank_strata)
        ncols = min(2, n_plots)
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows), squeeze=False)
        for i, r in enumerate(rank_strata):
            ax = axes[i // ncols, i % ncols]
            sub = classes[classes["rank"] == r].dropna(subset=["spec_gini", "log_spec_sum"])
            if len(sub) == 0:
                ax.set_title(f"rank {r} (n = 0)"); continue
            hb = ax.hexbin(sub["spec_gini"], sub["log_spec_sum"],
                          gridsize=40, cmap="YlOrRd", mincnt=1, bins="log")
            ax.set_xlabel("spec_gini")
            ax.set_ylabel("log(spec_sum)")
            ax.set_title(f"rank {r} (n = {len(sub):,})")
            plt.colorbar(hb, ax=ax, label="count (log)")
        for j in range(len(rank_strata), nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")
        plt.suptitle("Figure 1. Joint distribution of (spec_gini, log_spec_sum) by rank")
        plt.tight_layout()
        plt.savefig(output_dir / "paper3_concentration_2d.png", dpi=150, bbox_inches="tight")
        plt.close()

    section("DONE")
    print(f"""
Outputs in {output_dir}:
  paper3_nan_audit.csv               — diagnostic (which columns are NaN per rank)
  paper3_spectrum_by_rank.csv        — per-rank quantile atlas
  paper3_three_axis.csv              — scale × shape correlations
  paper3_extremes.csv                — top/bottom 5 classes per statistic
  paper3_regulator_regression.csv    — shape regression on log(regulator)
  paper3_concentration_2d.png        — 2D density (gini, log_spec_sum) by rank

If rank-2 is still empty after this run, send paper3_nan_audit.csv back and
we'll see exactly which CORE columns are NaN for rank-2 classes; that tells
us whether the spectral statistics need to be recomputed for rank-2 or
whether a different exclusion strategy can recover them.
""")


if __name__ == "__main__":
    main()
