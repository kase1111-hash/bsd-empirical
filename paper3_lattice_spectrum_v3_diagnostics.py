#!/usr/bin/env python3
"""paper3_lattice_spectrum_v3_diagnostics.py — diagnostic upgrade for Paper 3 v3.

Addresses two reviewer requests from the v1 review:

  1.  Conductor-stratified rank-2 vs rank-3 comparison (review point 4):
      compares spec_gini_norm, spec_entropy_norm, spec_participation, and
      spec_gini between rank 2 and rank 3 within a common conductor window
      (the rank-3 cohort sits at N ≥ 194,040, so we restrict both ranks to
      that window). Outputs paper3_conductor_stratified.csv with
      per-stratum quantile distributions for both ranks.

  2.  Gini-vs-participation scatter (review point 7):
      a scatter plot showing the near-bijective relationship between
      spec_gini and spec_participation, by rank. The reviewer described
      this as "devastatingly convincing" relative to reporting the
      correlation coefficient alone. Outputs paper3_gini_vs_participation.png
      (Figure 2).

Both outputs are inputs for Paper 3 v3.

Author: Kase Branham — Independent Researcher
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kase_utils import add_standard_args, resolve_args, banner, section, step, Timer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(p)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--cond-min", type=int, default=194040,
                   help="Minimum conductor for stratified comparison (default: 194040, the rank-3 floor)")
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("Paper 3 v3 diagnostics: conductor stratification + gini-vs-participation", {
        "Corpus":           str(args.corpus),
        "Output dir":       str(output_dir),
        "Conductor floor":  f"N >= {args.cond_min:,}",
    })

    # ---- 1. Load and aggregate ----
    step(1, 4, "Loading corpus and aggregating to class level (rank >= 2)...")
    corpus = pd.read_parquet(args.corpus)
    corpus = corpus[corpus["rank"] >= 2].copy()
    print(f"  Rank >= 2 rows: {len(corpus):,}")

    number_col = "curve_number" if "curve_number" in corpus.columns else None
    sort_cols = ["conductor", "iso"] + ([number_col] if number_col else [])
    spectral_cols = ["spec_sum", "spec_lambda_min", "spec_lambda_max", "spec_lambda_mean",
                     "spec_gini", "spec_gini_norm", "spec_entropy", "spec_entropy_norm",
                     "spec_participation", "hermite_quotient", "aspect_ratio",
                     "orthogonality_defect", "regulator_pari"]
    present = [c for c in spectral_cols if c in corpus.columns]

    agg_dict = {"rank": "first"}
    for c in present:
        agg_dict[c] = "first"
    with Timer("Class aggregation"):
        classes = corpus.sort_values(sort_cols).groupby(
            ["conductor", "iso"], as_index=False).agg(agg_dict)

    # Apply rank-aware NaN filter (rank 2 doesn't have spec_lambda_mid)
    core = [c for c in present if c != "spec_lambda_mid"]
    mask = np.ones(len(classes), dtype=bool)
    for c in core:
        mask &= np.isfinite(pd.to_numeric(classes[c], errors="coerce").values)
    classes = classes.loc[mask].copy()
    print(f"  Classes after filter: {len(classes):,}")
    for r in sorted(classes["rank"].unique()):
        print(f"    rank {r}: {(classes['rank']==r).sum():>8,d}")

    # ---- 2. Conductor-stratified comparison ----
    step(2, 4, f"Conductor-stratified comparison (N >= {args.cond_min:,})...")
    classes_high = classes[classes["conductor"] >= args.cond_min].copy()
    print(f"  Classes with N >= {args.cond_min:,}: {len(classes_high):,}")
    for r in sorted(classes_high["rank"].unique()):
        n_high = (classes_high["rank"] == r).sum()
        n_all  = (classes["rank"] == r).sum()
        print(f"    rank {r}: {n_high:>8,d} of {n_all:>8,d} ({100*n_high/n_all:.1f}%)")

    # Comparison statistics
    compare_stats = ["spec_gini", "spec_gini_norm", "spec_entropy", "spec_entropy_norm",
                     "spec_participation", "aspect_ratio", "orthogonality_defect", "hermite_quotient"]
    compare_stats = [c for c in compare_stats if c in classes.columns]

    rows = []
    for window_label, window_df in [("full_corpus", classes), (f"N>={args.cond_min}", classes_high)]:
        for r in [2, 3]:
            sub = window_df[window_df["rank"] == r]
            if len(sub) == 0: continue
            for c in compare_stats:
                vals = pd.to_numeric(sub[c], errors="coerce").dropna().values
                if len(vals) == 0: continue
                row = {
                    "window": window_label,
                    "rank": r,
                    "statistic": c,
                    "n": len(vals),
                    "mean": float(np.mean(vals)),
                    "sd":   float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                }
                qs = np.quantile(vals, QUANTILES)
                for q, qv in zip(QUANTILES, qs):
                    row[f"q{q:.3f}"] = float(qv)
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "paper3_conductor_stratified.csv", index=False)

    # Print headline: rank-2 vs rank-3 normalized comparisons in common window
    section(f"CONDUCTOR-STRATIFIED COMPARISON (N >= {args.cond_min:,})")
    print()
    print(f"  Within N >= {args.cond_min:,} (both ranks):")
    print(f"  {'statistic':<25s} {'rank':>5s} {'n':>8s} {'q5':>8s} {'q50':>8s} {'q95':>8s}")
    for c in ["spec_gini_norm", "spec_entropy_norm", "spec_participation", "aspect_ratio",
              "orthogonality_defect"]:
        if c not in compare_stats: continue
        for r in [2, 3]:
            sub = classes_high[classes_high["rank"] == r]
            vals = pd.to_numeric(sub[c], errors="coerce").dropna().values
            if len(vals) == 0: continue
            qs = np.quantile(vals, [0.05, 0.50, 0.95])
            print(f"  {c:<25s} {r:>5d} {len(vals):>8,d} {qs[0]:>8.4f} {qs[1]:>8.4f} {qs[2]:>8.4f}")
        print()

    print("  Full corpus (rank 2 includes all N; rank 3 is constrained at N>=194,040):")
    print(f"  {'statistic':<25s} {'rank':>5s} {'n':>8s} {'q5':>8s} {'q50':>8s} {'q95':>8s}")
    for c in ["spec_gini_norm", "spec_entropy_norm", "spec_participation", "aspect_ratio",
              "orthogonality_defect"]:
        if c not in compare_stats: continue
        for r in [2, 3]:
            sub = classes[classes["rank"] == r]
            vals = pd.to_numeric(sub[c], errors="coerce").dropna().values
            if len(vals) == 0: continue
            qs = np.quantile(vals, [0.05, 0.50, 0.95])
            print(f"  {c:<25s} {r:>5d} {len(vals):>8,d} {qs[0]:>8.4f} {qs[1]:>8.4f} {qs[2]:>8.4f}")
        print()

    # ---- 3. Gini-vs-participation scatter ----
    step(3, 4, "Generating Figure 2 — gini vs participation scatter by rank...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, r in zip(axes, [2, 3]):
        sub = classes[classes["rank"] == r].dropna(subset=["spec_gini", "spec_participation"])
        if len(sub) == 0:
            ax.set_title(f"rank {r} (n = 0)"); continue
        # Sub-sample for plotting if too many points
        if len(sub) > 100000:
            sub_plot = sub.sample(n=100000, random_state=42)
            n_label = f"n = {len(sub):,} ({len(sub_plot):,} shown)"
        else:
            sub_plot = sub
            n_label = f"n = {len(sub):,}"
        ax.hexbin(sub_plot["spec_gini"], sub_plot["spec_participation"],
                  gridsize=60, cmap="YlOrRd", mincnt=1, bins="log")
        corr = float(sub[["spec_gini", "spec_participation"]].corr().iloc[0, 1])
        ax.set_xlabel("spec_gini")
        ax.set_ylabel("spec_participation")
        ax.set_title(f"rank {r}  —  {n_label}\nPearson r = {corr:.4f}")
        ax.grid(alpha=0.3)
    plt.suptitle("Figure 2. Spec_gini and spec_participation are near-bijectively related at fixed rank",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "paper3_gini_vs_participation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'paper3_gini_vs_participation.png'}")

    # ---- 4. Save log ----
    step(4, 4, "Done.")
    section("FILES FOR PAPER 3 v3")
    print(f"""
Outputs in {output_dir}:

  paper3_conductor_stratified.csv   — rank-2 vs rank-3 within N >= {args.cond_min:,}
                                       (per-statistic quantile comparison)
  paper3_gini_vs_participation.png  — scatter showing -0.99 relationship visually

These feed Paper 3 v3:

  - The conductor-stratified comparison supplies the appendix table the
    reviewer asked for (review point 4): a "rank-2 vs rank-3 comparison
    in a common conductor window" that isolates the rank effect from
    the conductor effect.

  - The scatter figure makes the near-bijective gini-participation
    relationship visually compelling rather than asking the reader to
    interpret a -0.99 correlation coefficient.
""")


if __name__ == "__main__":
    main()
