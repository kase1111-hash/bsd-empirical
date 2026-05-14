#!/usr/bin/env python3
"""paper7_heights_v1.py — Naive vs canonical heights of MW generators.

For each rank-≥-1 isogeny class in the Cremona corpus:
  - Extracts the naive (Weil) heights of the MW basis generators
  - Extracts the canonical (Néron-Tate) heights from the regulator (rank 1) or
    the Gram matrix diagonal (rank ≥ 2)
  - Computes the height difference h(P) - ĥ(P), the empirical analog of
    Silverman's bound |h(P) - ĥ(P)| ≤ μ(E)
  - Computes ĥ_min(E) = min over generators of ĥ(P_i), the quantity that
    the Lang-Silverman conjecture predicts is bounded below by c·log(N(E))

Outputs:
  paper7_height_columns_inventory.csv  — diagnostic: which columns exist
  paper7_naive_height_distribution.csv  — naive height stats by rank
  paper7_canonical_height_distribution.csv — canonical height stats by rank
  paper7_height_difference.csv          — h - ĥ distribution and tightness
  paper7_lang_silverman_scan.csv        — ĥ_min vs log(N) per rank ≥ 1 class
  paper7_height_difference.png          — Figure 1: h - ĥ scatter and density
  paper7_lang_silverman.png             — Figure 2: ĥ_min vs log(N), with c bound
"""

import argparse
import ast
import warnings
from math import log10, log
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


def parse_list(s):
    """Parse a possibly-stringified list of floats from the parquet."""
    if isinstance(s, (list, tuple, np.ndarray)):
        if len(s) == 0:
            return None
        return np.asarray(s, dtype=float)
    try:
        if pd.isna(s):
            return None
    except (TypeError, ValueError):
        pass
    if not isinstance(s, str):
        return None
    try:
        return np.asarray(ast.literal_eval(s), dtype=float)
    except Exception:
        try:
            return np.asarray([float(x) for x in s.replace("[", "").replace("]", "").split(",")], dtype=float)
        except Exception:
            return None


def parse_gram_matrix(s, rank):
    """Parse a stringified Gram matrix for rank ≥ 2. Returns rank x rank np.array
    or None if parse fails. Different storage conventions handled."""
    if s is None: return None
    try:
        if pd.isna(s): return None
    except (TypeError, ValueError):
        pass
    if isinstance(s, np.ndarray):
        arr = s
    elif isinstance(s, (list, tuple)):
        arr = np.asarray(s, dtype=float)
    elif isinstance(s, str):
        try:
            arr = np.asarray(ast.literal_eval(s), dtype=float)
        except Exception:
            return None
    else:
        return None
    # Reshape to rank x rank if flat
    if arr.ndim == 1 and len(arr) == rank * rank:
        arr = arr.reshape(rank, rank)
    elif arr.ndim == 2 and arr.shape == (rank, rank):
        pass
    else:
        return None
    return arr


def main():
    p_arg = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(p_arg)
    p_arg.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = p_arg.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("Paper 7 v1: Naive vs canonical heights of MW generators", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Diagnostic: which height columns exist? ----
    step(1, 6, "Column inventory diagnostic...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")
    candidate_cols = [
        "naive_heights", "naive_h_max", "naive_h_min", "naive_h_sum",
        "gens_coord_max_log10", "canonical_heights", "canonical_h_max",
        "canonical_h_min", "canonical_h_sum",
        "gram_matrix", "gram_diag", "regulator", "rank",
        "conductor", "iso",
    ]
    inventory = []
    for col in candidate_cols:
        if col in corpus.columns:
            non_null = corpus[col].notna().sum()
            dtype = str(corpus[col].dtype)
            sample = corpus[col].dropna().iloc[0] if non_null > 0 else None
            sample_str = str(sample)[:80] if sample is not None else "—"
            inventory.append({
                "column": col, "exists": True, "dtype": dtype,
                "non_null": int(non_null), "sample": sample_str
            })
            print(f"  ✓ {col:<28s} dtype={dtype:<15s} non_null={non_null:>9,d}  sample={sample_str}")
        else:
            inventory.append({"column": col, "exists": False, "dtype": "",
                              "non_null": 0, "sample": ""})
            print(f"  ✗ {col:<28s} MISSING")
    pd.DataFrame(inventory).to_csv(output_dir / "paper7_height_columns_inventory.csv", index=False)

    # ---- 2. Aggregate to class level ----
    step(2, 6, "Class aggregation...")
    number_col = "curve_number" if "curve_number" in corpus.columns else None
    sort_cols = ["conductor", "iso"] + ([number_col] if number_col else [])
    agg_cols = ["rank", "is_cm", "regulator", "naive_heights", "naive_h_max",
                "naive_h_min", "naive_h_sum", "gens_coord_max_log10",
                "canonical_heights", "canonical_h_max", "canonical_h_min",
                "canonical_h_sum", "gram_matrix", "gram_diag"]
    agg_cols = [c for c in agg_cols if c in corpus.columns]
    agg_dict = {c: "first" for c in agg_cols}
    with Timer("Class aggregation"):
        classes = corpus.sort_values(sort_cols).groupby(
            ["conductor", "iso"], as_index=False).agg(agg_dict)
    print(f"  Total classes: {len(classes):,}")

    classes["rank"] = pd.to_numeric(classes["rank"], errors="coerce")
    rank_counts = classes["rank"].value_counts().sort_index()
    print(f"  Rank distribution: {dict(rank_counts.head(10))}")
    rank_ge_1 = classes[classes["rank"] >= 1].copy()
    print(f"  Rank ≥ 1 classes: {len(rank_ge_1):,}")

    # ---- 3. Naive height distribution ----
    step(3, 6, "Naive height distribution by rank...")
    naive_rows = []
    for r in [1, 2, 3, 4, 5]:
        sub = classes[classes["rank"] == r]
        if len(sub) == 0: continue
        naive_h_min_vals = pd.to_numeric(sub["naive_h_min"], errors="coerce").dropna() if "naive_h_min" in sub.columns else pd.Series()
        naive_h_max_vals = pd.to_numeric(sub["naive_h_max"], errors="coerce").dropna() if "naive_h_max" in sub.columns else pd.Series()
        naive_h_sum_vals = pd.to_numeric(sub["naive_h_sum"], errors="coerce").dropna() if "naive_h_sum" in sub.columns else pd.Series()
        gens_coord_log10_vals = pd.to_numeric(sub["gens_coord_max_log10"], errors="coerce").dropna() if "gens_coord_max_log10" in sub.columns else pd.Series()
        row = {"rank": r, "n_classes": len(sub)}
        if len(naive_h_min_vals):
            row.update({
                "naive_h_min_q25": float(naive_h_min_vals.quantile(0.25)),
                "naive_h_min_q50": float(naive_h_min_vals.quantile(0.50)),
                "naive_h_min_q75": float(naive_h_min_vals.quantile(0.75)),
                "naive_h_min_q95": float(naive_h_min_vals.quantile(0.95)),
                "naive_h_min_max": float(naive_h_min_vals.max()),
            })
        if len(naive_h_max_vals):
            row.update({
                "naive_h_max_q50": float(naive_h_max_vals.quantile(0.50)),
                "naive_h_max_q95": float(naive_h_max_vals.quantile(0.95)),
                "naive_h_max_max": float(naive_h_max_vals.max()),
            })
        if len(gens_coord_log10_vals):
            row.update({
                "gens_coord_max_log10_q50": float(gens_coord_log10_vals.quantile(0.50)),
                "gens_coord_max_log10_q95": float(gens_coord_log10_vals.quantile(0.95)),
                "gens_coord_max_log10_max": float(gens_coord_log10_vals.max()),
            })
        naive_rows.append(row)
        print(f"  rank={r}: n={len(sub):>7,d}  naive_h_min q50={row.get('naive_h_min_q50', np.nan):.4f}  "
              f"naive_h_max q50={row.get('naive_h_max_q50', np.nan):.4f}  "
              f"gens_coord_max_log10 q50={row.get('gens_coord_max_log10_q50', np.nan):.2f}")
    pd.DataFrame(naive_rows).to_csv(output_dir / "paper7_naive_height_distribution.csv", index=False)

    # ---- 4. Canonical height extraction ----
    step(4, 6, "Canonical height extraction (rank-1: regulator; rank≥2: Gram diag)...")
    # For rank=1 curves: canonical height = regulator (= ĥ(P_1))
    rank1 = classes[classes["rank"] == 1].copy()
    rank1["canonical_h_min"] = pd.to_numeric(rank1["regulator"], errors="coerce")
    rank1["canonical_h_max"] = rank1["canonical_h_min"]

    # For rank ≥ 2: extract from Gram matrix diagonal if available
    if "gram_matrix" in classes.columns:
        print("  Extracting Gram matrix diagonals for rank ≥ 2...")
        rank_ge_2 = classes[classes["rank"] >= 2].copy()
        canon_min, canon_max = [], []
        n_parsed, n_failed = 0, 0
        for _, row in rank_ge_2.iterrows():
            r = int(row["rank"])
            gm = parse_gram_matrix(row["gram_matrix"], r)
            if gm is None:
                n_failed += 1
                canon_min.append(np.nan); canon_max.append(np.nan)
                continue
            diag = np.diag(gm)
            canon_min.append(float(diag.min()))
            canon_max.append(float(diag.max()))
            n_parsed += 1
        rank_ge_2["canonical_h_min"] = canon_min
        rank_ge_2["canonical_h_max"] = canon_max
        print(f"  Parsed {n_parsed:,} gram matrices, failed {n_failed}")
    else:
        rank_ge_2 = pd.DataFrame()
        print("  No gram_matrix column — rank ≥ 2 canonical heights unavailable")

    canon_rows = []
    for r in [1, 2, 3, 4, 5]:
        if r == 1: sub = rank1
        else: sub = rank_ge_2[rank_ge_2["rank"] == r] if len(rank_ge_2) else pd.DataFrame()
        if len(sub) == 0: continue
        cmin = sub["canonical_h_min"].dropna()
        cmax = sub["canonical_h_max"].dropna()
        canon_rows.append({
            "rank": r, "n_classes": len(sub),
            "canonical_h_min_q05": float(cmin.quantile(0.05)) if len(cmin) else np.nan,
            "canonical_h_min_q25": float(cmin.quantile(0.25)) if len(cmin) else np.nan,
            "canonical_h_min_q50": float(cmin.quantile(0.50)) if len(cmin) else np.nan,
            "canonical_h_min_q75": float(cmin.quantile(0.75)) if len(cmin) else np.nan,
            "canonical_h_min_q95": float(cmin.quantile(0.95)) if len(cmin) else np.nan,
            "canonical_h_min_min": float(cmin.min()) if len(cmin) else np.nan,
            "canonical_h_min_max": float(cmin.max()) if len(cmin) else np.nan,
            "canonical_h_max_q50": float(cmax.quantile(0.50)) if len(cmax) else np.nan,
            "canonical_h_max_max": float(cmax.max()) if len(cmax) else np.nan,
        })
        print(f"  rank={r}: n={len(sub):>7,d}  ĥ_min q05={canon_rows[-1]['canonical_h_min_q05']:.4f}  "
              f"q50={canon_rows[-1]['canonical_h_min_q50']:.4f}  q95={canon_rows[-1]['canonical_h_min_q95']:.4f}  "
              f"global_min={canon_rows[-1]['canonical_h_min_min']:.6f}")
    pd.DataFrame(canon_rows).to_csv(output_dir / "paper7_canonical_height_distribution.csv", index=False)

    # ---- 5. Height difference h - ĥ (Silverman bound) ----
    step(5, 6, "Height difference h(P) - ĥ(P) (Silverman bound)...")
    # For rank-1 curves: naive_h_min is h(P_1) and canonical_h_min is ĥ(P_1)
    diff_rank1 = rank1[["conductor", "iso", "naive_h_min", "canonical_h_min"]].copy()
    diff_rank1 = diff_rank1.dropna()
    diff_rank1["h_diff"] = pd.to_numeric(diff_rank1["naive_h_min"], errors="coerce") - pd.to_numeric(diff_rank1["canonical_h_min"], errors="coerce")
    diff_rank1 = diff_rank1.dropna(subset=["h_diff"])
    print(f"  Rank-1 height-difference observations: {len(diff_rank1):,}")
    print(f"    h_diff min:    {diff_rank1['h_diff'].min():.4f}")
    print(f"    h_diff q05:    {diff_rank1['h_diff'].quantile(0.05):.4f}")
    print(f"    h_diff q50:    {diff_rank1['h_diff'].quantile(0.50):.4f}")
    print(f"    h_diff q95:    {diff_rank1['h_diff'].quantile(0.95):.4f}")
    print(f"    h_diff max:    {diff_rank1['h_diff'].max():.4f}")
    print(f"    h_diff |abs|:  q50={diff_rank1['h_diff'].abs().quantile(0.50):.4f}  q95={diff_rank1['h_diff'].abs().quantile(0.95):.4f}")
    diff_stats = {
        "n_observations": len(diff_rank1),
        "h_diff_min": float(diff_rank1["h_diff"].min()),
        "h_diff_q05": float(diff_rank1["h_diff"].quantile(0.05)),
        "h_diff_q50": float(diff_rank1["h_diff"].quantile(0.50)),
        "h_diff_q95": float(diff_rank1["h_diff"].quantile(0.95)),
        "h_diff_max": float(diff_rank1["h_diff"].max()),
        "h_diff_abs_q50": float(diff_rank1["h_diff"].abs().quantile(0.50)),
        "h_diff_abs_q95": float(diff_rank1["h_diff"].abs().quantile(0.95)),
        "h_diff_abs_max": float(diff_rank1["h_diff"].abs().max()),
    }
    pd.DataFrame([diff_stats]).to_csv(output_dir / "paper7_height_difference.csv", index=False)

    # Figure 1: h - ĥ distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Left: scatter h vs ĥ
    sample_idx = np.random.choice(len(diff_rank1), size=min(50000, len(diff_rank1)), replace=False)
    sample = diff_rank1.iloc[sample_idx]
    axes[0].scatter(sample["canonical_h_min"], sample["naive_h_min"], s=2, alpha=0.3, color="#1f4e79")
    lims = [min(sample["canonical_h_min"].min(), sample["naive_h_min"].min()),
            max(sample["canonical_h_min"].max(), sample["naive_h_min"].max())]
    axes[0].plot(lims, lims, "r--", linewidth=1, label="h = ĥ identity")
    axes[0].set_xlabel("canonical height ĥ(P)")
    axes[0].set_ylabel("naive height h(P)")
    axes[0].set_title(f"Naive vs canonical height, rank-1 curves\n(n = {len(diff_rank1):,}, sample n = {len(sample):,})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Right: histogram of h - ĥ
    h_diff_clipped = diff_rank1["h_diff"].clip(-5, 5)
    axes[1].hist(h_diff_clipped, bins=100, color="#5a8fd0", edgecolor="#1f4e79")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1, label="h = ĥ")
    axes[1].set_xlabel("h(P) - ĥ(P)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"Distribution of h - ĥ over rank-1 curves\nmedian = {diff_stats['h_diff_q50']:.3f}, q95 = {diff_stats['h_diff_q95']:.3f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "paper7_height_difference.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'paper7_height_difference.png'}")

    # ---- 6. Lang-Silverman test: ĥ_min vs log(N) ----
    step(6, 6, "Lang-Silverman test: ĥ_min vs log(N) across rank ≥ 1 cohort...")
    ls_data = []
    for _, row in rank1.iterrows():
        if pd.isna(row["canonical_h_min"]): continue
        try:
            N = int(row["conductor"])
        except (ValueError, TypeError):
            continue
        ls_data.append({"conductor": N, "rank": 1, "iso": row["iso"],
                        "log_N": log(N), "canonical_h_min": row["canonical_h_min"]})
    if len(rank_ge_2):
        for _, row in rank_ge_2.iterrows():
            if pd.isna(row["canonical_h_min"]): continue
            try:
                N = int(row["conductor"])
            except (ValueError, TypeError):
                continue
            ls_data.append({"conductor": N, "rank": int(row["rank"]), "iso": row["iso"],
                            "log_N": log(N), "canonical_h_min": row["canonical_h_min"]})
    ls_df = pd.DataFrame(ls_data)
    # Lang-Silverman ratio: ĥ_min / log(N)
    ls_df["ratio"] = ls_df["canonical_h_min"] / ls_df["log_N"]
    ls_df.to_csv(output_dir / "paper7_lang_silverman_scan.csv", index=False)
    print(f"  Total rank ≥ 1 observations: {len(ls_df):,}")
    print(f"  Min ratio ĥ_min / log(N): {ls_df['ratio'].min():.6f}")
    print(f"  q05 ratio:               {ls_df['ratio'].quantile(0.05):.6f}")
    print(f"  q50 ratio:               {ls_df['ratio'].quantile(0.50):.6f}")
    print(f"  q95 ratio:               {ls_df['ratio'].quantile(0.95):.6f}")
    # Smallest 5 ĥ_min observations
    print("\n  Smallest 5 ĥ_min observations (potential Lang-Silverman outliers):")
    smallest = ls_df.nsmallest(5, "canonical_h_min")
    for _, r in smallest.iterrows():
        label = f"{r['conductor']}{r['iso']}"
        print(f"    {label}: rank={r['rank']}  ĥ_min={r['canonical_h_min']:.6f}  log(N)={r['log_N']:.3f}  ratio={r['ratio']:.6f}")

    # Figure 2: ĥ_min vs log(N)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Scatter
    sample_idx = np.random.choice(len(ls_df), size=min(50000, len(ls_df)), replace=False)
    sample = ls_df.iloc[sample_idx]
    for r, col in [(1, "#1f4e79"), (2, "#d62728"), (3, "#2ca02c")]:
        sub = sample[sample["rank"] == r]
        axes[0].scatter(sub["log_N"], sub["canonical_h_min"], s=2, alpha=0.3, color=col, label=f"rank {r}")
    axes[0].set_xlabel("log(conductor N)")
    axes[0].set_ylabel("ĥ_min(E)")
    axes[0].set_title(f"Lang-Silverman test: ĥ_min vs log(N)\n(n = {len(ls_df):,} rank ≥ 1 classes)")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which="both")
    # Right: ratio distribution
    ratios_clipped = ls_df["ratio"].clip(0, 1)
    axes[1].hist(ratios_clipped, bins=200, color="#5a8fd0", edgecolor="#1f4e79")
    axes[1].set_xlabel("ratio ĥ_min(E) / log(N(E))")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"Distribution of ĥ_min / log(N)\nmin = {ls_df['ratio'].min():.5f}, q05 = {ls_df['ratio'].quantile(0.05):.4f}")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "paper7_lang_silverman.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'paper7_lang_silverman.png'}")

    section("PAPER 7 v1 — HEADLINE FINDINGS")
    print(f"""
Naive vs canonical heights of MW generators in the Cremona corpus:

  Total rank ≥ 1 isogeny classes:  {len(ls_df):,}
  Rank-1 observations:              {len(rank1):,}
  Rank ≥ 2 observations:            {len(rank_ge_2) if len(rank_ge_2) else 0}

Silverman bound (h - ĥ) on rank-1 cohort:
  median h - ĥ:  {diff_stats['h_diff_q50']:.4f}
  q95 |h - ĥ|:   {diff_stats['h_diff_abs_q95']:.4f}
  max |h - ĥ|:   {diff_stats['h_diff_abs_max']:.4f}

Lang-Silverman ratio ĥ_min / log(N):
  min:           {ls_df['ratio'].min():.6f}
  q05:           {ls_df['ratio'].quantile(0.05):.6f}
  q50:           {ls_df['ratio'].quantile(0.50):.6f}
  q95:           {ls_df['ratio'].quantile(0.95):.6f}
  global_min ĥ:  {ls_df['canonical_h_min'].min():.6f}
  (this is the corpus-wide minimum non-torsion canonical height)

Outputs in {output_dir}:
  paper7_height_columns_inventory.csv     — diagnostic: which columns available
  paper7_naive_height_distribution.csv    — naive height stats by rank
  paper7_canonical_height_distribution.csv — canonical height stats by rank
  paper7_height_difference.csv            — Silverman bound stats
  paper7_lang_silverman_scan.csv          — per-class ĥ_min, log(N), ratio
  paper7_height_difference.png            — Figure 1: h - ĥ scatter and density
  paper7_lang_silverman.png               — Figure 2: ĥ_min vs log(N), ratio histogram

These feed Paper 7 v1, which characterizes the empirical naive-vs-canonical
height geometry of MW generators in the Cremona corpus.
""")


if __name__ == "__main__":
    main()
