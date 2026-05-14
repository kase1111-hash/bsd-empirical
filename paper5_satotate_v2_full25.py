#!/usr/bin/env python3
"""paper5_satotate_v2_full25.py — Sato-Tate empirics extended to all 25 primes.

For each prime p in aps_list (the first 25 primes, p <= 97), compute:
  - variance of normalized a_p / (2 sqrt(p)) across non-CM classes
  - variance ratio (sample / 0.25)
  - KS statistic against semicircle CDF
  - support cardinality |S_p| = floor(4 sqrt(p)) + 1
  - Proposition 1 corollary lower bound on KS

Outputs:
  paper5_satotate_v2_full25.csv  — per-prime metrics across all 25 primes
  paper5_variance_vs_p.png        — variance ratio vs p (with horizontal y=1 ref)
  paper5_ks_vs_p.png              — KS vs p (with Proposition 1 corollary bound)

Optionally extends to CM analysis at higher primes too.
"""

import argparse
import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kase_utils import add_standard_args, resolve_args, banner, section, step, Timer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

# The 25 primes that aps_list covers
FIRST_25_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                   53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def semicircle_cdf(x):
    x = np.clip(x, -1.0, 1.0)
    return 0.5 + (x * np.sqrt(np.maximum(1.0 - x * x, 0.0)) + np.arcsin(x)) / np.pi


def parse_ap_list(s):
    if isinstance(s, (list, tuple, np.ndarray)):
        if len(s) == 0:
            return None
        return np.asarray(s)
    try:
        if pd.isna(s):
            return None
    except (TypeError, ValueError):
        pass
    if not isinstance(s, str):
        return None
    try:
        return np.asarray(ast.literal_eval(s))
    except Exception:
        try:
            return np.asarray([int(x) for x in s.replace("[", "").replace("]", "").split()])
        except Exception:
            return None


def proposition_1_lower_bound(p):
    """Compute the Proposition 1 corollary lower bound on KS distance against
    the semicircle for any probability distribution supported on
    S_p = {a/(2*sqrt(p)) : a in Z, |a| <= 2*sqrt(p)}.

    The rigorous derivation gives three contributions:
      - boundary cell on the left:   F_semi(s_0)
      - boundary cell on the right:  1 - F_semi(s_n)
      - interior cells (between consecutive support points s_i, s_{i+1}):
                                     (1/2) * (F_semi(s_{i+1}) - F_semi(s_i))

    The lower bound is the maximum of these three contributions.
    For p in the analyzed range (p <= 97), interior cells dominate (boundary
    contributions are subdominant since s_0 -> -1 as p grows)."""
    a_max = int(np.floor(2.0 * np.sqrt(p)))
    a_values = np.arange(-a_max, a_max + 1)
    s_values = a_values / (2.0 * np.sqrt(p))
    boundary_left  = float(semicircle_cdf(s_values[0]))
    boundary_right = 1.0 - float(semicircle_cdf(s_values[-1]))
    interior_rises = np.diff(semicircle_cdf(s_values))
    interior_max_half = 0.5 * float(interior_rises.max())
    return max(boundary_left, boundary_right, interior_max_half)


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

    banner("Paper 5 v2 full25: 25-prime Sato-Tate empirics", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load and aggregate to class level ----
    step(1, 5, "Loading corpus...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")
    number_col = "curve_number" if "curve_number" in corpus.columns else None
    sort_cols = ["conductor", "iso"] + ([number_col] if number_col else [])
    agg_cols = ["rank", "is_cm", "cm_discriminant", "aps_list", "n_aps"]
    agg_cols = [c for c in agg_cols if c in corpus.columns]
    agg_dict = {c: "first" for c in agg_cols}
    with Timer("Class aggregation"):
        classes = corpus.sort_values(sort_cols).groupby(
            ["conductor", "iso"], as_index=False).agg(agg_dict)
    print(f"  Classes: {len(classes):,}")

    # ---- 2. Extract aps_list into a per-class array (n_classes, 25) ----
    step(2, 5, "Extracting aps_list into a 2D array...")
    aps_per_curve = np.full((len(classes), 25), np.nan)
    with Timer("Parsing aps_list"):
        for i, ap_raw in enumerate(classes["aps_list"].values):
            parsed = parse_ap_list(ap_raw)
            if parsed is None or len(parsed) != 25:
                continue
            aps_per_curve[i, :] = parsed
    n_good = (~np.isnan(aps_per_curve[:, 0])).sum()
    print(f"  Classes with parsed aps_list of length 25: {n_good:,}")

    is_cm = classes["is_cm"].values if "is_cm" in classes.columns else np.zeros(len(classes), dtype=bool)
    n_noncm = (~is_cm).sum()
    n_cm = is_cm.sum()
    print(f"  Non-CM: {n_noncm:,}   CM: {n_cm:,}")

    # ---- 3. Per-prime statistics ----
    step(3, 5, "Per-prime variance, KS, and support-geometry analysis...")
    results = []
    for j, p_prime in enumerate(FIRST_25_PRIMES):
        a_col = aps_per_curve[:, j]
        ap_noncm = a_col[~is_cm]
        ap_noncm = ap_noncm[~np.isnan(ap_noncm)]
        if len(ap_noncm) == 0: continue

        # Normalize
        norm = ap_noncm / (2.0 * np.sqrt(p_prime))
        # Strict KS-eligible: |x| <= 1
        norm_strict = norm[np.abs(norm) <= 1.0 + 1e-9]
        norm_strict = norm_strict[np.abs(norm_strict) <= 1.0]
        # Statistics
        sample_var = float(np.var(norm_strict))
        sample_mean = float(np.mean(norm_strict))
        variance_ratio = sample_var / 0.25
        ks_stat, ks_p = stats.kstest(norm_strict, semicircle_cdf)
        # Support cardinality and Proposition 1 lower bound
        a_max = int(np.floor(2.0 * np.sqrt(p_prime)))
        support_card = 2 * a_max + 1
        prop1_lb = proposition_1_lower_bound(p_prime)
        results.append({
            "prime": p_prime,
            "log_p": float(np.log(p_prime)),
            "n": len(norm_strict),
            "support_cardinality": support_card,
            "support_spacing": 1.0 / np.sqrt(p_prime),
            "sample_mean": sample_mean,
            "sample_variance": sample_var,
            "variance_ratio": variance_ratio,
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p),
            "prop1_lower_bound": float(prop1_lb),
            "ks_excess_over_bound": float(ks_stat - prop1_lb),
        })
        print(f"  p={p_prime:>3d}: |S_p|={support_card:>3d} n={len(norm_strict):>8,d}"
              f"  var={sample_var:.4f}  ratio={variance_ratio:.4f}"
              f"  KS={ks_stat:.4f}  Prop1 LB={prop1_lb:.4f}  excess={ks_stat-prop1_lb:+.4f}")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "paper5_satotate_v2_full25.csv", index=False)

    # ---- 4. Figure: variance ratio vs p ----
    step(4, 5, "Generating Figure 2 — variance ratio vs prime...")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogx(df["prime"], df["variance_ratio"], "o-", color="#1f4e79",
                linewidth=1.5, markersize=7, label="sample variance / 0.25 (Sato-Tate)")
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="Sato–Tate asymptote (= 1)")
    ax.set_xlabel("prime p (log scale)", fontsize=11)
    ax.set_ylabel("variance ratio (sample / 0.25)", fontsize=11)
    ax.set_title("Variance ratio of normalized a_p / (2√p) for non-CM Cremona classes,\nacross primes p ≤ 97 (25-prime aps_list)", fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=10)
    for _, row in df.iterrows():
        ax.annotate(f"{row['variance_ratio']:.2f}",
                    (row["prime"], row["variance_ratio"]),
                    textcoords="offset points", xytext=(4, -10), fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "paper5_variance_vs_p.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'paper5_variance_vs_p.png'}")

    # ---- 5. Figure: KS vs p with Proposition 1 lower bound ----
    step(5, 5, "Generating Figure 3 — KS distance vs prime, with Proposition 1 bound...")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(df["prime"], df["ks_statistic"], "o-", color="#1f4e79",
              linewidth=1.5, markersize=7, label="empirical KS statistic")
    ax.loglog(df["prime"], df["prop1_lower_bound"], "s--", color="orange",
              linewidth=1.5, markersize=6, label="Proposition 1 corollary lower bound")
    # 1/sqrt(p) reference line, scaled
    p_grid = df["prime"].values
    ref_line = 0.4 / np.sqrt(p_grid)
    ax.loglog(p_grid, ref_line, ":", color="gray", linewidth=1, alpha=0.6,
              label="reference: 0.4 / √p")
    ax.set_xlabel("prime p (log scale)", fontsize=11)
    ax.set_ylabel("KS distance against semicircle", fontsize=11)
    ax.set_title("Kolmogorov–Smirnov distance against Sato–Tate semicircle,\nacross primes p ≤ 97 (with Proposition 1 corollary lower bound)", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "paper5_ks_vs_p.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'paper5_ks_vs_p.png'}")

    # Summary
    section("PAPER 5 v2 full25 — HEADLINE FINDINGS")
    print(f"""
Across the 25 primes p ≤ 97 in aps_list, n = {n_noncm:,} non-CM classes:

  Variance ratio: starts at {df.iloc[0]['variance_ratio']:.4f} (p=2), reaches
                  {df.iloc[len(df)//2]['variance_ratio']:.4f} (p={int(df.iloc[len(df)//2]['prime'])}),
                  ends at {df.iloc[-1]['variance_ratio']:.4f} (p=97).

  KS statistic:   starts at {df.iloc[0]['ks_statistic']:.4f} (p=2), reaches
                  {df.iloc[len(df)//2]['ks_statistic']:.4f} (p={int(df.iloc[len(df)//2]['prime'])}),
                  ends at {df.iloc[-1]['ks_statistic']:.4f} (p=97).

  Proposition 1 bound: starts at {df.iloc[0]['prop1_lower_bound']:.4f} (p=2),
                       ends at {df.iloc[-1]['prop1_lower_bound']:.4f} (p=97).

  Excess (empirical − bound):
                  starts at {df.iloc[0]['ks_excess_over_bound']:+.4f},
                  ends at {df.iloc[-1]['ks_excess_over_bound']:+.4f}.

Outputs in {output_dir}:

  paper5_satotate_v2_full25.csv  — 25-prime metrics table
  paper5_variance_vs_p.png        — variance ratio vs p with y=1 asymptote
  paper5_ks_vs_p.png              — KS vs p with Prop 1 bound and 1/√p reference

These feed Paper 5 v3, which will incorporate the 25-prime data into the
empirical convergence narrative and assess whether the KS statistic tracks
the Proposition 1 corollary lower bound (in which case discretization is the
dominant pre-asymptotic effect at all 25 primes) or whether structural
deviation from the semicircle persists beyond the support-geometry obstruction.
""")


if __name__ == "__main__":
    main()
