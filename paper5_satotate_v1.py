#!/usr/bin/env python3
"""paper5_satotate_v1.py — initial Sato-Tate empirics on the Cremona corpus.

Aggregates the Frobenius trace data (aps_list and the explicit a_p columns)
to the class level, computes the normalized a_p / (2 sqrt(p)) distribution
across the corpus, runs Kolmogorov-Smirnov tests against the Sato-Tate
semicircle distribution, and stratifies by CM status and by rank.

Outputs:
  - paper5_aps_diagnostic.csv         — structure of aps_list (how many primes
                                          per curve, what primes are present,
                                          what the value range looks like)
  - paper5_aps_by_prime_noncm.csv     — per-prime distribution statistics for
                                          non-CM classes (mean, sd, quantiles
                                          of normalized a_p / 2sqrt(p))
  - paper5_ks_semicircle.csv          — KS test statistics against the
                                          semicircle, per prime, with bootstrap
                                          confidence intervals
  - paper5_cm_distribution.csv        — CM cohort: count and discriminant
                                          distribution, a_p distribution at
                                          small primes
  - paper5_aps_distribution.png       — histograms of normalized a_p / 2sqrt(p)
                                          for several primes, overlaid with
                                          the semicircle reference density

The semicircle reference density is (2/π) sqrt(1 - x²) on [-1, 1].

Author: Kase Branham — Independent Researcher
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

# Small primes we have as direct columns; aps_list may have more
DIRECT_AP_PRIMES = [2, 3, 5, 7, 11, 13]


def semicircle_cdf(x):
    """CDF of the semicircle distribution on [-1, 1].
    density = (2/π) sqrt(1 - x²); CDF = (1/π)(x sqrt(1-x²) + arcsin(x)) + 1/2."""
    x = np.clip(x, -1.0, 1.0)
    return 0.5 + (x * np.sqrt(np.maximum(1.0 - x * x, 0.0)) + np.arcsin(x)) / np.pi


def parse_ap_list(s):
    """Return the aps as a numpy array. Handles three cases:
       (a) s is already a list/numpy array — return as-is
       (b) s is a string — try ast.literal_eval, then space-split fallback
       (c) s is missing or unparseable — return None"""
    # Case (a): already array-like
    if isinstance(s, (list, tuple, np.ndarray)):
        if len(s) == 0:
            return None
        return np.asarray(s)
    # Cases (b), (c): scalar (string or NaN)
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

    banner("Paper 5 v1: Sato-Tate empirics", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load and aggregate to class level ----
    step(1, 6, "Loading corpus...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")
    print(f"  is_cm present: {'is_cm' in corpus.columns}")
    print(f"  aps_list present: {'aps_list' in corpus.columns}")
    print(f"  n_aps present: {'n_aps' in corpus.columns}")
    for p_ in DIRECT_AP_PRIMES:
        col = f"a{p_}"
        print(f"  {col} present: {col in corpus.columns}")

    # Aggregate to class level — use optimal curve
    number_col = "curve_number" if "curve_number" in corpus.columns else None
    sort_cols = ["conductor", "iso"] + ([number_col] if number_col else [])
    agg_cols = ["rank", "is_cm", "cm_discriminant", "aps_list", "n_aps"] + [f"a{p_}" for p_ in DIRECT_AP_PRIMES]
    agg_cols = [c for c in agg_cols if c in corpus.columns]
    agg_dict = {c: "first" for c in agg_cols}
    with Timer("Class aggregation"):
        classes = corpus.sort_values(sort_cols).groupby(
            ["conductor", "iso"], as_index=False).agg(agg_dict)
    print(f"  Classes: {len(classes):,}")

    # ---- 2. Diagnose aps_list structure ----
    step(2, 6, "Diagnosing aps_list structure...")
    diagnostic_rows = []
    aps_lengths = []
    aps_prime_sets = {}  # length -> set of primes (estimated from list position)
    aps_value_ranges = {}  # length -> (min, max)

    if "aps_list" in classes.columns:
        sample_size = min(10000, len(classes))
        sample = classes.head(sample_size)  # quick sample
        parsed_count = 0
        for ap_str in sample["aps_list"]:
            parsed = parse_ap_list(ap_str)
            if parsed is None: continue
            parsed_count += 1
            length = len(parsed)
            aps_lengths.append(length)
            if length not in aps_value_ranges:
                aps_value_ranges[length] = (int(parsed[0]), int(parsed[-1]))
            else:
                lo, hi = aps_value_ranges[length]
                aps_value_ranges[length] = (min(lo, int(np.min(parsed))),
                                              max(hi, int(np.max(parsed))))

        print(f"  Sample size: {sample_size:,}, parsed: {parsed_count:,}")
        if aps_lengths:
            length_counts = pd.Series(aps_lengths).value_counts().sort_index()
            print(f"  aps_list length distribution (length: count):")
            for length, count in length_counts.head(10).items():
                lo, hi = aps_value_ranges.get(length, (0, 0))
                print(f"    length {length:>3d}: {count:>6,d} curves   ap_range=[{lo:>4d}, {hi:>4d}]")

    # Use n_aps if it exists
    if "n_aps" in classes.columns:
        n_aps_counts = classes["n_aps"].value_counts().sort_index()
        print(f"  n_aps distribution (count: classes):")
        for n, c in n_aps_counts.head(10).items():
            print(f"    n_aps={n:>4d}: {c:>8,d} classes")

    diagnostic_rows = []
    if aps_lengths:
        diagnostic_rows.append({"diagnostic": "median_aps_length", "value": int(np.median(aps_lengths))})
        diagnostic_rows.append({"diagnostic": "min_aps_length", "value": int(np.min(aps_lengths))})
        diagnostic_rows.append({"diagnostic": "max_aps_length", "value": int(np.max(aps_lengths))})
    pd.DataFrame(diagnostic_rows).to_csv(output_dir / "paper5_aps_diagnostic.csv", index=False)

    # ---- 3. Per-prime distributions for non-CM curves ----
    step(3, 6, "Per-prime distributions for non-CM curves...")
    if "is_cm" in classes.columns:
        noncm = classes[classes["is_cm"] == False].copy()
        cm    = classes[classes["is_cm"] == True].copy()
        print(f"  Non-CM classes: {len(noncm):,}")
        print(f"  CM classes:     {len(cm):,}")
    else:
        noncm = classes.copy()
        cm = pd.DataFrame()
        print(f"  Treating all {len(classes):,} classes as non-CM (no is_cm column)")

    # For each direct prime column, compute normalized a_p / 2sqrt(p) for non-CM curves
    aps_rows = []
    for p_ in DIRECT_AP_PRIMES:
        col = f"a{p_}"
        if col not in classes.columns: continue
        ap = pd.to_numeric(noncm[col], errors="coerce").dropna().values
        if len(ap) == 0: continue
        # Normalized a_p in [-1, 1] for primes of good reduction
        norm = ap / (2.0 * np.sqrt(p_))
        # Filter to "good reduction" — values in [-1, 1]; bad reduction (a_p = 0, ±1, etc. with N divisible by p) can leak in
        norm_good = norm[np.abs(norm) <= 1.0 + 1e-9]
        n_total = len(ap)
        n_good = len(norm_good)
        row = {
            "prime": p_,
            "n_total": n_total,
            "n_good_reduction_filter": n_good,
            "frac_passing": n_good / n_total if n_total > 0 else np.nan,
            "mean": float(np.mean(norm_good)),
            "sd":   float(np.std(norm_good, ddof=1)) if len(norm_good) > 1 else np.nan,
        }
        for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
            row[f"q{q:.3f}"] = float(np.quantile(norm_good, q))
        aps_rows.append(row)
        print(f"  p={p_:>2d}: n={n_good:>8,d}  mean={row['mean']:+.4f}  sd={row['sd']:.4f}  q5={row['q0.050']:+.3f}  q95={row['q0.950']:+.3f}")

    pd.DataFrame(aps_rows).to_csv(output_dir / "paper5_aps_by_prime_noncm.csv", index=False)

    # ---- 4. KS tests against the semicircle ----
    step(4, 6, "KS tests against the semicircle distribution...")
    # Semicircle: density (2/π) sqrt(1 - x²) on [-1, 1]; mean 0, variance 1/4
    ks_rows = []
    for p_ in DIRECT_AP_PRIMES:
        col = f"a{p_}"
        if col not in classes.columns: continue
        ap = pd.to_numeric(noncm[col], errors="coerce").dropna().values
        norm = ap / (2.0 * np.sqrt(p_))
        norm_good = norm[np.abs(norm) <= 1.0 + 1e-9]
        norm_good = norm_good[np.abs(norm_good) <= 1.0]  # strict for KS
        if len(norm_good) < 100: continue
        # KS against semicircle
        ks_stat, ks_p = stats.kstest(norm_good, semicircle_cdf)
        ks_rows.append({
            "prime": p_,
            "n": len(norm_good),
            "ks_statistic": float(ks_stat),
            "ks_p_value":   float(ks_p),
            "sample_mean":  float(np.mean(norm_good)),
            "sample_var":   float(np.var(norm_good)),
            "semicircle_var": 0.25,
        })
        print(f"  p={p_:>2d}: KS stat={ks_stat:.5f}  p-value={ks_p:.2e}  sample var={np.var(norm_good):.4f}  (semicircle var=0.25)")
    pd.DataFrame(ks_rows).to_csv(output_dir / "paper5_ks_semicircle.csv", index=False)

    # ---- 5. CM cohort ----
    step(5, 6, "CM cohort analysis...")
    if "is_cm" in classes.columns and len(cm) > 0:
        print(f"  CM classes: {len(cm):,} ({100*len(cm)/len(classes):.4f}% of corpus)")
        if "cm_discriminant" in classes.columns:
            cm_disc = pd.to_numeric(cm["cm_discriminant"], errors="coerce").dropna()
            disc_counts = cm_disc.value_counts().sort_index()
            print(f"\n  CM discriminant distribution:")
            for d, c in disc_counts.head(20).items():
                print(f"    D = {d:>5g}: {c:>6,d} classes")

            # CM curves at small primes: at split primes (a_p² = 4p), at inert primes (a_p = 0)
            # For each CM curve and each small prime, check whether a_p is 0 (inert) or non-zero (split or ramified)
            cm_rows = []
            for p_ in DIRECT_AP_PRIMES:
                col = f"a{p_}"
                if col not in classes.columns: continue
                ap = pd.to_numeric(cm[col], errors="coerce").dropna().values
                if len(ap) == 0: continue
                # Count zeros (inert) vs nonzeros (split)
                n_zero    = int((ap == 0).sum())
                n_nonzero = int((ap != 0).sum())
                cm_rows.append({
                    "prime": p_,
                    "n_total": len(ap),
                    "n_ap_zero": n_zero,
                    "n_ap_nonzero": n_nonzero,
                    "frac_zero": n_zero / len(ap) if len(ap) > 0 else np.nan,
                })
                print(f"  p={p_:>2d}: CM curves: a_p=0 (inert) {n_zero:>5,d}/{len(ap):,} ({100*n_zero/len(ap):.1f}%);  a_p≠0 (split) {n_nonzero:>5,d}")

            pd.DataFrame(cm_rows + [{"discriminant": d, "count": c} for d, c in disc_counts.items()]).to_csv(
                output_dir / "paper5_cm_distribution.csv", index=False)
        else:
            print("  No cm_discriminant column.")
    else:
        print("  No CM cohort found or is_cm column missing.")

    # ---- 6. Visualizations ----
    step(6, 6, "Generating Figure 1 — normalized a_p histograms with semicircle overlay...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    primes_to_plot = [2, 3, 5, 7, 11, 13]
    for ax, p_ in zip(axes.flat, primes_to_plot):
        col = f"a{p_}"
        if col not in classes.columns:
            ax.set_title(f"p = {p_} (no data)"); continue
        ap = pd.to_numeric(noncm[col], errors="coerce").dropna().values
        norm = ap / (2.0 * np.sqrt(p_))
        norm_good = norm[np.abs(norm) <= 1.0]
        if len(norm_good) == 0:
            ax.set_title(f"p = {p_} (no good reduction data)"); continue
        ax.hist(norm_good, bins=60, density=True, alpha=0.6, color="#5a8fd0", edgecolor="#1f4e79", linewidth=0.4)
        # Semicircle density overlay
        x = np.linspace(-1, 1, 200)
        density = (2.0 / np.pi) * np.sqrt(np.maximum(1.0 - x * x, 0.0))
        ax.plot(x, density, "r-", linewidth=2, label="Sato-Tate semicircle")
        ks_stat, ks_p = stats.kstest(norm_good, semicircle_cdf)
        ax.set_title(f"p = {p_}  (n = {len(norm_good):,})\nKS stat = {ks_stat:.4f}")
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("a_p / 2√p")
        if p_ == 2:
            ax.legend(loc="upper right", fontsize=9)
    plt.suptitle("Figure 1. Normalized a_p / 2√p distribution for non-CM Cremona classes,\nwith Sato-Tate semicircle (red) overlaid",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "paper5_aps_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'paper5_aps_distribution.png'}")

    section("PAPER 5 v1 OUTPUTS")
    print(f"""
Five outputs in {output_dir}:

  paper5_aps_diagnostic.csv         — structure of aps_list column
  paper5_aps_by_prime_noncm.csv     — normalized a_p / 2sqrt(p) per prime,
                                       non-CM classes only
  paper5_ks_semicircle.csv          — KS tests against semicircle distribution
  paper5_cm_distribution.csv        — CM cohort: discriminants and a_p at small p
  paper5_aps_distribution.png       — Figure 1, histograms vs semicircle

The headline questions Paper 5 v1 will address from these outputs:

  1. Does the non-CM cohort's a_p / 2sqrt(p) distribution match the Sato-Tate
     semicircle at small primes (p=2,3,5,7,11,13)? KS tests quantify the gap.

  2. At what primes (or sample sizes) does the empirical convergence toward
     the semicircle become statistically clean? At small p, finite-size and
     small-conductor effects contaminate the comparison.

  3. The CM cohort should show concentration at a_p = 0 for primes inert in
     the CM order. The breakdown by prime should reveal which small primes
     are inert vs split for the typical Cremona-corpus CM curve.

  4. Does aps_list contain enough higher primes (p > 13) to compute Sato-Tate
     statistics on the asymptotic regime? The diagnostic CSV answers this.

If aps_list has a useful number of higher primes (say up to p=100 or beyond),
a v2 will extend the analysis to those primes; the small-p analysis here is
the entry point.
""")


if __name__ == "__main__":
    main()
