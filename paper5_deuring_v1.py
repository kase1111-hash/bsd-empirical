#!/usr/bin/env python3
"""paper5_deuring_v1.py — Deuring distribution comparison.

For each prime p in aps_list (the first 25 primes, p ≤ 97), compute:
  - The Deuring probability distribution μ_p(a) = H(4p - a²) / (2p)
    where H is the Hurwitz class number (computed in pure Python here).
  - KS distance from the Cremona empirical distribution to the Deuring distribution
  - KS distance from the Deuring distribution to the semicircle
  - KS distance from Cremona to semicircle (recomputed for comparison)

The hypothesis tested: if D_KS(Cremona, Deuring) << D_KS(Cremona, semicircle),
then the Cremona empirical excess over the support-geometry bound is mostly
explained by the Deuring finite-p weighting effect; the corpus is well-modeled
by Deuring at finite p, with the "deviation from semicircle" mostly capturing
the Deuring-vs-semicircle gap rather than a separate Q-side structural feature.

If D_KS(Cremona, Deuring) is still comparable to D_KS(Cremona, semicircle),
then the corpus distribution differs from Deuring as well, indicating a
separate sampling-related deviation that is not captured by Deuring.

Outputs:
  paper5_deuring_v1.csv  — per-prime D_KS(Cremona,semi), D_KS(Cremona,Deuring),
                            D_KS(Deuring,semi), and the Hurwitz mass = 2p check
  paper5_deuring_vs_semi.png  — D_KS(Cremona,semi) and D_KS(Deuring,semi) on
                                  the same plot (= excess Cremona has over
                                  what Deuring would already produce)
  paper5_cremona_vs_deuring.png — D_KS(Cremona,Deuring) on its own plot, with
                                  the Prop 1 bound for reference
"""

import argparse
import ast
import warnings
from fractions import Fraction
from math import isqrt
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

FIRST_25_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                   53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def semicircle_cdf(x):
    x = np.clip(x, -1.0, 1.0)
    return 0.5 + (x * np.sqrt(np.maximum(1.0 - x * x, 0.0)) + np.arcsin(x)) / np.pi


def hurwitz_class_number(N):
    """H(N) Hurwitz class number, exact via Fraction.

    Sums over SL_2(Z)-equivalence classes of positive-definite binary quadratic
    forms (a, b, c) with b^2 - 4ac = -N, where each form's weight is 1/|Aut|:
      - 1/2 if equivalent to (k, 0, k)
      - 1/3 if equivalent to (k, k, k)
      - 1 otherwise
    Includes non-primitive forms (so this is the Hurwitz class number proper,
    not the Hurwitz-Kronecker class number). Verified against Eichler's mass
    formula sum_a H(4p - a^2) = 2p for primes p = 2, 3, ..., 97 (exact).
    """
    if N == 0:
        return Fraction(-1, 12)
    if N < 0:
        return Fraction(0)
    if N % 4 not in (0, 3):
        return Fraction(0)
    H = Fraction(0)
    a = 1
    while 3 * a * a <= N:
        for b in range(-(a - 1), a + 1):
            num = b * b + N
            if num % (4 * a) != 0:
                continue
            c = num // (4 * a)
            if c < a:
                continue
            if a == c and b < 0:
                continue  # reduction: when a = c, restrict to b >= 0
            if b == 0 and a == c:
                H += Fraction(1, 2)
            elif b == a and a == c:
                H += Fraction(1, 3)
            else:
                H += Fraction(1)
        a += 1
    return H


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


def deuring_distribution(p):
    """Return (a_values, masses) for the Deuring distribution at prime p.
    a_values are integers, masses are probabilities (normalized to sum to 1)."""
    a_lim = isqrt(4 * p)
    if a_lim * a_lim == 4 * p:
        a_lim -= 1  # exclude |a| = 2√p (the supersingular boundary)
    a_values = np.arange(-a_lim, a_lim + 1)
    H_values = np.array([float(hurwitz_class_number(4 * p - a * a)) for a in a_values])
    total = H_values.sum()
    # Eichler mass formula: total should be exactly 2p
    if abs(total - 2 * p) > 1e-9:
        raise ValueError(f"Deuring mass = {total:.6f}, expected 2p = {2*p}; Hurwitz implementation bug?")
    masses = H_values / total
    return a_values, masses


def empirical_ks(empirical_normalized, target_cdf):
    """KS distance between an empirical distribution (1D array of values) and a target CDF function."""
    return stats.kstest(empirical_normalized, target_cdf).statistic


def ks_discrete_to_discrete(empirical_normalized, target_support, target_masses):
    """KS distance between empirical sample (1D array on S_p) and a discrete target
    distribution (target_support, target_masses) on the same support S_p.

    For two distributions on the same finite support, the KS supremum is
    attained at one of the support points, so:

      D = max_i |F_emp(s_i) - F_target(s_i)|

    where F_emp(s_i) = (# emp samples ≤ s_i) / n_emp and F_target(s_i) is
    the cumulative target mass up to and including s_i."""
    sort_idx = np.argsort(target_support)
    sup_sorted = np.asarray(target_support)[sort_idx]
    m_sorted = np.asarray(target_masses)[sort_idx]
    target_cdf = np.cumsum(m_sorted)
    emp_sorted = np.sort(empirical_normalized)
    n = len(emp_sorted)
    # F_emp at each support point: count of samples <= s_i
    emp_counts = np.searchsorted(emp_sorted, sup_sorted, side='right')
    emp_cdf = emp_counts / n
    return float(np.max(np.abs(emp_cdf - target_cdf)))


def deuring_cdf(p, x_array):
    """Evaluate the Deuring CDF (normalized to support in [-1, 1]) at given x values."""
    a_values, masses = deuring_distribution(p)
    support_normalized = a_values / (2.0 * np.sqrt(p))
    sorted_idx = np.argsort(support_normalized)
    support_sorted = support_normalized[sorted_idx]
    masses_sorted = masses[sorted_idx]
    cdf_at_support = np.cumsum(masses_sorted)
    # For x < min(support_sorted): 0
    # For x >= max(support_sorted): 1
    # For x in between: piecewise constant
    result = np.zeros_like(x_array, dtype=float)
    for i, x in enumerate(x_array):
        idx = np.searchsorted(support_sorted, x, side='right')
        result[i] = cdf_at_support[idx - 1] if idx > 0 else 0.0
    return result


def ks_deuring_to_semicircle(p):
    """KS distance from the Deuring distribution (discrete on S_p) to the
    semicircle CDF (continuous on [-1, 1]).

    Both the just-before and just-after limits of F_Deuring at each support point
    are checked, since F_Deuring has jumps and the KS supremum |F_D - F_semi|
    can be attained on either side. Boundary cells [-1, s_0] and [s_n, 1] are
    also covered (F_Deuring = 0 on the left, F_Deuring = 1 on the right)."""
    a_values, masses = deuring_distribution(p)
    support_normalized = a_values / (2.0 * np.sqrt(p))
    sorted_idx = np.argsort(support_normalized)
    support_sorted = support_normalized[sorted_idx]
    masses_sorted = masses[sorted_idx]
    cdf_at_support = np.cumsum(masses_sorted)  # F_D(s_i) (after the jump at s_i)
    # F_D just before s_i: cumulative mass strictly before s_i
    cdf_just_before = np.concatenate([[0.0], cdf_at_support[:-1]])  # F_D(s_i -)
    # Sup discrepancy at each support point: max of |F_D(s_i-) - F_semi(s_i)| and |F_D(s_i+) - F_semi(s_i)|
    f_semi_at_support = semicircle_cdf(support_sorted)
    diff_before = np.abs(cdf_just_before - f_semi_at_support)
    diff_after = np.abs(cdf_at_support - f_semi_at_support)
    # Boundary cells: on (-1, s_0), F_D = 0, F_semi rises from 0 to F_semi(s_0-)
    # Sup on this cell = F_semi(s_0). On (s_n, 1), F_D = 1, sup = 1 - F_semi(s_n).
    boundary_left = f_semi_at_support[0]  # F_semi(s_0) - 0
    boundary_right = 1.0 - f_semi_at_support[-1]
    return float(max(diff_before.max(), diff_after.max(), boundary_left, boundary_right))


def proposition_1_lower_bound(p):
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

    banner("Paper 5 Deuring v1: Cremona vs Deuring comparison", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Verify Hurwitz / Eichler mass at every prime ----
    step(1, 5, "Verifying Eichler mass formula at 25 primes...")
    for p in FIRST_25_PRIMES:
        a_values, masses = deuring_distribution(p)
        if abs(masses.sum() - 1.0) > 1e-12:
            print(f"  ✗ p={p}: Deuring masses don't sum to 1: {masses.sum()}")
        else:
            pass  # print(f"  ✓ p={p}: |S_Deuring| = {len(a_values)}, mass sum = 1.0")
    print(f"  ✓ All 25 primes pass Eichler mass formula (sum H(4p-a²) = 2p exactly)")

    # ---- 2. Load corpus and extract aps_list ----
    step(2, 5, "Loading corpus and extracting aps_list...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")
    number_col = "curve_number" if "curve_number" in corpus.columns else None
    sort_cols = ["conductor", "iso"] + ([number_col] if number_col else [])
    agg_cols = ["rank", "is_cm", "aps_list", "n_aps"]
    agg_cols = [c for c in agg_cols if c in corpus.columns]
    agg_dict = {c: "first" for c in agg_cols}
    with Timer("Class aggregation"):
        classes = corpus.sort_values(sort_cols).groupby(
            ["conductor", "iso"], as_index=False).agg(agg_dict)
    print(f"  Classes: {len(classes):,}")

    aps_per_curve = np.full((len(classes), 25), np.nan)
    with Timer("Parsing aps_list"):
        for i, ap_raw in enumerate(classes["aps_list"].values):
            parsed = parse_ap_list(ap_raw)
            if parsed is None or len(parsed) != 25:
                continue
            aps_per_curve[i, :] = parsed
    is_cm = classes["is_cm"].values if "is_cm" in classes.columns else np.zeros(len(classes), dtype=bool)
    n_noncm = int((~is_cm).sum())
    print(f"  Non-CM classes: {n_noncm:,}")

    # ---- 3. Three KS distances per prime ----
    step(3, 5, "Computing three KS distances per prime...")
    results = []
    for j, p_prime in enumerate(FIRST_25_PRIMES):
        a_col = aps_per_curve[:, j]
        ap_noncm = a_col[~is_cm]
        ap_noncm = ap_noncm[~np.isnan(ap_noncm)]
        norm = ap_noncm / (2.0 * np.sqrt(p_prime))
        norm_strict = norm[np.abs(norm) <= 1.0]
        # D_KS(Cremona, semicircle)
        ks_cs = empirical_ks(norm_strict, semicircle_cdf)
        # D_KS(Deuring, semicircle)
        ks_ds = ks_deuring_to_semicircle(p_prime)
        # D_KS(Cremona, Deuring)
        a_d, m_d = deuring_distribution(p_prime)
        ks_cd = ks_discrete_to_discrete(norm_strict, a_d / (2.0 * np.sqrt(p_prime)), m_d)
        # Prop 1 bound (for reference)
        prop1 = proposition_1_lower_bound(p_prime)
        results.append({
            "prime": p_prime,
            "n_noncm": int(len(norm_strict)),
            "ks_cremona_semicircle": ks_cs,
            "ks_cremona_deuring":    ks_cd,
            "ks_deuring_semicircle": ks_ds,
            "prop1_lower_bound":     prop1,
            "ratio_CD_over_CS":      ks_cd / ks_cs,
            "ratio_DS_over_CS":      ks_ds / ks_cs,
        })
        print(f"  p={p_prime:>3d}: KS(C,sem)={ks_cs:.4f}  KS(D,sem)={ks_ds:.4f}  "
              f"KS(C,D)={ks_cd:.4f}  Prop1={prop1:.4f}  CD/CS={ks_cd/ks_cs:.3f}  DS/CS={ks_ds/ks_cs:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "paper5_deuring_v1.csv", index=False)

    # ---- 4. Figure: KS comparisons ----
    step(4, 5, "Generating Figure 4 — Cremona-vs-semi and Deuring-vs-semi on same plot...")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(df["prime"], df["ks_cremona_semicircle"], "o-", color="#1f4e79",
              linewidth=1.5, markersize=7, label="KS(Cremona, semicircle)")
    ax.loglog(df["prime"], df["ks_deuring_semicircle"], "^-", color="green",
              linewidth=1.5, markersize=7, label="KS(Deuring, semicircle)")
    ax.loglog(df["prime"], df["prop1_lower_bound"], "s--", color="orange",
              linewidth=1.5, markersize=5, label="Proposition 1 lower bound")
    ax.set_xlabel("prime p (log scale)", fontsize=11)
    ax.set_ylabel("KS distance vs semicircle", fontsize=11)
    ax.set_title("Cremona vs Deuring vs semicircle: KS distances at p ≤ 97\nDeuring (green) is the finite-prime reference for an across-corpus statistic", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "paper5_deuring_vs_semi.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'paper5_deuring_vs_semi.png'}")

    # ---- 5. Figure: Cremona vs Deuring directly ----
    step(5, 5, "Generating Figure 5 — KS(Cremona, Deuring) on its own...")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogx(df["prime"], df["ks_cremona_deuring"], "D-", color="red",
                linewidth=1.5, markersize=7, label="KS(Cremona, Deuring)")
    ax.semilogx(df["prime"], df["ks_cremona_semicircle"], "o-", color="#1f4e79",
                linewidth=1.5, markersize=7, alpha=0.5, label="KS(Cremona, semicircle) — for comparison")
    ax.set_xlabel("prime p (log scale)", fontsize=11)
    ax.set_ylabel("KS distance", fontsize=11)
    ax.set_title("Direct comparison: how much of the Cremona's KS distance from\nthe semicircle is the Deuring–vs–semicircle finite-prime effect?", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "paper5_cremona_vs_deuring.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'paper5_cremona_vs_deuring.png'}")

    # Summary
    section("PAPER 5 DEURING v1 — HEADLINE FINDINGS")
    print(f"""
Across the 25 primes p ≤ 97 in aps_list, n = {n_noncm:,} non-CM classes:

  KS(Cremona, semicircle):  ranges from {df['ks_cremona_semicircle'].min():.4f} to {df['ks_cremona_semicircle'].max():.4f}
  KS(Cremona, Deuring):     ranges from {df['ks_cremona_deuring'].min():.4f} to {df['ks_cremona_deuring'].max():.4f}
  KS(Deuring, semicircle):  ranges from {df['ks_deuring_semicircle'].min():.4f} to {df['ks_deuring_semicircle'].max():.4f}

  Mean ratio KS(C,D) / KS(C,sem):  {df['ratio_CD_over_CS'].mean():.4f}
  Mean ratio KS(D,sem) / KS(C,sem): {df['ratio_DS_over_CS'].mean():.4f}

Interpretation:
  - If KS(C,D) << KS(C,sem): Cremona ≈ Deuring at finite p, the apparent
    deviation from semicircle is mostly the Deuring-vs-semicircle finite-p effect.
  - If KS(C,D) ≈ KS(C,sem): Cremona differs from Deuring too; the corpus's
    Q-side conductor-bounded sampling produces a separate deviation that
    Deuring weighting alone does not explain.

Outputs in {output_dir}:
  paper5_deuring_v1.csv         — full 25-prime per-prime KS table
  paper5_deuring_vs_semi.png    — three lines (Cremona/Deuring/semi distance) vs p
  paper5_cremona_vs_deuring.png — direct Cremona-vs-Deuring comparison

These feed Paper 5 v5, which will integrate the Deuring closure and clarify
whether the empirical "deviation from semicircle" reported in §3.5 is a
Deuring-vs-semicircle finite-p effect or a separate Q-side structural feature.
""")


if __name__ == "__main__":
    main()
