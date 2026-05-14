#!/usr/bin/env python3
"""paper7_true_lambda1.py — True lattice minimum λ_1 computation.

For rank-2 classes: uses Gauss-Lagrange reduction (exact O(log) algorithm) to
compute the true minimum λ_1(E) = min over (m, n) ∈ Z² \\ {0} of
q(m, n) = m² G_11 + 2mn G_12 + n² G_22.

For rank-3 and rank-4 classes: uses brute-force enumeration over (m_1, ..., m_r)
∈ [-N, N]^r with N = 5 (sufficient for LLL-reduced bases; verified by checking
that increasing N does not change the result). For rank 3 this is 11³ = 1,331
lattice points per class; for rank 4 it is 11⁴ = 14,641 lattice points.

For each rank-≥-2 class, compares:
  - ĥ_min^basis(E) = min_i G_{ii} (the stored basis minimum)
  - λ_1(E) (the true lattice minimum)

Outputs:
  paper7_lambda1_rank2.csv          — per-class (conductor, iso, ĥ_min^basis, λ_1, ratio)
  paper7_lambda1_rank3.csv          — same for rank-3
  paper7_lambda1_rank4.csv          — the single rank-4 class
  paper7_lambda1_summary.csv        — summary by rank: how often λ_1 < ĥ_min^basis
  paper7_lambda1_lang_silverman.csv — updated Lang-Silverman ratios with TRUE λ_1
"""

import argparse
import ast
import warnings
from math import log, sqrt
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from kase_utils import add_standard_args, resolve_args, banner, section, step, Timer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")


def gauss_lagrange_rank2_lambda1(G):
    """Compute true λ_1 for a positive-definite 2x2 Gram matrix via Gauss-Lagrange reduction.

    Input: G is a 2x2 numpy array [[a, b], [b, c]] with a, c > 0, ac > b² (positive definite).
    Output: float, the minimum value of q(m, n) = a*m² + 2*b*m*n + c*n² over (m, n) ∈ Z² \\ {0}.

    Algorithm: alternate between
      (1) translation: if |2b| > a, replace b ← b - round(b/a)*a, c ← c - 2*round(b/a)*b + round(b/a)²*a
          (this is the column operation P_2 ← P_2 - k*P_1)
      (2) swap: if a > c, swap (a, c) ↔ (c, a) (and b unchanged in sign convention)
    Terminates with |2b| ≤ a ≤ c; then λ_1 = a.
    """
    a = float(G[0, 0])
    b = float(G[0, 1])
    c = float(G[1, 1])
    iterations = 0
    max_iter = 200
    while iterations < max_iter:
        iterations += 1
        if a > c:
            a, c = c, a
            # b is unchanged (form q(m, n) = a m² + 2b mn + c n²; swap m ↔ n gives same b)
        if abs(2 * b) <= a:
            break
        k = round(b / a)
        new_c = c - 2 * k * b + k * k * a
        new_b = b - k * a
        c = new_c
        b = new_b
    return a


def bruteforce_lambda1(G, max_coord=5):
    """Brute-force minimum of q(v) = v^T G v over v ∈ Z^r \\ {0} with |v_i| ≤ max_coord.
    Works for any rank; suitable for rank 3 and 4 (where LLL-reduced bases give small λ_1).
    """
    G = np.asarray(G, dtype=float)
    r = G.shape[0]
    best = float("inf")
    for vec in product(range(-max_coord, max_coord + 1), repeat=r):
        if all(v == 0 for v in vec):
            continue
        v = np.array(vec, dtype=float)
        q = float(v @ G @ v)
        if q < best:
            best = q
    return best


def bruteforce_lambda1_with_check(G, max_coord_start=5):
    """Brute-force with safety check: increase max_coord until the minimum stabilizes."""
    best = bruteforce_lambda1(G, max_coord_start)
    # Re-check with max_coord + 2 to ensure stability
    best2 = bruteforce_lambda1(G, max_coord_start + 2)
    if best2 < best * (1 - 1e-9):
        # Disagreement — try even larger
        best3 = bruteforce_lambda1(G, max_coord_start + 5)
        return best3, max_coord_start + 5, "expanded"
    return best, max_coord_start, "stable"


def parse_gram_matrix(s, rank):
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

    banner("Paper 7 true-λ_1 computation: actual MW lattice minima", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # Self-test
    step(1, 5, "Self-test of Gauss-Lagrange and brute-force algorithms...")
    # Test case: G = [[100, 99], [99, 100]] — basis min = 100, true λ_1 = (1,-1) gives 2
    G_test = np.array([[100.0, 99.0], [99.0, 100.0]])
    lam_gauss = gauss_lagrange_rank2_lambda1(G_test)
    lam_brute = bruteforce_lambda1(G_test, max_coord=3)
    print(f"  Test G = [[100, 99], [99, 100]]: basis min = 100, expected λ_1 = 2")
    print(f"    Gauss-Lagrange: λ_1 = {lam_gauss:.6f}")
    print(f"    Brute-force:    λ_1 = {lam_brute:.6f}")
    assert abs(lam_gauss - 2) < 1e-9, f"Gauss-Lagrange returns {lam_gauss}, expected 2"
    assert abs(lam_brute - 2) < 1e-9, f"Brute-force returns {lam_brute}, expected 2"
    # Test case 2: G = [[1, 0], [0, 5]] — basis min = 1 = λ_1
    G_test2 = np.array([[1.0, 0.0], [0.0, 5.0]])
    lam_gauss2 = gauss_lagrange_rank2_lambda1(G_test2)
    print(f"  Test G = [[1, 0], [0, 5]]: expected λ_1 = 1")
    print(f"    Gauss-Lagrange: λ_1 = {lam_gauss2:.6f}")
    assert abs(lam_gauss2 - 1) < 1e-9
    # Test case 3: G = [[2, 1], [1, 2]] — already reduced, λ_1 = 2
    G_test3 = np.array([[2.0, 1.0], [1.0, 2.0]])
    lam_gauss3 = gauss_lagrange_rank2_lambda1(G_test3)
    print(f"  Test G = [[2, 1], [1, 2]]: expected λ_1 = 2")
    print(f"    Gauss-Lagrange: λ_1 = {lam_gauss3:.6f}")
    assert abs(lam_gauss3 - 2) < 1e-9
    # Test case 4: rank 3 brute force
    G_test4 = np.diag([1.0, 2.0, 3.0])
    G_test4[0, 1] = G_test4[1, 0] = 0.5
    lam_brute4 = bruteforce_lambda1(G_test4, max_coord=3)
    print(f"  Test rank-3 diagonal+off-diag: λ_1 = {lam_brute4:.4f}")
    print(f"  ✓ All algorithm tests pass")

    # Load corpus
    step(2, 5, "Loading corpus and isolating rank ≥ 2 classes...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")
    number_col = "curve_number" if "curve_number" in corpus.columns else None
    sort_cols = ["conductor", "iso"] + ([number_col] if number_col else [])
    agg_cols = ["rank", "gram_matrix", "regulator"]
    agg_cols = [c for c in agg_cols if c in corpus.columns]
    agg_dict = {c: "first" for c in agg_cols}
    with Timer("Class aggregation"):
        classes = corpus.sort_values(sort_cols).groupby(
            ["conductor", "iso"], as_index=False).agg(agg_dict)
    classes["rank"] = pd.to_numeric(classes["rank"], errors="coerce")
    rank_ge_2 = classes[classes["rank"] >= 2].copy()
    print(f"  Rank ≥ 2 classes: {len(rank_ge_2):,}")
    print(f"  Rank distribution: {dict(rank_ge_2['rank'].value_counts().sort_index())}")

    # Rank 2: Gauss-Lagrange
    step(3, 5, "Computing true λ_1 for rank-2 classes via Gauss-Lagrange...")
    rank2 = rank_ge_2[rank_ge_2["rank"] == 2].copy()
    print(f"  Rank-2 classes to process: {len(rank2):,}")
    rank2_results = []
    n_basis_min_equals_lambda1 = 0
    n_basis_min_gt_lambda1 = 0
    max_relative_diff = 0.0
    with Timer("Rank-2 Gauss-Lagrange"):
        for _, row in rank2.iterrows():
            G = parse_gram_matrix(row["gram_matrix"], 2)
            if G is None:
                continue
            basis_min = float(min(G[0, 0], G[1, 1]))
            true_lam = gauss_lagrange_rank2_lambda1(G)
            log_N = log(int(row["conductor"]))
            ratio_basis = basis_min / log_N
            ratio_true = true_lam / log_N
            rank2_results.append({
                "conductor": int(row["conductor"]), "iso": row["iso"], "rank": 2,
                "h_min_basis": basis_min, "lambda_1_true": true_lam,
                "log_N": log_N, "ratio_basis": ratio_basis, "ratio_true": ratio_true,
                "basis_optimal": (abs(basis_min - true_lam) < 1e-9 * basis_min),
            })
            if abs(basis_min - true_lam) < 1e-9 * basis_min:
                n_basis_min_equals_lambda1 += 1
            else:
                n_basis_min_gt_lambda1 += 1
                rel_diff = (basis_min - true_lam) / basis_min
                max_relative_diff = max(max_relative_diff, rel_diff)
    print(f"  Rank-2 cases where ĥ_min^basis = λ_1 (basis-optimal): {n_basis_min_equals_lambda1:,}")
    print(f"  Rank-2 cases where ĥ_min^basis > λ_1 (basis-suboptimal): {n_basis_min_gt_lambda1:,}")
    print(f"  Max relative gap (ĥ_min^basis - λ_1) / ĥ_min^basis: {max_relative_diff:.6f}")
    pd.DataFrame(rank2_results).to_csv(output_dir / "paper7_lambda1_rank2.csv", index=False)

    # Rank 3: brute force
    step(4, 5, "Computing true λ_1 for rank-3 classes via brute-force enumeration...")
    rank3 = rank_ge_2[rank_ge_2["rank"] == 3].copy()
    print(f"  Rank-3 classes to process: {len(rank3):,}")
    rank3_results = []
    n_r3_basis_eq = 0
    n_r3_basis_gt = 0
    n_r3_expanded = 0
    with Timer("Rank-3 brute-force"):
        for _, row in rank3.iterrows():
            G = parse_gram_matrix(row["gram_matrix"], 3)
            if G is None:
                continue
            basis_min = float(min(G[0, 0], G[1, 1], G[2, 2]))
            true_lam, used_max_coord, stability = bruteforce_lambda1_with_check(G, max_coord_start=5)
            if stability == "expanded":
                n_r3_expanded += 1
            log_N = log(int(row["conductor"]))
            rank3_results.append({
                "conductor": int(row["conductor"]), "iso": row["iso"], "rank": 3,
                "h_min_basis": basis_min, "lambda_1_true": true_lam,
                "log_N": log_N,
                "ratio_basis": basis_min / log_N, "ratio_true": true_lam / log_N,
                "max_coord_used": used_max_coord,
                "basis_optimal": (abs(basis_min - true_lam) < 1e-9 * basis_min),
            })
            if abs(basis_min - true_lam) < 1e-9 * basis_min:
                n_r3_basis_eq += 1
            else:
                n_r3_basis_gt += 1
    print(f"  Rank-3 cases where ĥ_min^basis = λ_1: {n_r3_basis_eq:,}")
    print(f"  Rank-3 cases where ĥ_min^basis > λ_1: {n_r3_basis_gt:,}")
    print(f"  Rank-3 cases requiring expanded max_coord: {n_r3_expanded:,}")
    pd.DataFrame(rank3_results).to_csv(output_dir / "paper7_lambda1_rank3.csv", index=False)

    # Rank 4: brute force on the single class
    rank4 = rank_ge_2[rank_ge_2["rank"] == 4]
    rank4_results = []
    if len(rank4) > 0:
        for _, row in rank4.iterrows():
            G = parse_gram_matrix(row["gram_matrix"], 4)
            if G is None:
                continue
            basis_min = float(np.diag(G).min())
            # Brute force rank 4 with max_coord = 5 (= 11^4 - 1 = 14640 vectors)
            true_lam, used_max_coord, stability = bruteforce_lambda1_with_check(G, max_coord_start=5)
            log_N = log(int(row["conductor"]))
            rank4_results.append({
                "conductor": int(row["conductor"]), "iso": row["iso"], "rank": 4,
                "h_min_basis": basis_min, "lambda_1_true": true_lam,
                "log_N": log_N,
                "ratio_basis": basis_min / log_N, "ratio_true": true_lam / log_N,
                "max_coord_used": used_max_coord,
                "basis_optimal": (abs(basis_min - true_lam) < 1e-9 * basis_min),
            })
            print(f"  Rank-4 class {int(row['conductor'])}{row['iso']}: "
                  f"basis_min = {basis_min:.6f}, λ_1 = {true_lam:.6f}, ratio = {true_lam/log_N:.6f}")
    pd.DataFrame(rank4_results).to_csv(output_dir / "paper7_lambda1_rank4.csv", index=False)

    # Summary
    step(5, 5, "Combining results and recomputing Lang-Silverman with true λ_1...")
    all_rank2_3_4 = rank2_results + rank3_results + rank4_results
    all_df = pd.DataFrame(all_rank2_3_4)
    print(f"  Total rank ≥ 2 classes with both basis and true minima: {len(all_df):,}")
    print(f"\n  Comparison:")
    print(f"    cases ĥ_min^basis = λ_1 (basis-optimal): {(all_df['basis_optimal']).sum():,}  ({100*(all_df['basis_optimal']).mean():.2f}%)")
    print(f"    cases ĥ_min^basis > λ_1 (basis-suboptimal): {(~all_df['basis_optimal']).sum():,}  ({100*(~all_df['basis_optimal']).mean():.2f}%)")
    print(f"\n  Lang-Silverman ratios:")
    print(f"    min ratio (basis): {all_df['ratio_basis'].min():.6f}")
    print(f"    min ratio (TRUE):  {all_df['ratio_true'].min():.6f}")
    print(f"\n  Top 10 cases where basis-min > λ_1 (largest gaps):")
    suboptimal = all_df[~all_df['basis_optimal']].copy()
    suboptimal["gap"] = suboptimal["h_min_basis"] - suboptimal["lambda_1_true"]
    suboptimal = suboptimal.nlargest(10, "gap")
    print(suboptimal[["conductor", "iso", "rank", "h_min_basis", "lambda_1_true", "gap"]].to_string(index=False))

    # Recompute Lang-Silverman with TRUE λ_1 for rank-≥-2; rank-1 already exact
    print(f"\n  Smallest 10 TRUE Lang-Silverman ratios (rank ≥ 2 only):")
    smallest_true = all_df.nsmallest(10, "ratio_true")
    print(smallest_true[["conductor", "iso", "rank", "lambda_1_true", "log_N", "ratio_true"]].to_string(index=False))

    # Save summary
    summary = []
    for r in [2, 3, 4]:
        sub = all_df[all_df["rank"] == r] if "rank" in all_df.columns else pd.DataFrame()
        if len(sub) == 0: continue
        summary.append({
            "rank": r, "n_classes": len(sub),
            "n_basis_optimal": int(sub["basis_optimal"].sum()),
            "frac_basis_optimal": float(sub["basis_optimal"].mean()),
            "min_basis_ratio": float(sub["ratio_basis"].min()),
            "min_true_ratio": float(sub["ratio_true"].min()),
            "min_lambda_1": float(sub["lambda_1_true"].min()),
        })
    pd.DataFrame(summary).to_csv(output_dir / "paper7_lambda1_summary.csv", index=False)

    # Combined Lang-Silverman scan (rank-≥-2 with true λ_1; rank-1 from existing CSV)
    all_df["true_lambda_1"] = all_df["lambda_1_true"]
    all_df.to_csv(output_dir / "paper7_lambda1_lang_silverman.csv", index=False)

    section("PAPER 7 TRUE-λ_1 — HEADLINE FINDINGS")
    print(f"""
True lattice minimum λ_1(E) computed for the rank ≥ 2 cohort:
  rank 2: {n_basis_min_equals_lambda1 + n_basis_min_gt_lambda1:,} classes processed via Gauss-Lagrange reduction
  rank 3: {n_r3_basis_eq + n_r3_basis_gt:,} classes processed via brute-force enumeration
  rank 4: {len(rank4):,} class processed via brute-force enumeration

Basis-vs-true comparison:
  cases ĥ_min^basis = λ_1 (basis-optimal): {(all_df['basis_optimal']).sum():,} ({100*(all_df['basis_optimal']).mean():.2f}%)
  cases ĥ_min^basis > λ_1 (basis-suboptimal): {(~all_df['basis_optimal']).sum():,}

Lang-Silverman ratio update:
  Previous c_emp^basis (across rank ≥ 1):  0.001068 (curve 220110a)
  New c_emp^true (across rank ≥ 2):         {all_df['ratio_true'].min():.6f}
  (combined with rank-1 c_emp^rank1 = 0.001075, the corpus-wide c_emp^true is the smaller of these)

Outputs in {output_dir}:
  paper7_lambda1_rank2.csv          — per rank-2 class: basis_min, λ_1, ratios
  paper7_lambda1_rank3.csv          — per rank-3 class: same
  paper7_lambda1_rank4.csv          — the single rank-4 class
  paper7_lambda1_summary.csv        — summary by rank
  paper7_lambda1_lang_silverman.csv — all rank ≥ 2 classes with both ratios

These feed Paper 7 v3, which incorporates true λ_1 in §6 and eliminates the
basis-minimum-vs-true-minimum ambiguity that v2 had to caveat.
""")


if __name__ == "__main__":
    main()
