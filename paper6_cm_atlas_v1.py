#!/usr/bin/env python3
"""paper6_cm_atlas_v1.py — Structural atlas of CM elliptic curves.

Analyses the 2,670 CM isogeny classes in the Cremona corpus across:
  - conductor distribution by discriminant
  - rank distribution by discriminant
  - Kronecker symbol predictions for a_p across all 25 primes in aps_list
  - BSD invariants (Tamagawa product, modular degree, Sha)
  - isogeny class sizes and available degrees

The central theoretical anchor is the CM trace prediction:
  For a CM elliptic curve E with CM by an order in Q(√D), at a prime p of
  good reduction:
    - if (D/p) = -1 (p inert in K), then a_p = 0
    - if (D/p) = +1 (p splits), then a_p = π + π̄ with N(π) = p (in particular, a_p ≠ 0)
    - if (D/p) = 0 (p ramified), a_p is special

The empirical analysis tests this prediction at each of the 25 primes for
each of the 2,670 CM classes; exceptions (predicted inert but a_p ≠ 0, or
predicted split but a_p = 0) are flagged and attributed to bad reduction
(p | conductor) where possible.

Outputs:
  paper6_cm_conductor_distribution.csv — quantiles of conductor by discriminant
  paper6_cm_rank_distribution.csv      — rank cross-tab by discriminant
  paper6_cm_kronecker_predictions.csv  — per-(D, p): predicted vs observed a_p=0 rates
  paper6_cm_kronecker_exceptions.csv   — flagged (class, p) pairs with prediction mismatch
  paper6_cm_bsd_invariants.csv         — Tamagawa, deg_phi, sha_an stats by discriminant
  paper6_cm_isogeny_structure.csv      — isogeny class sizes and degrees by discriminant

Author: Kase Branham — Independent Researcher
"""

import argparse
import ast
import warnings
from math import gcd
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

FIRST_25_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                   53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# The 9 Heegner discriminants in canonical order
HEEGNER_DISCS = [-3, -4, -7, -8, -11, -19, -43, -67, -163]


def legendre_symbol(a, p):
    """Legendre symbol (a/p) for odd prime p, a integer.
    Returns -1, 0, or 1."""
    a = a % p
    if a == 0:
        return 0
    result = pow(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else result


def kronecker_symbol(D, p):
    """Kronecker symbol (D/p) for integer D and prime p.
    Returns -1, 0, or 1.
    Predicts CM Frobenius trace behavior:
      -1: p inert in Q(sqrt(D)), a_p = 0 at good reduction
      +1: p splits, a_p != 0 (specifically a_p = pi + pi-bar)
       0: p ramified"""
    if p == 2:
        # Special case for p=2: (D/2) depends on D mod 8
        D_mod_8 = D % 8
        if D % 2 == 0:
            return 0  # 2 | D, ramified
        elif D_mod_8 == 1:
            return 1
        elif D_mod_8 == 7:  # D ≡ -1 mod 8
            return 1
        elif D_mod_8 == 3:
            return -1
        elif D_mod_8 == 5:
            return -1
        else:
            return 0
    else:
        return legendre_symbol(D, p)


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

    banner("Paper 6 v1: CM elliptic curves structural atlas", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load corpus and isolate CM cohort ----
    step(1, 6, "Loading corpus and isolating CM cohort...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")

    number_col = "curve_number" if "curve_number" in corpus.columns else None
    sort_cols = ["conductor", "iso"] + ([number_col] if number_col else [])
    # Take 'first' for most columns; class-level data
    agg_cols_first = ["rank", "is_cm", "cm_discriminant", "aps_list", "n_aps",
                       "torsion_order", "tamagawa_product", "real_period", "L_value",
                       "regulator", "sha_an", "deg_phi", "max_isog_degree",
                       "n_isog_primes", "class_size_from_matrix"]
    agg_cols_first = [c for c in agg_cols_first if c in corpus.columns]
    agg_dict = {c: "first" for c in agg_cols_first}
    with Timer("Class aggregation"):
        classes = corpus.sort_values(sort_cols).groupby(
            ["conductor", "iso"], as_index=False).agg(agg_dict)
    print(f"  Total classes: {len(classes):,}")

    cm = classes[classes["is_cm"] == True].copy()
    cm["cm_discriminant"] = pd.to_numeric(cm["cm_discriminant"], errors="coerce")
    print(f"  CM classes: {len(cm):,} ({100*len(cm)/len(classes):.4f}%)")
    print(f"  Discriminants present: {sorted(cm['cm_discriminant'].dropna().unique().astype(int))}")

    # ---- 2. Conductor and rank distribution by discriminant ----
    step(2, 6, "Conductor and rank distributions by discriminant...")
    cond_rows = []
    rank_rows = []
    for D in HEEGNER_DISCS:
        sub = cm[cm["cm_discriminant"] == D]
        if len(sub) == 0: continue
        # Conductor
        conds = pd.to_numeric(sub["conductor"], errors="coerce").dropna().astype(int)
        cond_rows.append({
            "discriminant": D, "n_classes": len(sub),
            "conductor_min": int(conds.min()),
            "conductor_q25": int(conds.quantile(0.25)),
            "conductor_q50": int(conds.quantile(0.50)),
            "conductor_q75": int(conds.quantile(0.75)),
            "conductor_max": int(conds.max()),
        })
        # Rank
        ranks = pd.to_numeric(sub["rank"], errors="coerce").dropna().astype(int)
        rank_counts = ranks.value_counts().sort_index().to_dict()
        row = {"discriminant": D, "n_classes": len(sub),
               "mean_rank": float(ranks.mean()),
               "max_rank": int(ranks.max())}
        for r in range(0, 5):
            row[f"rank_{r}"] = int(rank_counts.get(r, 0))
        rank_rows.append(row)
        print(f"  D={D:>5d}: n={len(sub):>5d}  conductor q50={int(conds.quantile(0.5)):>6d}  "
              f"rank counts {dict(rank_counts)}")

    pd.DataFrame(cond_rows).to_csv(output_dir / "paper6_cm_conductor_distribution.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(output_dir / "paper6_cm_rank_distribution.csv", index=False)

    # ---- 3. Kronecker predictions vs observed across 25 primes ----
    step(3, 6, "Kronecker predictions vs observed a_p across 25 primes...")
    # Extract aps_list for CM curves
    aps_cm = np.full((len(cm), 25), np.nan)
    for i, ap_raw in enumerate(cm["aps_list"].values):
        parsed = parse_ap_list(ap_raw)
        if parsed is None or len(parsed) != 25: continue
        aps_cm[i, :] = parsed

    cm_discs = cm["cm_discriminant"].values.astype(int)
    cm_conds = pd.to_numeric(cm["conductor"], errors="coerce").values

    pred_rows = []
    exception_rows = []
    cm_labels = []
    for idx, (D, N) in enumerate(zip(cm_discs, cm_conds)):
        # Get curve label for exception reporting
        cond_val = cm.iloc[idx]["conductor"]
        iso_val = cm.iloc[idx]["iso"]
        label = f"{int(cond_val)}{iso_val}"
        cm_labels.append(label)

    for j, p_prime in enumerate(FIRST_25_PRIMES):
        ap_col = aps_cm[:, j]
        for D in HEEGNER_DISCS:
            mask_D = (cm_discs == D)
            ap_for_D = ap_col[mask_D]
            valid_mask = ~np.isnan(ap_for_D)
            ap_valid = ap_for_D[valid_mask].astype(int)
            if len(ap_valid) == 0: continue
            n_zero = int((ap_valid == 0).sum())
            n_nonzero = int((ap_valid != 0).sum())
            kron = kronecker_symbol(D, p_prime)
            pred = "inert" if kron == -1 else ("split" if kron == 1 else "ramified")
            pred_a0 = (kron == -1)
            # Bad reduction at p: how many of these curves have p | conductor
            cond_for_D = cm_conds[mask_D][valid_mask].astype(int)
            bad_red_mask = (cond_for_D % p_prime == 0)
            n_bad = int(bad_red_mask.sum())
            n_good = len(ap_valid) - n_bad
            # Good-reduction-only: at good red, prediction should hold
            ap_good = ap_valid[~bad_red_mask]
            n_zero_good = int((ap_good == 0).sum())
            n_nonzero_good = int((ap_good != 0).sum())
            # Prediction accuracy at good reduction
            if pred_a0:
                n_correct = n_zero_good  # predicted a_p=0 and a_p=0
                n_wrong = n_nonzero_good
            elif kron == 1:
                n_correct = n_nonzero_good  # predicted a_p!=0 and a_p!=0
                n_wrong = n_zero_good
            else:  # ramified, no prediction
                n_correct = -1
                n_wrong = -1
            pred_rows.append({
                "discriminant": D, "prime": p_prime,
                "kronecker_symbol": kron, "prediction": pred,
                "n_classes": len(ap_valid),
                "n_bad_reduction_at_p": n_bad,
                "n_good_reduction_at_p": n_good,
                "n_ap_zero_good": n_zero_good,
                "n_ap_nonzero_good": n_nonzero_good,
                "n_prediction_correct_good": n_correct,
                "n_prediction_wrong_good": n_wrong,
            })

            # Record exceptions at good-reduction primes
            if kron == -1 or kron == 1:
                # Iterate over the curves where prediction is wrong
                wrong_mask = bad_red_mask * 0  # boolean over the valid_mask subset
                if kron == -1:
                    # Prediction: a_p = 0. Exception: a_p != 0 at good reduction.
                    wrong_idx = np.where((~bad_red_mask) & (ap_valid != 0))[0]
                else:
                    wrong_idx = np.where((~bad_red_mask) & (ap_valid == 0))[0]
                # Map back to original class indices
                cm_indices_for_D = np.where(mask_D)[0][valid_mask]
                for k in wrong_idx:
                    cls_idx = cm_indices_for_D[k]
                    exception_rows.append({
                        "label": cm_labels[cls_idx],
                        "discriminant": D,
                        "conductor": int(cm_conds[cls_idx]),
                        "prime": p_prime,
                        "predicted": "a_p=0" if kron == -1 else "a_p!=0",
                        "observed_ap": int(aps_cm[cls_idx, j]),
                        "p_divides_conductor": False,
                    })

    pd.DataFrame(pred_rows).to_csv(output_dir / "paper6_cm_kronecker_predictions.csv", index=False)
    pd.DataFrame(exception_rows).to_csv(output_dir / "paper6_cm_kronecker_exceptions.csv", index=False)

    # Headline accuracy
    pred_df = pd.DataFrame(pred_rows)
    pred_clean = pred_df[pred_df["n_prediction_correct_good"] >= 0]
    total_correct = int(pred_clean["n_prediction_correct_good"].sum())
    total_wrong = int(pred_clean["n_prediction_wrong_good"].sum())
    total_predictions = total_correct + total_wrong
    print(f"\n  Kronecker prediction accuracy at good-reduction primes:")
    print(f"    Total predictions made: {total_predictions:,}")
    print(f"    Correct:                {total_correct:,} ({100*total_correct/total_predictions:.4f}%)")
    print(f"    Wrong (exceptions):     {total_wrong:,} ({100*total_wrong/total_predictions:.4f}%)")

    # ---- 4. BSD invariants by discriminant ----
    step(4, 6, "BSD invariants by discriminant...")
    bsd_rows = []
    for D in HEEGNER_DISCS:
        sub = cm[cm["cm_discriminant"] == D]
        if len(sub) == 0: continue
        tama = pd.to_numeric(sub["tamagawa_product"], errors="coerce").dropna() if "tamagawa_product" in sub.columns else pd.Series()
        deg = pd.to_numeric(sub["deg_phi"], errors="coerce").dropna() if "deg_phi" in sub.columns else pd.Series()
        sha = pd.to_numeric(sub["sha_an"], errors="coerce").dropna() if "sha_an" in sub.columns else pd.Series()
        bsd_rows.append({
            "discriminant": D, "n_classes": len(sub),
            "tamagawa_q50": float(tama.quantile(0.50)) if len(tama) else np.nan,
            "tamagawa_max": float(tama.max()) if len(tama) else np.nan,
            "deg_phi_q50": float(deg.quantile(0.50)) if len(deg) else np.nan,
            "deg_phi_q95": float(deg.quantile(0.95)) if len(deg) else np.nan,
            "deg_phi_max": float(deg.max()) if len(deg) else np.nan,
            "sha_an_q50": float(sha.quantile(0.50)) if len(sha) else np.nan,
            "sha_an_max": float(sha.max()) if len(sha) else np.nan,
            "n_sha_eq_1": int((sha == 1).sum()) if len(sha) else 0,
            "n_sha_gt_1": int((sha > 1).sum()) if len(sha) else 0,
        })
        print(f"  D={D:>5d}: tamagawa q50={bsd_rows[-1]['tamagawa_q50']:.2f}  "
              f"deg_phi q50={bsd_rows[-1]['deg_phi_q50']:.2f}  "
              f"sha_an q50={bsd_rows[-1]['sha_an_q50']:.2f}")
    pd.DataFrame(bsd_rows).to_csv(output_dir / "paper6_cm_bsd_invariants.csv", index=False)

    # ---- 5. Isogeny structure ----
    step(5, 6, "Isogeny structure by discriminant...")
    isog_rows = []
    for D in HEEGNER_DISCS:
        sub = cm[cm["cm_discriminant"] == D]
        if len(sub) == 0: continue
        class_size = pd.to_numeric(sub["class_size_from_matrix"], errors="coerce").dropna() if "class_size_from_matrix" in sub.columns else pd.Series()
        max_isog = pd.to_numeric(sub["max_isog_degree"], errors="coerce").dropna() if "max_isog_degree" in sub.columns else pd.Series()
        n_isog = pd.to_numeric(sub["n_isog_primes"], errors="coerce").dropna() if "n_isog_primes" in sub.columns else pd.Series()
        row = {
            "discriminant": D, "n_classes": len(sub),
            "class_size_q50": float(class_size.quantile(0.50)) if len(class_size) else np.nan,
            "class_size_max": int(class_size.max()) if len(class_size) else 0,
            "max_isog_degree_q50": float(max_isog.quantile(0.50)) if len(max_isog) else np.nan,
            "max_isog_degree_max": int(max_isog.max()) if len(max_isog) else 0,
            "n_isog_primes_q50": float(n_isog.quantile(0.50)) if len(n_isog) else np.nan,
        }
        # Available isogeny degrees indicator columns
        for d in [2, 3, 5, 7, 11, 13, 17, 19, 37, 43, 67, 163]:
            col = f"has_{d}_isog"
            if col in sub.columns:
                row[f"frac_has_{d}_isog"] = float(sub[col].mean())
        isog_rows.append(row)
        print(f"  D={D:>5d}: class_size q50={row['class_size_q50']:.1f}  "
              f"max_isog q50={row['max_isog_degree_q50']:.1f}")
    pd.DataFrame(isog_rows).to_csv(output_dir / "paper6_cm_isogeny_structure.csv", index=False)

    # ---- 6. Figure: Kronecker accuracy by prime ----
    step(6, 6, "Generating Figure 1 — Kronecker prediction accuracy by prime...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Left: per-prime accuracy (across all discriminants)
    p_acc = []
    for p_prime in FIRST_25_PRIMES:
        sub = pred_clean[pred_clean["prime"] == p_prime]
        tc = int(sub["n_prediction_correct_good"].sum())
        tw = int(sub["n_prediction_wrong_good"].sum())
        tt = tc + tw
        p_acc.append({"prime": p_prime, "accuracy": tc/tt if tt > 0 else 1.0, "n": tt})
    p_acc_df = pd.DataFrame(p_acc)
    axes[0].semilogx(p_acc_df["prime"], p_acc_df["accuracy"], "o-", color="#1f4e79", markersize=8)
    axes[0].axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="Perfect prediction")
    axes[0].set_xlabel("prime p (log scale)")
    axes[0].set_ylabel("Kronecker prediction accuracy")
    axes[0].set_title("Kronecker symbol prediction accuracy by prime\n(non-trivial predictions only; good-reduction primes)")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which="both")
    # Right: per-discriminant accuracy
    d_acc = []
    for D in HEEGNER_DISCS:
        sub = pred_clean[pred_clean["discriminant"] == D]
        tc = int(sub["n_prediction_correct_good"].sum())
        tw = int(sub["n_prediction_wrong_good"].sum())
        tt = tc + tw
        if tt > 0:
            d_acc.append({"discriminant": D, "accuracy": tc/tt, "n": tt})
    d_acc_df = pd.DataFrame(d_acc)
    axes[1].bar(range(len(d_acc_df)), d_acc_df["accuracy"], color="#5a8fd0", edgecolor="#1f4e79")
    axes[1].set_xticks(range(len(d_acc_df)))
    axes[1].set_xticklabels([f"D={int(d)}" for d in d_acc_df["discriminant"]], rotation=45)
    axes[1].axhline(y=1.0, color="red", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Kronecker prediction accuracy")
    axes[1].set_title("Kronecker prediction accuracy by Heegner discriminant\n(non-trivial predictions, good-reduction primes)")
    axes[1].set_ylim(0.9, 1.005)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "paper6_cm_kronecker_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'paper6_cm_kronecker_accuracy.png'}")

    section("PAPER 6 v1 — HEADLINE FINDINGS")
    print(f"""
The 2,670 CM isogeny classes in the Cremona corpus, analyzed across:

  Conductor range:    {cm['conductor'].min():,} to {cm['conductor'].max():,}
  Discriminants:      9 Heegner values; D=-3 dominates ({100*len(cm[cm['cm_discriminant']==-3])/len(cm):.1f}%)
  Total Kronecker tests: {total_predictions:,} (over 25 primes × 9 discriminants, restricted to good red.)
  Kronecker accuracy:   {100*total_correct/total_predictions:.4f}% correct, {total_wrong:,} exceptions

Outputs in {output_dir}:

  paper6_cm_conductor_distribution.csv  — conductor quantiles by discriminant
  paper6_cm_rank_distribution.csv       — rank crosstab by discriminant
  paper6_cm_kronecker_predictions.csv   — per-(D,p) prediction vs observed (225 rows)
  paper6_cm_kronecker_exceptions.csv    — individual class-prime mismatches
  paper6_cm_bsd_invariants.csv          — Tamagawa, deg_phi, sha_an by discriminant
  paper6_cm_isogeny_structure.csv       — class sizes, isogeny degrees by discriminant
  paper6_cm_kronecker_accuracy.png      — Figure 1: prediction accuracy by prime and by D

These feed Paper 6 v1: a structural atlas of CM curves covering conductor,
rank, BSD invariants, isogeny structure, and the Kronecker-symbol prediction
test for Frobenius traces.
""")


if __name__ == "__main__":
    main()
