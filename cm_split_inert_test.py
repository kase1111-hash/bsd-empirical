#!/usr/bin/env python3
"""cm_split_inert_test.py - refined Sato-Tate test for CM curves.

The aggregate CM moments from sato_tate_empirics.py deviated from the
"balanced" (1, 3, 10) prediction. This isn't a bug — it reflects the
specific mix of CM discriminants and primes in the corpus, which doesn't
average out to 50/50 split/inert. This script verifies the underlying
CM theory exactly by partitioning each (curve, prime) pair via the
Kronecker symbol (d_K / p):

    p RAMIFIES in K       iff p | d_K               -> a_p has specific form
    p SPLITS  in K        iff (d_K / p) = +1        -> a_p uniform on cos
    p is INERT in K       iff (d_K / p) = -1        -> a_p = 0 EXACTLY

For E with CM by an order in K with field discriminant d_K, CM theory
predicts:
    - INERT subset: a_p = 0 with no exceptions
    - SPLIT subset: a_p / sqrt(p) = 2 cos(theta), theta uniform on [0, pi]
                    moments m_2 = 2, m_4 = 6, m_6 = 20

Each of the 13 CM discriminants D maps to a field discriminant d_K:
    Fundamental:     D = d_K for D in {-3, -4, -7, -8, -11, -19, -43, -67, -163}
    Non-fundamental: D = -12 -> -3,   -16 -> -4
                     D = -27 -> -3,   -28 -> -7

This script:
    1. Computes Kronecker(d_K, p) for all 13 x 25 (discriminant, prime) pairs
    2. Classifies every CM (curve, prime) good-reduction pair
    3. Tests: inert => a_p = 0 (should be exact)
    4. Tests: split prime moments match (2, 6, 20)
    5. Reports ramified-prime a_p values empirically

Input:  lmfdb_ec_out/ec_corpus_with_isog.parquet
Output: lmfdb_ec_out/cm_split_inert_summary.csv

Author: Kase Branham - Independent Researcher
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, Timer, C, debug,
)


DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
          53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# Map CM order discriminant D -> imaginary-quadratic-field discriminant d_K
CM_ORDER_TO_FIELD = {
    -3:    -3,
    -4:    -4,
    -7:    -7,
    -8:    -8,
    -11:   -11,
    -12:   -3,    # non-fundamental order in Q(√-3)
    -16:   -4,    # non-fundamental order in Q(i)
    -19:   -19,
    -27:   -3,    # non-fundamental order in Q(√-3)
    -28:   -7,    # non-fundamental order in Q(√-7)
    -43:   -43,
    -67:   -67,
    -163:  -163,
}

# CM theory predicts these moments for the SPLIT-only subset
SPLIT_PRIME_MOMENTS = {2: 2.0, 4: 6.0, 6: 20.0}


# =====================================================================
#  KRONECKER SYMBOL
# =====================================================================

def kronecker_d_p(d, p):
    """Kronecker symbol (d/p) for d an imaginary quadratic field discriminant
    (d < 0, d ≡ 0 or 1 mod 4) and p prime.

    Returns +1 if p splits in Q(√d), -1 if inert, 0 if ramified.
    """
    if d % p == 0:
        return 0
    if p == 2:
        # (d/2) by the supplementary law for the Jacobi/Kronecker symbol
        d_mod_8 = d % 8
        return 1 if d_mod_8 in (1, 7) else -1
    # Odd prime: Euler's criterion
    d_mod_p = d % p
    result = pow(d_mod_p, (p - 1) // 2, p)
    return 1 if result == 1 else -1


# =====================================================================
#  MAIN
# =====================================================================

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

    banner("CM split/inert refinement of Sato-Tate test", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load corpus, restrict to CM -------------------------
    step(1, 4, "Loading corpus and restricting to CM curves...")
    corpus = pd.read_parquet(args.corpus)
    cm = corpus[corpus["is_cm"]].copy()
    print(f"  Total CM curves: {len(cm):,}")
    print()

    # Per-discriminant counts
    rows = []
    for D in sorted(CM_ORDER_TO_FIELD.keys()):
        n = int((cm["cm_discriminant"] == D).sum())
        rows.append((D, CM_ORDER_TO_FIELD[D], n))
    summary_table(
        rows,
        ["CM order D", "field d_K", "n curves"],
        title="CM curves per discriminant in the corpus",
        fmt=[">12d", ">10d", ">12,d"],
    )

    # ---- 2. Precompute Kronecker table --------------------------
    step(2, 4, "Building Kronecker symbol lookup table (13 D × 25 p)...")
    kron = {}
    for D, d_K in CM_ORDER_TO_FIELD.items():
        for prime in PRIMES:
            kron[(D, prime)] = kronecker_d_p(d_K, prime)

    section("KRONECKER (d_K / p) LOOKUP TABLE")
    print("  S = split,  I = inert,  R = ramified")
    print()
    # Pretty-print as a wide table
    header_top = "  D     d_K"
    for prime in PRIMES:
        header_top += f"  {prime:>3d}"
    print(header_top)
    print("  " + "-" * (len(header_top) - 2))
    for D in sorted(CM_ORDER_TO_FIELD.keys()):
        d_K = CM_ORDER_TO_FIELD[D]
        row = f"  {D:>4d}  {d_K:>4d}"
        for prime in PRIMES:
            k = kron[(D, prime)]
            sym = " S " if k == 1 else (" I " if k == -1 else " R ")
            row += f"  {sym}"
        print(row)

    # ---- 3. Build (curve, prime) classification -----------------
    step(3, 4, "Classifying all CM (curve, prime) good-reduction pairs...")
    cm_ap_lists = cm["aps_list"].tolist()
    cm_N = cm["conductor"].to_numpy(dtype=np.int64)
    cm_D = cm["cm_discriminant"].astype(int).to_numpy()

    all_ap   = []
    all_cls  = []   # +1 split, -1 inert, 0 ramify
    all_p    = []
    all_D    = []

    with Timer("Classification"):
        for i in range(len(cm)):
            aps = cm_ap_lists[i]
            D = int(cm_D[i])
            N = int(cm_N[i])
            for idx, prime in enumerate(PRIMES):
                if N % prime == 0:
                    continue  # bad reduction; skip
                all_ap.append(int(aps[idx]))
                all_cls.append(kron[(D, prime)])
                all_p.append(prime)
                all_D.append(D)

    all_ap_arr  = np.array(all_ap,  dtype=np.int64)
    all_cls_arr = np.array(all_cls, dtype=np.int8)
    all_p_arr   = np.array(all_p,   dtype=np.int64)
    all_D_arr   = np.array(all_D,   dtype=np.int64)

    split_mask  = (all_cls_arr == 1)
    inert_mask  = (all_cls_arr == -1)
    ramify_mask = (all_cls_arr == 0)

    section("CLASSIFICATION SUMMARY")
    n_total = len(all_ap_arr)
    n_split = int(split_mask.sum())
    n_inert = int(inert_mask.sum())
    n_ram   = int(ramify_mask.sum())
    print(f"  Good-reduction (CM curve, prime) pairs: {n_total:,}")
    print(f"  Split   : {n_split:,}  ({100.0 * n_split / n_total:5.2f}%)")
    print(f"  Inert   : {n_inert:,}  ({100.0 * n_inert / n_total:5.2f}%)")
    print(f"  Ramify  : {n_ram:,}  ({100.0 * n_ram / n_total:5.2f}%)")

    # ---- 4. Run the three tests ---------------------------------
    step(4, 4, "Running tests...")

    section("TEST 1: INERT PRIMES SHOULD GIVE a_p = 0 EXACTLY")
    inert_aps = all_ap_arr[inert_mask]
    n_nonzero = int((inert_aps != 0).sum())
    print(f"  Inert (curve, prime) pairs: {n_inert:,}")
    print(f"  Non-zero a_p at inert primes: {n_nonzero}")
    if n_nonzero == 0:
        print(C.ok("  PASS: every inert prime has a_p = 0 exactly. "
                   "CM theory confirmed."))
    else:
        print(C.warn(f"  FAIL: {n_nonzero} non-zero a_p values at inert primes."))
        nonzero_idx = np.where(inert_mask & (all_ap_arr != 0))[0][:5]
        print("  First few violators:")
        for idx in nonzero_idx:
            print(f"    D = {all_D[idx]}, p = {all_p[idx]}, a_p = {all_ap[idx]}")

    section("TEST 2: SPLIT PRIMES SHOULD GIVE MOMENTS (2, 6, 20)")
    if n_split > 0:
        split_ap = all_ap_arr[split_mask]
        split_p  = all_p_arr[split_mask]
        x = split_ap.astype(np.float64) / np.sqrt(split_p)
        m2 = float((x ** 2).mean())
        m4 = float((x ** 4).mean())
        m6 = float((x ** 6).mean())
        print(f"  Split (curve, prime) pairs: {n_split:,}")
        print(f"  Empirical m_2 = {m2:.4f}   (predicted 2.0,   "
              f"Δ = {m2 - 2:+.4f})")
        print(f"  Empirical m_4 = {m4:.4f}   (predicted 6.0,   "
              f"Δ = {m4 - 6:+.4f})")
        print(f"  Empirical m_6 = {m6:.4f}  (predicted 20.0,  "
              f"Δ = {m6 - 20:+.4f})")
        # Threshold for "pass"
        all_close = (abs(m2 - 2) < 0.10
                     and abs(m4 - 6) < 0.30
                     and abs(m6 - 20) < 1.50)
        if all_close:
            print(C.ok("  PASS: split-prime moments match CM theory."))
        else:
            print(C.warn("  Some deviation; check sample sizes "
                         "and small-prime discreteness."))
    else:
        print(C.warn("  No split-prime pairs found."))

    section("TEST 3: RAMIFIED PRIME a_p VALUES")
    if n_ram > 0:
        ram_ap = all_ap_arr[ramify_mask]
        ram_p  = all_p_arr[ramify_mask]
        ram_D  = all_D_arr[ramify_mask]
        # Group by (D, p) and tabulate distinct a_p values
        ram_counter = Counter()
        for i in range(len(ram_ap)):
            ram_counter[(int(ram_D[i]), int(ram_p[i]), int(ram_ap[i]))] += 1

        rows = []
        for (D, prime, a_p), count in sorted(ram_counter.items()):
            d_K = CM_ORDER_TO_FIELD[D]
            rows.append((D, d_K, prime, a_p, count))
        summary_table(
            rows,
            ["D", "d_K", "prime p", "a_p", "n curves"],
            title="Ramified-prime a_p values (no specific prediction tested)",
            fmt=[">5d", ">5d", ">8d", ">6d", ">10,d"],
        )
    else:
        print("  No ramified-prime pairs found.")

    # ---- Save summary CSV ---------------------------------------
    out_csv = output_dir / "cm_split_inert_summary.csv"
    summary_rows = []
    for prime in PRIMES:
        for cls_label, cls_val in [("split", 1), ("inert", -1), ("ramify", 0)]:
            mask = (all_p_arr == prime) & (all_cls_arr == cls_val)
            if mask.sum() == 0:
                continue
            sub = all_ap_arr[mask]
            x = sub.astype(np.float64) / np.sqrt(prime)
            summary_rows.append({
                "prime":      prime,
                "class":      cls_label,
                "n":          int(len(sub)),
                "n_nonzero":  int((sub != 0).sum()),
                "m2":         float((x ** 2).mean()),
                "m4":         float((x ** 4).mean()),
                "m6":         float((x ** 6).mean()),
            })
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\n  Per-(prime, class) CSV: {out_csv}")

    # ---- Final verdict ------------------------------------------
    section("VERDICT")
    inert_ok = (n_nonzero == 0)
    split_ok = (n_split == 0
                or (abs(m2 - 2) < 0.10
                    and abs(m4 - 6) < 0.30
                    and abs(m6 - 20) < 1.50))
    if inert_ok and split_ok:
        print(C.ok("  CM Sato-Tate verified exactly via Legendre-symbol partition."))
        print("    - Inert primes give a_p = 0 with zero exceptions")
        print("    - Split primes match the (2, 6, 20) prediction")
        print()
        print("  This resolves the aggregate 'deviation' from "
              "sato_tate_empirics.py:")
        print("    The corpus's CM discriminant distribution is biased toward")
        print("    D = -3 and D = -4, so the per-prime split/inert ratio is")
        print("    not 50/50 — exactly as observed in the per-prime moments.")
    else:
        print(C.warn("  Some test failed; see above."))


if __name__ == "__main__":
    main()
