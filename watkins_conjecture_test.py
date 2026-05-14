#!/usr/bin/env python3
"""watkins_conjecture_test.py - corpus-scale test of Watkins's conjecture.

Watkins (2002) conjectured that for an elliptic curve E/Q of rank r, the
modular degree deg(phi_E) of the optimal curve in the isogeny class of E
satisfies:

    2^rank(E) | deg(phi_E_optimal)

Watkins verified this for low-conductor curves at the time of his paper.
This script extends the verification across the full Cremona elliptic
curve database (2,164,260 isogeny classes, conductors up to 499,998).

Method:
    1. Restrict to isogeny classes with deg_phi recorded (one per class)
    2. For each class, compute v_2(deg_phi) (the 2-adic valuation)
    3. The conjecture predicts v_2(deg_phi) >= rank for every class
    4. For each rank r >= 1, report: count, pass rate, and the full
       distribution of v_2(deg_phi) within the rank-r cohort
    5. List any classes that violate the conjecture (none expected)

The distribution of v_2(deg_phi) per rank is itself interesting: it
tells us whether Watkins's lower bound is typically tight (v_2 = r) or
loose (v_2 >> r), which speaks to whether deg_phi has more 2-adic
content than the bound forces.

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet
Output: lmfdb_ec_out/watkins_test_results.csv
        lmfdb_ec_out/watkins_failures.csv  (only if violations exist)

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


DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_spectrum.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")


def vectorized_v_2(arr):
    """Vectorized 2-adic valuation.

    For positive integer n, v_2(n) is the largest k such that 2^k divides n.
    Uses the bit-trick: for n > 0, (n & -n) extracts the lowest set bit,
    which equals 2^{v_2(n)}, and log2 of that gives v_2.
    """
    arr = np.asarray(arr, dtype=np.int64)
    out = np.zeros(len(arr), dtype=np.int64)
    pos = arr > 0
    lowest_bit = arr[pos] & -arr[pos]
    out[pos] = np.log2(lowest_bit).astype(np.int64)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(p)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--max-failures-to-list", type=int, default=30)
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("Watkins's conjecture test: 2^rank | deg_phi", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    # ---- 1. Load optimal curves ----------------------------------
    step(1, 3, "Loading optimal-curve subset...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    corpus = corpus[corpus["deg_phi"].notna()].copy()
    corpus["deg_phi"]  = corpus["deg_phi"].astype(np.int64)
    corpus["rank_int"] = corpus["rank"].astype(np.int64)
    print(f"  Optimal curves with deg_phi: {len(corpus):,}")
    print(f"  (one per isogeny class)")

    # ---- 2. v_2(deg_phi) -----------------------------------------
    step(2, 3, "Computing v_2(deg_phi)...")
    with Timer("v_2 vectorized"):
        corpus["v2_deg_phi"] = vectorized_v_2(corpus["deg_phi"].values)

    min_v2 = int(corpus["v2_deg_phi"].min())
    max_v2 = int(corpus["v2_deg_phi"].max())
    print(f"  v_2(deg_phi) range across corpus: [{min_v2}, {max_v2}]")
    print(f"  (max possible value bounded by log2(max deg_phi); deg_phi can be odd)")

    # ---- 3. Per-rank Watkins test --------------------------------
    step(3, 3, "Per-rank conjecture test...")

    section("RANK DISTRIBUTION IN CORPUS")
    rank_counts = corpus["rank_int"].value_counts().sort_index()
    rows = [(int(r), int(c), 100.0 * c / len(corpus))
            for r, c in rank_counts.items()]
    summary_table(
        rows,
        ["rank", "n classes", "% of corpus"],
        title="Rank distribution",
        fmt=[">5d", ">14,d", ">12.3f"],
    )

    max_rank = int(corpus["rank_int"].max())
    print(f"  Maximum rank observed: {max_rank}")
    print()

    section("WATKINS'S CONJECTURE BY RANK")
    print("  For rank r >= 1: does 2^r divide deg_phi for every class?")
    print("  (rank 0 reported for context; conjecture is 2^0 = 1, trivially true)")
    print()

    summary_rows = []
    failures_to_save = []

    for r in range(0, max_rank + 1):
        sub = corpus[corpus["rank_int"] == r]
        n = int(len(sub))
        if n == 0:
            continue

        if r == 0:
            avg_v2 = float(sub["v2_deg_phi"].mean())
            min_v2 = int(sub["v2_deg_phi"].min())
            max_v2 = int(sub["v2_deg_phi"].max())
            print(f"  Rank {r}: {n:>10,} classes  "
                  f"(Watkins trivially: 2^0=1 divides everything)")
            print(f"           v_2(deg_phi) range [{min_v2}, {max_v2}], "
                  f"mean {avg_v2:.2f}")
            summary_rows.append((r, n, n, 100.0,
                                 float(sub["v2_deg_phi"].mean()),
                                 int(sub["v2_deg_phi"].max())))
            continue

        passes = sub["v2_deg_phi"] >= r
        n_pass = int(passes.sum())
        n_fail = n - n_pass
        rate   = 100.0 * n_pass / n if n > 0 else 0.0

        avg_v2 = float(sub["v2_deg_phi"].mean())
        med_v2 = float(sub["v2_deg_phi"].median())
        max_v2 = int(sub["v2_deg_phi"].max())
        min_v2 = int(sub["v2_deg_phi"].min())

        summary_rows.append((r, n, n_pass, rate, avg_v2, max_v2))

        if n_fail > 0:
            print(C.warn(f"  Rank {r}: {n:>10,} classes  "
                         f"PASS {n_pass:,} ({rate:.4f}%)  "
                         f"FAIL {n_fail:,}"))
            failures = sub[~passes].copy()
            failures_to_save.append(failures)
            print(f"           Watkins's conjecture FAILS at "
                  f"{min(n_fail, args.max_failures_to_list)} listed cases:")
            for fr in failures.head(args.max_failures_to_list).itertuples():
                print(f"             {fr.label:<12}  "
                      f"rank={r}  "
                      f"deg_phi={fr.deg_phi:,}  "
                      f"v2(deg_phi)={fr.v2_deg_phi}  "
                      f"(needed >= {r})")
        else:
            print(C.ok(f"  Rank {r}: {n:>10,} classes  "
                       f"ALL PASS (100.0000%)"))
        print(f"           v_2(deg_phi) range [{min_v2}, {max_v2}], "
              f"median {med_v2:.1f}, mean {avg_v2:.2f}")
        print(f"           "
              f"excess over lower bound r={r}: median {med_v2-r:.1f}, "
              f"mean {avg_v2-r:.2f}")

    print()
    section("SUMMARY TABLE")
    summary_table(
        summary_rows,
        ["rank", "n classes", "n pass", "pass rate %",
         "avg v_2(dp)", "max v_2(dp)"],
        title="Watkins's conjecture by rank",
        fmt=[">5d", ">12,d", ">12,d", ">14.4f",
             ">14.3f", ">14d"],
    )

    # ---- v_2 distribution per rank -------------------------------
    section("DISTRIBUTION OF v_2(deg_phi) WITHIN EACH RANK COHORT")
    print("  How tightly does Watkins's bound hold? If v_2 = r typically,")
    print("  the bound is sharp; if v_2 >> r typically, deg_phi has extra")
    print("  2-adic content beyond what the conjecture requires.")
    print()
    for r in range(0, max_rank + 1):
        sub = corpus[corpus["rank_int"] == r]
        if len(sub) == 0:
            continue
        v2_counts = Counter(sub["v2_deg_phi"].astype(int))
        if not v2_counts:
            continue
        all_keys = sorted(v2_counts.keys())
        print(f"  Rank {r}: n = {len(sub):,}")
        for k in all_keys:
            c = v2_counts[k]
            pct = 100.0 * c / len(sub)
            bar = "█" * min(40, int(pct / 2))
            marker = ""
            if r >= 1 and k < r:
                marker = C.warn("  VIOLATES Watkins (k < r)")
            elif r >= 1 and k == r:
                marker = "  (tight, k = r)"
            print(f"    v_2 = {k:>2d}: {c:>10,} ({pct:>6.3f}%)  {bar}{marker}")
        print()

    # ---- Save ----------------------------------------------------
    out_csv = output_dir / "watkins_test_results.csv"
    pd.DataFrame(summary_rows, columns=[
        "rank", "n_classes", "n_pass", "pass_rate", "avg_v2", "max_v2"
    ]).to_csv(out_csv, index=False)
    print(f"  Summary CSV: {out_csv}")

    if failures_to_save:
        fail_csv = output_dir / "watkins_failures.csv"
        all_failures = pd.concat(failures_to_save)
        keep_cols = [c for c in ["label", "conductor", "iso", "rank_int",
                                  "deg_phi", "v2_deg_phi", "torsion_order",
                                  "tamagawa_product", "sha_an"]
                     if c in all_failures.columns]
        all_failures[keep_cols].to_csv(fail_csv, index=False)
        print(f"  Failures CSV: {fail_csv}  ({len(all_failures):,} rows)")
    else:
        print(C.ok("  No failures found across the corpus."))


if __name__ == "__main__":
    main()
