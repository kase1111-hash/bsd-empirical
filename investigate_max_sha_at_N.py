#!/usr/bin/env python3
"""investigate_max_sha_at_N.py - deep dive on the max-Sha specimen at a conductor.

Default target: N = 165066, the conductor hosting the corpus-wide maximum
analytic Sha_an = 5625, i.e. |Sha| = 75. This is the most extreme Sha
specimen in the entire 3M-curve Cremona corpus.

Why this specimen is interesting:
    - |Sha| = 75 = 3 * 5^2 is the only product-of-distinct-primes Sha order
      among the corpus's top extremes (nearly all others are pure prime
      powers: 4, 9, 16, 25, 64, ...).
    - N = 165066 = 2 * 3 * 11 * 41 * 61 is SQUAREFREE with 5 distinct primes.
    - 5 is NOT a prime factor of N, so the curve has good reduction at 5
      and yet hosts non-trivial 5-part of Sha (5^2 | |Sha|).
    - This contradicts the naive intuition that Sha[p] requires structural
      involvement of p in the conductor.

Output:
    1. Conductor factorization
    2. |Sha| factorization with prime-by-prime check vs N's primes
    3. The specimen(s) at max-Sha (full per-curve invariants)
    4. The full isogeny class containing the specimen
    5. BSD self-consistency table across the class
    6. All non-trivial Sha curves at this conductor
    7. Per-class summary at this conductor

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet
Output: lmfdb_ec_out/max_sha_at_<N>.csv

Author: Kase Branham - Independent Researcher
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, C, debug,
)


DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_spectrum.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")


def factor_dict(n):
    """Trial-division factorization of n."""
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def fmt_factorization(n):
    return " * ".join(
        f"{p}^{e}" if e > 1 else str(p)
        for p, e in sorted(factor_dict(n).items())
    )


def shorten(s, n=80):
    s = str(s)
    return s if len(s) <= n else s[: n - 3] + "..."


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(p)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--conductor", type=int, default=165066)
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    banner(f"Max-Sha specimen at N = {args.conductor}", {
        "Corpus":     str(args.corpus),
        "Conductor":  args.conductor,
        "Output dir": str(output_dir),
    })

    # ---- 1. Load and filter ---------------------------------------
    step(1, 4, "Loading and filtering...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    at_N = corpus[corpus["conductor"] == args.conductor].copy()
    print(f"  Curves at N={args.conductor}: {len(at_N)}")

    if len(at_N) == 0:
        print(C.fail(f"  No curves at N={args.conductor}; aborting."))
        return

    at_N["sha_int"] = at_N["sha_an"].round().astype(int)

    # ---- Conductor factorization ----------------------------------
    section("CONDUCTOR FACTORIZATION")
    f_N = factor_dict(args.conductor)
    print(f"  N = {args.conductor} = {fmt_factorization(args.conductor)}")
    print(f"  omega(N) = {len(f_N)} distinct prime factors")
    print(f"  primes:   {sorted(f_N.keys())}")
    print(f"  squarefree: {'YES' if all(e == 1 for e in f_N.values()) else 'NO'}")

    # ---- 2. Identify max-Sha specimens ----------------------------
    step(2, 4, "Identifying max-Sha specimens...")
    max_sha = int(at_N["sha_int"].max())
    section(f"MAX SHA AT N = {args.conductor}")
    print(f"  Maximum Sha_an = {max_sha}")
    sq = int(round(np.sqrt(max_sha)))
    if sq * sq == max_sha:
        print(C.ok(f"  |Sha| = sqrt({max_sha}) = {sq}  "
                   f"(Cassels-Tate square root)"))
        print(C.info(f"  |Sha| factorization: {sq} = {fmt_factorization(sq)}"))
        sha_factors = factor_dict(sq)
        print()
        print("  Prime-by-prime check  (does p | N?):")
        for p_sha, e_sha in sorted(sha_factors.items()):
            divides_N = (args.conductor % p_sha == 0)
            if divides_N:
                print(f"    p = {p_sha:<3d} (exp {e_sha}):  "
                      f"{C.ok('YES, p | N')}")
            else:
                print(f"    p = {p_sha:<3d} (exp {e_sha}):  "
                      f"{C.warn('NO, p does NOT divide N')}  "
                      "(good reduction at p, yet non-trivial Sha[p])")
    else:
        print(C.warn(f"  {max_sha} is NOT a perfect square -- this would"
                     " violate Cassels-Tate."))

    # ---- The specimens themselves ---------------------------------
    section(f"SPECIMEN(S) WITH SHA = {max_sha}")
    specimens = at_N[at_N["sha_int"] == max_sha].sort_values("curve_number")
    print(f"  Number of curves at this Sha level: {len(specimens)}")
    print()
    for r in specimens.itertuples():
        print(C.info(f"  ============== {r.label} =============="))
        print(f"    ainvs:           {r.ainvs}")
        print(f"    iso class:       {args.conductor}{r.iso}")
        print(f"    rank:            {int(r.rank)}")
        print(f"    torsion_order:   {int(r.torsion_order)}")
        print(f"    tamagawa:        {int(r.tamagawa_product)}")
        print(f"    sha_an:          {r.sha_an:.4f}    (|Sha| = {sq})")
        print(f"    regulator:       {r.regulator:.10f}")
        print(f"    real_period:     {r.real_period:.10f}")
        print(f"    L^(r)(E,1)/r!:   {r.L_value:.10f}")
        if pd.notna(r.deg_phi):
            print(f"    deg_phi:         {int(r.deg_phi)}")
        if pd.notna(r.naive_h_max):
            print(f"    naive_h_max:     {r.naive_h_max:.6f}")
        if pd.notna(r.gens_coord_max_log10):
            print(f"    gen coord log10: {r.gens_coord_max_log10:.4f}")
        if pd.notna(r.generators_str):
            try:
                gens = json.loads(r.generators_str)
                if gens:
                    print(f"    generators ({len(gens)}):")
                    for g in gens:
                        print(f"      [X:Y:Z] =")
                        print(f"        X = {shorten(g[0])}")
                        print(f"        Y = {shorten(g[1])}")
                        print(f"        Z = {shorten(g[2])}")
            except Exception as e:
                print(f"    (gen parse error: {e})")
        print()

    # ---- 3. Isogeny class + non-trivial Sha at N ------------------
    step(3, 4, "Analyzing the isogeny class...")

    iso_letter = specimens.iloc[0]["iso"]
    cls = at_N[at_N["iso"] == iso_letter].copy().sort_values("curve_number")
    section(f"FULL ISOGENY CLASS {args.conductor}{iso_letter}  ({len(cls)} curves)")
    print(f"  {'label':<14} {'#':>3} {'rk':>3} {'|T|':>3} {'c':>5} "
          f"{'sha_an':>10} {'R':>10} {'Omega':>11}")
    print("  " + "-" * 75)
    for r in cls.itertuples():
        print(f"  {r.label:<14} "
              f"{int(r.curve_number):>3} "
              f"{int(r.rank):>3} "
              f"{int(r.torsion_order):>3} "
              f"{int(r.tamagawa_product):>5} "
              f"{r.sha_an:>10.4f} "
              f"{r.regulator:>10.4f} "
              f"{r.real_period:>11.6f}")

    print()
    print(f"  Sha values along class: {cls['sha_int'].tolist()}")
    print(f"  Rank values along class: {cls['rank'].astype(int).tolist()}")
    print(f"  Torsion values:          {cls['torsion_order'].astype(int).tolist()}")
    print(f"  Tamagawa values:         {cls['tamagawa_product'].astype(int).tolist()}")

    # ---- BSD self-consistency ------------------------------------
    section("BSD SELF-CONSISTENCY ACROSS THE CLASS")
    print("  BSD: L^(r)(E,1)/r! = Omega * R * |Sha|_an * prod(c_p) / |T|^2")
    print()
    rows = []
    for r in cls.itertuples():
        product = (r.real_period * r.regulator * r.sha_an *
                   r.tamagawa_product / (r.torsion_order ** 2))
        rows.append((
            r.label,
            float(r.L_value),
            float(product),
            float(product - r.L_value),
        ))
    summary_table(
        rows,
        ["label", "L (stored)", "Omega*R*Sha*c/T^2", "diff"],
        title=f"BSD product across class {args.conductor}{iso_letter}",
        fmt=["<14s", ">15.10f", ">22.10f", ">14.2e"],
    )

    L_max_diff = float(cls["L_value"].max() - cls["L_value"].min())
    print(f"\n  L max-min across class: {L_max_diff:.3e}")
    if L_max_diff < 1e-6:
        print(C.ok("  L is constant across the class (isogeny invariance OK)."))

    # ---- All non-trivial Sha curves at N --------------------------
    section(f"ALL NON-TRIVIAL SHA CURVES AT N = {args.conductor}")
    nontriv = at_N[at_N["sha_int"] >= 4].sort_values(
        ["sha_int", "iso", "curve_number"], ascending=[False, True, True])
    print(f"  Total curves with Sha_an >= 4: {len(nontriv)}")
    print()
    print(f"  {'label':<14} {'rank':>4} {'|T|':>3} {'c':>5} "
          f"{'R':>10} {'sha_an':>10}")
    print("  " + "-" * 60)
    for r in nontriv.itertuples():
        print(f"  {r.label:<14} "
              f"{int(r.rank):>4} {int(r.torsion_order):>3} "
              f"{int(r.tamagawa_product):>5} "
              f"{r.regulator:>10.4f} {r.sha_an:>10.4f}")

    # ---- Per-class summary at N -----------------------------------
    section(f"ALL ISOGENY CLASSES AT N = {args.conductor}")
    classes_summary = at_N.groupby("iso").agg(
        n_curves=("curve_number", "count"),
        max_sha=("sha_int", "max"),
        n_sha_nontriv=("sha_int", lambda s: int((s >= 4).sum())),
        ranks=("rank", lambda s: sorted(s.unique().tolist())),
    ).reset_index().sort_values("max_sha", ascending=False)

    rows = []
    for r in classes_summary.itertuples():
        rows.append((
            r.iso, int(r.n_curves), int(r.max_sha),
            int(r.n_sha_nontriv), str(r.ranks),
        ))
    summary_table(
        rows[:30],
        ["iso", "n_curves", "max_sha", "n_sha+", "ranks"],
        title=f"Classes at N={args.conductor} (top 30 by max_sha)",
        fmt=["<6s", ">10d", ">10d", ">8d", "<16s"],
    )

    # ---- 4. Save --------------------------------------------------
    step(4, 4, "Saving...")
    out_csv = output_dir / f"max_sha_at_{args.conductor}.csv"
    save_cols = ["label", "conductor", "iso", "curve_number", "ainvs",
                 "rank", "torsion_order", "tamagawa_product",
                 "real_period", "L_value", "regulator", "sha_an",
                 "deg_phi", "naive_h_max", "gens_coord_max_log10",
                 "generators_str"]
    save_cols = [c for c in save_cols if c in at_N.columns]
    at_N.sort_values("curve_number")[save_cols].to_csv(out_csv, index=False)
    print(f"  CSV: {out_csv}  ({len(at_N)} rows, {len(save_cols)} cols)")


if __name__ == "__main__":
    main()
