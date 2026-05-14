#!/usr/bin/env python3
"""investigate_177450dr.py - deep dive on the largest Sha=4 isogeny cluster.

In the rank-1 Sha=4 cohort (18,668 curves), the isogeny class 177450dr is
the most concentrated specimen: 6 curves, all carrying Sha=4. They are
2-isogenous within Q (with possible 5-isogenies depending on the graph),
forming a connected component of curves where the 2-part of Sha stays
constrained throughout.

This script pulls everything in conductor 177450, focuses on class dr,
shows:
    1. Conductor factorization (for arithmetic context)
    2. Per-curve summary table for all 6 specimens
    3. Detailed per-curve data including explicit generators
    4. BSD invariant check: L' should match Omega*R*|Sha|*c/|T|^2 for every curve
       AND L' should be the SAME across all 6 (isogeny invariance)
    5. Pairwise |Sha| consistency
    6. Other isogeny classes at N=177450 for context

Input:  lmfdb_ec_out/ec_corpus_with_spectrum.parquet
Output: lmfdb_ec_out/isogeny_class_177450dr.csv

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
TARGET_CONDUCTOR   = 177450
TARGET_CLASS       = "dr"


def factor(n):
    """Trial division. Returns dict {p: e}."""
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
        for p, e in sorted(factor(n).items())
    )


def fmt_int_or_q(x):
    if pd.isna(x):
        return "?"
    return str(int(x))


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(p)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--conductor", type=int, default=TARGET_CONDUCTOR)
    p.add_argument("--iso", default=TARGET_CLASS)
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    target = f"{args.conductor}{args.iso}"
    banner(f"Deep dive: isogeny class {target}", {
        "Corpus":     str(args.corpus),
        "Conductor":  args.conductor,
        "Class":      args.iso,
    })

    # ---- 1. Load --------------------------------------------------
    step(1, 3, "Loading and filtering...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)

    at_N = corpus[corpus["conductor"] == args.conductor].copy()
    cls = at_N[at_N["iso"] == args.iso].copy().sort_values("curve_number")
    print(f"  Curves at N={args.conductor}: {len(at_N)}")
    print(f"  Curves in class {target}: {len(cls)}")

    if len(cls) == 0:
        print(C.fail(f"  Class {target} not found; aborting."))
        return

    # ---- Conductor factorization ----------------------------------
    section("CONDUCTOR FACTORIZATION")
    print(f"  N = {args.conductor} = {fmt_factorization(args.conductor)}")
    f = factor(args.conductor)
    squares = [p for p, e in f.items() if e >= 2]
    if squares:
        print(C.info(f"  Square prime factors (p^2 | N): {squares}"))
        print("  (Multiple p^2 | N can give extra structure to local")
        print("   arithmetic and 2-Selmer / 2-descent computations.)")

    step(2, 3, "Inspecting...")

    # ---- 2. Per-curve summary -------------------------------------
    section(f"ALL {len(cls)} CURVES IN CLASS {target}")
    print(f"  {'label':<14} {'#':>3} {'rk':>3} {'|T|':>3} {'c':>4} "
          f"{'sha_an':>8} {'R':>11} {'Omega':>11} {'deg_phi':>8}")
    print("  " + "-" * 84)
    for row in cls.itertuples():
        print(f"  {row.label:<14} "
              f"{int(row.curve_number):>3} "
              f"{int(row.rank):>3} "
              f"{int(row.torsion_order):>3} "
              f"{int(row.tamagawa_product):>4} "
              f"{row.sha_an:>8.4f} "
              f"{row.regulator:>11.6f} "
              f"{row.real_period:>11.6f} "
              f"{fmt_int_or_q(row.deg_phi):>8}")

    # ---- 3. Detailed per-curve dump -------------------------------
    section("DETAILED PER-CURVE DATA")
    for row in cls.itertuples():
        print(C.info(f"  --- {row.label} (curve #{int(row.curve_number)} in class) ---"))
        print(f"    ainvs:         {row.ainvs}")
        print(f"    rank:          {int(row.rank)}")
        print(f"    torsion_order: {int(row.torsion_order)}")
        print(f"    tamagawa:      {int(row.tamagawa_product)}")
        print(f"    sha_an:        {row.sha_an:.6f}")
        print(f"    regulator:     {row.regulator:.10f}")
        print(f"    real_period:   {row.real_period:.10f}")
        print(f"    L'(E,1):       {row.L_value:.10f}")
        print(f"    deg_phi:       {fmt_int_or_q(row.deg_phi)}")
        if hasattr(row, "naive_h_max") and pd.notna(row.naive_h_max):
            print(f"    naive_h_max:   {row.naive_h_max:.6f}")
        if pd.notna(row.generators_str):
            try:
                gens = json.loads(row.generators_str)
                print(f"    generators ({len(gens)}):")
                for i, g in enumerate(gens, 1):
                    # Truncate huge coordinates for display
                    def short(s, n=80):
                        return s if len(s) <= n else s[:n-3] + "..."
                    print(f"      P{i} = [{short(str(g[0]))}")
                    print(f"           :{short(str(g[1]))}")
                    print(f"           :{short(str(g[2]))}]")
            except Exception as e:
                print(f"    (generator parse error: {e})")
        print()

    # ---- 4. BSD invariant check -----------------------------------
    section("BSD INVARIANT CHECK")
    print("  Within an isogeny class, L'(E,1) is the SAME for every member.")
    print("  BSD: L' = Omega * R * |Sha| * prod_c / |T|^2.")
    print("  All 6 should produce the same L' from their individual invariants.")
    print()
    rows = []
    for row in cls.itertuples():
        product = (row.real_period * row.regulator * row.sha_an *
                   row.tamagawa_product / (row.torsion_order ** 2))
        rows.append((
            row.label,
            float(row.L_value),
            float(row.real_period),
            float(row.regulator),
            float(row.sha_an),
            int(row.tamagawa_product),
            int(row.torsion_order),
            float(product),
            float(product - row.L_value),
        ))
    summary_table(
        rows,
        ["label", "L'(stored)", "Omega", "R", "Sha", "c", "|T|",
         "Omega*R*Sha*c/|T|^2", "diff"],
        title="BSD product across the class",
        fmt=["<12s", ">12.6f", ">10.6f", ">11.4f", ">6.2f",
             ">4d", ">4d", ">22.6f", ">12.2e"],
    )

    # L'-invariance across class
    L_values = cls["L_value"].values
    L_max_diff = float(L_values.max() - L_values.min())
    print(f"\n  L' max-min across class: {L_max_diff:.3e}")
    if L_max_diff < 1e-6:
        print(C.ok("  L' is constant across the class to machine precision (as expected)."))
    else:
        print(C.warn(f"  L' VARIES across the class -- unexpected!"))

    # ---- 5. Sha pattern in class ----------------------------------
    section("SHA ACROSS THE CLASS")
    sha_int_values = cls["sha_an"].round().astype(int).tolist()
    print(f"  |Sha|_an rounded: {sha_int_values}")
    if len(set(sha_int_values)) == 1:
        print(C.ok(f"  Constant |Sha| = {sha_int_values[0]} across all "
                   f"{len(cls)} curves -- the 2-Selmer / Sha structure is"))
        print(C.ok("  uniform across the isogeny graph."))
    else:
        print(C.info(f"  |Sha| varies: {sorted(set(sha_int_values))}"))

    # Per-curve regulator * Tamagawa / torsion^2 should give the same
    # value when divided by |Sha|: that's Omega·R*c/(|T|^2 * |Sha|) = L'/|Sha|
    print()
    print("  Reduced invariant: R * c / (|T|^2 * Sha) "
          "= L' / (Omega * Sha^2)  -- should be uniform")
    for row in cls.itertuples():
        reduced = (row.regulator * row.tamagawa_product /
                   (row.torsion_order ** 2 * row.sha_an))
        print(f"    {row.label:<14}  R*c/(T^2*Sha) = {reduced:.8f}")

    # ---- 6. Context: other classes at same N ----------------------
    section(f"OTHER ISOGENY CLASSES AT N = {args.conductor}")
    other = at_N[at_N["iso"] != args.iso]
    classes = sorted(other["iso"].unique())
    print(f"  {len(classes)} other classes at N={args.conductor}")
    print()

    # For each, compute summary: max rank, max Sha, count
    rows = []
    for cls_letter in classes:
        sub = other[other["iso"] == cls_letter]
        if len(sub) == 0:
            continue
        max_rk  = int(sub["rank"].max())
        max_sha = int(round(float(sub["sha_an"].max())))
        any_sha4 = int((sub["sha_an"].round() == 4).sum())
        rows.append((cls_letter, len(sub), max_rk, max_sha, any_sha4))
    rows.sort(key=lambda r: (-r[4], r[0]))  # sort by Sha=4 count desc

    summary_table(
        rows[:20],
        ["iso", "n", "max_rank", "max_sha", "Sha=4 count"],
        title=f"Other classes at N=177450 (top 20 by Sha=4 count)",
        fmt=["<6s", ">5d", ">9d", ">8d", ">12d"],
    )

    total_sha4_at_N = int((at_N["sha_an"].round() == 4).sum())
    print(f"\n  Total Sha=4 curves at N={args.conductor}: {total_sha4_at_N}")
    print(f"  Of which in class {args.iso}: {len(cls[cls['sha_an'].round() == 4])}")

    # ---- Save -----------------------------------------------------
    step(3, 3, "Saving...")
    out_csv = output_dir / f"isogeny_class_{target}.csv"
    save_cols = [
        "label", "conductor", "iso", "curve_number", "ainvs",
        "rank", "torsion_order", "tamagawa_product",
        "real_period", "L_value", "regulator", "sha_an",
        "deg_phi", "naive_h_max", "gens_coord_max_log10",
        "generators_str",
    ]
    save_cols = [c for c in save_cols if c in cls.columns]
    cls[save_cols].to_csv(out_csv, index=False)
    print(f"  CSV: {out_csv}  ({len(cls)} rows, {len(save_cols)} cols)")


if __name__ == "__main__":
    main()
