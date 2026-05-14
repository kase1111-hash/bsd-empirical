#!/usr/bin/env python3
"""analyze_mw_spectrum.py - eigenvalue spectrum analysis of MW lattices.

For every rank>=2 curve with a canonical-height Gram matrix in
ec_corpus_with_heights.parquet, computes statistics of the spectrum
{lambda_1, ..., lambda_r}:

    - spec_lambda_min/max/mean/mid    smallest, largest, mean, median eigenvalue
    - spec_sum                        trace of Gram matrix
    - spec_gini                       Gini coefficient of the spectrum
    - spec_gini_norm                  Gini / (r-1)/r  -- rank-normalized [0,1]
    - spec_entropy                    Shannon entropy of normalized spectrum
    - spec_entropy_norm               entropy / log(r)  -- rank-normalized [0,1]
    - spec_participation              (sum lambda)^2 / (r * sum lambda^2)  in [1/r, 1]

Then aggregates by rank, reports percentile distributions, and highlights
the most-spherical and most-elongated specimens. Finishes with a deep dive
on the rank-4 singleton.

Vectorized via numpy.linalg.eigvalsh on (n_rank, r, r) arrays per rank
group. Runs in seconds; no parallel processing needed.

Input:  lmfdb_ec_out/ec_corpus_with_heights.parquet
Output: lmfdb_ec_out/ec_corpus_with_spectrum.parquet

Author: Kase Branham - Independent Researcher
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    summary_table, Timer, C, debug,
)


# =====================================================================
#  CONFIG
# =====================================================================

DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_heights.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")


# =====================================================================
#  SPECTRUM STATISTICS (vectorized)
# =====================================================================

def spectrum_stats(eigs):
    """eigs : (n, r) array of eigenvalues sorted ascending along axis 1.
    Returns dict of feature arrays of length n.

    Definitions:
      Gini       G = (2 * sum_i (i+1) * lambda_(i)) / (r * sum lambda) - (r+1)/r
                 ascending-sorted, 1-indexed, in [0, (r-1)/r].
      Gini_norm  G / ((r-1)/r)  -> [0, 1] independent of r.
      Entropy    H = -sum (lambda_i / S) * log(lambda_i / S),  S = sum lambda
      H_norm     H / log(r)     -> [0, 1] (uniform = 1).
      PR         (sum lambda)^2 / (r * sum lambda^2),  in [1/r, 1].
                 Equals 1 when all eigenvalues equal, 1/r when one dominates.
    """
    n, r = eigs.shape
    lam_sum = eigs.sum(axis=1)

    # Gini
    idx = np.arange(1, r + 1)
    gini = (2 * (idx * eigs).sum(axis=1) / (r * lam_sum)) - (r + 1) / r
    gini_max = (r - 1) / r if r > 1 else 1.0
    gini_norm = gini / gini_max if gini_max > 0 else gini

    # Shannon entropy of normalized spectrum
    p = eigs / lam_sum[:, None]
    p_safe = np.where(p > 0, p, 1.0)  # log(1) = 0; cells with p=0 contribute 0
    entropy = -np.sum(np.where(p > 0, p * np.log(p_safe), 0.0), axis=1)
    entropy_norm = entropy / np.log(r) if r > 1 else np.zeros_like(entropy)

    # Participation ratio (normalized to [1/r, 1])
    lam_sq_sum = (eigs ** 2).sum(axis=1)
    participation = (lam_sum ** 2) / (r * lam_sq_sum)

    return {
        "spec_lambda_min":   eigs[:, 0],
        "spec_lambda_max":   eigs[:, -1],
        "spec_lambda_mean":  eigs.mean(axis=1),
        "spec_lambda_mid":   np.median(eigs, axis=1) if r >= 3 else np.full(n, np.nan),
        "spec_sum":          lam_sum,
        "spec_gini":         gini,
        "spec_gini_norm":    gini_norm,
        "spec_entropy":      entropy,
        "spec_entropy_norm": entropy_norm,
        "spec_participation": participation,
    }


# =====================================================================
#  PER-RANK PROCESSING
# =====================================================================

def process_rank(df_rank, r):
    """Extract Gram matrices for one rank group; return stats DataFrame."""
    n = len(df_rank)
    print(f"  rank {r}: parsing {n:,} Gram matrices...")

    matrices = np.empty((n, r, r), dtype=float)
    bad_indices = []
    gram_strs = df_rank["gram_matrix"].values

    for i, s in enumerate(gram_strs):
        try:
            arr = np.array(json.loads(s), dtype=float)
            if arr.shape != (r, r):
                bad_indices.append(i)
                matrices[i] = np.eye(r)
                continue
            # Symmetrize numerically (Gram matrices ARE symmetric, but be safe)
            matrices[i] = 0.5 * (arr + arr.T)
        except Exception:
            bad_indices.append(i)
            matrices[i] = np.eye(r)

    if bad_indices:
        print(C.warn(f"    {len(bad_indices)} parse failures (identity placeholder)"))

    print(f"  rank {r}: batched eigendecomposition...")
    eigs = np.linalg.eigvalsh(matrices)  # ascending-sorted along axis 1

    print(f"  rank {r}: computing spectrum statistics...")
    stats = spectrum_stats(eigs)
    stats["label"] = df_rank["label"].values

    # Mark bad rows as NaN
    if bad_indices:
        for key, arr in stats.items():
            if key == "label":
                continue
            arr[bad_indices] = np.nan

    return pd.DataFrame(stats)


# =====================================================================
#  MAIN
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_standard_args(p)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)

    banner("MW Lattice Spectrum Analysis", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
        "Preset":     args.preset,
    })

    # ---- 1. Load ---------------------------------------------------
    step(1, 4, "Loading corpus...")
    debug("loading: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    print(f"  Loaded {len(corpus):,} rows")

    eligible = corpus[
        (corpus["ch_status"] == "ok")
        & corpus["gram_matrix"].notna()
    ]
    print(f"  Eligible (have Gram matrix): {len(eligible):,}")

    # ---- 2. Compute spectra ----------------------------------------
    step(2, 4, "Computing spectra by rank...")
    spectrum_dfs = []
    with Timer("Spectrum"):
        for r in sorted(eligible["rank"].unique()):
            df_rank = eligible[eligible["rank"] == r]
            if df_rank.empty:
                continue
            spectrum_dfs.append(process_rank(df_rank, int(r)))
    spectrum_df = pd.concat(spectrum_dfs, ignore_index=True)
    print(f"  Total curves with spectrum: {len(spectrum_df):,}")

    # ---- 3. Merge and analyze --------------------------------------
    step(3, 4, "Merging and analyzing...")
    merged = corpus.merge(spectrum_df, on="label", how="left")
    has_spec = merged["spec_gini"].notna()
    print(f"  Curves with spectrum data after merge: {int(has_spec.sum()):,}")

    # ── Medians by rank ──────────────────────────────────────────
    section("SPECTRUM MEDIANS BY RANK")
    rows = []
    for r in sorted(merged.loc[has_spec, "rank"].unique()):
        sub = merged[has_spec & (merged["rank"] == r)]
        if sub.empty:
            continue
        rows.append((
            int(r), int(len(sub)),
            float(sub["spec_gini"].median()),
            float(sub["spec_gini_norm"].median()),
            float(sub["spec_entropy_norm"].median()),
            float(sub["spec_participation"].median()),
        ))
    summary_table(
        rows,
        ["rank", "count", "med_gini", "med_gini_norm",
         "med_H_norm", "med_PR"],
        title="Spectrum medians",
        fmt=[">4d", ">10,d", ">10.4f", ">15.4f", ">13.4f", ">10.4f"],
    )

    # ── Percentiles of rank-normalized Gini ──────────────────────
    section("NORMALIZED GINI PERCENTILES BY RANK")
    pcts = [5, 25, 50, 75, 95]
    rows = []
    for r in sorted(merged.loc[has_spec, "rank"].unique()):
        sub = merged.loc[has_spec & (merged["rank"] == r), "spec_gini_norm"]
        if sub.empty:
            continue
        row = [int(r)] + [float(np.percentile(sub, p)) for p in pcts]
        rows.append(tuple(row))
    summary_table(
        rows,
        ["rank"] + [f"p{p}" for p in pcts],
        title="spec_gini_norm percentiles",
        fmt=[">4d"] + [">10.4f"] * len(pcts),
    )

    # ── Extreme shapes ───────────────────────────────────────────
    section("EXTREME SPECTRUM SHAPES")
    for r in sorted(merged.loc[has_spec, "rank"].unique()):
        sub = merged[has_spec & (merged["rank"] == r)]
        if len(sub) < 5:
            continue
        print(C.info(f"  --- rank {r} (n={len(sub):,}) ---"))

        most_round = sub.nsmallest(3, "spec_gini_norm")
        most_elong = sub.nlargest(3, "spec_gini_norm")

        print("    Most spherical (lowest gini_norm):")
        for row in most_round.itertuples():
            print(f"      {row.label:<14}  gini_norm={row.spec_gini_norm:.5f}  "
                  f"λ=[{row.spec_lambda_min:.4f}..{row.spec_lambda_max:.4f}]  "
                  f"N={int(row.conductor)}")
        print("    Most elongated (highest gini_norm):")
        for row in most_elong.itertuples():
            print(f"      {row.label:<14}  gini_norm={row.spec_gini_norm:.5f}  "
                  f"λ=[{row.spec_lambda_min:.4f}..{row.spec_lambda_max:.4f}]  "
                  f"N={int(row.conductor)}")
        print()

    # ── Rank-4 deep dive ─────────────────────────────────────────
    r4 = merged[has_spec & (merged["rank"] == 4)]
    if len(r4) >= 1:
        section("RANK-4 SPECIMEN - DEEP DIVE")
        row = r4.iloc[0]
        print(f"  Label:         {row['label']}")
        print(f"  Conductor:     {int(row['conductor']):,}")
        print(f"  ainvs:         {row['ainvs']}")
        print(f"  Sha_an:        {row['sha_an']}")
        print(f"  Regulator:    {row['regulator']:.12f}")
        print()
        M = np.array(json.loads(row["gram_matrix"]))
        print("  Canonical height Gram matrix:")
        for line in M:
            print("    [ " + "  ".join(f"{x:>10.6f}" for x in line) + " ]")
        eigs = np.linalg.eigvalsh(M)
        print(f"\n  Eigenvalues:   {[round(float(e), 6) for e in eigs]}")
        print(f"  Spectrum sum:  {float(eigs.sum()):.6f}  (= trace)")
        print(f"  Spectrum prod: {float(np.prod(eigs)):.6f}  (= regulator)")
        print()
        print(f"  Gini:           {row['spec_gini']:.5f}")
        print(f"  Gini (norm):    {row['spec_gini_norm']:.5f}")
        print(f"  Entropy:        {row['spec_entropy']:.5f}   max={np.log(4):.5f}")
        print(f"  Entropy (norm): {row['spec_entropy_norm']:.5f}")
        print(f"  Participation:  {row['spec_participation']:.5f}   "
              f"(1.0 = uniform, 0.25 = dominated)")

    # ---- 4. Save ---------------------------------------------------
    step(4, 4, "Saving augmented corpus...")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = output_dir / "ec_corpus_with_spectrum.parquet"
    debug("writing %s", out_parquet)
    merged.to_parquet(out_parquet, index=False)
    sz = out_parquet.stat().st_size / 1e6
    print(f"  {C.info('parquet:')} {out_parquet}  ({sz:.1f} MB)")

    sub_csv = output_dir / "ec_corpus_with_spectrum_rank3plus.csv"
    merged[merged["rank"] >= 3].to_csv(sub_csv, index=False)
    print(f"  {C.info('csv (rank>=3):')} {sub_csv}")


if __name__ == "__main__":
    main()
