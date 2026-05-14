#!/usr/bin/env python3
"""diagnose_aps_coverage.py - Check whether a_p coverage gap is
conductor-skewed or uniform.

After add_aplist.py reported only 46.85% of curves had a_p sequences
attached, we need to know WHY. Three hypotheses:

  A. Coverage drops with conductor (aplist not computed at the frontier).
     If this is the case, we should compute the missing a_p ourselves
     via PARI's ellap.

  B. Coverage is uniformly ~47% across all conductors.
     If this is the case, the aplist files systematically exclude some
     classes; we'd need a different strategy.

  C. Coverage is bimodal: full at some conductor regions, zero at others.
     Some other systematic effect (e.g., aplist files missing).

This script prints coverage by conductor bin, by rank, by torsion order,
and by isogeny class size to triangulate.

Author: Kase Branham - Independent Researcher
"""

from pathlib import Path
import pandas as pd
import numpy as np

CORPUS = Path("./lmfdb_ec_out/ec_corpus_with_aps.parquet")


def pct_table(df, group_col, title):
    """Print a coverage table grouped by group_col."""
    g = (df.groupby(group_col, observed=False)
           .agg(n=('has_aps', 'count'), n_with=('has_aps', 'sum')))
    g['pct'] = 100.0 * g['n_with'] / g['n']
    print(f"\n  {title}")
    print(f"  {'─' * 70}")
    print(f"  {'group':<24s} {'n curves':>14s} {'with a_p':>14s} {'pct':>10s}")
    for idx, row in g.iterrows():
        label = str(idx)
        print(f"  {label:<24s} {int(row['n']):>14,d} "
              f"{int(row['n_with']):>14,d} {row['pct']:>9.2f}%")


def main():
    print("Loading corpus...")
    df = pd.read_parquet(CORPUS)
    df['has_aps'] = df['aps_list'].notna()
    print(f"Total curves: {len(df):,}")
    print(f"With a_p:     {df['has_aps'].sum():,}  "
          f"({100.0 * df['has_aps'].mean():.2f}%)")

    # By conductor bin
    df['cond_bin'] = pd.cut(
        df['conductor'],
        bins=[0, 1000, 10000, 50000, 100000, 200000, 300000, 400000, 500000],
    )
    pct_table(df, 'cond_bin', "Coverage by conductor bin")

    # By rank
    pct_table(df, 'rank', "Coverage by rank")

    # By torsion order
    pct_table(df, 'torsion_order', "Coverage by torsion order")

    # By isogeny class size (count curves per (N, iso))
    class_sizes = df.groupby(['conductor', 'iso']).size().rename('class_size')
    df = df.merge(class_sizes, on=['conductor', 'iso'], how='left')
    pct_table(df, 'class_size', "Coverage by isogeny class size")

    # Look at the highest conductors specifically
    print(f"\n  {'─' * 70}")
    print(f"  Sample of curves WITH a_p (highest conductors):")
    with_aps = df[df['has_aps']].sort_values('conductor', ascending=False).head(5)
    for _, row in with_aps.iterrows():
        print(f"    conductor={int(row['conductor'])}, iso={row['iso']}")

    print(f"\n  Sample of curves WITHOUT a_p (lowest conductors):")
    without = df[~df['has_aps']].sort_values('conductor').head(10)
    for _, row in without.iterrows():
        print(f"    conductor={int(row['conductor'])}, iso={row['iso']}")


if __name__ == "__main__":
    main()
