#!/usr/bin/env python3
"""diagnose_cm_alignment.py - check whether is_cm got misaligned during
the add_allisog merge step.

Hypothesis: the merge operation in add_allisog.py (joining the class_size
groupby back onto the corpus) reordered rows, causing the is_cm column to
no longer match the ainvs in each row.

Test: for each is_cm=True curve, recompute CM-ness directly from the
current ainvs. Count mismatches. If all match, my hypothesis is wrong;
if many mismatch, alignment is broken.
"""

import math
import json
from pathlib import Path

import pandas as pd
import numpy as np


CORPUS = Path("./lmfdb_ec_out/ec_corpus_with_jinv.parquet")

# 13 CM j-invariants as strings (for quick set check)
CM_J_STR_SET = {"0", "1728", "-3375", "8000", "-32768", "54000", "287496",
                "-884736", "-12288000", "16581375", "-884736000",
                "-147197952000", "-262537412640768000"}


def normalize_ainvs(x):
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]
    if isinstance(x, str):
        s = x.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [int(v) for v in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        s = s.strip("[]() ").replace(" ", "")
        return [int(v) for v in s.split(",") if v]
    raise ValueError(f"Bad ainvs: {x!r}")


def main():
    print(f"Loading {CORPUS}...")
    df = pd.read_parquet(CORPUS)
    print(f"Rows: {len(df):,}")
    print()

    # Check 1: Show the first 10 is_cm=True curves with their j_str
    print("=" * 70)
    print("First 10 is_cm=True rows: do their j_str values look like CM?")
    print("=" * 70)
    cm_rows = df[df["is_cm"]].head(10)
    for _, row in cm_rows.iterrows():
        j = row["j_str"]
        ainvs = row["ainvs"]
        is_actual_cm = j in CM_J_STR_SET
        flag = "✓ CM j-value" if is_actual_cm else "✗ NOT CM j-value (BUG)"
        print(f"  N={int(row['conductor'])}, iso={row['iso']}, "
              f"ainvs={ainvs}, j={j[:40]}...  {flag}")
    print()

    # Check 2: For 1000 random is_cm rows, recompute j from ainvs and
    # compare to stored j_str.
    print("=" * 70)
    print("Recompute j from ainvs for 1000 random is_cm rows.")
    print("If alignment is correct, recomputed j matches stored j_str.")
    print("=" * 70)
    sample = df[df["is_cm"]].sample(min(1000, int(df["is_cm"].sum())),
                                     random_state=42)
    n_match = 0
    n_mismatch = 0
    n_actual_cm = 0
    for _, row in sample.iterrows():
        try:
            ainvs = normalize_ainvs(row["ainvs"])
            a1, a2, a3, a4, a6 = ainvs
            b2 = a1 * a1 + 4 * a2
            b4 = 2 * a4 + a1 * a3
            b6 = a3 * a3 + 4 * a6
            b8 = (a1 * a1 * a6 - a1 * a3 * a4
                  + 4 * a2 * a6 + a2 * a3 * a3 - a4 * a4)
            c4 = b2 * b2 - 24 * b4
            delta = (-b2 * b2 * b8 - 8 * b4 ** 3 - 27 * b6 ** 2
                     + 9 * b2 * b4 * b6)
            if delta == 0:
                continue
            j_num = c4 ** 3
            j_den = delta
            g = math.gcd(abs(j_num), abs(j_den))
            j_num //= g
            j_den //= g
            if j_den < 0:
                j_num = -j_num
                j_den = -j_den
            recomputed_j = str(j_num) if j_den == 1 else f"{j_num}/{j_den}"

            if recomputed_j == row["j_str"]:
                n_match += 1
            else:
                n_mismatch += 1
            if recomputed_j in CM_J_STR_SET:
                n_actual_cm += 1
        except Exception as e:
            print(f"  Error on row N={row['conductor']}, iso={row['iso']}: {e}")
    print(f"  Sampled is_cm=True rows: {len(sample)}")
    print(f"  j_str recomputed == stored: {n_match}")
    print(f"  j_str recomputed != stored: {n_mismatch}")
    print(f"  Of recomputed values, in CM set: {n_actual_cm}")
    print()
    if n_mismatch > 0:
        print("  -> j_str does not match recomputed value.")
        print("     Either ainvs was reordered after j_str was computed,")
        print("     or there's a parsing inconsistency.")
    else:
        print("  -> ainvs and j_str are aligned correctly.")
        print()
        if n_actual_cm < len(sample) * 0.9:
            print(f"  But only {n_actual_cm}/{len(sample)} sampled is_cm rows")
            print("  have actual CM j-invariants. The is_cm column itself")
            print("  is misaligned with the ainvs column.")

    # Check 3: Look at curves WITH CM j-string. Are they flagged is_cm?
    print("=" * 70)
    print("Curves with CM j-string: are they flagged is_cm=True?")
    print("=" * 70)
    has_cm_j = df["j_str"].isin(CM_J_STR_SET)
    n_cm_j = int(has_cm_j.sum())
    n_cm_j_and_is_cm = int((has_cm_j & df["is_cm"]).sum())
    print(f"  Curves with j_str in CM set: {n_cm_j:,}")
    print(f"  Of those, is_cm=True:        {n_cm_j_and_is_cm:,}")
    print(f"  Of those, is_cm=False:       {n_cm_j - n_cm_j_and_is_cm:,}")
    print()
    # Sample a few CM-j curves that are NOT flagged is_cm
    bad = df[has_cm_j & ~df["is_cm"]].head(5)
    print("  First 5 missed CM curves (CM j-string but is_cm=False):")
    for _, row in bad.iterrows():
        print(f"    N={int(row['conductor'])}, iso={row['iso']}, "
              f"ainvs={row['ainvs']}, j={row['j_str']}")


if __name__ == "__main__":
    main()
