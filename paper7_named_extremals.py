#!/usr/bin/env python3
"""paper7_named_extremals.py — Explicit data for Paper 7's named extremal curves.

For each of 8 isogeny classes identified as extremal in Paper 7 (small λ_1,
small Lang-Silverman ratio, largest λ_1, or unique rank), extract:
  - Weierstrass coefficients (a1, a2, a3, a4, a6)
  - Minimal discriminant Δ(E)
  - j-invariant
  - Torsion structure / order
  - Isogeny degrees (sizes of the isogeny class and the maximal cyclic isogeny)
  - CM discriminant (if CM)
  - Analytic Sha order
  - Generator coordinates (the actual rational points)
  - Regulator and ĥ_min = λ_1

Outputs:
  paper7_named_extremals.csv     — machine-readable table
  paper7_named_extremals_appendix.md — Appendix A in markdown for Paper 7 v5

The 8 named extremals (in order they appear in Paper 7 v4 §6 Tables 5–6):
  3990v       rank 1   λ_1 = 0.00891   corpus-wide minimum non-torsion canonical height
  220110a     rank 2   λ_1 = 0.01314   smallest Lang-Silverman ratio (0.001068)
  3630y       rank 1   λ_1 = 0.00904   2nd smallest λ_1 (rank 1)
  1430k       rank 1   λ_1 = 0.00974
  1470l       rank 1   λ_1 = 0.00996
  431970hw    rank 2   λ_1 = 0.01897   2nd smallest rank-2 ratio (0.001462)
  234446a     rank 4   λ_1 = 0.984     unique rank-4 isogeny class in corpus
  417582j     rank 1   λ_1 = 7941.886  largest λ_1 in corpus, candidate extreme-deg φ specimen
"""

import argparse
import ast
import warnings
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from kase_utils import add_standard_args, resolve_args, banner, section, step, Timer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

# The 8 named extremals (conductor, iso) with the role each plays in Paper 7
NAMED_EXTREMALS = [
    {"conductor": 3990, "iso": "v",
     "role": "Corpus-wide minimum non-torsion canonical height (λ_1 = 0.00891)"},
    {"conductor": 220110, "iso": "a",
     "role": "Smallest Lang-Silverman ratio in corpus (R = 0.001068, rank 2)"},
    {"conductor": 3630, "iso": "y",
     "role": "2nd-smallest λ_1 in rank-1 cohort (0.00904)"},
    {"conductor": 1430, "iso": "k",
     "role": "Small λ_1 specimen (0.00974, rank 1)"},
    {"conductor": 1470, "iso": "l",
     "role": "Small λ_1 specimen (0.00996, rank 1)"},
    {"conductor": 431970, "iso": "hw",
     "role": "2nd-smallest rank-2 Lang-Silverman ratio (R = 0.001462)"},
    {"conductor": 234446, "iso": "a",
     "role": "Unique rank-4 isogeny class in the corpus (λ_1 = 0.984)"},
    {"conductor": 417582, "iso": "j",
     "role": "Largest λ_1 in corpus (7,941.886, rank 1) — candidate extreme-deg φ"},
]


def safe_get(row, col, default="—"):
    """Return row[col] if present and non-null, else default."""
    if col not in row or pd.isna(row[col]):
        return default
    return row[col]


def parse_list_str(s, default=None):
    """Parse Python-list string representations like '[0, 1, 2]' safely."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return default
    if isinstance(s, (list, tuple, np.ndarray)):
        return list(s)
    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except Exception:
            return default
    return default


def format_weierstrass(a_invariants):
    """Format [a1, a2, a3, a4, a6] as the equation y² + a1 xy + a3 y = x³ + a2 x² + a4 x + a6."""
    if a_invariants is None or len(a_invariants) < 5:
        return "[unavailable]"
    a1, a2, a3, a4, a6 = a_invariants[:5]
    # Build LHS
    lhs_terms = ["y²"]
    if a1 != 0:
        lhs_terms.append(f"{'+' if a1 > 0 else '−'} {abs(a1)}xy" if a1 != 1 else "+ xy")
    if a3 != 0:
        lhs_terms.append(f"{'+' if a3 > 0 else '−'} {abs(a3)}y" if abs(a3) != 1 else f"{'+' if a3 > 0 else '−'} y")
    lhs = " ".join(lhs_terms)
    # Build RHS
    rhs_terms = ["x³"]
    if a2 != 0:
        rhs_terms.append(f"{'+' if a2 > 0 else '−'} {abs(a2)}x²" if abs(a2) != 1 else f"{'+' if a2 > 0 else '−'} x²")
    if a4 != 0:
        rhs_terms.append(f"{'+' if a4 > 0 else '−'} {abs(a4)}x" if abs(a4) != 1 else f"{'+' if a4 > 0 else '−'} x")
    if a6 != 0:
        rhs_terms.append(f"{'+' if a6 > 0 else '−'} {abs(a6)}")
    rhs = " ".join(rhs_terms)
    return f"{lhs} = {rhs}"


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

    banner("Paper 7 named extremals: Appendix A data", {
        "Corpus":        str(args.corpus),
        "Output dir":    str(output_dir),
        "Curves to pull": str(len(NAMED_EXTREMALS)),
    })

    # Load corpus
    step(1, 4, "Loading corpus...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")
    print(f"  Columns ({len(corpus.columns)}): {sorted(corpus.columns)}")

    # Filter to extremals
    step(2, 4, "Filtering to the 8 named extremals...")
    rows_collected = []
    for extremal in NAMED_EXTREMALS:
        mask = (corpus["conductor"] == extremal["conductor"]) & (corpus["iso"] == extremal["iso"])
        sub = corpus[mask].copy()
        if len(sub) == 0:
            print(f"  ✗ MISSING: {extremal['conductor']}{extremal['iso']}")
            continue
        # Sort by curve_number to get the optimal curve (curve_number = 1 conventionally) first
        if "curve_number" in sub.columns:
            sub = sub.sort_values("curve_number")
        rows_collected.append((extremal, sub))
        n_curves = len(sub)
        print(f"  ✓ {extremal['conductor']}{extremal['iso']}: {n_curves} curve(s) in class")

    # Build the extraction table
    step(3, 4, "Extracting per-curve data...")
    extractions = []
    for extremal, sub in rows_collected:
        # The class-level data is the same across curves; the curve-level data
        # differs. We'll record the optimal curve (curve_number=1) as the canonical
        # representative, plus note the full isogeny class size.
        opt = sub.iloc[0]
        n_in_class = len(sub)
        # All curves in class share these
        conductor = int(opt["conductor"])
        iso = opt["iso"]
        rank = int(opt["rank"]) if not pd.isna(opt["rank"]) else None
        log_N = log(conductor)
        regulator = float(opt["regulator"]) if "regulator" in opt and not pd.isna(opt["regulator"]) else None

        # The Gram matrix gives ĥ_min = λ_1 (verified basis-optimal in §6.3)
        gram_str = safe_get(opt, "gram_matrix", None)
        gram = parse_list_str(gram_str)
        if gram is not None:
            gram_arr = np.array(gram, dtype=float)
            if gram_arr.ndim == 1 and len(gram_arr) == rank * rank:
                gram_arr = gram_arr.reshape(rank, rank)
            lambda_1 = float(np.diag(gram_arr).min()) if gram_arr.ndim == 2 else regulator
        else:
            lambda_1 = regulator  # rank 1 case
        ratio = lambda_1 / log_N if lambda_1 is not None else None

        # Curve-level data (per the optimal curve)
        a_inv = parse_list_str(safe_get(opt, "ainvs", None) if "ainvs" in opt else
                               safe_get(opt, "a_invariants", None))
        discriminant = safe_get(opt, "disc", safe_get(opt, "discriminant", None))
        j_inv = safe_get(opt, "j_invariant", safe_get(opt, "jinv", None))
        torsion_str = safe_get(opt, "torsion_structure", None)
        torsion = parse_list_str(torsion_str)
        if torsion is None:
            torsion_order = safe_get(opt, "torsion_order", "—")
        else:
            torsion_order = (1 if not torsion else
                             torsion[0] if len(torsion) == 1 else
                             f"{torsion[0]}×{torsion[1]}")
        cm = safe_get(opt, "cm_discriminant", 0)
        cm_flag = "yes (D = " + str(int(cm)) + ")" if cm and cm != 0 and not pd.isna(cm) else "no"
        sha_an = safe_get(opt, "sha_an", "—")
        gens_str = safe_get(opt, "gens", None)
        gens = parse_list_str(gens_str)

        # Isogeny degree info
        isog_matrix_str = safe_get(opt, "isogeny_matrix", None)
        isog_matrix = parse_list_str(isog_matrix_str)
        if isog_matrix is not None:
            try:
                M = np.array(isog_matrix)
                # The isogeny degrees in the class are the entries of the isogeny matrix
                # The max non-trivial isogeny is the largest entry
                if M.ndim == 2:
                    max_isog = int(np.max(M)) if M.size > 0 else 1
                else:
                    max_isog = "—"
            except Exception:
                max_isog = "—"
        else:
            max_isog = "—"

        # naive_h_min, gens_coord_max_log10 for context
        naive_h_min = safe_get(opt, "naive_h_min", "—")
        gens_coord_max_log10 = safe_get(opt, "gens_coord_max_log10", "—")

        # Modular degree (if available)
        deg_phi = safe_get(opt, "modular_degree",
                  safe_get(opt, "deg_phi",
                  safe_get(opt, "degphi", "—")))

        # Anal sha
        analytic_rank = safe_get(opt, "analytic_rank", "—")

        entry = {
            "extremal_role": extremal["role"],
            "label": f"{conductor}{iso}",
            "conductor": conductor,
            "isogeny_class": iso,
            "n_curves_in_class": n_in_class,
            "rank": rank,
            "ainvs": a_inv,
            "weierstrass_equation": format_weierstrass(a_inv) if a_inv else "—",
            "discriminant": discriminant,
            "j_invariant": j_inv,
            "torsion_structure": torsion,
            "torsion_order": torsion_order,
            "cm": cm_flag,
            "modular_degree": deg_phi,
            "max_isogeny_degree": max_isog,
            "sha_an": sha_an,
            "regulator": regulator,
            "lambda_1": lambda_1,
            "log_N": log_N,
            "ratio_R": ratio,
            "naive_h_min": naive_h_min,
            "gens_coord_max_log10": gens_coord_max_log10,
            "gens": gens,
            "analytic_rank": analytic_rank,
        }
        extractions.append(entry)
        print(f"\n  {entry['label']} (rank {rank}, λ_1 = {lambda_1:.6f}):")
        print(f"    Weierstrass: {entry['weierstrass_equation']}")
        print(f"    Disc: {discriminant}, j: {j_inv}")
        print(f"    Torsion: {torsion_order}, CM: {cm_flag}")
        print(f"    Sha_an: {sha_an}, max isog deg: {max_isog}")
        print(f"    Modular degree: {deg_phi}")
        print(f"    Gens: {gens}")

    # Save CSV
    step(4, 4, "Saving CSV and building markdown appendix...")
    df = pd.DataFrame(extractions)
    # Stringify list columns for CSV
    for col in ["ainvs", "torsion_structure", "gens"]:
        df[col] = df[col].apply(lambda v: str(v) if v is not None else "—")
    df.to_csv(output_dir / "paper7_named_extremals.csv", index=False)
    print(f"  Wrote {output_dir}/paper7_named_extremals.csv ({len(df)} rows)")

    # Build markdown Appendix A
    md = []
    md.append("## Appendix A: Named Extremals — Explicit Data\n")
    md.append("This appendix records explicit arithmetic data for the eight isogeny classes identified as extremal in §6. For each class, the optimal curve (Cremona's curve number 1 within the class) is reported as the canonical representative; properties shared across the isogeny class (rank, regulator, λ_1, conductor, modular degree) are class-level, while curve-specific properties (Weierstrass coefficients, discriminant, j-invariant, torsion, generators) refer to this optimal curve.\n")
    for entry in extractions:
        md.append(f"### A.{extractions.index(entry)+1}. Curve {entry['label']} — {entry['extremal_role']}\n")
        md.append(f"**Isogeny class:** {entry['label']}, containing {entry['n_curves_in_class']} curve(s)")
        md.append(f"  ")
        md.append(f"**Rank:** {entry['rank']}")
        md.append(f"  ")
        md.append(f"**Conductor:** N = {entry['conductor']:,} (log N = {entry['log_N']:.4f})")
        md.append(f"  ")
        md.append(f"**λ_1 (Mordell-Weil lattice minimum):** {entry['lambda_1']:.6f}")
        md.append(f"  ")
        md.append(f"**Lang-Silverman ratio:** R = λ_1 / log(N) = {entry['ratio_R']:.6f}")
        md.append(f"  ")
        md.append(f"**Regulator:** {entry['regulator']:.6f}" if entry['regulator'] else "**Regulator:** —")
        md.append(f"  ")
        md.append(f"**Optimal-curve Weierstrass equation:** ")
        md.append(f"`{entry['weierstrass_equation']}`")
        md.append(f"  ")
        md.append(f"**a-invariants (Cremona convention):** `{entry['ainvs']}`")
        md.append(f"  ")
        md.append(f"**Minimal discriminant Δ:** `{entry['discriminant']}`")
        md.append(f"  ")
        md.append(f"**j-invariant:** `{entry['j_invariant']}`")
        md.append(f"  ")
        md.append(f"**Torsion structure:** {entry['torsion_order']}" + (f" (= {entry['torsion_structure']})" if entry['torsion_structure'] else ""))
        md.append(f"  ")
        md.append(f"**CM:** {entry['cm']}")
        md.append(f"  ")
        md.append(f"**Maximum isogeny degree in class:** {entry['max_isogeny_degree']}")
        md.append(f"  ")
        md.append(f"**Modular degree deg φ:** {entry['modular_degree']}")
        md.append(f"  ")
        md.append(f"**Analytic Sha (|Sha_an|):** {entry['sha_an']}")
        md.append(f"  ")
        md.append(f"**naive_h_min:** {entry['naive_h_min']}")
        md.append(f"  ")
        md.append(f"**gens_coord_max_log10:** {entry['gens_coord_max_log10']}")
        md.append(f"  ")
        md.append(f"**Mordell-Weil generators (stored basis, Cremona convention):**")
        md.append(f"`{entry['gens']}`")
        md.append("")
        md.append("---")
        md.append("")

    md_text = "\n".join(md)
    appendix_path = output_dir / "paper7_named_extremals_appendix.md"
    appendix_path.write_text(md_text, encoding="utf-8")
    print(f"  Wrote {appendix_path}")

    section("PAPER 7 NAMED EXTREMALS — SUMMARY")
    print(f"""
Extracted explicit arithmetic data for {len(extractions)} isogeny classes
identified as extremal in Paper 7 v4 §6 (Tables 5-6).

For each class:
  - Weierstrass equation of the optimal curve
  - Minimal discriminant, j-invariant
  - Torsion structure
  - Isogeny degrees (max in class)
  - CM discriminant (if CM)
  - Analytic Sha order
  - Modular degree deg φ
  - Mordell-Weil generators (stored basis)
  - λ_1, regulator, Lang-Silverman ratio

Outputs in {output_dir}:
  paper7_named_extremals.csv              — machine-readable table
  paper7_named_extremals_appendix.md      — Appendix A draft for Paper 7 v5

Next step: review the appendix output, then integrate as Appendix A
in Paper 7 v5 (after §8 and before References).
""")


if __name__ == "__main__":
    main()
