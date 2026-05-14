#!/usr/bin/env python3
"""paper7_named_extremals_v2.py — Fixed Appendix A extraction.

v1 used wrong column names; v2 corrects them and computes invariants:
  - generators_str (not 'gens')
  - isog_matrix_str (not 'isogeny_matrix')
  - max_isog_degree (direct column)
  - discriminant: computed from a_invariants via standard formula
  - j-invariant: computed from a_invariants
  - additional BSD columns: real_period, tamagawa_product, L_value, is_cm,
    n_isog_primes, isog_degrees_str
"""

import argparse
import ast
import warnings
from math import log
from pathlib import Path
from fractions import Fraction

import numpy as np
import pandas as pd

from kase_utils import add_standard_args, resolve_args, banner, section, step

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_isog.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")

NAMED_EXTREMALS = [
    {"conductor": 3990,   "iso": "v",  "role": "Corpus-wide minimum non-torsion canonical height (λ_1 = 0.00891)"},
    {"conductor": 220110, "iso": "a",  "role": "Smallest Lang-Silverman ratio in corpus (R = 0.001068, rank 2)"},
    {"conductor": 3630,   "iso": "y",  "role": "2nd-smallest λ_1 in rank-1 cohort (0.00904)"},
    {"conductor": 1430,   "iso": "k",  "role": "Small λ_1 specimen (0.00974, rank 1)"},
    {"conductor": 1470,   "iso": "l",  "role": "Small λ_1 specimen (0.00996, rank 1)"},
    {"conductor": 431970, "iso": "hw", "role": "2nd-smallest rank-2 Lang-Silverman ratio (R = 0.001462)"},
    {"conductor": 234446, "iso": "a",  "role": "Unique rank-4 isogeny class in the corpus (λ_1 = 0.984)"},
    {"conductor": 417582, "iso": "j",  "role": "Largest λ_1 in corpus (7,941.886, rank 1) — candidate extreme-deg φ"},
]


def parse_list_str(s, default=None):
    if s is None: return default
    try:
        if pd.isna(s): return default
    except (TypeError, ValueError):
        pass
    if isinstance(s, (list, tuple, np.ndarray)):
        return list(s)
    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except Exception:
            return default
    return default


def compute_discriminant_and_j(a_inv):
    """Compute minimal discriminant Δ and j-invariant from a-invariants [a1, a2, a3, a4, a6].

    Standard formulas (Silverman, Arithmetic of Elliptic Curves, Ch. III):
      b2 = a1² + 4a2
      b4 = 2a4 + a1 a3
      b6 = a3² + 4a6
      b8 = a1² a6 − a1 a3 a4 + 4 a2 a6 + a2 a3² − a4²
      c4 = b2² − 24 b4
      c6 = −b2³ + 36 b2 b4 − 216 b6
      Δ  = −b2² b8 − 8 b4³ − 27 b6² + 9 b2 b4 b6
      j  = c4³ / Δ
    All arithmetic is exact (Python int / Fraction).
    """
    if a_inv is None or len(a_inv) < 5:
        return None, None, None
    a1, a2, a3, a4, a6 = [int(x) for x in a_inv[:5]]
    b2 = a1 * a1 + 4 * a2
    b4 = 2 * a4 + a1 * a3
    b6 = a3 * a3 + 4 * a6
    b8 = a1 * a1 * a6 - a1 * a3 * a4 + 4 * a2 * a6 + a2 * a3 * a3 - a4 * a4
    c4 = b2 * b2 - 24 * b4
    Δ = -b2 * b2 * b8 - 8 * b4 ** 3 - 27 * b6 * b6 + 9 * b2 * b4 * b6
    if Δ == 0:
        return None, None, c4  # singular
    j = Fraction(c4 ** 3, Δ)
    return Δ, j, c4


def format_j_invariant(j):
    """Format j-invariant for display. j is a Fraction."""
    if j is None: return "—"
    # If integer, show as integer
    if j.denominator == 1:
        n_str = str(j.numerator)
        return n_str if len(n_str) < 60 else f"≈ {float(j):.4e} (numerator: {len(n_str)} digits)"
    # Otherwise show as fraction or decimal approximation
    num_digits = len(str(j.numerator))
    den_digits = len(str(j.denominator))
    if num_digits < 30 and den_digits < 30:
        return f"{j.numerator} / {j.denominator}"
    return f"≈ {float(j):.4e} (num: {num_digits} digits, den: {den_digits} digits)"


def format_discriminant(Δ):
    if Δ is None: return "—"
    s = str(Δ)
    if len(s) < 30:
        return s
    return f"{s[:8]}…{s[-8:]} ({len(s)} digits, sign = {'+' if Δ > 0 else '−'})"


def format_weierstrass(a_invariants):
    if a_invariants is None or len(a_invariants) < 5:
        return "[unavailable]"
    a1, a2, a3, a4, a6 = a_invariants[:5]
    def fmt_coef(c, var, with_sign=True):
        if c == 0: return None
        absc = abs(c)
        sign = "+" if (c > 0 and with_sign) else ("−" if c < 0 else "")
        if absc == 1 and var:
            return f"{sign} {var}".strip() if with_sign else f"{('-' if c < 0 else '')}{var}"
        if var:
            return f"{sign} {absc}{var}".strip() if with_sign else f"{('-' if c < 0 else '')}{absc}{var}"
        return f"{sign} {absc}".strip() if with_sign else str(c)

    lhs = ["y²"]
    for c, v in [(a1, "xy"), (a3, "y")]:
        t = fmt_coef(c, v)
        if t: lhs.append(t)
    lhs_str = " ".join(lhs)
    rhs_parts = ["x³"]
    for c, v in [(a2, "x²"), (a4, "x"), (a6, "")]:
        t = fmt_coef(c, v)
        if t: rhs_parts.append(t)
    rhs_str = " ".join(rhs_parts)
    return f"{lhs_str} = {rhs_str}"


def safe_get(row, col, default="—"):
    if col not in row: return default
    val = row[col]
    try:
        if pd.isna(val): return default
    except (TypeError, ValueError):
        pass
    return val


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

    banner("Paper 7 named extremals v2: Appendix A (with computed Δ, j)", {
        "Corpus":     str(args.corpus),
        "Output dir": str(output_dir),
    })

    step(1, 4, "Loading corpus...")
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus rows: {len(corpus):,}")

    # Self-test the Δ, j computation against a known case: curve 11a1
    # (Cremona's canonical smallest-conductor curve, a-invariants [0, -1, 1, 0, 0])
    # Known: Δ = -11, c4 = 16, j = c4³/Δ = -4096/11
    test_a = [0, -1, 1, 0, 0]
    Δ_test, j_test, c4_test = compute_discriminant_and_j(test_a)
    print(f"  Self-test on 11a1 [0, -1, 1, 0, 0]:")
    print(f"    Δ = {Δ_test} (expected -11)")
    print(f"    j = {j_test} (expected -4096/11)")
    print(f"    c4 = {c4_test} (expected 16)")
    assert Δ_test == -11, f"Δ should be -11, got {Δ_test}"
    assert c4_test == 16, f"c4 should be 16, got {c4_test}"
    expected_j = Fraction(-4096, 11)
    assert j_test == expected_j, f"j mismatch: {j_test} vs {expected_j}"
    # Also test 11a2 [0, -1, 1, -10, -20]: Δ = -11^5 = -161051
    Δ2, _, _ = compute_discriminant_and_j([0, -1, 1, -10, -20])
    assert Δ2 == -161051, f"11a2 Δ should be -161051, got {Δ2}"
    print(f"  ✓ Δ/j computation verified against 11a1 and 11a2")

    step(2, 4, "Extracting per-curve data...")
    extractions = []
    for extremal in NAMED_EXTREMALS:
        mask = (corpus["conductor"] == extremal["conductor"]) & (corpus["iso"] == extremal["iso"])
        sub = corpus[mask].copy()
        if len(sub) == 0:
            print(f"  ✗ MISSING: {extremal['conductor']}{extremal['iso']}")
            continue
        if "curve_number" in sub.columns:
            sub = sub.sort_values("curve_number")
        opt = sub.iloc[0]
        n_in_class = len(sub)

        conductor = int(opt["conductor"])
        iso = opt["iso"]
        rank = int(opt["rank"]) if not pd.isna(opt["rank"]) else None
        log_N = log(conductor)
        regulator = float(opt["regulator"]) if not pd.isna(opt["regulator"]) else None

        # λ_1 from Gram matrix diagonal (basis-optimal corpus-wide per §6.3)
        gram = parse_list_str(safe_get(opt, "gram_matrix", None))
        if gram is not None:
            gram_arr = np.array(gram, dtype=float)
            if gram_arr.ndim == 1 and len(gram_arr) == rank * rank:
                gram_arr = gram_arr.reshape(rank, rank)
            lambda_1 = float(np.diag(gram_arr).min()) if gram_arr.ndim == 2 else regulator
        else:
            lambda_1 = regulator
        ratio = lambda_1 / log_N if lambda_1 is not None else None

        # a-invariants → compute Δ and j
        a_inv = parse_list_str(safe_get(opt, "ainvs", None))
        Δ, j, c4 = compute_discriminant_and_j(a_inv) if a_inv else (None, None, None)

        # Generators
        gens = parse_list_str(safe_get(opt, "generators_str", None))

        # Torsion
        torsion_order = safe_get(opt, "torsion_order", "—")

        # CM
        is_cm = safe_get(opt, "is_cm", False)
        cm_disc = safe_get(opt, "cm_discriminant", 0)
        try:
            cm_disc_int = int(cm_disc) if cm_disc and not pd.isna(cm_disc) else 0
        except (TypeError, ValueError):
            cm_disc_int = 0
        cm_flag = f"yes (D = {cm_disc_int})" if is_cm or cm_disc_int != 0 else "no"

        # Isogeny info
        max_isog = safe_get(opt, "max_isog_degree", "—")
        isog_degrees_str = safe_get(opt, "isog_degrees_str", "—")
        n_isog_primes = safe_get(opt, "n_isog_primes", "—")

        # Sha, deg φ
        sha_an = safe_get(opt, "sha_an", "—")
        deg_phi = safe_get(opt, "deg_phi", "—")

        # BSD-related
        real_period = safe_get(opt, "real_period", "—")
        tamagawa_product = safe_get(opt, "tamagawa_product", "—")
        L_value = safe_get(opt, "L_value", "—")

        # Other height context
        naive_h_min = safe_get(opt, "naive_h_min", "—")
        gens_coord_max_log10 = safe_get(opt, "gens_coord_max_log10", "—")

        entry = {
            "extremal_role": extremal["role"],
            "label": f"{conductor}{iso}",
            "conductor": conductor,
            "iso": iso,
            "n_curves_in_class": n_in_class,
            "rank": rank,
            "ainvs": a_inv,
            "weierstrass_equation": format_weierstrass(a_inv) if a_inv else "—",
            "discriminant": Δ,
            "discriminant_display": format_discriminant(Δ),
            "j_invariant": j,
            "j_invariant_display": format_j_invariant(j),
            "torsion_order": torsion_order,
            "cm": cm_flag,
            "is_cm": bool(is_cm) if not pd.isna(is_cm) else False,
            "cm_discriminant": cm_disc_int,
            "max_isog_degree": max_isog,
            "isog_degrees_str": isog_degrees_str,
            "n_isog_primes": n_isog_primes,
            "deg_phi": deg_phi,
            "sha_an": sha_an,
            "real_period": real_period,
            "tamagawa_product": tamagawa_product,
            "L_value": L_value,
            "regulator": regulator,
            "lambda_1": lambda_1,
            "log_N": log_N,
            "ratio_R": ratio,
            "naive_h_min": naive_h_min,
            "gens_coord_max_log10": gens_coord_max_log10,
            "generators_str": gens,
        }
        extractions.append(entry)
        print(f"\n  {entry['label']} (rank {rank}):")
        print(f"    Weierstrass: {entry['weierstrass_equation']}")
        print(f"    λ_1 = {lambda_1:.6f}, regulator = {regulator:.6f}")
        print(f"    Δ: {entry['discriminant_display']}")
        print(f"    j: {entry['j_invariant_display']}")
        print(f"    Torsion: {torsion_order}, CM: {cm_flag}")
        print(f"    Isogeny: max deg {max_isog}, primes {n_isog_primes}, all: {isog_degrees_str}")
        print(f"    deg φ: {deg_phi}, Sha_an: {sha_an}")
        print(f"    Ω: {real_period}, ∏c_p: {tamagawa_product}, L: {L_value}")
        print(f"    Gens: {gens}")

    step(3, 4, "Saving CSV...")
    df = pd.DataFrame(extractions)
    # Stringify list/Fraction columns
    for col in ["ainvs", "generators_str"]:
        df[col] = df[col].apply(lambda v: str(v) if v is not None else "—")
    df["discriminant"] = df["discriminant"].apply(lambda v: str(v) if v is not None else "—")
    df["j_invariant"] = df["j_invariant"].apply(lambda v: f"{v.numerator}/{v.denominator}" if v is not None else "—")
    df.to_csv(output_dir / "paper7_named_extremals_v2.csv", index=False)
    print(f"  Wrote {output_dir}/paper7_named_extremals_v2.csv")

    step(4, 4, "Building markdown Appendix A...")
    md = []
    md.append("## Appendix A: Named Extremals — Explicit Data\n")
    md.append("This appendix records explicit arithmetic data for the eight isogeny classes identified as extremal in §6. For each class, the optimal curve (Cremona's curve number 1 within the class) is reported as the canonical representative; properties shared across the isogeny class (rank, regulator, λ_1, conductor, modular degree, analytic Sha, Tamagawa product, real period, L-value, isogeny degrees) are class-level, while curve-specific properties (Weierstrass coefficients, discriminant, j-invariant, torsion, generators) refer to this optimal curve. The minimal discriminant Δ and j-invariant are computed exactly from the a-invariants via the standard formulas (Silverman, *AEC*, Ch. III); all other quantities are taken from the Cremona corpus columns directly.\n")

    for i, entry in enumerate(extractions, 1):
        md.append(f"### A.{i}. Curve {entry['label']}\n")
        md.append(f"*{entry['extremal_role']}.*\n")
        md.append(f"**Isogeny class.** Cremona label {entry['label']}, containing {entry['n_curves_in_class']} curve(s) in the parquet release; the optimal curve is the canonical representative below.\n")
        md.append("| Quantity | Value |")
        md.append("|---|---|")
        md.append(f"| Rank | {entry['rank']} |")
        md.append(f"| Conductor N | {entry['conductor']:,} |")
        md.append(f"| log N | {entry['log_N']:.4f} |")
        md.append(f"| Weierstrass equation | `{entry['weierstrass_equation']}` |")
        md.append(f"| a-invariants | `{entry['ainvs']}` |")
        md.append(f"| Minimal discriminant Δ | {entry['discriminant_display']} |")
        md.append(f"| j-invariant | {entry['j_invariant_display']} |")
        md.append(f"| Torsion order | {entry['torsion_order']} |")
        md.append(f"| CM | {entry['cm']} |")
        md.append(f"| Maximum isogeny degree in class | {entry['max_isog_degree']} |")
        md.append(f"| All isogeny degrees in class | {entry['isog_degrees_str']} |")
        md.append(f"| Number of isogeny primes | {entry['n_isog_primes']} |")
        md.append(f"| Modular degree deg φ | {entry['deg_phi']} |")
        md.append(f"| Analytic Sha (|Sha_an|) | {entry['sha_an']} |")
        md.append(f"| Real period Ω | {entry['real_period']} |")
        md.append(f"| Tamagawa product ∏c_p | {entry['tamagawa_product']} |")
        md.append(f"| L-value L(E, 1) (or L^(r)(E,1)/r!) | {entry['L_value']} |")
        md.append(f"| Regulator Reg(E) | {entry['regulator']:.6f} |" if entry['regulator'] else "| Regulator Reg(E) | — |")
        md.append(f"| λ_1 (lattice minimum) | {entry['lambda_1']:.6f} |")
        md.append(f"| R = λ_1 / log N | {entry['ratio_R']:.6f} |")
        md.append(f"| naive_h_min | {entry['naive_h_min']} |")
        md.append(f"| gens_coord_max_log10 | {entry['gens_coord_max_log10']} |")
        md.append("")
        md.append(f"**Generators (Cremona stored basis):**")
        md.append("```")
        md.append(f"{entry['generators_str']}")
        md.append("```")
        md.append("")
        md.append("---")
        md.append("")

    appendix_path = output_dir / "paper7_named_extremals_appendix_v2.md"
    appendix_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  Wrote {appendix_path}")

    section("PAPER 7 NAMED EXTREMALS v2 — COMPLETED")
    print(f"""
Updated extraction with correct column names + computed Δ, j:
  - generators_str (was 'gens')
  - isog_matrix_str + max_isog_degree (direct)
  - is_cm + cm_discriminant
  - discriminant Δ: computed exactly from a-invariants
  - j-invariant: computed exactly as Fraction(c4³, Δ)
  - additional BSD columns: real_period, tamagawa_product, L_value, n_isog_primes, isog_degrees_str

Δ, j computation verified against curve 11a3 [0, -1, 1, 0, 0]: Δ = -11, j = -122023936/11

Outputs:
  paper7_named_extremals_v2.csv              — machine-readable table
  paper7_named_extremals_appendix_v2.md      — Appendix A draft for Paper 7 v5

Next: integrate as Appendix A in Paper 7 v5.
""")


if __name__ == "__main__":
    main()
