#!/usr/bin/env python3
"""compute_canonical_heights.py - canonical Neron-Tate heights via PARI/GP.

For every rank->=2 curve in ec_corpus_with_gens.parquet, computes the
canonical height pairing matrix M_ij = <P_i, P_j>_NT using PARI's
ellheightmatrix(). This matrix is the Gram matrix of the Mordell-Weil
lattice (modulo torsion) under the canonical height pairing. Its
determinant is the regulator R_E - which we cross-check against the
regulator already in the corpus (from Cremona's allbsd) as validation
of the projective->affine conversion.

For each successfully computed curve we add:
    - gram_matrix              JSON: r x r symmetric matrix as nested list
    - regulator_pari           det(M) computed by PARI
    - nt_diag_min/max/mean     canonical height of individual generators
    - nt_offdiag_max_abs       largest |<P_i, P_j>|, i != j
    - gram_trace               sum of diagonal entries
    - eigenvalue_min/max       endpoints of the spectrum
    - hermite_quotient         (trace/r) / det^(1/r)   shape, >=1 (=1 orthogonal)
    - aspect_ratio             eigenvalue_max / eigenvalue_min
    - orthogonality_defect     (prod of diag) / det   shape, >=1 (=1 orthogonal)
    - ch_status                'ok' | 'missing' | 'shape_mismatch_*' | ...

Requirements:
    PARI/GP installed and 'gp' on PATH.
    Windows: install from https://pari.math.u-bordeaux.fr/download.html
    Test:    gp --version

Usage:
    python compute_canonical_heights.py --validate     # 10-curve cross-check
    python compute_canonical_heights.py --quick        # 100 curves smoke
    python compute_canonical_heights.py --rank-min 3   # rank-3+ only (~9.5K, ~2 min)
    python compute_canonical_heights.py                # all rank>=2 (~358K)

Author: Kase Branham - Independent Researcher
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    parallel_map, summary_table, Timer, C, debug,
)


# =====================================================================
#  CONFIGURATION
# =====================================================================

DEFAULT_CORPUS     = Path("./lmfdb_ec_out/ec_corpus_with_gens.parquet")
DEFAULT_OUTPUT_DIR = Path("./lmfdb_ec_out")
DEFAULT_GP_BIN     = "gp"
BATCH_SIZE         = 100
GP_TIMEOUT_SEC     = 180
PRECISION_DIGITS   = 19

SENTINEL_LABEL = "--LBL:"
SENTINEL_END   = "--END--"

# PARI sometimes prints scientific notation with a space: "1.234 E-15".
# Python's float() doesn't accept that; we strip the space before parsing.
PARI_SCI_RE = re.compile(r"(\d)\s+E([+-]?\d)")


# =====================================================================
#  PARI ENVIRONMENT
# =====================================================================

def check_gp_available(gp_bin):
    """Return PARI version string, or None if `gp` cannot be invoked."""
    for flag in ("--version-short", "--version"):
        try:
            r = subprocess.run(
                [gp_bin, flag],
                capture_output=True, text=True, timeout=10,
            )
            out = (r.stdout or r.stderr).strip().splitlines()
            if out:
                return out[0]
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None
    return None


# =====================================================================
#  GP SCRIPT GENERATION + OUTPUT PARSING
# =====================================================================

def build_batch_script(curves, precision):
    """Build a multi-curve GP script. Each curve emits:
        --LBL:<label>
        m11 m12 ... mrr     (row-major, space-separated)
    Then a final --END-- sentinel.

    curves = [(label, ainvs_str, gens), ...]
        ainvs_str: e.g. "[0,1,1,-2,0]" (already valid PARI list literal)
        gens:      list of [X_str, Y_str, Z_str] (arbitrary-precision)
    """
    lines = [
        f"default(realprecision, {precision});",
        "default(parisize, 200000000);",
    ]
    for label, ainvs_str, gens in curves:
        affine = ", ".join(f"[{X}/{Z}, {Y}/{Z}]" for X, Y, Z in gens)
        lines.append(f"E = ellinit({ainvs_str});")
        lines.append(f"PTS = [{affine}];")
        lines.append("M = ellheightmatrix(E, PTS);")
        lines.append(f'print("{SENTINEL_LABEL}{label}");')
        lines.append(
            "for(i=1, matsize(M)[1], "
            "for(j=1, matsize(M)[2], print1(M[i,j], \" \"))); print(\"\");"
        )
    lines.append(f'print("{SENTINEL_END}");')
    return "\n".join(lines)


def parse_gp_output(text):
    """Parse stdout into {label: [matrix entries in row-major]}."""
    results = {}
    current = None
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(SENTINEL_LABEL):
            current = line[len(SENTINEL_LABEL):]
            continue
        if line == SENTINEL_END:
            break
        if current is None:
            continue
        # Strip space inside scientific notation
        cleaned = PARI_SCI_RE.sub(r"\1E\2", line)
        try:
            entries = [float(tok) for tok in cleaned.split()]
            results[current] = entries
            current = None
        except ValueError:
            # Probably a continuation line for a very wide matrix; skip
            pass
    return results


# =====================================================================
#  WORKER
# =====================================================================

def features_from_matrix(M, r):
    """Compute the per-curve scalar features from a Gram matrix."""
    M = 0.5 * (M + M.T)  # numerical symmetrize
    diag = np.diag(M)
    offd = M - np.diag(diag)
    det = float(np.linalg.det(M))
    tr = float(np.trace(M))

    # Eigenvalues for shape invariants
    evals = np.linalg.eigvalsh(M)
    ev_min = float(evals.min())
    ev_max = float(evals.max())

    if det > 0:
        hermite_q = (tr / r) / (det ** (1.0 / r))
        orth_def  = float(np.prod(diag) / det)
    else:
        hermite_q = float("nan")
        orth_def  = float("nan")

    return {
        "ch_status":             "ok",
        "gram_matrix":           json.dumps(M.tolist()),
        "regulator_pari":        det,
        "nt_diag_min":           float(diag.min()),
        "nt_diag_max":           float(diag.max()),
        "nt_diag_mean":          float(diag.mean()),
        "nt_offdiag_max_abs":    float(np.abs(offd).max()) if r >= 2 else 0.0,
        "gram_trace":            tr,
        "eigenvalue_min":        ev_min,
        "eigenvalue_max":        ev_max,
        "hermite_quotient":      float(hermite_q),
        "aspect_ratio":          float(ev_max / ev_min) if ev_min > 0 else float("inf"),
        "orthogonality_defect":  float(orth_def),
    }


def worker(item):
    """Process one batch through a single GP invocation.

    The script is piped to GP on stdin; GP exits cleanly when it hits EOF.
    (Passing the script as a positional file leaves GP sitting at the
    interactive prompt, which is what caused the original timeout.)
    """
    n_in = len(item["curves"])
    try:
        debug("worker: batch=%d", n_in)
        script = build_batch_script(item["curves"], item["precision"])

        res = subprocess.run(
            [item["gp_bin"], "-q", "-f"],
            input=script,
            capture_output=True, text=True,
            timeout=GP_TIMEOUT_SEC,
            encoding="ascii", errors="replace",
        )

        if res.returncode != 0:
            return {"error": f"gp rc={res.returncode}: {res.stderr[:200]}",
                    "batch": n_in}

        parsed = parse_gp_output(res.stdout)

        rows = []
        for label, _ainvs, gens in item["curves"]:
            r = len(gens)
            entries = parsed.get(label)
            if entries is None:
                rows.append({"label": label, "ch_status": "missing"})
                continue
            if len(entries) != r * r:
                rows.append({"label": label,
                             "ch_status": f"shape_{len(entries)}vs{r*r}"})
                continue
            M = np.array(entries, dtype=float).reshape(r, r)
            feat = features_from_matrix(M, r)
            feat["label"] = label
            rows.append(feat)

        return {"rows": rows, "batch": n_in}

    except subprocess.TimeoutExpired:
        return {"error": "gp timeout", "batch": n_in}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "batch": n_in}


# =====================================================================
#  VALIDATION
# =====================================================================

def validate(corpus, gp_bin, precision, tmpdir):
    section("VALIDATE - PARI heights vs Cremona regulators")
    sample = corpus[
        (corpus["rank"] >= 2)
        & (corpus["regulator"].notna())
        & (corpus["generators_str"].notna())
    ].head(10)
    if sample.empty:
        print(C.fail("  no rank>=2 curves with generators in corpus"))
        return False

    curves = []
    for _, row in sample.iterrows():
        gens = [[c for c in g] for g in json.loads(row["generators_str"])]
        curves.append((row["label"], row["ainvs"], gens))

    result = worker({"curves": curves, "gp_bin": gp_bin,
                     "tmpdir": tmpdir, "precision": precision})
    if "error" in result:
        print(C.fail(f"  GP error: {result['error']}"))
        return False

    print(f"  {'label':<14} {'rank':>4} {'reg(Cremona)':>18} "
          f"{'reg(PARI)':>18} {'ratio':>12}  status")
    matches = 0
    for r_out, (_, src) in zip(result["rows"], sample.iterrows()):
        if r_out["ch_status"] != "ok":
            print(f"  {src['label']:<14} {int(src['rank']):>4} "
                  f"{src['regulator']:>18.12f}  {'---':>18} {'---':>12}  "
                  f"{r_out['ch_status']}")
            continue
        ratio = r_out["regulator_pari"] / src["regulator"] if src["regulator"] else float("nan")
        ok = abs(ratio - 1.0) < 1e-6
        if ok:
            matches += 1
        tag = C.ok("OK") if ok else C.fail("MISMATCH")
        print(f"  {src['label']:<14} {int(src['rank']):>4} "
              f"{src['regulator']:>18.12f} {r_out['regulator_pari']:>18.12f} "
              f"{ratio:>12.9f}  {tag}")
    print()
    if matches == len(result["rows"]):
        print(C.ok(f"  ALL {matches}/{len(result['rows'])} matched. "
                   "Projective convention OK."))
        return True
    print(C.fail(f"  {matches}/{len(result['rows'])} matched. "
                 "Possible convention mismatch."))
    return False


# =====================================================================
#  WORK LIST + REDUCE
# =====================================================================

def build_work_list(corpus, args):
    df = corpus[
        (corpus["rank"] >= args.rank_min)
        & (corpus["generators_str"].notna())
    ]
    print(f"  Eligible curves (rank >= {args.rank_min}): {len(df):,}")

    # NOTE: we deliberately do NOT honor args.n_samples here. The standard
    # preset sets it to 500, which silently truncates full-corpus runs.
    # Use --quick for smoke tests, --rank-min for shape-based filtering.
    if args.preset == "quick":
        df = df.head(100)
        print(f"  --quick: keeping {len(df)}")

    curves = []
    bad_parse = 0
    for _, row in df.iterrows():
        try:
            gens = json.loads(row["generators_str"])
            if not gens:
                continue
        except Exception:
            bad_parse += 1
            continue
        curves.append((row["label"], row["ainvs"], gens))

    if bad_parse:
        print(C.warn(f"  {bad_parse} rows skipped (bad generators_str)"))
    print(f"  Curves to process: {len(curves):,}")

    batches = []
    bs = args.batch_size
    for i in range(0, len(curves), bs):
        batches.append({
            "curves":    curves[i : i + bs],
            "gp_bin":    args.gp,
            "tmpdir":    args.tmpdir,
            "precision": args.precision,
        })
    print(f"  Batches: {len(batches)} (size {bs})")
    return batches


def analyze(results, corpus, output_dir):
    section("REDUCE")
    errors = [r for r in results if "error" in r]
    good   = [r for r in results if "error" not in r]
    if errors:
        print(C.fail(f"  {len(errors)} batches failed:"))
        for e in errors[:5]:
            print(f"    {e}")

    rows = []
    for r in good:
        rows.extend(r["rows"])
    if not rows:
        print(C.fail("  no successful rows; aborting"))
        return

    n_ok = sum(1 for r in rows if r.get("ch_status") == "ok")
    n_bad = len(rows) - n_ok
    print(f"  Successful: {C.ok(f'{n_ok:,}')}    Failed: {C.warn(f'{n_bad:,}')}")

    ch_df = pd.DataFrame(rows)

    section("MERGE + CROSS-CHECK")
    merged = corpus.merge(ch_df, on="label", how="left")
    ok_mask = merged["ch_status"] == "ok"
    print(f"  Curves with canonical heights: {int(ok_mask.sum()):,}")

    sub = merged[ok_mask & merged["regulator"].notna() & (merged["regulator"] > 0)]
    if len(sub):
        ratios = sub["regulator_pari"] / sub["regulator"]
        delta = (ratios - 1.0).abs()
        for thresh, label in [(1e-2, "1%"), (1e-4, "1e-4"), (1e-6, "1ppm")]:
            n = int((delta < thresh).sum())
            print(f"  PARI vs Cremona regulator agreement (< {label}): "
                  f"{n:,} / {len(sub):,}  ({100*n/len(sub):.3f}%)")
        worst = sub.loc[delta.idxmax()]
        print(f"  Worst single ratio: {worst['label']} -> {ratios.loc[worst.name]:.6e}")

    section("MORDELL-WEIL LATTICE SHAPE BY RANK")
    rows = []
    for rk in sorted(merged.loc[ok_mask, "rank"].unique()):
        sub = merged[(merged["rank"] == rk) & ok_mask]
        if sub.empty:
            continue
        rows.append((
            int(rk), int(len(sub)),
            float(sub["nt_diag_mean"].median()),
            float(sub["hermite_quotient"].median()),
            float(sub["aspect_ratio"].median()),
            float(sub["orthogonality_defect"].median()),
        ))
    summary_table(
        rows,
        ["rank", "count", "med_NT_h", "med_hermite_q",
         "med_aspect", "med_ortho_def"],
        title="MW lattice shape (medians by rank)",
        fmt=[">4d", ">10,d", ">12.4f", ">14.4f", ">12.4f", ">14.4f"],
    )

    section("OUTPUT FILES")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = output_dir / "ec_corpus_with_heights.parquet"
    debug("writing %s", out_parquet)
    merged.to_parquet(out_parquet, index=False)
    sz = out_parquet.stat().st_size / 1e6
    print(f"  {C.info('parquet:')} {out_parquet}  ({sz:.1f} MB)")

    sub_csv = output_dir / "ec_corpus_with_heights_rank3plus.csv"
    merged[merged["rank"] >= 3].to_csv(sub_csv, index=False)
    print(f"  {C.info('csv (rank>=3):')} {sub_csv}")


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
    p.add_argument("--gp", default=DEFAULT_GP_BIN,
                   help="path to gp executable (default: 'gp' on PATH)")
    p.add_argument("--tmpdir", default=None,
                   help="dir for temp .gp scripts (default: OS temp)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="curves per GP batch (default 100)")
    p.add_argument("--rank-min", type=int, default=2,
                   help="minimum rank to process (default 2)")
    p.add_argument("--precision", type=int, default=PRECISION_DIGITS,
                   help="PARI realprecision in digits (default 19)")
    p.add_argument("--validate", action="store_true",
                   help="cross-check PARI on 10 curves vs Cremona regulator, exit")
    args = p.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)

    banner("Canonical Heights via PARI/GP", {
        "Corpus":       str(args.corpus),
        "Output dir":   str(output_dir),
        "GP binary":    args.gp,
        "Batch size":   args.batch_size,
        "Workers":      args.workers,
        "Rank min":     args.rank_min,
        "Precision":    f"{args.precision} digits",
        "Validate":     args.validate,
        "Preset":       args.preset,
    })

    # ---- 1. Sanity --------------------------------------------------
    step(1, 4, "Checking environment...")
    if not args.corpus.exists():
        raise FileNotFoundError(f"{args.corpus} missing - run add_allgens.py first")
    version = check_gp_available(args.gp)
    if version is None:
        print(C.fail(f"  '{args.gp}' not found / not runnable"))
        print(C.fail("  Install PARI/GP: https://pari.math.u-bordeaux.fr/download.html"))
        print(C.fail("  After install, ensure 'gp' is on PATH (test: 'gp --version')."))
        sys.exit(1)
    print(f"  PARI/GP: {version}")
    debug("loading corpus: %s", args.corpus)
    corpus = pd.read_parquet(args.corpus)
    print(f"  Corpus loaded: {len(corpus):,} curves")

    if args.validate:
        validate(corpus, args.gp, args.precision, args.tmpdir)
        return

    # ---- 2. Work list ----------------------------------------------
    step(2, 4, "Building work list...")
    work_list = build_work_list(corpus, args)
    if not work_list:
        print(C.fail("  no work to do"))
        return

    # ---- 3. Compute ------------------------------------------------
    step(3, 4, f"Computing canonical heights ({len(work_list)} batches)...")
    with Timer("PARI"):
        results = parallel_map(worker, work_list, args.workers, desc="Batches")

    # ---- 4. Reduce -------------------------------------------------
    step(4, 4, "Reducing and saving...")
    analyze(results, corpus, output_dir)


if __name__ == "__main__":
    mp.freeze_support()
    main()
