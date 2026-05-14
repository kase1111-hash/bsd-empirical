#!/usr/bin/env python3
"""add_allgens.py - augment ec_corpus.parquet with Mordell-Weil generators.

Parses Cremona's allgens/ files: per curve, gives the rational generators
of E(Q) / torsion as projective triples [X:Y:Z], plus the torsion generators.

This script:
    1. Loads the existing ec_corpus.parquet (from pull_lmfdb_ec.py).
    2. Parses all allgens.* files in parallel (one work item per range).
    3. For each curve, computes naive-height proxies and aggregate stats.
    4. Merges onto the corpus and writes ec_corpus_with_gens.parquet.

CAVEAT - naive vs canonical heights:
    The "naive height" stored here is  log_10 max(|X|, Z^2)  computed from the
    projective coordinates. This is a CRUDE proxy for the canonical Neron-Tate
    height; it agrees up to O(1) but is NOT the true MW-lattice metric. For
    genuine lattice geometry (Gram matrices, regulator reconstruction, shape
    invariants) the next stage is PARI/GP or Sage computing canonical heights
    per curve. This script gets you the generators and a fast first look.

Schema notes (verified by --validate against ecdata master 2026-05):
    allgens: <N> <iso> <num> <ainvs> <rank> <torsion_struct>
             <P_1> ... <P_r> <T_1> ... <T_t>
    Points are [X:Y:Z] in projective coordinates. Non-torsion generators
    come first, torsion last.

Usage:
    python add_allgens.py --validate              # confirm format on first range
    python add_allgens.py --quick                 # first 2 ranges, ~10 sec
    python add_allgens.py                         # full corpus
    python add_allgens.py --corpus PATH           # path to existing Parquet

Author: Kase Branham - Independent Researcher
"""

import argparse
import json
import math
import multiprocessing as mp
import re
from pathlib import Path

import pandas as pd

from kase_utils import (
    add_standard_args, resolve_args, banner, section, step,
    parallel_map, summary_table, Timer, C, debug,
)


# =====================================================================
#  CONFIGURATION
# =====================================================================

DEFAULT_DATA_DIR    = Path("./ecdata")
DEFAULT_CORPUS_PATH = Path("./lmfdb_ec_out/ec_corpus.parquet")
DEFAULT_OUTPUT_DIR  = Path("./lmfdb_ec_out")

SUBDIR    = "allgens"
RANGE_RE  = re.compile(r"^allgens\.(\d+)-(\d+)$")
LOG10_OF_2 = math.log10(2.0)


# =====================================================================
#  PARSING
# =====================================================================

def parse_point(s: str):
    """Parse '[X:Y:Z]' -> [X, Y, Z] using arbitrary-precision Python ints."""
    s = s.strip().strip("[]")
    return [int(p) for p in s.split(":")]


def parse_allgens(line):
    """allgens line: N iso num ainvs rank tors_struct P_1 ... P_r T_1 ... T_t

    Whitespace-separated; bracketed groups never contain whitespace internally.
    """
    p = line.split()
    N = int(p[0])
    iso = p[1]
    num = int(p[2])
    # p[3] = ainvs (already in corpus, skip)
    rank = int(p[4])
    # p[5] = torsion_structure: "[]", "[n]", or "[n,m]"
    tors_str = p[5].strip("[]")
    tors_structure = [int(x) for x in tors_str.split(",") if x] if tors_str else []
    t = len(tors_structure)

    point_tokens = p[6 : 6 + rank + t]
    points = [parse_point(tok) for tok in point_tokens]

    return {
        "conductor":        N,
        "iso":              iso,
        "curve_number":     num,
        "rank":             rank,
        "torsion_struct":   tors_structure,
        "generators":       points[:rank],
        "torsion_pts":      points[rank : rank + t],
    }


def parse_file(path):
    rows, skipped = [], 0
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(parse_allgens(line))
            except (ValueError, IndexError):
                skipped += 1
    return rows, skipped


# =====================================================================
#  FEATURE COMPUTATION
# =====================================================================

def log10_bigint(n: int) -> float:
    """log_10 |n| with arbitrary-precision fallback for huge ints."""
    n = abs(n)
    if n == 0:
        return 0.0
    if n < 10**300:
        return math.log10(n)
    return (n.bit_length() - 1) * LOG10_OF_2


def naive_log_height(point) -> float:
    """log_10 max(|X|, Z^2). Proxy for log H(x(P)).
    Correct up to an O(1) constant; NOT the canonical Neron-Tate height.
    """
    X, _Y, Z = point
    h = max(abs(X), Z * Z)
    return log10_bigint(h)


def compute_features(parsed):
    """Aggregate generator features per curve.
    For rank-0 curves we still emit a row (label + zeros / Nones) so the
    merge keeps every label.
    """
    label = f"{parsed['conductor']}{parsed['iso']}{parsed['curve_number']}"
    gens = parsed["generators"]

    if not gens:
        return {
            "label":                 label,
            "n_gens_parsed":         0,
            "generators_str":        "[]",
            "naive_heights":         "[]",
            "naive_h_min":           None,
            "naive_h_max":           None,
            "naive_h_mean":          None,
            "naive_h_ratio":         None,
            "gens_coord_max_log10":  None,
        }

    heights = [naive_log_height(g) for g in gens]
    coord_max = max(max(abs(X), abs(Y), abs(Z)) for X, Y, Z in gens)
    coord_max_log10 = log10_bigint(coord_max)

    # Ratio only meaningful for rank >= 2 with all heights positive.
    if len(heights) >= 2 and min(heights) > 0:
        ratio = max(heights) / min(heights)
    else:
        ratio = None

    # Stringify coordinates so arbitrary-precision ints survive Parquet.
    gens_str = json.dumps([[str(c) for c in g] for g in gens])

    return {
        "label":                 label,
        "n_gens_parsed":         len(gens),
        "generators_str":        gens_str,
        "naive_heights":         json.dumps([round(h, 4) for h in heights]),
        "naive_h_min":           float(min(heights)),
        "naive_h_max":           float(max(heights)),
        "naive_h_mean":          float(sum(heights) / len(heights)),
        "naive_h_ratio":         float(ratio) if ratio is not None else None,
        "gens_coord_max_log10":  float(coord_max_log10),
    }


# =====================================================================
#  WORKER
# =====================================================================

def worker(item):
    try:
        debug("worker: range=%s", item["range"])
        path = Path(item["data_dir"]) / SUBDIR / f"{SUBDIR}.{item['range']}"
        if not path.exists():
            return {"range": item["range"], "n": 0, "skipped": 0, "rows": []}
        parsed, skipped = parse_file(path)
        rows = [compute_features(p) for p in parsed]
        return {"range": item["range"], "n": len(rows),
                "skipped": skipped, "rows": rows}
    except Exception as e:
        return {"error": str(e), "range": item.get("range", "?")}


# =====================================================================
#  DISCOVERY / VALIDATION
# =====================================================================

def discover_ranges(data_dir: Path):
    sub = data_dir / SUBDIR
    if not sub.is_dir():
        raise FileNotFoundError(f"{sub} not found")
    ranges = []
    for p in sub.iterdir():
        m = RANGE_RE.match(p.name)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            ranges.append((lo, hi, p.name.split(".", 1)[1]))
    ranges.sort()
    return ranges


def validate(data_dir: Path):
    section("VALIDATE - schema sanity check")
    ranges = discover_ranges(data_dir)
    if not ranges:
        print(C.fail("  no allgens files found"))
        return
    _, _, rng = ranges[0]
    path = data_dir / SUBDIR / f"{SUBDIR}.{rng}"
    with open(path) as f:
        lines = [next(f, "").rstrip() for _ in range(6)]

    print(C.info(f"  {SUBDIR}.{rng} (first 6 lines):"))
    for ln in lines:
        if ln:
            print(f"    {ln}")
    print()

    print(C.info("  Parse attempts:"))
    for ln in lines:
        if not ln:
            continue
        try:
            p = parse_allgens(ln)
            lbl = f"{p['conductor']}{p['iso']}{p['curve_number']}"
            print(f"    {C.ok('OK:')} {lbl}  rank={p['rank']}  "
                  f"tors_struct={p['torsion_struct']}  "
                  f"n_gens={len(p['generators'])}  n_tors={len(p['torsion_pts'])}")
            if p["generators"]:
                X, Y, Z = p["generators"][0]
                h = naive_log_height([X, Y, Z])
                print(f"      first gen: [{X}:{Y}:{Z}]  log10 h_naive = {h:.3f}")
        except Exception as e:
            print(f"    {C.fail('FAIL:')} {e}")
            print(f"      line: {ln}")


# =====================================================================
#  WORK LIST / REDUCE
# =====================================================================

def build_work_list(args, data_dir):
    ranges = discover_ranges(data_dir)
    print(f"  Found {len(ranges)} allgens range files")
    items = []
    for lo, _hi, rng in ranges:
        if args.conductor_max is not None and lo > args.conductor_max:
            continue
        items.append({"range": rng, "data_dir": str(data_dir)})
    if args.preset == "quick":
        items = items[:2]
        print(f"  --quick: keeping first {len(items)} range(s)")
    elif args.n_samples and args.n_samples < len(items):
        items = items[:args.n_samples]
    return items


def analyze(results, corpus_path: Path, output_dir: Path):
    section("REDUCE")
    errors = [r for r in results if "error" in r]
    good   = [r for r in results if "error" not in r]
    if errors:
        print(C.fail(f"  {len(errors)} ranges failed:"))
        for e in errors[:5]:
            print(f"    {e}")

    skipped = sum(r.get("skipped", 0) for r in good)
    if skipped:
        print(C.warn(f"  {skipped} malformed lines skipped"))

    rows = []
    for r in good:
        rows.extend(r["rows"])
    if not rows:
        print(C.fail("  no rows; aborting"))
        return

    gens_df = pd.DataFrame(rows)
    print(f"  {C.ok(f'{len(gens_df):,}')} curves with generator data")

    # ---- Merge onto existing corpus --------------------------------
    section("MERGE")
    debug("loading corpus: %s", corpus_path)
    corpus = pd.read_parquet(corpus_path)
    print(f"  Loaded corpus: {len(corpus):,} rows")

    merged = corpus.merge(gens_df, on="label", how="left")
    matched = merged["n_gens_parsed"].notna().sum()
    print(f"  Merged: {len(merged):,} rows")
    print(f"  Generators attached: {matched:,} ({100*matched/len(merged):.1f}%)")

    # Sanity: rank should equal n_gens_parsed for matched curves.
    sane_mask = merged["n_gens_parsed"].notna()
    rank_check = (merged.loc[sane_mask, "rank"]
                  == merged.loc[sane_mask, "n_gens_parsed"]).sum()
    print(f"  rank == n_gens_parsed: {rank_check:,} / {int(sane_mask.sum()):,}")

    # ---- Naive-height stats by rank --------------------------------
    section("NAIVE LOG-HEIGHT BY RANK")
    rows = []
    for rk in sorted(merged["rank"].unique()):
        if rk == 0:
            continue
        sub = merged[(merged["rank"] == rk) & merged["naive_h_max"].notna()]
        if sub.empty:
            continue
        rows.append((
            int(rk),
            int(len(sub)),
            float(sub["naive_h_min"].min()),
            float(sub["naive_h_mean"].median()),
            float(sub["naive_h_max"].max()),
        ))
    summary_table(
        rows,
        ["rank", "count", "min(min_h)", "median(mean_h)", "max(max_h)"],
        title="Naive log-height summary",
        fmt=[">4d", ">10,d", ">11.3f", ">15.3f", ">12.3f"],
    )

    # Top-10 most-complex generators (largest coordinate magnitudes)
    section("LARGEST GENERATORS (log10 max coordinate)")
    top = merged.dropna(subset=["gens_coord_max_log10"]).nlargest(
        10, "gens_coord_max_log10")
    rows = [(r.label, int(r.rank), round(r.naive_h_max, 2),
             round(r.gens_coord_max_log10, 2))
            for r in top.itertuples()]
    summary_table(
        rows,
        ["label", "rank", "max_naive_h", "max_log10_coord"],
        title="Most arithmetically complex generators",
        fmt=["<14s", ">4d", ">12.2f", ">16.2f"],
    )

    # ---- Save ------------------------------------------------------
    section("OUTPUT FILES")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = output_dir / "ec_corpus_with_gens.parquet"
    debug("writing %s", out_parquet)
    merged.to_parquet(out_parquet, index=False)
    sz = out_parquet.stat().st_size / 1e6
    print(f"  {C.info('parquet:')} {out_parquet}  ({sz:.1f} MB)")

    # Skip CSV by default for the augmented corpus — generator strings are
    # large enough that the CSV bloats to ~1 GB. Save just rank>=2 subset
    # as a convenience for spreadsheet inspection.
    sub_csv = output_dir / "ec_corpus_with_gens_rank2plus.csv"
    debug("writing rank>=2 csv: %s", sub_csv)
    merged[merged["rank"] >= 2].to_csv(sub_csv, index=False)
    print(f"  {C.info('csv (rank>=2):')} {sub_csv}")


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_standard_args(parser)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH,
                        help="path to ec_corpus.parquet from pull_lmfdb_ec.py")
    parser.add_argument("--validate", action="store_true",
                        help="show first lines of allgens file and parse-check")
    parser.add_argument("--conductor-max", type=int, default=None)
    args = parser.parse_args()
    resolve_args(args)

    if args.output in (None, "results", "."):
        args.output = str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(args.output)

    banner("Augment EC Corpus with MW Generators", {
        "Data dir":      str(args.data_dir),
        "Corpus":        str(args.corpus),
        "Output dir":    str(output_dir),
        "Workers":       args.workers,
        "Validate only": args.validate,
        "Conductor max": args.conductor_max,
        "Preset":        args.preset,
    })

    step(1, 4, "Checking inputs...")
    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"{args.data_dir} missing - run pull_lmfdb_ec.py first")
    if not args.validate and not args.corpus.exists():
        raise FileNotFoundError(
            f"{args.corpus} missing - run pull_lmfdb_ec.py first")

    if args.validate:
        validate(args.data_dir)
        return

    step(2, 4, "Building work list...")
    work_list = build_work_list(args, args.data_dir)
    if not work_list:
        print(C.fail("  no ranges to process"))
        return

    step(3, 4, f"Parsing {len(work_list)} allgens ranges...")
    with Timer("Parse"):
        results = parallel_map(worker, work_list, args.workers, desc="Ranges")

    step(4, 4, "Reducing and saving...")
    analyze(results, args.corpus, output_dir)


if __name__ == "__main__":
    mp.freeze_support()
    main()
