#!/usr/bin/env python3
"""
kase_utils.py — Shared utilities for all computation/test scripts.

Drop this in your Python path or next to your scripts.
Every script imports: from kase_utils import *

Author: Kase Branham — Independent Researcher
"""

import multiprocessing as mp
import os
import sys
import time
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from functools import partial

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ═══════════════════════════════════════════════════════════════════
#  COLOR OUTPUT (Windows CMD compatible)
# ═══════════════════════════════════════════════════════════════════

# Enable ANSI escape codes on Windows 10+ CMD
if sys.platform == 'win32':
    os.system('')  # triggers VT100 mode in CMD

class _Colors:
    """ANSI color codes. Degrades gracefully if redirected to file."""
    def __init__(self):
        self._enabled = sys.stdout.isatty()

    def _wrap(self, code, text):
        if self._enabled:
            return f"\033[{code}m{text}\033[0m"
        return text

    # Core colors
    def red(self, text):      return self._wrap('91', text)
    def green(self, text):    return self._wrap('92', text)
    def yellow(self, text):   return self._wrap('93', text)
    def blue(self, text):     return self._wrap('94', text)
    def magenta(self, text):  return self._wrap('95', text)
    def cyan(self, text):     return self._wrap('96', text)
    def white(self, text):    return self._wrap('97', text)
    def bold(self, text):     return self._wrap('1', text)
    def dim(self, text):      return self._wrap('2', text)

    # Semantic aliases — use these in scripts
    def ok(self, text):       return self.green(text)
    def warn(self, text):     return self.yellow(text)
    def fail(self, text):     return self.red(text)
    def info(self, text):     return self.cyan(text)
    def highlight(self, text): return self.bold(self.white(text))
    def muted(self, text):    return self.dim(text)

    # Result coloring by threshold
    def result(self, value, good_below=1.0, warn_below=10.0, fmt='.4f'):
        """Color a numeric result: green if good, yellow if marginal, red if bad."""
        text = f"{value:{fmt}}"
        if value < good_below:
            return self.green(text)
        elif value < warn_below:
            return self.yellow(text)
        else:
            return self.red(text)

    def pass_fail(self, passed):
        """Green PASS or red FAIL."""
        return self.green("PASS") if passed else self.red("FAIL")

C = _Colors()  # Global instance — usage: C.green("text"), C.ok("text")


# ═══════════════════════════════════════════════════════════════════
#  DEBUG LOGGING
# ═══════════════════════════════════════════════════════════════════

_DEBUG = False

def set_debug(enabled):
    """Enable/disable debug output globally."""
    global _DEBUG
    _DEBUG = enabled

def debug(msg, *args):
    """Print a debug message (only when --debug is active).
    States what the script is ABOUT TO DO, not what it did.
    
    Usage:
        debug("Computing overlap integrals for alpha=%s", alpha)
        debug("Loading checkpoint from %s", filepath)
        debug("Worker received item: %s", item)
    """
    if _DEBUG:
        if args:
            msg = msg % args
        print(C.muted(f"  [DEBUG] {msg}"))


# ═══════════════════════════════════════════════════════════════════
#  PRESETS: --quick vs --standard vs --long
# ═══════════════════════════════════════════════════════════════════

PRESETS = {
    'quick': {
        'restarts': 3,
        'popsize': 15,
        'maxiter': 200,
        'grid': 80,
        'n_samples': 50,
        'description': 'Fast sanity check (~2-5 min)',
    },
    'standard': {
        'restarts': 10,
        'popsize': 25,
        'maxiter': 500,
        'grid': 150,
        'n_samples': 500,
        'description': 'Publication-quality (~30-60 min)',
    },
    'long': {
        'restarts': 50,
        'popsize': 40,
        'maxiter': 2000,
        'grid': 300,
        'n_samples': 5000,
        'description': 'Last-ditch / high-precision (~hours)',
    },
}


def add_standard_args(parser, description=None):
    """Add the standard --quick/--long/--workers/--output/--seed args."""
    if description:
        parser.description = description

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_const', dest='preset', const='quick',
                      help=f"Fast sanity check: {PRESETS['quick']['description']}")
    mode.add_argument('--long', action='store_const', dest='preset', const='long',
                      help=f"High-precision: {PRESETS['long']['description']}")
    parser.set_defaults(preset='standard')

    parser.add_argument('--workers', type=int, default=-1,
                        help='Parallel workers (-1 = cpu_count - 1, 1 = serial)')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output (default: True)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug mode: print what each step is about to do')

    # Override individual preset values
    parser.add_argument('--restarts', type=int, default=None,
                        help='Override: DE restarts per point')
    parser.add_argument('--popsize', type=int, default=None,
                        help='Override: DE population size')
    parser.add_argument('--maxiter', type=int, default=None,
                        help='Override: DE max iterations per restart')
    parser.add_argument('--grid', type=int, default=None,
                        help='Override: spatial grid points')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Override: number of samples/trials')

    return parser


def resolve_args(args):
    """Merge preset defaults with any explicit overrides. Returns updated args."""
    preset = PRESETS[args.preset]

    # Apply preset, then override with anything explicitly set
    for key in ['restarts', 'popsize', 'maxiter', 'grid', 'n_samples']:
        arg_key = key.replace('-', '_')
        if getattr(args, arg_key, None) is None:
            setattr(args, arg_key, preset[key])

    # Resolve workers
    if args.workers == -1:
        args.workers = max(1, os.cpu_count() - 1)
    elif args.workers == 0:
        args.workers = os.cpu_count()

    # Set seed
    np.random.seed(args.seed)

    # Create output dir
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Enable debug if requested
    if getattr(args, 'debug', False):
        set_debug(True)
        debug("Debug mode enabled")
        debug("Preset: %s, Workers: %d, Seed: %d", args.preset, args.workers, args.seed)

    return args


# ═══════════════════════════════════════════════════════════════════
#  BANNER / LOGGING
# ═══════════════════════════════════════════════════════════════════

def banner(title, params=None, width=72):
    """Print a formatted banner with optional parameter table."""
    print()
    print(C.highlight("=" * width))
    print(C.highlight(f"  {title}"))
    print(C.highlight("=" * width))
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Host: {os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME', '?')}")
    print(f"  CPUs: {os.cpu_count()}")
    if params:
        print()
        max_key = max(len(str(k)) for k in params)
        for k, v in params.items():
            print(f"  {C.cyan(str(k) + ':'):>{max_key + 10}s} {v}")
    print(C.highlight("=" * width))
    print()
    debug("Banner printed, starting execution")


def section(title, width=72):
    """Print a section header."""
    print(f"\n{C.blue('─' * width)}")
    print(f"  {C.bold(title)}")
    print(f"{C.blue('─' * width)}\n")
    debug("Entering section: %s", title)


def step(n, total, msg):
    """Print a step marker: [3/7] Building work list..."""
    print(f"  {C.cyan(f'[{n}/{total}]')} {msg}")
    debug("Step %d/%d: %s", n, total, msg)


# ═══════════════════════════════════════════════════════════════════
#  PARALLEL MAP-REDUCE
# ═══════════════════════════════════════════════════════════════════

def parallel_map(worker_fn, work_list, n_workers, desc="Computing",
                 chunksize=None, callback=None):
    """
    Run worker_fn over work_list with n_workers processes.
    
    Returns list of results. Handles pool lifecycle correctly.
    Worker must be a top-level function (picklable).
    
    Args:
        worker_fn: Module-level function(item) -> result dict
        work_list: List of work items
        n_workers: Number of parallel workers (1 = serial)
        desc: Progress bar description
        chunksize: Items per chunk (None = auto)
        callback: Optional fn(result) called per completed item
    """
    if chunksize is None:
        # Auto: bigger chunks for small items, smaller for large
        chunksize = max(1, min(100, len(work_list) // (n_workers * 4)))

    results = []
    t0 = time.time()
    debug("parallel_map: %d items, %d workers, chunksize=%d", len(work_list), n_workers, chunksize)

    if n_workers <= 1:
        # Serial mode
        print(f"  Running serial (1 worker)...")
        debug("Serial mode — no pool, iterating work_list directly")
        iterator = work_list
        if HAS_TQDM:
            iterator = tqdm(work_list, desc=desc)
        for i, item in enumerate(iterator):
            result = worker_fn(item)
            results.append(result)
            if callback:
                callback(result)
            if not HAS_TQDM and (i + 1) % max(1, len(work_list) // 20) == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(work_list) - i - 1)
                print(f"    {i+1}/{len(work_list)} "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s left]")
    else:
        print(f"  Running {len(work_list)} jobs on {C.cyan(str(n_workers))} workers "
              f"(chunksize={chunksize})...")
        debug("Creating mp.Pool with %d workers", n_workers)
        pool = mp.Pool(n_workers)
        debug("Pool created, starting imap_unordered")
        try:
            if HAS_TQDM:
                for result in tqdm(
                    pool.imap_unordered(worker_fn, work_list, chunksize=chunksize),
                    total=len(work_list),
                    desc=desc
                ):
                    results.append(result)
                    if callback:
                        callback(result)
            else:
                for i, result in enumerate(
                    pool.imap_unordered(worker_fn, work_list, chunksize=chunksize)
                ):
                    results.append(result)
                    if callback:
                        callback(result)
                    if (i + 1) % max(1, len(work_list) // 20) == 0:
                        elapsed = time.time() - t0
                        eta = elapsed / (i + 1) * (len(work_list) - i - 1)
                        print(f"    {i+1}/{len(work_list)} "
                              f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s left]")
            pool.close()
            pool.join()
        except Exception:
            pool.terminate()
            pool.join()
            raise
        finally:
            del pool

    elapsed = time.time() - t0
    errors = [r for r in results if isinstance(r, dict) and 'error' in r]
    good = len(results) - len(errors)
    debug("parallel_map complete: %d good, %d errors, %.1fs", good, len(errors), elapsed)

    status = C.ok(f"✓ {good} results") if not errors else C.warn(f"✓ {good} results")
    err_str = C.fail(f", {len(errors)} errors") if errors else ""
    print(f"\n  {status}{err_str} in {elapsed:.1f}s "
          f"({len(results)/elapsed:.1f} items/sec)")

    return results


# ═══════════════════════════════════════════════════════════════════
#  RESULTS: SAVE / LOAD / CHECKPOINT
# ═══════════════════════════════════════════════════════════════════

def save_results(results, filepath, metadata=None, indent=2):
    """Save results to JSON with metadata header."""
    debug("Saving %d results to %s", len(results), filepath)
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'author': 'Kase Branham, Independent Researcher',
            'n_results': len(results),
            **(metadata or {}),
        },
        'results': results,
    }
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=indent, default=_json_default)
    print(f"  Saved: {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")
    return filepath


def save_csv(results, filepath, fieldnames=None):
    """Save list of dicts to CSV."""
    if not results:
        print(f"  No results to save.")
        return
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(results[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {filepath} ({len(results)} rows)")
    return filepath


def save_checkpoint(data, filepath):
    """Save checkpoint (overwrite). Atomic-ish via temp file."""
    debug("Saving checkpoint to %s", filepath)
    filepath = Path(filepath)
    tmp = filepath.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, default=_json_default)
    tmp.replace(filepath)


def load_checkpoint(filepath):
    """Load checkpoint if it exists, else return None."""
    filepath = Path(filepath)
    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
        print(f"  Resumed from checkpoint: {filepath}")
        return data
    return None


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


# ═══════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════

def summary_table(rows, headers, title=None, fmt=None, highlight_fn=None):
    """
    Print an aligned summary table.
    
    Args:
        rows: list of tuples/lists
        headers: list of column headers  
        title: optional table title
        fmt: list of format strings per column (e.g. ['>8s', '>10.4f', '>6d'])
             If None, auto-detect from first row.
        highlight_fn: optional fn(row) -> str suffix (e.g. ' ← BEST')
    """
    if not rows:
        print("  (no results)")
        return

    # Auto-format if not provided
    if fmt is None:
        fmt = []
        for val in rows[0]:
            if isinstance(val, float):
                fmt.append('>12.6f')
            elif isinstance(val, int):
                fmt.append('>8d')
            else:
                fmt.append('>12s')

    # Extract column widths from format specs
    import re
    widths = []
    for h, f in zip(headers, fmt):
        m = re.search(r'(\d+)', f)
        w = int(m.group(1)) if m else 8
        widths.append(max(len(h), w))

    if title:
        print(f"  {title}")
    header_str = "  ".join(f"{h:>{w}s}" for h, w in zip(headers, widths))
    print(f"  {header_str}")
    print(f"  {'─' * len(header_str)}")

    for row in rows:
        parts = []
        for val, f in zip(row, fmt):
            try:
                parts.append(f"{val:{f}}")
            except (ValueError, TypeError):
                parts.append(f"{str(val):>12s}")
        line = "  " + "  ".join(parts)
        if highlight_fn:
            suffix = highlight_fn(row)
            if suffix:
                line += suffix
        print(line)
    print()


# ═══════════════════════════════════════════════════════════════════
#  INTERPRETATION
# ═══════════════════════════════════════════════════════════════════

def interpret(metric_name, value, thresholds, messages):
    """
    Print an interpretation based on thresholds.
    
    Args:
        metric_name: e.g. "χ²/dof"
        value: the number
        thresholds: list of (threshold, message) in ascending order
        messages: final message if above all thresholds
    
    Example:
        interpret("χ²/dof", 0.43, [
            (1.0, "EXCELLENT fit — χ²/dof < 1"),
            (2.0, "GOOD fit — mild tensions"),
            (5.0, "FAIR — some observables off"),
        ], "POOR — significant tensions")
    """
    print(f"\n  {C.bold(metric_name)} = {C.highlight(str(value))}")
    for i, (threshold, msg) in enumerate(thresholds):
        if value < threshold:
            # First threshold = best, color green; middle = yellow
            color_fn = C.ok if i == 0 else C.warn
            print(f"  → {color_fn(msg)}")
            return
    print(f"  → {C.fail(messages)}")


# ═══════════════════════════════════════════════════════════════════
#  TIMER CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════

class Timer:
    """Context manager for timing blocks with automatic reporting."""
    
    def __init__(self, label=""):
        self.label = label
        self.elapsed = 0
    
    def __enter__(self):
        self.start = time.time()
        if self.label:
            print(f"  ⏱ {self.label}...")
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.label:
            if self.elapsed < 60:
                print(f"  ⏱ {self.label}: {self.elapsed:.1f}s")
            elif self.elapsed < 3600:
                print(f"  ⏱ {self.label}: {self.elapsed/60:.1f} min")
            else:
                print(f"  ⏱ {self.label}: {self.elapsed/3600:.1f} hr")
