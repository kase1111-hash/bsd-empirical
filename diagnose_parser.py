#!/usr/bin/env python3
"""diagnose_parser.py - Run add_aplist's parser on ONE aplist file and
check whether expected (N, iso) keys are present in the result.

The grep_aplist.py output showed that 101a, 113a, etc. exist in the
aplist file, but the main add_aplist.py reports only 47% coverage —
meaning the parser is silently dropping records.

This script:
  1. Imports parse_aplist_file from add_aplist
  2. Runs it on the first aplist file
  3. Checks specific (N, iso) keys we know should be present
  4. Also counts total file lines vs parsed records to find the drop rate
  5. If a known-present record is missing from parsed result, hunt for
     it in the file and show what the parser is failing on.

Author: Kase Branham - Independent Researcher
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")
from add_aplist import parse_aplist_file, _parse_ap_token, detect_per_curve_format

APLIST_FILE = Path("./ecdata/aplist/aplist.00000-09999")

# Known to be in the file (from grep_aplist.py)
EXPECTED_KEYS = [
    (11, "a"), (14, "a"), (15, "a"), (26, "a"), (26, "b"),
    (101, "a"), (109, "a"), (113, "a"), (131, "a"),
    (139, "a"), (163, "a"), (179, "a"), (197, "a"), (202, "a"),
    (9999, "a"), (9999, "b"),
]


def main():
    print(f"File: {APLIST_FILE}")
    print()

    # Count raw lines in the file
    with open(APLIST_FILE, "r") as f:
        all_lines = f.readlines()
    n_raw = len(all_lines)
    n_nonempty = sum(1 for ln in all_lines
                     if ln.strip() and not ln.strip().startswith("#"))
    print(f"Raw lines:         {n_raw:,}")
    print(f"Non-empty lines:   {n_nonempty:,}")

    # Detect format
    fmt = detect_per_curve_format(APLIST_FILE)
    print(f"Format detected:   {'per-curve' if fmt else 'per-class'}")
    print()

    # Run the parser
    print("Running parse_aplist_file...")
    result = parse_aplist_file(APLIST_FILE)
    print(f"Records parsed:    {len(result):,}")
    print(f"Drop rate:         {100.0 * (n_nonempty - len(result)) / n_nonempty:.1f}%")
    print()

    # Check expected keys
    print("Checking expected (N, iso) keys:")
    print("-" * 70)
    missing = []
    for key in EXPECTED_KEYS:
        if key in result:
            aps = result[key]
            preview = " ".join(str(x) for x in aps[:6])
            print(f"  {key}: PRESENT ({len(aps)} a_p, first 6: {preview} ...)")
        else:
            print(f"  {key}: MISSING from parsed result")
            missing.append(key)

    # For missing keys, find the line in the raw file and try parsing it
    # token by token
    if missing:
        print()
        print("=" * 70)
        print("DIAGNOSING WHY THE FIRST FEW MISSING KEYS ARE DROPPED")
        print("=" * 70)
        for key in missing[:3]:
            N, iso = key
            print(f"\nLooking for ({N}, '{iso}') in raw file...")
            target_prefix = f"{N} {iso} "  # also try variants
            for i, line in enumerate(all_lines):
                stripped = line.lstrip().rstrip("\n")
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    continue
                if parts[0] == str(N) and parts[1] == iso:
                    print(f"  Found at line {i+1}:")
                    display = stripped if len(stripped) <= 120 else stripped[:117] + "..."
                    print(f"    {display}")
                    print(f"    Tokens: {parts}")
                    aps_start = 3 if fmt else 2
                    print(f"    aps_start = {aps_start}")
                    print(f"    parts[0:aps_start] = {parts[:aps_start]}")
                    print(f"    Trying to parse a_p tokens parts[{aps_start}:]:")
                    try:
                        aps = [_parse_ap_token(x) for x in parts[aps_start:]]
                        print(f"    Parsed OK: {aps}")
                        print(f"    -> parser WOULD have accepted this. "
                              f"Possible cause: collision with another (N, iso) "
                              f"seen earlier (first-wins dedup).")
                    except (ValueError, IndexError) as e:
                        # Identify which token failed
                        for j, tok in enumerate(parts[aps_start:]):
                            try:
                                _parse_ap_token(tok)
                            except (ValueError, IndexError):
                                print(f"    FAILED at token index {j}: "
                                      f"'{tok}' (overall position {aps_start+j})")
                                break
                        print(f"    Error: {e}")
                    break
            else:
                print(f"  ({N}, '{iso}') not found in raw lines either.")


if __name__ == "__main__":
    main()
