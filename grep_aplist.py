#!/usr/bin/env python3
"""grep_aplist.py - Check whether specific conductors are present in
Cremona's aplist files.

Hypothesis: aplist is a historical snapshot, missing curves added later.
Test: search for specific low-conductor curves that the diagnostic said
had no a_p (101a, 113a). If they're absent from the file entirely,
hypothesis confirmed. If present with unusual labels, we have a parsing
mismatch to fix.

Author: Kase Branham - Independent Researcher
"""

from pathlib import Path

APLIST_FILE = Path("./ecdata/aplist/aplist.00000-09999")

# Conductors flagged by the coverage diagnostic as missing a_p data:
TEST_CONDUCTORS = [11, 14, 15, 26, 101, 109, 113, 131, 139, 163, 179, 197, 202,
                   500, 1000, 5000, 9000, 9999]


def main():
    print(f"File: {APLIST_FILE}")
    print(f"Exists: {APLIST_FILE.exists()}")
    if not APLIST_FILE.exists():
        print("File not found.")
        return

    size_kb = APLIST_FILE.stat().st_size / 1024
    print(f"Size: {size_kb:.0f} KB")
    print()

    # Read all lines once
    with open(APLIST_FILE, "r") as f:
        lines = f.readlines()
    print(f"Total lines in file: {len(lines):,}")
    print()

    # For each test conductor, find all matching lines
    print("Searching for specific conductors:")
    print("-" * 70)
    for N in TEST_CONDUCTORS:
        prefix = f"{N} "  # conductor followed by space
        # Also try with leading space (the dump showed leading space)
        matches = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith(prefix):
                matches.append(stripped.rstrip())

        if matches:
            print(f"  N = {N:>5d}: {len(matches)} record(s)")
            for m in matches[:3]:
                # Truncate long lines for display
                display = m if len(m) <= 100 else m[:97] + "..."
                print(f"      {display}")
            if len(matches) > 3:
                print(f"      ... and {len(matches)-3} more")
        else:
            print(f"  N = {N:>5d}: NOT FOUND in this file")

    # Show overall conductor coverage of the file
    print()
    print("-" * 70)
    print("Conductor range of records in this file:")
    conductors_in_file = set()
    for line in lines:
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            conductors_in_file.add(int(parts[0]))

    if conductors_in_file:
        sorted_N = sorted(conductors_in_file)
        print(f"  min conductor: {sorted_N[0]}")
        print(f"  max conductor: {sorted_N[-1]}")
        print(f"  distinct conductors with records: {len(sorted_N):,}")
        # Show a few from each end
        print(f"  first 10: {sorted_N[:10]}")
        print(f"  last 10:  {sorted_N[-10:]}")
    else:
        print("  No numeric records found.")


if __name__ == "__main__":
    main()
