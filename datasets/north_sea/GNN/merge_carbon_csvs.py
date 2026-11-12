"""
Merge many CSVs (e.g. the per-run CodeCarbon dumps under carbon_logs/, or
archived stage2_carbon_log__pre_*.bak.csv schema-drift backups) into one file.

Usage:
    python merge_carbon_csvs.py "carbon_logs/*.csv" -o merged_carbon_logs.csv
    python merge_carbon_csvs.py "stage2_ckpt/*.csv" "stage2_ckpt/carbon_logs/*.csv" -o all.csv

Files with different columns are unioned (missing values become NaN) rather
than raising, since the CodeCarbon dumps, the incremental run log, and any
archived .bak.csv from a schema change don't all share the same schema.
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="+", help="One or more glob patterns matching CSVs to merge")
    ap.add_argument("-o", "--output", default="merged.csv", help="Output CSV path (default: merged.csv)")
    ap.add_argument("--no-source-col", action="store_true",
                    help="Don't add a 'source_file' column recording which file each row came from")
    args = ap.parse_args()

    paths = []
    for pattern in args.patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            print(f"  [warn] pattern matched nothing: {pattern}", file=sys.stderr)
        paths.extend(matches)

    # de-dupe while preserving order, in case two patterns overlap
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]

    if not paths:
        print("No CSVs matched any pattern — nothing to merge.", file=sys.stderr)
        sys.exit(1)

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  [skip] {p}: could not parse ({e})", file=sys.stderr)
            continue
        if not args.no_source_col:
            df.insert(0, "source_file", Path(p).name)
        frames.append(df)
        print(f"  + {p}  ({len(df)} rows, {len(df.columns)} cols)")

    if not frames:
        print("No CSVs could be read — nothing to merge.", file=sys.stderr)
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged.to_csv(args.output, index=False)
    print(f"\nMerged {len(frames)} file(s), {len(merged)} total row(s), "
          f"{len(merged.columns)} column(s) -> {args.output}")


if __name__ == "__main__":
    main()
