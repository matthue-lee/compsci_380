#!/usr/bin/env python3

"""Filter master_table.csv rows based on secondary-structure content."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_COLUMN = "ss_fraction_strand"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="master_table.csv", help="Source master table CSV")
    parser.add_argument(
        "--output",
        default="master_table_beta.csv",
        help="Destination CSV containing filtered rows",
    )
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help="Column with beta content/fraction to evaluate",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=0.4,
        help="Minimum value (inclusive) required to keep a row",
    )
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset at {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input file lacks a header row")
        return reader.fieldnames, [row for row in reader]


def save_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_rows(
    rows: list[dict[str, str]], column: str, min_value: float
) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        raw_value = (row.get(column) or "").strip()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if value >= min_value:
            filtered.append(row)
    return filtered


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    fieldnames, rows = load_rows(input_path)
    if args.column not in fieldnames:
        raise ValueError(f"Column '{args.column}' not found in {input_path}")
    filtered_rows = filter_rows(rows, args.column, args.min_value)
    save_rows(output_path, fieldnames, filtered_rows)
    print(
        f"Wrote {len(filtered_rows)} beta-rich rows (column {args.column} >= {args.min_value}) to {output_path}"
    )


if __name__ == "__main__":
    main()
