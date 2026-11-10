"""Command-line report: ``backtest-overfitting returns.csv``.

The CSV has one header row with trial names and one row per period, oldest first.
Non-numeric columns (such as a date column) are ignored.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .report import analyze


def load_trials(path: str) -> tuple[list[str], np.ndarray]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        rows = [row for row in reader if row]
    if not header or not rows:
        raise ValueError("CSV must have a header row and at least one data row")
    kept, columns = [], []
    for j, name in enumerate(header):
        try:
            col = [float(row[j]) for row in rows]
        except (ValueError, IndexError):
            continue  # non-numeric column, e.g. dates
        kept.append(name)
        columns.append(col)
    if len(columns) < 2:
        raise ValueError("need at least two numeric trial columns")
    return kept, np.column_stack(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="backtest-overfitting", description=__doc__.splitlines()[0]
    )
    parser.add_argument("csv", help="CSV of returns: header row, one column per trial")
    parser.add_argument("--periods-per-year", type=float, default=252.0)
    parser.add_argument(
        "--splits", type=int, default=10, help="even number of CSCV blocks"
    )
    parser.add_argument(
        "--split", type=int, default=None, help="row index for the chronological check"
    )
    args = parser.parse_args(argv)
    try:
        names, matrix = load_trials(args.csv)
        report = analyze(
            matrix,
            periods_per_year=args.periods_per_year,
            n_splits=args.splits,
            split=args.split,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"best trial: {names[report.best_trial]}")
    print(report.summary())
    return 0
