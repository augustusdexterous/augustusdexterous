"""Command-line report: ``backtest-overfitting returns.csv``.

The CSV has one header row with trial names and one row per period, oldest first.
Each trial column must hold decimal returns (0.01 means 1%). Columns are skipped, with
a note on stderr, when they contain text (such as dates), or when any value is at least
1 in magnitude (which usually means prices or YYYYMMDD dates). Use ``--columns`` to choose
the trial columns yourself. A missing value in a trial column is an error, not a silent drop.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .report import analyze


def _as_float(cell: str) -> float | None:
    try:
        return float(cell)
    except ValueError:
        return None


def load_trials(
    path: str, columns: list[str] | None = None
) -> tuple[list[str], np.ndarray, list[str]]:
    """Read trial columns. Returns (names, matrix of shape (T, N), skipped-column notes)."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        rows = [row for row in reader if row]
    if not header or not rows:
        raise ValueError("CSV must have a header row and at least one data row")

    names: list[str] = []
    data: list[np.ndarray] = []
    skipped: list[str] = []
    for j, name in enumerate(header):
        cells = [row[j].strip() if j < len(row) else "" for row in rows]
        if columns is not None:
            if name not in columns:
                continue
            values = [_as_float(c) for c in cells]
            for i, (cell, value) in enumerate(zip(cells, values)):
                if value is None:
                    raise ValueError(
                        f"column {name!r} is not numeric or has a missing value at data row {i + 1}"
                    )
            names.append(name)
            data.append(np.asarray(values, dtype=np.float64))
            continue

        present = [c for c in cells if c]
        if not present:
            skipped.append(f"{name} (empty)")
            continue
        if any(_as_float(c) is None for c in present):
            skipped.append(f"{name} (non-numeric, e.g. dates or labels)")
            continue
        missing = [i for i, c in enumerate(cells) if not c]
        if missing:
            raise ValueError(
                f"column {name!r} has a missing value at data row {missing[0] + 1}; "
                "fill it in, or leave the column out with --columns"
            )
        values = np.asarray([float(c) for c in cells], dtype=np.float64)
        if np.max(np.abs(values)) >= 1:
            skipped.append(
                f"{name} (values of 1 or more: looks like prices or dates; returns are decimals)"
            )
            continue
        names.append(name)
        data.append(values)

    if columns is not None:
        absent = [c for c in columns if c not in names]
        if absent:
            raise ValueError(f"requested columns not found: {', '.join(absent)}")
    if len(names) < 2:
        raise ValueError(
            f"need at least two numeric return columns, found {len(names)}"
        )
    return names, np.column_stack(data), skipped


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
    parser.add_argument(
        "--columns", default=None, help="comma-separated trial columns to use"
    )
    args = parser.parse_args(argv)
    columns = [c.strip() for c in args.columns.split(",")] if args.columns else None
    try:
        names, matrix, skipped = load_trials(args.csv, columns)
        report = analyze(
            matrix,
            periods_per_year=args.periods_per_year,
            n_splits=args.splits,
            split=args.split,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for note in skipped:
        print(f"skipped column: {note}", file=sys.stderr)
    print(f"best trial: {names[report.best_trial]}")
    print(report.summary())
    return 0
