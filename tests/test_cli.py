import csv
import subprocess
import sys
from pathlib import Path

from backtest_overfitting.cli import load_trials, main

DEMO = Path(__file__).resolve().parent.parent / "examples" / "demo_returns.csv"


def _write(path, rows, header):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def test_load_trials_skips_non_numeric_columns(tmp_path):
    p = tmp_path / "r.csv"
    _write(
        p,
        [["2020-01-01", "0.01", "-0.02"], ["2020-01-02", "0.00", "0.03"]],
        ["date", "a", "b"],
    )
    names, matrix = load_trials(p)
    assert names == ["a", "b"]
    assert matrix.shape == (2, 2)


def test_main_prints_report_for_demo():
    assert DEMO.exists()
    assert main([str(DEMO)]) == 0


def test_main_reports_bad_input(tmp_path, capsys):
    p = tmp_path / "one_column.csv"
    _write(p, [["0.01"], ["0.02"]], ["only"])
    assert main([str(p)]) == 2
    assert "error" in capsys.readouterr().err


def test_module_entry_point_runs():
    out = subprocess.run(
        [sys.executable, "-m", "backtest_overfitting", str(DEMO)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Deflated Sharpe Ratio" in out.stdout
