"""Tests for the example sweep helpers (regression for the aggregate-curve bug).

    python tests/test_examples.py
    pytest tests/test_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from midcap_scan import aggregate_curve  # noqa: E402


def test_aggregate_curve_uses_common_window_and_rebases() -> None:
    """Bug #4: curves with different histories must be intersected to a common
    window and re-based — no leading-NaN splicing / drifting constituent set."""
    idx_a = pd.date_range("2020-01-01", periods=10, freq="D")
    idx_b = pd.date_range("2020-01-04", periods=10, freq="D")  # starts 3 days later
    a = pd.Series(np.linspace(100, 120, 10), index=idx_a)      # each already starts at 100
    b = pd.Series(np.linspace(100, 140, 10), index=idx_b)

    agg = aggregate_curve({"a": a, "b": b})
    common = idx_a.intersection(idx_b)

    assert agg is not None
    assert list(agg.index) == list(common)        # only the overlapping dates
    assert agg.notna().all()                       # no NaN splice
    assert np.isclose(agg.iloc[0], 100.0)          # re-based to 100 at the common start
    # Membership is constant across the whole window (always exactly 2 names).
    assert len(agg) == len(common)


def test_aggregate_curve_no_overlap_returns_none() -> None:
    a = pd.Series([100.0, 110.0], index=pd.date_range("2020-01-01", periods=2, freq="D"))
    c = pd.Series([100.0, 105.0], index=pd.date_range("2021-01-01", periods=2, freq="D"))
    assert aggregate_curve({"a": a, "c": c}) is None
    assert aggregate_curve({}) is None


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} example tests …\n")
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__}: {exc}")
        else:
            passed += 1
            print(f"  ✓ {t.__name__}")
    print(f"\n{passed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
