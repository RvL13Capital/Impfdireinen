"""Tests for vpts.validation — Combinatorial Purged CV (splitter + evaluator).

The splitter tests are pure-index and deterministic; the evaluator test is an
offline smoke run on synthetic data.

    python tests/test_validation.py
    pytest tests/test_validation.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import (  # noqa: E402
    Backtester,
    CombinatorialPurgedCV,
    CPCVResult,
    SignalGenerator,
)
from vpts.validation.cpcv import _contiguous_blocks  # noqa: E402


def _osc(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100 + 7 * np.sin(t / 26.0) + 3 * np.sin(t / 8.0) + np.cumsum(rng.normal(0, 0.1, n))
    close = np.maximum(close, 5)
    rngs = 1.0 + 0.5 * np.abs(np.sin(t / 26.0))
    high, low = close + rngs / 2 + 0.2, close - rngs / 2 - 0.2
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = (3e6 + 1e6 * np.cos(t / 26.0) + rng.integers(-3e5, 3e5, n)).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": vol}, index=idx)


# --------------------------------------------------------------------------- #
# Splitter (pure index logic)
# --------------------------------------------------------------------------- #
def test_split_count_disjoint_and_in_range() -> None:
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    n = 60
    splits = list(cv.split(n))
    assert len(splits) == cv.n_splits() == math.comb(6, 2) == 15
    for s in splits:
        assert set(s.test_idx).isdisjoint(set(s.train_idx))
        assert s.test_idx.min() >= 0 and s.test_idx.max() < n
        assert len(_contiguous_blocks(s.test_idx)) <= 2  # k=2 groups -> <=2 blocks


def test_purge_and_embargo_remove_train_neighbours() -> None:
    n, purge, embargo_pct = 60, 3, 0.1
    embargo_bars = math.ceil(embargo_pct * n)  # 6
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge=purge,
                               embargo_pct=embargo_pct)
    for s in cv.split(n):
        train = set(int(x) for x in s.train_idx)
        for a, b in _contiguous_blocks(s.test_idx):
            assert not any(a - purge <= t < a for t in train), "purge leak before block"
            assert not any(b < t <= b + embargo_bars for t in train), "embargo leak after block"


def test_contiguous_blocks() -> None:
    assert _contiguous_blocks(np.array([0, 1, 2, 5, 6, 9])) == [(0, 2), (5, 6), (9, 9)]
    assert _contiguous_blocks(np.array([], dtype=int)) == []


def test_split_validation() -> None:
    for kwargs in (
        {"n_groups": 1},
        {"n_groups": 5, "n_test_groups": 5},
        {"n_groups": 5, "n_test_groups": 0},
        {"embargo_pct": 1.0},
        {"purge": -1},
    ):
        try:
            CombinatorialPurgedCV(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {kwargs}")


# --------------------------------------------------------------------------- #
# Evaluator (backtest paths)
# --------------------------------------------------------------------------- #
def test_backtest_paths_structure() -> None:
    df = _osc(600)
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2)
    bt = Backtester(
        lookback=60, recompute_stride=2,
        signal_generator=SignalGenerator(style="breakout", min_quality=40, min_abs_bias=10),
    )
    res = cv.backtest_paths(df, bt, symbol="OSC", interval="1d")

    assert isinstance(res, CPCVResult)
    assert res.n_groups == 5
    assert res.group_results[0].skipped          # first group has no warm-up history
    assert res.n_usable_groups == 4
    assert res.n_paths == math.comb(4, 2) == 6   # only fully-usable combinations
    assert len(res.path_returns) == 6
    assert res.return_min <= res.return_mean <= res.return_max
    assert 0.0 <= res.pct_paths_profitable <= 100.0
    assert "CPCV" in res.summary()
    assert len(res.groups_dataframe()) == 5 and len(res.paths_dataframe()) == 6
    json.dumps(res.as_dict())


def test_backtest_paths_too_few_bars_raises() -> None:
    df = _osc(80)  # 5 groups of 16; lookback 60 leaves no usable group
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2)
    bt = Backtester(lookback=60)
    try:
        cv.backtest_paths(df, bt)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError when no usable paths")


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} validation tests …\n")
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  ✓ {t.__name__}")
    print(f"\n{passed} passed, {failed} failed.")
    if not failed:
        print("\nSample CPCV on synthetic data:")
        df = _osc(600)
        bt = Backtester(lookback=60, recompute_stride=2,
                        signal_generator=SignalGenerator(style="breakout",
                                                         min_quality=40, min_abs_bias=10))
        print(CombinatorialPurgedCV(n_groups=5, n_test_groups=2).backtest_paths(
            df, bt, symbol="OSC", interval="1d").summary())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
