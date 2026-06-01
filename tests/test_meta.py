"""Tests for vpts.ml triple-barrier labeling + meta-labeling.

    python tests/test_meta.py
    pytest tests/test_meta.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import (  # noqa: E402
    LogisticMetaModel,
    MetaCVResult,
    build_meta_dataset,
    cpcv_meta_eval,
    triple_barrier_labels,
)
from vpts.ml.meta_model import _auc  # noqa: E402
from vpts.ml.models import MetaDataset  # noqa: E402


# --------------------------------------------------------------------------- #
# Triple barrier (crafted price paths; vol=1%, pt=sl=2x -> ±2% barriers)
# --------------------------------------------------------------------------- #
def _tb(close, high, low, side):
    return triple_barrier_labels(
        np.array(close, float), np.array(high, float), np.array(low, float),
        np.array([0.01] * len(close)), np.array([0]), np.array([side]),
        horizon=2, pt_mult=2.0, sl_mult=2.0)


def test_triple_barrier_long_profit() -> None:
    out, ret, win, xp = _tb([100, 101.5], [100.5, 103], [99.5, 101], side=1)
    assert out[0] == 1 and win[0] == 1 and np.isclose(ret[0], 0.02) and xp[0] == 1


def test_triple_barrier_long_stop() -> None:
    out, ret, win, _ = _tb([100, 100.0], [100.5, 100.5], [99.5, 97.0], side=1)
    assert out[0] == -1 and win[0] == 0 and np.isclose(ret[0], -0.02)


def test_triple_barrier_vertical() -> None:
    out, ret, win, _ = _tb([100, 100.5, 101], [100.5, 101, 101.5], [99.5, 100, 100.5], side=1)
    assert out[0] == 0 and np.isclose(ret[0], 0.01) and win[0] == 1


def test_triple_barrier_same_bar_is_stop() -> None:
    out, _, win, _ = _tb([100, 101.0], [100.5, 103], [99.5, 97.0], side=1)
    assert out[0] == -1 and win[0] == 0          # both barriers in one bar -> stop first


def test_triple_barrier_short_profit() -> None:
    out, ret, win, _ = _tb([100, 99.0], [100.5, 101], [99.5, 97.0], side=-1)
    assert out[0] == 1 and win[0] == 1 and np.isclose(ret[0], 0.02)  # short profits when price falls


# --------------------------------------------------------------------------- #
# Logistic meta-model
# --------------------------------------------------------------------------- #
def test_logistic_learns_separable_signal() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 3))
    p = 1.0 / (1.0 + np.exp(-(X @ np.array([2.0, -1.0, 0.0]))))
    y = (rng.uniform(size=600) < p).astype(int)
    m = LogisticMetaModel(l2=0.01, iters=1500).fit(X, y)
    assert _auc(y, m.predict_proba(X)) > 0.75
    assert m.w_[0] > 0 and m.w_[1] < 0


def test_logistic_single_class_and_guards() -> None:
    m = LogisticMetaModel().fit(np.random.normal(size=(50, 3)), np.ones(50))
    assert m.predict_proba(np.random.normal(size=(5, 3))).mean() > 0.9  # base-rate constant
    for bad in (lambda: LogisticMetaModel(l2=-1.0),
                lambda: LogisticMetaModel().predict_proba(np.zeros((2, 3)))):
        try:
            bad()
        except (ValueError, RuntimeError):
            pass
        else:  # pragma: no cover
            raise AssertionError("expected a guard to raise")


# --------------------------------------------------------------------------- #
# CPCV meta-evaluation
# --------------------------------------------------------------------------- #
def _meta_dataset(n=420, signal=0.0, seed=0) -> MetaDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    p = 1.0 / (1.0 + np.exp(-(signal * X[:, 0])))
    win = (rng.uniform(size=n) < p).astype(int)
    ret = np.where(win == 1, np.abs(rng.normal(0.02, 0.01, n)),
                   -np.abs(rng.normal(0.02, 0.01, n)))
    return MetaDataset(X=X, meta_label=win, side=np.ones(n, int), realized_return=ret,
                       feature_names=("a", "b", "c", "d"), horizon=1, stride=1, symbol="SYN")


def test_meta_eval_detects_filterable_signal() -> None:
    res = cpcv_meta_eval(_meta_dataset(n=480, signal=1.6, seed=2), threshold=0.5)
    assert isinstance(res, MetaCVResult)
    assert res.oos_auc_mean > 0.55
    assert res.oos_precision_mean > res.base_win_rate          # picks winners better
    assert res.return_improvement_mean > 0                     # and improves return


def test_meta_eval_no_signal_no_help() -> None:
    res = cpcv_meta_eval(_meta_dataset(n=480, signal=0.0, seed=4), threshold=0.5)
    assert abs(res.oos_auc_mean - 0.5) < 0.08
    assert "Meta-labeling CPCV" in res.summary()
    json.dumps(res.as_dict())


# --------------------------------------------------------------------------- #
# Dataset builder (real pipeline, synthetic OHLCV)
# --------------------------------------------------------------------------- #
def test_build_meta_dataset_shape() -> None:
    rng = np.random.default_rng(3)
    n = 240
    t = np.arange(n)
    close = 100 + 6 * np.sin(t / 22.0) + np.cumsum(rng.normal(0, 0.2, n))
    close = np.maximum(close, 5)
    high, low = close + 1.0, close - 1.0
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                       "Volume": (3e6 + rng.integers(-3e5, 3e5, n)).astype(float)},
                      index=pd.date_range("2020-01-01", periods=n, freq="B"))
    ds = build_meta_dataset(df, lookback=60, horizon=10, stride=5, min_abs_bias=5.0, symbol="SYN")
    assert len(ds) > 0
    assert ds.X.shape[1] == 6
    assert set(np.unique(ds.meta_label)).issubset({0, 1})
    assert set(np.unique(ds.side)).issubset({-1, 1})
    assert np.isfinite(ds.realized_return).all()


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} meta-labeling tests …\n")
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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
