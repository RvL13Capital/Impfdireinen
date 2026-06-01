"""Walk bars and assemble the structural feature matrix → ``FactorDataset`` / ``MetaDataset``.

Each decision bar gets the per-window structural features from
:mod:`vpts.structure.analytics` plus the two rolling/time-series signals that
need history — the **value-area compression z-score** and the **POC-migration
slope** — computed from a trailing window of *previously sampled* bars, so there
is no look-ahead. The shared bar-walk feeds two builders:

* :func:`build_structural_dataset` — features → forward return (a
  :class:`~vpts.ml.models.FactorDataset` for the CPCV factor harness);
* :func:`build_structural_meta_dataset` — features → **MFE/MAE triple-barrier**
  win/loss for a long (or fixed-side) bet (a :class:`~vpts.ml.models.MetaDataset`
  for the meta-labeling harness / an XGBoost classifier).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

from vpts.profile.calculator import VolumeProfileCalculator
from vpts.regime.indicators import atr, ensure_ohlcv
from vpts.structure.analytics import (
    SHAPE_B,
    SHAPE_P,
    SHAPE_b,
    classify_shape,
    decayed_poc,
    detect_ledges,
    poor_high,
    synthetic_delta_stats,
    value_area_compression_ratio,
    weighted_moments,
)
from vpts.structure.models import STRUCTURAL_FEATURES, StructuralFeatures
from vpts.ml.labeling import triple_barrier_labels
from vpts.ml.models import FactorDataset, MetaDataset

_EPS = 1e-12


def _walk_structural(
    df: pd.DataFrame,
    *,
    lookback: int,
    horizon: int,
    stride: int,
    vacr_window: int,
    poc_window: int,
    halflife: float,
    symbol: Optional[str],
    interval: Optional[str],
    profile_calculator: Optional[VolumeProfileCalculator],
) -> tuple[list[np.ndarray], list[int], list, list[StructuralFeatures]]:
    """Shared no-look-ahead walk → ``(feature_vectors, event_positions, timestamps, rows)``.

    A feature row is emitted only once the rolling warm-up
    (``max(vacr_window, poc_window)`` previously sampled bars) is satisfied and the
    whole vector is finite. ``event_positions`` are the bar indices ``t`` (for
    label construction); features use only data ``<= t``.
    """
    ensure_ohlcv(df, min_bars=lookback + horizon + 2)
    pc = profile_calculator or VolumeProfileCalculator(bin_mode="auto")
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    volume = df["Volume"].to_numpy(float)
    atr_a = atr(df["High"], df["Low"], df["Close"], 14).to_numpy(float)
    n = len(df)
    is_dt = isinstance(df.index, pd.DatetimeIndex)

    hist_vacr: list[float] = []
    hist_poc: list[float] = []
    feats: list[np.ndarray] = []
    ev_pos: list[int] = []
    ts: list = []
    rows: list[StructuralFeatures] = []

    for t in range(lookback - 1, n - horizon, max(1, stride)):
        window = df.iloc[t - lookback + 1 : t + 1]
        try:
            profile = pc.calculate(window, symbol, interval)
        except (ValueError, ZeroDivisionError):
            continue
        sl = slice(t - lookback + 1, t + 1)
        wh, wl, wc, wv = high[sl], low[sl], close[sl], volume[sl]

        net, poc_d = synthetic_delta_stats(wh, wl, wc, wv, profile)
        _, _, skew, kurt = weighted_moments(profile.bin_centers, profile.volume_distribution)
        shape = classify_shape(profile, skew)
        loc_rng = profile.price_high - profile.price_low
        poc_loc = float(np.clip((profile.poc - profile.price_low) / loc_rng, 0, 1)) \
            if loc_rng > _EPS else 0.5
        n_ledges = float(len(detect_ledges(profile)))
        ph = 1.0 if poor_high(profile) else 0.0
        atr_t = atr_a[t] if (np.isfinite(atr_a[t]) and atr_a[t] > 0) else max(profile.bin_size, _EPS)
        cbm = float((decayed_poc(wh, wl, wc, wv, profile, halflife) - profile.poc) / atr_t)
        vacr_raw = value_area_compression_ratio(profile, close[t])

        # --- rolling features from PRIOR sampled history (no look-ahead) ---
        warm = len(hist_vacr) >= vacr_window and len(hist_poc) >= poc_window - 1
        vacr_z = 0.0
        poc_slope = 0.0
        if len(hist_vacr) >= vacr_window:
            b = np.asarray(hist_vacr[-vacr_window:], float)
            sd = float(b.std())
            vacr_z = float((vacr_raw - b.mean()) / sd) if sd > _EPS else 0.0
        if len(hist_poc) >= poc_window - 1:
            pocs = np.asarray(hist_poc[-(poc_window - 1):] + [profile.poc], float)
            xs = np.arange(pocs.size, dtype=float)
            slope = float(linregress(xs, pocs).slope) if pocs.size >= 2 else 0.0
            poc_slope = slope / close[t] if close[t] > _EPS else 0.0

        hist_vacr.append(float(vacr_raw))
        hist_poc.append(float(profile.poc))
        if not warm:
            continue

        sf = StructuralFeatures(
            delta_net=net, delta_poc=poc_d, skew=float(skew), kurtosis=float(kurt),
            poc_loc=poc_loc, vacr_z=vacr_z, poc_slope=poc_slope,
            cost_basis_migration=cbm, n_ledges=n_ledges, poor_high=ph,
            is_P=1.0 if shape == SHAPE_P else 0.0,
            is_b=1.0 if shape == SHAPE_b else 0.0,
            is_B=1.0 if shape == SHAPE_B else 0.0,
            shape_class=int(shape), poc=float(profile.poc), vacr=float(vacr_raw),
            timestamp=df.index[t] if is_dt else None,
        )
        vec = sf.to_vector()
        if not np.all(np.isfinite(vec)):
            continue
        feats.append(vec)
        ev_pos.append(t)
        ts.append(df.index[t] if is_dt else t)
        rows.append(sf)
    return feats, ev_pos, ts, rows


def build_structural_dataset(
    df: pd.DataFrame,
    *,
    lookback: int = 120,
    horizon: int = 20,
    stride: int = 3,
    vacr_window: int = 20,
    poc_window: int = 5,
    halflife: float = 21.0,
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    profile_calculator: Optional[VolumeProfileCalculator] = None,
) -> FactorDataset:
    """Structural features → strictly-future forward return (no look-ahead).

    The single-feature ``baseline`` is the synthetic delta at the POC.
    """
    feats, ev_pos, ts, rows = _walk_structural(
        df, lookback=lookback, horizon=horizon, stride=stride, vacr_window=vacr_window,
        poc_window=poc_window, halflife=halflife, symbol=symbol, interval=interval,
        profile_calculator=profile_calculator)
    close = df["Close"].to_numpy(float)
    ys = [float(close[t + horizon] / close[t] - 1.0) for t in ev_pos]
    base = [r.delta_poc for r in rows]
    is_dt = isinstance(df.index, pd.DatetimeIndex) and bool(ts)
    return FactorDataset(
        X=np.array(feats, dtype=float).reshape(-1, len(STRUCTURAL_FEATURES)),
        y=np.array(ys, dtype=float),
        baseline=np.array(base, dtype=float),
        feature_names=STRUCTURAL_FEATURES,
        horizon=horizon,
        stride=max(1, stride),
        timestamps=pd.DatetimeIndex(ts) if is_dt else None,
        symbol=symbol,
    )


def build_structural_meta_dataset(
    df: pd.DataFrame,
    *,
    lookback: int = 120,
    horizon: int = 20,
    stride: int = 3,
    vacr_window: int = 20,
    poc_window: int = 5,
    halflife: float = 21.0,
    side: int = 1,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    atr_period: int = 14,
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    profile_calculator: Optional[VolumeProfileCalculator] = None,
) -> MetaDataset:
    """Structural features → **MFE/MAE triple-barrier** win/loss for a fixed-``side`` bet.

    For each decision bar a bet is taken in ``side`` (default long); the
    volatility-scaled triple barrier decides win (profit-take touched first) vs
    loss (stop or adverse vertical) — i.e. whether Maximum Favorable Excursion
    beat Maximum Adverse Excursion. An XGBoost/logistic classifier then learns
    ``P(win)`` from the structural features. No look-ahead: features ≤ t, barriers
    strictly after t.
    """
    feats, ev_pos, ts, _ = _walk_structural(
        df, lookback=lookback, horizon=horizon, stride=stride, vacr_window=vacr_window,
        poc_window=poc_window, halflife=halflife, symbol=symbol, interval=interval,
        profile_calculator=profile_calculator)
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    vol = (atr(df["High"], df["Low"], df["Close"], atr_period) / df["Close"]).to_numpy(float)

    ev_arr = np.array(ev_pos, dtype=int)
    side_arr = np.full(ev_arr.size, int(side), dtype=int)
    if ev_arr.size:
        _, ret, win, exit_pos = triple_barrier_labels(
            close, high, low, vol, ev_arr, side_arr, horizon, pt_mult, sl_mult)
        holding = (exit_pos - ev_arr).astype(int)        # bars entry → first-touch exit, in [1, horizon]
    else:
        ret = np.array([], dtype=float)
        win = np.array([], dtype=int)
        holding = np.array([], dtype=int)

    is_dt = isinstance(df.index, pd.DatetimeIndex) and bool(ts)
    return MetaDataset(
        X=np.array(feats, dtype=float).reshape(-1, len(STRUCTURAL_FEATURES)),
        meta_label=win,
        side=side_arr,
        realized_return=ret,
        feature_names=STRUCTURAL_FEATURES,
        horizon=horizon,
        stride=max(1, stride),
        timestamps=pd.DatetimeIndex(ts) if is_dt else None,
        symbol=symbol,
        holding_bars=holding,
    )
