"""Enriched feature set — genuinely new inputs beyond the coarse confluence factors.

Adds time-series **momentum** (the most robust empirical equity factor),
**volatility**, **volume microstructure** and a continuous **distance-to-POC** to
the four confluence factors, then reuses the same
:func:`~vpts.ml.factor_model.cpcv_factor_eval` harness. All features at bar ``t``
use only data ``<= t``; the label is the strictly-future forward return.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from vpts.profile.calculator import VolumeProfileCalculator
from vpts.regime.indicators import atr, ensure_ohlcv
from vpts.regime.patterns import VolumePatternDetector
from vpts.regime.quiet import QuietPhaseDetector
from vpts.scoring.scorer import ConfluenceScorer
from vpts.ml.models import FactorDataset

ENRICHED_FEATURES = (
    "value_area", "key_level", "quiet", "patterns",     # confluence
    "mom_20", "mom_60", "mom_120_20",                    # momentum (incl. 12-1 style)
    "vol_20", "atr_frac",                                # volatility
    "vol_trend",                                         # volume microstructure
    "poc_dist",                                          # continuous structure
)


def build_enriched_factor_dataset(
    df: pd.DataFrame,
    *,
    lookback: int = 120,
    horizon: int = 20,
    stride: int = 3,
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    profile_calculator: Optional[VolumeProfileCalculator] = None,
    quiet_detector: Optional[QuietPhaseDetector] = None,
    pattern_detector: Optional[VolumePatternDetector] = None,
    scorer: Optional[ConfluenceScorer] = None,
) -> FactorDataset:
    """Build an enriched (confluence + momentum/vol/microstructure) factor dataset."""
    ensure_ohlcv(df, min_bars=lookback + horizon + 2)
    pc = profile_calculator or VolumeProfileCalculator(bin_mode="auto")
    qd = quiet_detector or QuietPhaseDetector()
    pat = pattern_detector or VolumePatternDetector()
    sc = scorer or ConfluenceScorer()

    close = df["Close"]
    ret = close.pct_change()
    mom20 = (close / close.shift(20) - 1.0).to_numpy()
    mom60 = (close / close.shift(60) - 1.0).to_numpy()
    mom120_20 = (close.shift(20) / close.shift(120) - 1.0).to_numpy()   # 12-1 style
    vol20 = ret.rolling(20).std().to_numpy()
    atr_s = atr(df["High"], df["Low"], df["Close"], 14)
    atr_a = atr_s.to_numpy()
    atr_frac = (atr_s / close).to_numpy()
    volu = df["Volume"]
    vol_trend = (volu.rolling(10).mean() / volu.rolling(60).mean() - 1.0).to_numpy()
    close_a = close.to_numpy()
    n = len(df)

    feats: list[list[float]] = []
    ys: list[float] = []
    base: list[float] = []
    ts: list = []
    for t in range(lookback - 1, n - horizon, max(1, stride)):
        pre = (mom20[t], mom60[t], mom120_20[t], vol20[t], atr_frac[t], vol_trend[t])
        if not np.all(np.isfinite(pre)):           # warm-up / missing precomputed feature
            continue
        window = df.iloc[t - lookback + 1 : t + 1]
        try:
            profile = pc.calculate(window, symbol, interval)
            quiet = qd.detect(window, symbol, interval)
            patterns = pat.detect(window, profile=profile, symbol=symbol, interval=interval)
            score = sc.score(window, profile, quiet, patterns, symbol=symbol, interval=interval)
        except (ValueError, ZeroDivisionError):
            continue
        comps = {c.name: c for c in score.components}
        atr_t = atr_a[t] if np.isfinite(atr_a[t]) and atr_a[t] > 0 else max(profile.bin_size, 1e-9)
        feats.append([
            comps["value_area"].strength * comps["value_area"].direction,
            comps["key_level"].strength * comps["key_level"].direction,
            comps["quiet"].strength,
            comps["patterns"].strength * comps["patterns"].direction,
            float(mom20[t]), float(mom60[t]), float(mom120_20[t]),
            float(vol20[t]), float(atr_frac[t]), float(vol_trend[t]),
            float((close_a[t] - profile.poc) / atr_t),
        ])
        ys.append(float(close_a[t + horizon] / close_a[t] - 1.0))
        base.append(float(score.bias_score))
        ts.append(df.index[t])

    is_dt = isinstance(df.index, pd.DatetimeIndex) and ts
    return FactorDataset(
        X=np.array(feats, dtype=float).reshape(-1, len(ENRICHED_FEATURES)),
        y=np.array(ys, dtype=float),
        baseline=np.array(base, dtype=float),
        feature_names=ENRICHED_FEATURES,
        horizon=horizon,
        stride=max(1, stride),
        timestamps=pd.DatetimeIndex(ts) if is_dt else None,
        symbol=symbol,
    )
