"""Triple-barrier labeling + meta-labeling dataset (López de Prado, AFML ch. 3).

For each event the label is decided by which of three barriers is touched first,
over a forward horizon, with the profit/stop barriers scaled by volatility:

* **profit-take** — price moves ``pt_mult × vol`` in the bet's direction → win,
* **stop-loss** — price moves ``sl_mult × vol`` against it → loss,
* **vertical** — ``horizon`` bars elapse → labeled by the sign of the return.

**Meta-labeling**: the *primary* model (here, the confluence `bias`) picks the
**side**; the triple barrier is applied in that side's direction, giving a binary
**meta-label** (did the bet win?). A *secondary* model then learns *whether* to
take the primary signal — i.e. it filters, it does not pick direction.
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
from vpts.ml.models import MetaDataset

_META_FEATURES = ("value_area", "key_level", "quiet", "patterns",
                  "setup_quality", "bias_mag")


def triple_barrier_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    vol: np.ndarray,
    event_pos: np.ndarray,
    side: np.ndarray,
    horizon: int,
    pt_mult: float,
    sl_mult: float,
):
    """First-touch triple-barrier outcomes for *events* taken in *side* direction.

    ``vol`` is a per-bar **fractional** volatility (e.g. ATR/close). Returns
    ``(outcome, ret, win, exit_pos)`` where ``outcome`` is ``+1`` profit / ``-1``
    stop / ``0`` vertical, ``ret`` is the realised return in the bet's direction,
    and ``win`` is ``1`` iff ``ret > 0``. If both barriers fall in the same bar,
    the **stop** is assumed first (conservative).
    """
    n = len(close)
    E = len(event_pos)
    outcome = np.zeros(E, dtype=int)
    ret = np.zeros(E, dtype=float)
    exit_pos = np.zeros(E, dtype=int)
    for k in range(E):
        i = int(event_pos[k])
        s = int(side[k])
        entry = close[i]
        v = vol[i]
        if not np.isfinite(v) or v <= 0:
            v = 0.01
        pt, sl = pt_mult * v, sl_mult * v
        end = min(i + horizon, n - 1)
        hit, xp = 0, end
        for j in range(i + 1, end + 1):
            if s == 1:
                up = high[j] >= entry * (1.0 + pt)
                dn = low[j] <= entry * (1.0 - sl)
            else:
                up = low[j] <= entry * (1.0 - pt)     # profit for a short = price down
                dn = high[j] >= entry * (1.0 + sl)    # stop for a short = price up
            if dn:                                     # stop assumed first on a tie
                hit, xp = -1, j
                break
            if up:
                hit, xp = 1, j
                break
        if hit == 1:
            exit_price = entry * (1.0 + pt) if s == 1 else entry * (1.0 - pt)
        elif hit == -1:
            exit_price = entry * (1.0 - sl) if s == 1 else entry * (1.0 + sl)
        else:
            exit_price = close[end]
        outcome[k] = hit
        ret[k] = s * (exit_price - entry) / entry
        exit_pos[k] = xp
    win = (ret > 0).astype(int)
    return outcome, ret, win, exit_pos


def build_meta_dataset(
    df: pd.DataFrame,
    *,
    lookback: int = 120,
    horizon: int = 20,
    stride: int = 3,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    min_abs_bias: float = 10.0,
    atr_period: int = 14,
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    profile_calculator: Optional[VolumeProfileCalculator] = None,
    quiet_detector: Optional[QuietPhaseDetector] = None,
    pattern_detector: Optional[VolumePatternDetector] = None,
    scorer: Optional[ConfluenceScorer] = None,
) -> MetaDataset:
    """Build a meta-labeling dataset: primary side + triple-barrier win/loss + features.

    The **primary side** is ``sign(bias_score)`` at events where
    ``|bias_score| >= min_abs_bias`` (i.e. only where the strategy would lean).
    """
    ensure_ohlcv(df, min_bars=lookback + horizon + 2)
    pc = profile_calculator or VolumeProfileCalculator(bin_mode="auto")
    qd = quiet_detector or QuietPhaseDetector()
    pat = pattern_detector or VolumePatternDetector()
    sc = scorer or ConfluenceScorer()

    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    vol = (atr(df["High"], df["Low"], df["Close"], atr_period) / df["Close"]).to_numpy(float)
    n = len(df)

    feats: list[list[float]] = []
    sides: list[int] = []
    ev_pos: list[int] = []
    ts: list = []
    for t in range(lookback - 1, n - horizon, max(1, stride)):
        window = df.iloc[t - lookback + 1 : t + 1]
        try:
            profile = pc.calculate(window, symbol, interval)
            quiet = qd.detect(window, symbol, interval)
            patterns = pat.detect(window, profile=profile, symbol=symbol, interval=interval)
            score = sc.score(window, profile, quiet, patterns, symbol=symbol, interval=interval)
        except (ValueError, ZeroDivisionError):
            continue
        if abs(score.bias_score) < min_abs_bias:      # no primary signal -> no event
            continue
        comps = {c.name: c for c in score.components}
        feats.append([
            comps["value_area"].strength * comps["value_area"].direction,
            comps["key_level"].strength * comps["key_level"].direction,
            comps["quiet"].strength,
            comps["patterns"].strength * comps["patterns"].direction,
            score.setup_quality / 100.0,
            abs(score.bias_score) / 100.0,
        ])
        sides.append(1 if score.bias_score > 0 else -1)
        ev_pos.append(t)
        ts.append(df.index[t])

    ev_pos_arr = np.array(ev_pos, dtype=int)
    side_arr = np.array(sides, dtype=int)
    if ev_pos_arr.size:
        _, ret, win, _ = triple_barrier_labels(
            close, high, low, vol, ev_pos_arr, side_arr, horizon, pt_mult, sl_mult)
    else:
        ret = np.array([], dtype=float)
        win = np.array([], dtype=int)

    is_dt = isinstance(df.index, pd.DatetimeIndex) and ts
    return MetaDataset(
        X=np.array(feats, dtype=float).reshape(-1, len(_META_FEATURES)),
        meta_label=win,
        side=side_arr,
        realized_return=ret,
        feature_names=_META_FEATURES,
        horizon=horizon,
        stride=max(1, stride),
        timestamps=pd.DatetimeIndex(ts) if is_dt else None,
        symbol=symbol,
    )
