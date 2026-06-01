"""Phase 2 — small, dependency-free technical-indicator helpers.

These intentionally avoid ``pandas-ta`` (fragile on numpy ≥ 2.0 / Python < 3.12)
and implement only what the quiet-phase and volume-pattern detectors need, in
plain pandas/numpy. Everything operates on aligned :class:`pandas.Series` and
returns a Series on the same index, so detectors can compose them freely.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "true_range",
    "atr",
    "rolling_percentile_rank",
    "rolling_slope",
    "bollinger_bandwidth",
    "ensure_ohlcv",
]


def ensure_ohlcv(
    df: pd.DataFrame,
    required: tuple[str, ...] = ("High", "Low", "Close", "Volume"),
    min_bars: int = 1,
) -> None:
    """Validate that *df* is a usable OHLCV frame, else raise ``ValueError``."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("`df` must be a pandas DataFrame.")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required column(s): {missing}.")
    if len(df) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} bars, got {len(df)}. "
            "Use a longer period or a coarser interval."
        )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = ``max(H-L, |H-prevC|, |L-prevC|)``; first bar uses ``H-L``."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return tr.rename("true_range")


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    method: str = "wilder",
) -> pd.Series:
    """Average True Range as a Series.

    ``method="wilder"`` uses Wilder's RMA (``ewm`` with ``alpha=1/period``);
    ``method="sma"`` uses a simple rolling mean.
    """
    tr = true_range(high, low, close)
    if method == "wilder":
        out = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    elif method == "sma":
        out = tr.rolling(period, min_periods=max(2, period // 2)).mean()
    else:
        raise ValueError("atr method must be 'wilder' or 'sma'.")
    return out.rename("atr")


def rolling_percentile_rank(
    series: pd.Series, window: int, min_periods: Optional[int] = None
) -> pd.Series:
    """Trailing percentile rank of each point within its window, in ``(0, 1]``.

    The value is the fraction of the trailing *window* (including the current
    point) that is ``<=`` the current value. A small rank means "low relative to
    recent history" — e.g. a quiet, contracted reading.
    """
    if min_periods is None:
        min_periods = max(3, window // 4)

    def _rank(x: np.ndarray) -> float:
        return float(np.mean(x <= x[-1]))

    return series.rolling(window, min_periods=min_periods).apply(_rank, raw=True)


def rolling_slope(
    series: pd.Series, window: int, min_periods: Optional[int] = None
) -> pd.Series:
    """Rolling ordinary-least-squares slope (change in *series* per bar)."""
    if min_periods is None:
        min_periods = max(2, window // 2)

    def _slope(y: np.ndarray) -> float:
        n = len(y)
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        denom = float(((x - x_mean) ** 2).sum())
        if denom == 0.0:
            return 0.0
        return float(((x - x_mean) * (y - y.mean())).sum() / denom)

    return series.rolling(window, min_periods=min_periods).apply(_slope, raw=True)


def bollinger_bandwidth(
    close: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Bollinger Bandwidth = ``(upper - lower) / mid`` — a range/squeeze gauge.

    Low bandwidth = a tight, compressed (quiet) range relative to price.
    """
    if min_periods is None:
        min_periods = max(2, window // 2)
    mid = close.rolling(window, min_periods=min_periods).mean()
    sd = close.rolling(window, min_periods=min_periods).std(ddof=0)
    return ((2.0 * n_std * sd) / mid.replace(0.0, np.nan)).rename("bandwidth")
