"""Phase 1 — robust, free market-data fetching via ``yfinance``.

``yfinance`` (Yahoo Finance) is free but quirky, especially for intraday data:

* **Interval lookback limits** — Yahoo only serves a limited history per
  interval (e.g. ``1m`` ≈ last 7 days, ``5m/15m/30m`` ≈ 60 days,
  ``1h`` ≈ 730 days). Asking for more silently returns an empty frame.
* **Flaky responses** — transient network errors and empty payloads are common.
* **Inconsistent columns** — depending on version, a single-ticker download may
  return either flat columns or a ``(field, ticker)`` ``MultiIndex``.

:class:`MarketDataFetcher` wraps all of that with period clamping, retries with
exponential backoff, column normalisation, validation, and an on-disk cache so
repeated runs (and Yahoo rate limits) stay friendly — all with zero paid
services.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Final, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class DataFetchError(RuntimeError):
    """Raised when data cannot be retrieved after all retries."""


class InsufficientDataError(DataFetchError):
    """Raised when fewer than the required number of bars are returned."""


class NoVolumeError(DataFetchError):
    """Raised when the returned data has no usable (positive) volume."""


# --------------------------------------------------------------------------- #
# Fetcher
# --------------------------------------------------------------------------- #
class MarketDataFetcher:
    """Fetch and cache clean OHLCV data from Yahoo Finance (free).

    Parameters
    ----------
    cache_dir:
        Directory for the on-disk cache. Created on demand.
    cache_ttl:
        Cache lifetime in seconds (default 1 hour). Set to ``0`` to disable
        reading from the cache (fresh data is still written).
    max_retries:
        Number of download attempts before giving up.
    retry_backoff:
        Base seconds for exponential backoff between retries (2, 4, 8, …).
    auto_adjust:
        Pass-through to ``yfinance``; adjusts OHLC for splits/dividends.
    min_bars:
        Minimum acceptable number of bars; fewer raises
        :class:`InsufficientDataError`.
    """

    #: Maximum lookback Yahoo serves per intraday interval, in days.
    #: ``None`` means effectively unlimited (daily and coarser).
    INTERVAL_MAX_DAYS: Final[dict[str, Optional[int]]] = {
        "1m": 7,
        "2m": 60,
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "60m": 730,
        "90m": 60,
        "1h": 730,
        "1d": None,
        "5d": None,
        "1wk": None,
        "1mo": None,
        "3mo": None,
    }

    #: Roughly how many calendar days a ``period`` string represents (for clamp
    #: comparisons only — Yahoo still does the real windowing).
    _PERIOD_DAYS: Final[dict[str, int]] = {
        "1d": 1,
        "5d": 5,
        "7d": 7,
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 366,
        "2y": 730,  # ~730d matches the 1h/60m Yahoo limit, so 2y of 1h is allowed
        "5y": 1827,
        "10y": 3653,
        "ytd": 366,
        "max": 1_000_000,
    }

    def __init__(
        self,
        cache_dir: str | Path = ".cache/vpts",
        cache_ttl: int = 3600,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
        auto_adjust: bool = True,
        min_bars: int = 20,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_ttl = int(cache_ttl)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff = float(retry_backoff)
        self.auto_adjust = bool(auto_adjust)
        self.min_bars = int(min_bars)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fetch(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return a clean OHLCV DataFrame for *symbol*.

        Parameters
        ----------
        symbol:
            Yahoo ticker, e.g. ``"AAPL"``, ``"SPY"``, ``"^GDAXI"``, ``"BTC-USD"``.
        period:
            Look-back window (``"1d"``, ``"5d"``, ``"1mo"``, ``"6mo"``, ``"1y"``,
            ``"max"``, …). Ignored if *start*/*end* are given. Automatically
            clamped to the interval's Yahoo limit.
        interval:
            Bar size (``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"1d"``, …).
        start, end:
            Explicit date bounds (``"YYYY-MM-DD"``). Override *period*.
        use_cache:
            Read from the on-disk cache when a fresh entry exists.
        force_refresh:
            Ignore any cached entry and re-download (the fresh result is cached).

        Returns
        -------
        pandas.DataFrame
            Indexed by timestamp with columns ``Open, High, Low, Close, Volume``.

        Raises
        ------
        ValueError
            For an unknown *interval*.
        InsufficientDataError, NoVolumeError, DataFetchError
            On data-quality or network failures.
        """
        interval = interval.lower()
        if interval not in self.INTERVAL_MAX_DAYS:
            raise ValueError(
                f"Unsupported interval {interval!r}. "
                f"Valid: {sorted(self.INTERVAL_MAX_DAYS)}."
            )

        using_range = bool(start or end)
        if not using_range:
            period = self._clamp_period(period, interval)

        cache_path = self._cache_path(symbol, period, interval, start, end)
        if use_cache and not force_refresh:
            cached = self._read_cache(cache_path)
            if cached is not None:
                logger.info(
                    "Cache hit for %s [%s/%s] (%d bars).",
                    symbol,
                    period if not using_range else f"{start}->{end}",
                    interval,
                    len(cached),
                )
                return cached

        raw = self._download_with_retries(
            symbol, period, interval, start, end, using_range
        )
        df = self._normalize(raw, symbol)
        self._validate(df, symbol)
        self._write_cache(cache_path, df)
        return df

    # ------------------------------------------------------------------ #
    # Period / interval handling
    # ------------------------------------------------------------------ #
    @classmethod
    def _clamp_period(cls, period: str, interval: str) -> str:
        """Clamp *period* down to what Yahoo will actually serve for *interval*.

        Pure function (no network) — used by :meth:`fetch` and unit-tested
        directly.
        """
        max_days = cls.INTERVAL_MAX_DAYS.get(interval)
        if max_days is None:
            return period  # daily+ : no intraday cap

        requested_days = cls._PERIOD_DAYS.get(period.lower())
        if requested_days is None:
            logger.warning("Unknown period %r; passing through unchanged.", period)
            return period

        if requested_days <= max_days:
            return period

        clamped = cls._largest_period_within(max_days)
        logger.warning(
            "Interval %r only serves ~%d days of history on Yahoo; "
            "clamping period %r -> %r.",
            interval,
            max_days,
            period,
            clamped,
        )
        return clamped

    @classmethod
    def _largest_period_within(cls, max_days: int) -> str:
        """Return the largest standard period string that fits in *max_days*."""
        candidates = {
            k: v
            for k, v in cls._PERIOD_DAYS.items()
            if v <= max_days and k not in ("max", "ytd")
        }
        return max(candidates, key=candidates.get) if candidates else "1d"

    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #
    def _download_with_retries(
        self,
        symbol: str,
        period: str,
        interval: str,
        start: Optional[str],
        end: Optional[str],
        using_range: bool,
    ) -> pd.DataFrame:
        """Call yfinance with retries + exponential backoff."""
        import yfinance as yf  # lazy import keeps the package importable offline

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                ticker = yf.Ticker(symbol)
                if using_range:
                    raw = ticker.history(
                        start=start,
                        end=end,
                        interval=interval,
                        auto_adjust=self.auto_adjust,
                        actions=False,
                    )
                else:
                    raw = ticker.history(
                        period=period,
                        interval=interval,
                        auto_adjust=self.auto_adjust,
                        actions=False,
                    )
                if raw is not None and not raw.empty:
                    return raw
                last_err = DataFetchError("Yahoo returned an empty dataset.")
            except Exception as exc:  # noqa: BLE001 - retry on any download error
                last_err = exc
                logger.debug("Attempt %d/%d failed: %s", attempt, self.max_retries, exc)

            if attempt < self.max_retries:
                delay = self.retry_backoff * (2 ** (attempt - 1))
                logger.info("Retrying %s in %.0fs …", symbol, delay)
                time.sleep(delay)

        raise DataFetchError(
            f"Failed to download {symbol!r} ({interval}) after "
            f"{self.max_retries} attempt(s): {last_err}"
        )

    # ------------------------------------------------------------------ #
    # Cleaning & validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Flatten columns, standardise names, sort, and drop bad rows."""
        df = raw.copy()

        # Flatten a (field, ticker) MultiIndex down to the field level.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardise capitalisation: Open/High/Low/Close/Volume.
        rename = {c: str(c).strip().title() for c in df.columns}
        df = df.rename(columns=rename)
        # "Adj Close" -> title() => "Adj Close" already; keep only what we need.

        keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        df = df[keep]

        # Coerce numerics, sort by time, de-duplicate the index.
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.dropna(subset=[c for c in ("High", "Low", "Close") if c in df.columns])
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].fillna(0.0)
        df.attrs["symbol"] = symbol
        return df

    def _validate(self, df: pd.DataFrame, symbol: str) -> None:
        """Raise informative errors for empty / short / volume-less data."""
        required = {"High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise DataFetchError(
                f"{symbol}: missing columns after normalisation: {sorted(missing)}."
            )
        if len(df) < self.min_bars:
            raise InsufficientDataError(
                f"{symbol}: only {len(df)} bar(s) returned (need >= {self.min_bars}). "
                "Try a longer period or a coarser interval."
            )
        if float(df["Volume"].sum()) <= 0:
            raise NoVolumeError(
                f"{symbol}: data has no volume. Cash indices (e.g. '^GDAXI', "
                "'^GSPC') report zero volume on Yahoo and cannot form a volume "
                "profile. Use a tradable proxy: an ETF ('SPY', 'EWG') or a "
                "futures contract."
            )

    # ------------------------------------------------------------------ #
    # Cache (pickle: no extra dependency, preserves dtypes & tz-aware index)
    # ------------------------------------------------------------------ #
    def _cache_path(
        self,
        symbol: str,
        period: str,
        interval: str,
        start: Optional[str],
        end: Optional[str],
    ) -> Path:
        safe_symbol = re.sub(r"[^A-Za-z0-9._-]", "_", symbol)
        key = f"{symbol}|{period}|{interval}|{start}|{end}|adj={self.auto_adjust}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:10]
        return self.cache_dir / f"{safe_symbol}_{interval}_{digest}.pkl"

    def _read_cache(self, path: Path) -> Optional[pd.DataFrame]:
        if self.cache_ttl <= 0 or not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.cache_ttl:
            logger.debug("Cache stale (%.0fs > %ds): %s", age, self.cache_ttl, path.name)
            return None
        try:
            return pd.read_pickle(path)
        except Exception as exc:  # noqa: BLE001 - corrupt cache shouldn't be fatal
            logger.warning("Ignoring unreadable cache %s: %s", path.name, exc)
            return None

    def _write_cache(self, path: Path, df: pd.DataFrame) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_pickle(path)
        except Exception as exc:  # noqa: BLE001 - caching is best-effort
            logger.warning("Could not write cache %s: %s", path.name, exc)
