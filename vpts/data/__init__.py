"""Data access package (free sources only).

Phase 1 ships a robust :class:`~vpts.data.fetcher.MarketDataFetcher` around
``yfinance`` with interval-limit handling, retries and on-disk caching.
"""
from __future__ import annotations

from vpts.data.fetcher import (
    DataFetchError,
    InsufficientDataError,
    MarketDataFetcher,
    NoVolumeError,
)

__all__ = [
    "MarketDataFetcher",
    "DataFetchError",
    "InsufficientDataError",
    "NoVolumeError",
]
