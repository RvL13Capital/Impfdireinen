"""Quiet-Volume — a free, modular Volume Profile trading system.

The package is built in *phases*; each phase is a self-contained, importable
module so the pieces connect seamlessly:

    Phase 1  vpts.profile   -> Volume Profile Calculator (POC, VAH/VAL, HVN/LVN)
    Phase 2  vpts.regime    -> Quiet-phase detector + volume pattern recognition
    Phase 3  vpts.scoring   -> Confluence & scoring engine (0-100)
    Phase 4  vpts.signals   -> Signal generator with natural-language reasoning
    Phase 5  vpts.dashboard -> Streamlit dashboard
    Phase 6  vpts.backtest  -> Backtester with realistic (free) cost simulation

Only Phase 1 is implemented so far.

Typical Phase 1 usage
----------------------
>>> from vpts import MarketDataFetcher, VolumeProfileCalculator
>>> df = MarketDataFetcher().fetch("AAPL", period="6mo", interval="1d")
>>> profile = VolumeProfileCalculator(num_bins=100).calculate(df)
>>> print(profile.summary())
"""
from __future__ import annotations

__version__ = "0.1.0"  # Phase 1

# Re-export the public Phase 1 API at the package root for convenience.
from vpts.data.fetcher import (
    DataFetchError,
    InsufficientDataError,
    MarketDataFetcher,
    NoVolumeError,
)
from vpts.profile.calculator import VolumeProfileCalculator
from vpts.profile.models import VolumeNode, VolumeProfile

__all__ = [
    "__version__",
    # data
    "MarketDataFetcher",
    "DataFetchError",
    "InsufficientDataError",
    "NoVolumeError",
    # profile
    "VolumeProfileCalculator",
    "VolumeProfile",
    "VolumeNode",
]
