"""Quiet-Volume — a free, modular Volume Profile trading system.

The package is built in *phases*; each phase is a self-contained, importable
module so the pieces connect seamlessly:

    Phase 1  vpts.profile   -> Volume Profile Calculator (POC, VAH/VAL, HVN/LVN)
    Phase 2  vpts.regime    -> Quiet-phase detector + volume pattern recognition
    Phase 3  vpts.scoring   -> Confluence & scoring engine (0-100)
    Phase 4  vpts.signals   -> Signal generator with natural-language reasoning
    Phase 5  vpts.dashboard -> Streamlit dashboard
    Phase 6  vpts.backtest  -> Backtester with realistic (free) cost simulation

All six phases are implemented. (The Phase 5 dashboard depends on the optional
``streamlit``/``plotly`` extras and is therefore imported on demand from
:mod:`vpts.dashboard`, not at this package root.)

Typical Phase 1 usage
----------------------
>>> from vpts import MarketDataFetcher, VolumeProfileCalculator
>>> df = MarketDataFetcher().fetch("AAPL", period="6mo", interval="1d")
>>> profile = VolumeProfileCalculator(num_bins=100).calculate(df)
>>> print(profile.summary())
"""
from __future__ import annotations

__version__ = "1.3.0"  # + triple-barrier meta-labeling (CPCV-evaluated)

# Re-export the public API at the package root for convenience.
from vpts.data.fetcher import (
    DataFetchError,
    InsufficientDataError,
    MarketDataFetcher,
    NoVolumeError,
)
from vpts.profile.calculator import VolumeProfileCalculator
from vpts.profile.models import VolumeNode, VolumeProfile
from vpts.regime.patterns import (
    VolumePattern,
    VolumePatternDetector,
    VolumePatternResult,
    VolumePatternType,
)
from vpts.regime.quiet import QuietPhaseDetector, QuietPhaseResult, QuietState
from vpts.scoring.models import ConfluenceComponent, ConfluenceScore
from vpts.scoring.scorer import ConfluenceScorer
from vpts.signals.generator import SignalGenerator
from vpts.signals.models import SignalAction, TradeSignal
from vpts.backtest.engine import Backtester
from vpts.backtest.models import BacktestResult, CostModel, Trade
from vpts.validation.cpcv import CombinatorialPurgedCV
from vpts.validation.models import CPCVResult, GroupResult
from vpts.ml.factor_model import RidgeFactorModel, build_factor_dataset, cpcv_factor_eval
from vpts.ml.labeling import build_meta_dataset, triple_barrier_labels
from vpts.ml.meta_model import LogisticMetaModel, cpcv_meta_eval
from vpts.ml.models import FactorCVResult, FactorDataset, MetaCVResult, MetaDataset

__all__ = [
    "__version__",
    # data
    "MarketDataFetcher",
    "DataFetchError",
    "InsufficientDataError",
    "NoVolumeError",
    # profile (Phase 1)
    "VolumeProfileCalculator",
    "VolumeProfile",
    "VolumeNode",
    # regime (Phase 2)
    "QuietPhaseDetector",
    "QuietPhaseResult",
    "QuietState",
    "VolumePatternDetector",
    "VolumePatternResult",
    "VolumePattern",
    "VolumePatternType",
    # scoring (Phase 3)
    "ConfluenceScorer",
    "ConfluenceScore",
    "ConfluenceComponent",
    # signals (Phase 4)
    "SignalGenerator",
    "TradeSignal",
    "SignalAction",
    # backtest (Phase 6)
    "Backtester",
    "BacktestResult",
    "Trade",
    "CostModel",
    # validation (CPCV)
    "CombinatorialPurgedCV",
    "CPCVResult",
    "GroupResult",
    # ml (learned factor weights)
    "build_factor_dataset",
    "RidgeFactorModel",
    "cpcv_factor_eval",
    "FactorDataset",
    "FactorCVResult",
    # ml (triple-barrier meta-labeling)
    "triple_barrier_labels",
    "build_meta_dataset",
    "LogisticMetaModel",
    "cpcv_meta_eval",
    "MetaDataset",
    "MetaCVResult",
]
