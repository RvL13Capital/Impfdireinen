"""Phase 6 — walk-forward backtester with realistic free cost simulation.

* :class:`~vpts.backtest.engine.Backtester` — no-look-ahead event loop over the
  full signal stack.
* :class:`~vpts.backtest.models.BacktestResult` / :class:`~vpts.backtest.models.Trade`
  / :class:`~vpts.backtest.models.CostModel`.
"""
from __future__ import annotations

from vpts.backtest.engine import Backtester
from vpts.backtest.models import BacktestResult, CostModel, Trade

__all__ = ["Backtester", "BacktestResult", "Trade", "CostModel"]
