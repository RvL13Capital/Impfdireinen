"""Phase 4 — signal generation.

* :class:`~vpts.signals.generator.SignalGenerator` — turns a confluence score +
  profile into an actionable, explainable trade plan.
* :class:`~vpts.signals.models.TradeSignal` / :class:`~vpts.signals.models.SignalAction`.
"""
from __future__ import annotations

from vpts.signals.generator import SignalGenerator
from vpts.signals.models import SignalAction, TradeSignal

__all__ = ["SignalGenerator", "TradeSignal", "SignalAction"]
