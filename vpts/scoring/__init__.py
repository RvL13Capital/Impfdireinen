"""Phase 3 — confluence & scoring engine.

* :class:`~vpts.scoring.scorer.ConfluenceScorer` — fuses Phase-1 profile location
  with Phase-2 quiet-regime and volume patterns into a single explainable score.
* :class:`~vpts.scoring.models.ConfluenceScore` /
  :class:`~vpts.scoring.models.ConfluenceComponent` — immutable results.
"""
from __future__ import annotations

from vpts.scoring.models import ConfluenceComponent, ConfluenceScore
from vpts.scoring.scorer import ConfluenceScorer

__all__ = ["ConfluenceScorer", "ConfluenceScore", "ConfluenceComponent"]
