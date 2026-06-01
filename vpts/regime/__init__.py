"""Phase 2 — market-regime analysis.

* :class:`~vpts.regime.quiet.QuietPhaseDetector` — quiet / low-energy phase scoring.
* :class:`~vpts.regime.patterns.VolumePatternDetector` — Dry-up / Accumulation /
  Divergence / Climax recognition, optionally anchored to Phase-1 profile levels.
"""
from __future__ import annotations

from vpts.regime.patterns import (
    VolumePattern,
    VolumePatternDetector,
    VolumePatternResult,
    VolumePatternType,
)
from vpts.regime.quiet import QuietPhaseDetector, QuietPhaseResult, QuietState

__all__ = [
    "QuietPhaseDetector",
    "QuietPhaseResult",
    "QuietState",
    "VolumePatternDetector",
    "VolumePatternResult",
    "VolumePattern",
    "VolumePatternType",
]
