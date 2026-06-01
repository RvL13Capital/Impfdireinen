"""Phase 3 — immutable result objects for the confluence scoring engine.

A :class:`ConfluenceScore` answers two questions about *one moment* in the market:

* **How good is the setup?**  -> ``setup_quality`` in ``0..100`` (how much
  aligned/relevant evidence is present right now).
* **Which way does it lean?** -> ``bias`` (``"bullish"`` / ``"bearish"`` /
  ``"neutral"``) with a signed ``bias_score`` in ``-100..100``.

Both are derived from a transparent list of :class:`ConfluenceComponent` factors,
each carrying its own weight, strength, direction and a short human reason — so
nothing about the score is a black box.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

_ARROW = {1: "↑ bull", -1: "↓ bear", 0: "· neut"}
_SHORT = {1: "bull", -1: "bear", 0: "neut"}


@dataclass(frozen=True)
class ConfluenceComponent:
    """One weighted factor contributing to the overall confluence score.

    Attributes
    ----------
    name:
        Short identifier, e.g. ``"value_area"``.
    weight:
        Relative importance (as configured on the scorer).
    strength:
        How strongly this factor is in play, in ``0..1``.
    direction:
        Directional lean: ``+1`` bullish, ``-1`` bearish, ``0`` neutral.
    reason:
        One concise, human-readable phrase explaining the reading.
    """

    name: str
    weight: float
    strength: float
    direction: int
    reason: str

    @property
    def weighted_strength(self) -> float:
        """``weight × strength`` — contribution to setup quality."""
        return self.weight * self.strength

    @property
    def signed_contribution(self) -> float:
        """``weight × strength × direction`` — contribution to the bias."""
        return self.weight * self.strength * self.direction

    @property
    def bias_label(self) -> str:
        return _SHORT[self.direction]


@dataclass(frozen=True)
class ConfluenceScore:
    """The fused, explainable confluence reading for a single bar."""

    setup_quality: float          # 0..100
    bias: str                     # "bullish" | "bearish" | "neutral"
    bias_score: float             # -100..100 (sign = direction, |.| = conviction)
    components: tuple[ConfluenceComponent, ...]
    price: float
    rationale: str
    timestamp: Optional[pd.Timestamp] = None
    symbol: Optional[str] = None
    interval: Optional[str] = None
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def breakdown(self) -> dict[str, dict]:
        """Return a clean ``{name: {weight, strength, direction, reason}}`` dict."""
        return {
            c.name: {
                "weight": round(c.weight, 4),
                "strength": round(c.strength, 4),
                "direction": c.direction,
                "reason": c.reason,
            }
            for c in self.components
        }

    def as_dict(self) -> dict:
        """Flat, JSON-serialisable summary."""
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "timestamp": None if self.timestamp is None else str(self.timestamp),
            "price": self.price,
            "setup_quality": self.setup_quality,
            "bias": self.bias,
            "bias_score": self.bias_score,
            "rationale": self.rationale,
            "components": self.breakdown(),
        }

    def summary(self) -> str:
        """Readable, aligned multi-line report (rationale first)."""
        sym = self.symbol or "data"
        itv = f" {self.interval}" if self.interval else ""
        lines = [
            f"Confluence — {sym}{itv} @ {self.price:.4f}",
            "-" * 56,
            f"  Setup quality : {self.setup_quality:.0f}/100",
            f"  Directional   : {self.bias.upper()}  (bias {self.bias_score:+.0f})",
            f"  Rationale     : {self.rationale}",
            "  Components:",
        ]
        # Sort the breakdown strongest-first for readability.
        for c in sorted(self.components, key=lambda x: x.weighted_strength, reverse=True):
            bar = _strength_bar(c.strength)
            lines.append(
                f"    {c.name:<12} [w{c.weight:>3.1f}] {bar} {c.strength:.2f}  "
                f"{_ARROW[c.direction]:<7} {c.reason}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _strength_bar(strength: float, width: int = 6) -> str:
    """Render a 0..1 strength as a small block-bar like ``████░░``."""
    strength = 0.0 if strength != strength else max(0.0, min(1.0, strength))  # NaN-safe
    filled = int(round(strength * width))
    return "█" * filled + "░" * (width - filled)
