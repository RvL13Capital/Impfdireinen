"""Phase 3 — Confluence & Scoring Engine.

Fuses the Phase-1 volume profile (where price sits vs POC / value area / nearest
HVN-LVN) with the Phase-2 regime read (quiet-phase score + active volume patterns)
into a single, explainable :class:`~vpts.scoring.models.ConfluenceScore`:

* **setup_quality** ``0..100`` — weighted amount of aligned evidence present now.
* **bias** — ``bullish`` / ``bearish`` / ``neutral`` with a signed ``bias_score``.

By construction ``|bias_score| <= setup_quality``: directional conviction can
never exceed the quality of the evidence behind it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from vpts.profile.calculator import VolumeProfileCalculator
from vpts.profile.models import VolumeProfile
from vpts.regime.indicators import atr, ensure_ohlcv
from vpts.regime.patterns import (
    VolumePatternDetector,
    VolumePatternResult,
    VolumePatternType,
)
from vpts.regime.quiet import QuietPhaseDetector, QuietPhaseResult
from vpts.scoring.models import ConfluenceComponent, ConfluenceScore

_DEFAULT_WEIGHTS = {
    "value_area": 1.0,
    "key_level": 1.0,
    "quiet": 1.5,      # the system's edge — quiet phases are weighted higher
    "patterns": 1.5,
}


class ConfluenceScorer:
    """Combine profile location + quiet regime + volume patterns into a score.

    Parameters
    ----------
    weights:
        Per-component weights. Keys: ``value_area``, ``key_level``, ``quiet``,
        ``patterns``. Missing keys fall back to the defaults
        ``{value_area:1, key_level:1, quiet:1.5, patterns:1.5}``.
    neutral_band:
        ``|bias_score|`` below this (in 0..100 units) is reported as
        ``"neutral"`` (default ``10``).
    node_tolerance_atr:
        Price is "at" an HVN/LVN when within this many ATRs of it (default
        ``0.5``).
    pattern_recency:
        Only volume patterns whose anchor bar is within this many bars of the
        end count as "active" (default ``10``).
    atr_period:
        ATR look-back used for level tolerances (default ``14``).
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        neutral_band: float = 10.0,
        node_tolerance_atr: float = 0.5,
        pattern_recency: int = 10,
        atr_period: int = 14,
    ) -> None:
        merged = dict(_DEFAULT_WEIGHTS)
        if weights:
            unknown = set(weights) - set(_DEFAULT_WEIGHTS)
            if unknown:
                raise ValueError(f"Unknown weight key(s): {sorted(unknown)}.")
            merged.update(weights)
        if any(w < 0 for w in merged.values()):
            raise ValueError("weights must be non-negative.")
        if sum(merged.values()) <= 0:
            raise ValueError("weights must not all be zero.")
        if not 0.0 <= neutral_band < 100.0:
            raise ValueError("neutral_band must be in [0, 100).")
        if node_tolerance_atr <= 0:
            raise ValueError("node_tolerance_atr must be > 0.")
        if int(pattern_recency) < 1:
            raise ValueError("pattern_recency must be >= 1.")
        if int(atr_period) < 1:
            raise ValueError("atr_period must be >= 1.")

        self.weights = merged
        self.neutral_band = float(neutral_band)
        self.node_tolerance_atr = float(node_tolerance_atr)
        self.pattern_recency = int(pattern_recency)
        self.atr_period = int(atr_period)

    # ------------------------------------------------------------------ #
    # Convenience: build the Phase 1/2 inputs then score in one call.
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        df: pd.DataFrame,
        *,
        profile: Optional[VolumeProfile] = None,
        quiet_result: Optional[QuietPhaseResult] = None,
        pattern_result: Optional[VolumePatternResult] = None,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> ConfluenceScore:
        """Compute any missing Phase 1/2 inputs with defaults, then score.

        Pass pre-computed *profile* / *quiet_result* / *pattern_result* to reuse
        work and your own detector settings; anything omitted is built here.
        """
        ensure_ohlcv(df, min_bars=2)
        if profile is None:
            profile = VolumeProfileCalculator().calculate(df, symbol, interval)
        if quiet_result is None:
            quiet_result = QuietPhaseDetector().detect(df, symbol, interval)
        if pattern_result is None:
            pattern_result = VolumePatternDetector().detect(
                df, profile=profile, symbol=symbol, interval=interval
            )
        return self.score(df, profile, quiet_result, pattern_result,
                          symbol=symbol, interval=interval)

    # ------------------------------------------------------------------ #
    def score(
        self,
        df: pd.DataFrame,
        profile: VolumeProfile,
        quiet_result: QuietPhaseResult,
        pattern_result: VolumePatternResult,
        *,
        price: Optional[float] = None,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> ConfluenceScore:
        """Score the latest bar from already-computed Phase 1/2 results."""
        ensure_ohlcv(df, min_bars=2)
        if price is None:
            price = float(df["Close"].iloc[-1])
        atr_val = self._latest_atr(df, profile)

        components = [
            self._component_value_area(price, profile),
            self._component_key_level(price, profile, atr_val),
            self._component_quiet(quiet_result),
            self._component_patterns(pattern_result, len(df)),
        ]

        total_w = sum(c.weight for c in components)
        quality = 100.0 * sum(c.weighted_strength for c in components) / total_w
        bias_score = 100.0 * sum(c.signed_contribution for c in components) / total_w

        if bias_score > self.neutral_band:
            bias = "bullish"
        elif bias_score < -self.neutral_band:
            bias = "bearish"
        else:
            bias = "neutral"

        rationale = self._rationale(bias, quality, bias_score, components)
        ts = (
            df.index[-1]
            if isinstance(df.index, pd.DatetimeIndex) and len(df.index)
            else None
        )
        symbol = symbol or profile.symbol
        interval = interval or profile.interval
        return ConfluenceScore(
            setup_quality=round(float(quality), 2),
            bias=bias,
            bias_score=round(float(bias_score), 2),
            components=tuple(components),
            price=float(price),
            rationale=rationale,
            timestamp=ts,
            symbol=symbol,
            interval=interval,
            extra={
                "atr": atr_val,
                "weights": dict(self.weights),
                "quiet_score": float(quiet_result.latest.quiet_score),
                "is_quiet": bool(quiet_result.latest.is_quiet),
            },
        )

    # ------------------------------------------------------------------ #
    # Components
    # ------------------------------------------------------------------ #
    def _component_value_area(
        self, price: float, profile: VolumeProfile
    ) -> ConfluenceComponent:
        """Where price sits relative to the value area."""
        val, vah = profile.val, profile.vah
        width = max(vah - val, 1e-9)
        loc = (price - val) / width  # 0 at VAL, 1 at VAH

        if 0.0 <= loc <= 1.0:
            edge = abs(loc - 0.5) * 2.0  # 0 centre … 1 at an edge
            strength = 0.2 + 0.7 * edge
            if loc <= 0.45:
                direction, where = 1, f"near value-area low (VAL {val:.2f}) — bounce zone"
            elif loc >= 0.55:
                direction, where = -1, f"near value-area high (VAH {vah:.2f}) — rejection zone"
            else:
                direction, where = 0, f"balanced inside the value area (POC {profile.poc:.2f})"
        elif loc < 0.0:
            # Continuous with the inside-edge strength (0.9) and rising with the
            # stretch (more extended below value = stronger reversion pull).
            direction = 1
            strength = min(0.98, 0.9 + abs(loc) * 0.25)
            where = f"below value (under VAL {val:.2f}) — stretched, reversion pull up"
        else:  # loc > 1
            direction = -1
            strength = min(0.98, 0.9 + (loc - 1.0) * 0.25)
            where = f"above value (over VAH {vah:.2f}) — stretched, reversion pull down"

        return ConfluenceComponent(
            "value_area", self.weights["value_area"], float(strength), direction, where
        )

    def _component_key_level(
        self, price: float, profile: VolumeProfile, atr_val: float
    ) -> ConfluenceComponent:
        """Proximity to the nearest HVN (support/resistance) or LVN (air pocket)."""
        tol = self.node_tolerance_atr * atr_val
        hvn = profile.nearest_node(price, "HVN")
        lvn = profile.nearest_node(price, "LVN")
        d_hvn = abs(price - hvn.price) if hvn else np.inf
        d_lvn = abs(price - lvn.price) if lvn else np.inf

        if min(d_hvn, d_lvn) > tol or not np.isfinite(min(d_hvn, d_lvn)):
            return ConfluenceComponent(
                "key_level", self.weights["key_level"], 0.1, 0,
                "no major volume node nearby",
            )
        if d_hvn <= d_lvn:
            strength = 0.3 + 0.6 * (1.0 - d_hvn / tol)
            if price >= hvn.price:
                direction, reason = 1, f"holding above HVN {hvn.price:.2f} (support)"
            else:
                direction, reason = -1, f"capped below HVN {hvn.price:.2f} (resistance)"
        else:
            strength = 0.3 + 0.5 * (1.0 - d_lvn / tol)
            direction, reason = 0, f"inside LVN {lvn.price:.2f} (air pocket — fast moves)"
        return ConfluenceComponent(
            "key_level", self.weights["key_level"], float(strength), direction, reason
        )

    def _component_quiet(self, quiet_result: QuietPhaseResult) -> ConfluenceComponent:
        """Quiet-phase regime — a non-directional quality amplifier (the edge)."""
        state = quiet_result.latest
        strength = max(0.0, min(1.0, state.quiet_score / 100.0))
        if state.is_quiet:
            reason = f"quiet coil (score {state.quiet_score:.0f}/100) — primed for a move"
        else:
            reason = f"active market (quiet score {state.quiet_score:.0f}/100)"
        return ConfluenceComponent("quiet", self.weights["quiet"], strength, 0, reason)

    def _component_patterns(
        self, pattern_result: VolumePatternResult, n_bars: int
    ) -> ConfluenceComponent:
        """Net direction & strength of recently active volume patterns."""
        cutoff = n_bars - 1 - self.pattern_recency
        active = [p for p in pattern_result.patterns if p.index_pos >= cutoff]
        if not active:
            return ConfluenceComponent(
                "patterns", self.weights["patterns"], 0.0, 0,
                "no recent volume patterns",
            )

        signed = 0.0
        total = 0.0
        peak = 0.0
        for p in active:
            direction, strength = _pattern_dir_strength(p)
            recency = 1.0 - (n_bars - 1 - p.index_pos) / max(self.pattern_recency, 1)
            recency = max(0.0, min(1.0, recency))
            w = strength * recency
            signed += direction * w
            total += w
            peak = max(peak, w)

        if total <= 0:
            net_dir = 0
        else:
            ratio = signed / total
            net_dir = 1 if ratio > 0.15 else -1 if ratio < -0.15 else 0

        latest = max(active, key=lambda p: p.index_pos)
        reason = _shorten(latest.explanation)
        if len(active) > 1:
            reason = f"{len(active)} active; latest: {reason}"
        return ConfluenceComponent(
            "patterns", self.weights["patterns"],
            float(max(0.0, min(1.0, peak))), net_dir, reason,
        )

    # ------------------------------------------------------------------ #
    def _latest_atr(self, df: pd.DataFrame, profile: VolumeProfile) -> float:
        series = atr(df["High"], df["Low"], df["Close"], self.atr_period)
        val = float(series.iloc[-1]) if pd.notna(series.iloc[-1]) else np.nan
        if not np.isfinite(val) or val <= 0:
            val = float((df["High"] - df["Low"]).tail(self.atr_period).mean())
        if not np.isfinite(val) or val <= 0:
            val = max(profile.bin_size, 1e-9)
        return val

    @staticmethod
    def _rationale(
        bias: str,
        quality: float,
        bias_score: float,
        components: list[ConfluenceComponent],
    ) -> str:
        """Compose 1-2 concise, trader-facing sentences."""
        ranked = sorted(components, key=lambda c: c.weighted_strength, reverse=True)
        leads = [c.reason for c in ranked if c.strength >= 0.25][:2]
        lead = {
            "bullish": "Bullish setup",
            "bearish": "Bearish setup",
            "neutral": "Neutral / balanced setup",
        }[bias]
        body = "; ".join(leads) if leads else "little decisive evidence yet"
        sentence = f"{lead} (quality {quality:.0f}/100): {body}."

        # Flag the strongest factor pulling against the net bias, if any.
        if bias != "neutral":
            net = 1 if bias == "bullish" else -1
            against = [
                c for c in ranked if c.direction == -net and c.strength >= 0.4
            ]
            if against:
                sentence += f" Caution: {against[0].reason}."
        return sentence


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pattern_dir_strength(p) -> tuple[int, float]:
    """Map a :class:`~vpts.regime.patterns.VolumePattern` to (direction, strength)."""
    t = p.type
    s = float(p.strength) if np.isfinite(p.strength) else 0.0
    if t == VolumePatternType.ACCUMULATION:
        return 1, min(1.0, max(0.4, s))
    if t == VolumePatternType.DRY_UP:
        return 0, min(1.0, max(0.3, s))  # coiling: boosts quality, no direction
    if t == VolumePatternType.DIVERGENCE:
        return (1 if p.direction == "bullish" else -1), 0.6
    if t == VolumePatternType.CLIMAX:
        return (1 if p.direction == "bullish" else -1), min(1.0, max(0.3, (s - 1.0) / 3.0))
    return 0, 0.0


def _shorten(text: str, limit: int = 70) -> str:
    """Trim a pattern explanation to its lead clause for compact rationales."""
    head = text.split(" — ")[0].split(":")[0]
    return (head[:limit] + "…") if len(head) > limit else head
