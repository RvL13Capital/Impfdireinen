"""Phase 4 — Signal Generator with trade suggestions.

Turns a Phase-3 :class:`~vpts.scoring.models.ConfluenceScore` (plus the Phase-1
profile, for structure) into a concrete, explainable
:class:`~vpts.signals.models.TradeSignal`:

* **Gating** — only act when setup quality and directional conviction clear
  configurable thresholds (and, optionally, only in quiet phases).
* **Plan** — entry / stop / targets built from volume-profile structure:

  ``"reversion"`` (default)
      Fade value-area edges / HVN support-resistance back toward the POC.
  ``"breakout"``
      Trade the bias direction as price expands beyond the coil's edge.
* **Risk** — structure-based stop (ATR fallback), a minimum R:R filter, and a
  free **fixed-fractional** position size (``risk %`` ÷ stop distance).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from vpts.profile.calculator import VolumeProfileCalculator
from vpts.profile.models import VolumeProfile
from vpts.regime.indicators import ensure_ohlcv
from vpts.regime.patterns import VolumePatternDetector
from vpts.regime.quiet import QuietPhaseDetector
from vpts.scoring.models import ConfluenceScore
from vpts.scoring.scorer import ConfluenceScorer
from vpts.signals.models import SignalAction, TradeSignal

_VALID_STYLES = ("reversion", "breakout")


class SignalGenerator:
    """Generate explainable trade signals from a confluence score + profile.

    Parameters
    ----------
    style:
        ``"reversion"`` (default) or ``"breakout"`` (see module docstring).
    min_quality:
        Minimum ``setup_quality`` (0–100) required to trade.
    min_abs_bias:
        Minimum ``|bias_score|`` (0–100) of directional conviction required.
    require_quiet:
        If ``True``, only signal when the bar is in a quiet phase.
    min_quiet_score:
        Quiet-score threshold used when *require_quiet* is set.
    min_rr:
        Minimum risk:reward (to the first target); weaker plans become NO_TRADE.
    atr_stop_mult:
        ATR multiple for the stop when no structural level is available, and the
        coil-stop distance for breakouts.
    stop_buffer_atr:
        Extra ATR buffer placed beyond the structural level for the stop.
    breakout_buffer_atr:
        ATR buffer beyond the level for a breakout entry trigger.
    target_rr:
        R:R used for the measured-move target when structure runs out.
    risk_fraction:
        Fraction of account equity risked per trade (default ``0.01`` = 1%).
    account_equity:
        Default account size used for position sizing.
    """

    def __init__(
        self,
        style: str = "reversion",
        min_quality: float = 55.0,
        min_abs_bias: float = 15.0,
        require_quiet: bool = False,
        min_quiet_score: float = 60.0,
        min_rr: float = 1.5,
        atr_stop_mult: float = 1.5,
        stop_buffer_atr: float = 0.25,
        breakout_buffer_atr: float = 0.10,
        target_rr: float = 2.0,
        risk_fraction: float = 0.01,
        account_equity: float = 10_000.0,
    ) -> None:
        if style not in _VALID_STYLES:
            raise ValueError(f"style must be one of {_VALID_STYLES}, got {style!r}.")
        if not 0.0 <= min_quality <= 100.0:
            raise ValueError("min_quality must be in [0, 100].")
        if not 0.0 <= min_abs_bias <= 100.0:
            raise ValueError("min_abs_bias must be in [0, 100].")
        if min_rr <= 0 or target_rr <= 0:
            raise ValueError("min_rr and target_rr must be > 0.")
        if atr_stop_mult <= 0:
            raise ValueError("atr_stop_mult must be > 0.")
        if stop_buffer_atr < 0 or breakout_buffer_atr < 0:
            raise ValueError("buffer ATR multiples must be >= 0.")
        if not 0.0 < risk_fraction < 1.0:
            raise ValueError("risk_fraction must be in (0, 1).")
        if account_equity <= 0:
            raise ValueError("account_equity must be > 0.")

        self.style = style
        self.min_quality = float(min_quality)
        self.min_abs_bias = float(min_abs_bias)
        self.require_quiet = bool(require_quiet)
        self.min_quiet_score = float(min_quiet_score)
        self.min_rr = float(min_rr)
        self.atr_stop_mult = float(atr_stop_mult)
        self.stop_buffer_atr = float(stop_buffer_atr)
        self.breakout_buffer_atr = float(breakout_buffer_atr)
        self.target_rr = float(target_rr)
        self.risk_fraction = float(risk_fraction)
        self.account_equity = float(account_equity)

    # ------------------------------------------------------------------ #
    def analyze(
        self,
        df: pd.DataFrame,
        *,
        scorer: Optional[ConfluenceScorer] = None,
        account_equity: Optional[float] = None,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> TradeSignal:
        """One-call pipeline: build profile/regime/score from *df*, then signal."""
        ensure_ohlcv(df, min_bars=2)
        scorer = scorer or ConfluenceScorer()
        profile = VolumeProfileCalculator().calculate(df, symbol, interval)
        quiet = QuietPhaseDetector().detect(df, symbol, interval)
        patterns = VolumePatternDetector().detect(
            df, profile=profile, symbol=symbol, interval=interval
        )
        score = scorer.score(df, profile, quiet, patterns,
                             symbol=symbol, interval=interval)
        return self.from_score(score, profile, account_equity=account_equity)

    # ------------------------------------------------------------------ #
    def from_score(
        self,
        score: ConfluenceScore,
        profile: VolumeProfile,
        *,
        atr: Optional[float] = None,
        account_equity: Optional[float] = None,
    ) -> TradeSignal:
        """Build a :class:`TradeSignal` from a pre-computed confluence score."""
        equity = float(account_equity) if account_equity is not None else self.account_equity
        atr_val = float(atr) if atr is not None else float(score.extra.get("atr", 0.0) or 0.0)
        if not np.isfinite(atr_val) or atr_val <= 0:
            atr_val = max(profile.bin_size, 1e-9)

        direction = {"bullish": 1, "bearish": -1}.get(score.bias, 0)

        # --- Gating ---------------------------------------------------- #
        reject = self._gate(score, direction)
        if reject is not None:
            return self._no_trade(score, reject, equity)

        # --- Build the plan from profile structure --------------------- #
        entry, stop, targets = self._plan(direction, score.price, profile, atr_val)
        if entry is None or stop is None or not targets:
            return self._no_trade(score, "could not build a valid level plan", equity)

        risk = abs(entry - stop)
        if risk <= 0:
            return self._no_trade(score, "degenerate stop (zero risk)", equity)

        rr = abs(targets[0] - entry) / risk
        if rr < self.min_rr:
            return self._no_trade(
                score, f"risk:reward {rr:.2f} below minimum {self.min_rr:.2f}", equity
            )

        risk_amount = equity * self.risk_fraction
        size = math.floor(risk_amount / risk) if risk > 0 else 0
        if size <= 0:
            return self._no_trade(
                score,
                f"risk budget ({risk_amount:.2f}) too small to size one unit "
                f"at {risk:.2f}/unit risk",
                equity,
            )
        action = SignalAction.LONG if direction == 1 else SignalAction.SHORT
        rationale = self._rationale(action, score, entry, stop, targets, rr)

        return TradeSignal(
            action=action,
            style=self.style,
            price=score.price,
            rationale=rationale,
            entry=round(entry, 6),
            stop=round(stop, 6),
            targets=tuple(round(t, 6) for t in targets),
            risk_per_unit=round(risk, 6),
            risk_reward_ratio=round(rr, 3),
            suggested_size=float(size),
            risk_amount=round(risk_amount, 2),
            account_equity=equity,
            setup_quality=score.setup_quality,
            bias=score.bias,
            bias_score=score.bias_score,
            timestamp=score.timestamp,
            symbol=score.symbol,
            interval=score.interval,
            extra={"atr": atr_val},
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _gate(self, score: ConfluenceScore, direction: int) -> Optional[str]:
        """Return a rejection reason, or ``None`` if the setup may be traded."""
        if direction == 0 or score.bias == "neutral":
            return "no directional bias"
        if score.setup_quality < self.min_quality:
            return (
                f"setup quality {score.setup_quality:.0f} below "
                f"minimum {self.min_quality:.0f}"
            )
        if abs(score.bias_score) < self.min_abs_bias:
            return (
                f"directional conviction {abs(score.bias_score):.0f} below "
                f"minimum {self.min_abs_bias:.0f}"
            )
        if self.require_quiet:
            qs = float(score.extra.get("quiet_score", 0.0))
            if qs < self.min_quiet_score:
                return f"not a quiet phase (quiet score {qs:.0f})"
        return None

    def _plan(
        self, direction: int, price: float, profile: VolumeProfile, atr: float
    ) -> tuple[Optional[float], Optional[float], list[float]]:
        """Return ``(entry, stop, targets)`` from profile structure for *direction*."""
        # Significant levels price reacts to (HVN/POC/value edges). LVNs are
        # deliberately excluded as targets — price slides *through* low-volume gaps.
        sig = sorted(
            {profile.poc, profile.vah, profile.val}
            | {n.price for n in profile.hvn}
        )
        eps = 0.05 * atr
        above = [lvl for lvl in sig if lvl > price + eps]
        below = [lvl for lvl in sig if lvl < price - eps]

        if self.style == "reversion":
            # Reversion targets the value magnet (POC) and the far value edge —
            # not just the closest node, which may be only inches away.
            if direction == 1:  # fade up from support toward value
                entry = price
                support = max(below) if below else None
                stop = (support - self.stop_buffer_atr * atr) if support is not None \
                    else entry - self.atr_stop_mult * atr
                targets = [lvl for lvl in (profile.poc, profile.vah) if lvl > price + eps]
            else:               # fade down from resistance toward value
                entry = price
                resistance = min(above) if above else None
                stop = (resistance + self.stop_buffer_atr * atr) if resistance is not None \
                    else entry + self.atr_stop_mult * atr
                targets = [lvl for lvl in (profile.poc, profile.val) if lvl < price - eps]
        else:  # breakout — trade the expansion beyond the coil edge
            if direction == 1:
                trigger = min(above) if above else None
                entry = (trigger + self.breakout_buffer_atr * atr) if trigger is not None \
                    else price + self.breakout_buffer_atr * atr
                stop = entry - self.atr_stop_mult * atr
                targets = [lvl for lvl in sig if lvl > entry + eps][:1]
            else:
                trigger = max(below) if below else None
                entry = (trigger - self.breakout_buffer_atr * atr) if trigger is not None \
                    else price - self.breakout_buffer_atr * atr
                stop = entry + self.atr_stop_mult * atr
                targets = sorted([lvl for lvl in sig if lvl < entry - eps], reverse=True)[:1]

        # Guarantee a correctly-placed stop and at least one target (measured move).
        if direction == 1 and (stop is None or stop >= entry):
            stop = entry - self.atr_stop_mult * atr
        if direction == -1 and (stop is None or stop <= entry):
            stop = entry + self.atr_stop_mult * atr

        targets = [t for t in targets if direction * (t - entry) > eps]
        if not targets:
            risk = abs(entry - stop)
            targets = [entry + direction * risk * self.target_rr]

        # Order nearest-first in the trade direction.
        targets = sorted(set(targets), reverse=(direction == -1))
        return entry, stop, targets[:2]

    def _no_trade(
        self, score: ConfluenceScore, reason: str, equity: float
    ) -> TradeSignal:
        return TradeSignal(
            action=SignalAction.NO_TRADE,
            style=self.style,
            price=score.price,
            rationale=reason,
            account_equity=equity,
            setup_quality=score.setup_quality,
            bias=score.bias,
            bias_score=score.bias_score,
            timestamp=score.timestamp,
            symbol=score.symbol,
            interval=score.interval,
        )

    def _rationale(
        self,
        action: SignalAction,
        score: ConfluenceScore,
        entry: float,
        stop: float,
        targets: list[float],
        rr: float,
    ) -> str:
        """Compose a concise 'why' from the confluence read and the plan."""
        verb = "Long" if action == SignalAction.LONG else "Short"
        plan = "fade toward value" if self.style == "reversion" else "trade the breakout"
        return (
            f"{verb} ({self.style}, {plan}): {score.rationale} "
            f"Plan: enter {entry:.2f}, stop {stop:.2f}, "
            f"first target {targets[0]:.2f} (R:R {rr:.2f})."
        )
