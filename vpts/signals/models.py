"""Phase 4 — immutable result objects for the signal generator.

A :class:`TradeSignal` is the system's final, actionable output for one bar:
a clear ``LONG`` / ``SHORT`` / ``NO_TRADE`` call with a concrete trade plan
(entry, stop, targets), an explicit **risk:reward**, a free **fixed-fractional
position size**, and a journal-ready ``explain()`` write-up of *why*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class SignalAction(str, Enum):
    """The actionable call for the current bar."""

    LONG = "long"
    SHORT = "short"
    NO_TRADE = "no_trade"


@dataclass(frozen=True)
class TradeSignal:
    """A concrete, explainable trade plan (or a reasoned no-trade)."""

    action: SignalAction
    style: str                       # "reversion" | "breakout"
    price: float                     # reference price the plan was built from
    rationale: str

    # Trade plan (None / empty for NO_TRADE) -------------------------------- #
    entry: Optional[float] = None
    stop: Optional[float] = None
    targets: tuple[float, ...] = ()
    risk_per_unit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    suggested_size: Optional[float] = None
    risk_amount: Optional[float] = None
    account_equity: Optional[float] = None

    # Context copied from the confluence read ------------------------------ #
    setup_quality: float = 0.0
    bias: str = "neutral"
    bias_score: float = 0.0

    # Bookkeeping ---------------------------------------------------------- #
    timestamp: Optional[pd.Timestamp] = None
    symbol: Optional[str] = None
    interval: Optional[str] = None
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def is_actionable(self) -> bool:
        """``True`` for LONG/SHORT, ``False`` for NO_TRADE."""
        return self.action in (SignalAction.LONG, SignalAction.SHORT)

    @property
    def direction(self) -> int:
        """``+1`` long, ``-1`` short, ``0`` no-trade."""
        return {SignalAction.LONG: 1, SignalAction.SHORT: -1}.get(self.action, 0)

    @property
    def final_target(self) -> Optional[float]:
        """The furthest target, if any."""
        return self.targets[-1] if self.targets else None

    # ------------------------------------------------------------------ #
    def as_dict(self) -> dict:
        """Flat, JSON-serialisable representation."""
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "timestamp": None if self.timestamp is None else str(self.timestamp),
            "action": self.action.value,
            "style": self.style,
            "price": self.price,
            "entry": self.entry,
            "stop": self.stop,
            "targets": list(self.targets),
            "risk_per_unit": self.risk_per_unit,
            "risk_reward_ratio": self.risk_reward_ratio,
            "suggested_size": self.suggested_size,
            "risk_amount": self.risk_amount,
            "account_equity": self.account_equity,
            "setup_quality": self.setup_quality,
            "bias": self.bias,
            "bias_score": self.bias_score,
            "rationale": self.rationale,
        }

    def explain(self) -> str:
        """Return a journal-ready, human-readable summary of the signal."""
        sym = self.symbol or "data"
        itv = f" {self.interval}" if self.interval else ""
        when = f"  ({self.timestamp})" if self.timestamp is not None else ""
        head = f"TRADE SIGNAL — {self.action.value.upper()} {sym}{itv} [{self.style}]{when}"
        lines = [head, "-" * max(56, len(head))]
        lines.append(
            f"  Setup       : quality {self.setup_quality:.0f}/100, "
            f"bias {self.bias.upper()} ({self.bias_score:+.0f})"
        )

        if not self.is_actionable:
            lines.append(f"  Decision    : NO TRADE — {self.rationale}")
            return "\n".join(lines)

        tgt = ", ".join(f"{t:.4f}" for t in self.targets) if self.targets else "—"
        lines.append(f"  Reference   : {self.price:.4f}")
        lines.append(f"  Entry       : {self.entry:.4f}")
        lines.append(
            f"  Stop        : {self.stop:.4f}   (risk {self.risk_per_unit:.4f}/unit)"
        )
        lines.append(f"  Target(s)   : {tgt}")
        if self.risk_reward_ratio is not None:
            lines.append(
                f"  R:R         : {self.risk_reward_ratio:.2f}  (to first target)"
            )
        if self.suggested_size is not None and self.risk_amount is not None:
            equity = self.account_equity or 0.0
            pct = (self.risk_amount / equity * 100.0) if equity else 0.0
            lines.append(
                f"  Size        : {self.suggested_size:g} units  "
                f"(risk {self.risk_amount:,.2f} = {pct:.1f}% of {equity:,.0f})"
            )
        lines.append(f"  Why         : {self.rationale}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if not self.is_actionable:
            return f"NO_TRADE [{self.style}] — {self.rationale}"
        tgt = "/".join(f"{t:.2f}" for t in self.targets)
        return (
            f"{self.action.value.upper()} {self.symbol or ''} @ {self.entry:.2f} "
            f"| stop {self.stop:.2f} | T {tgt} | R:R "
            f"{self.risk_reward_ratio:.2f} | quality {self.setup_quality:.0f} "
            f"bias {self.bias_score:+.0f}"
        ).strip()
