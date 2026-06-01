"""Phase 6 — cost model and immutable result objects for the backtester."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Realistic, free-retail transaction costs (all adverse to the trader).

    Costs are applied to every fill: the buy price is nudged up and the sell
    price down by ``slippage_bps + spread_bps/2`` basis points, plus a
    commission (bps of notional and/or per share).

    Defaults model a commission-free retail broker with light slippage.
    """

    slippage_bps: float = 5.0
    spread_bps: float = 0.0
    commission_bps: float = 0.0
    commission_per_share: float = 0.0

    def __post_init__(self) -> None:
        for name in ("slippage_bps", "spread_bps", "commission_bps",
                     "commission_per_share"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0.")

    def fill_price(self, price: float, side: int) -> float:
        """Adverse fill price. ``side`` is ``+1`` for a buy, ``-1`` for a sell."""
        adverse = (self.slippage_bps + self.spread_bps / 2.0) / 1e4
        return float(price * (1.0 + side * adverse))

    def commission(self, price: float, shares: float) -> float:
        """Commission for trading *shares* at *price* (one side)."""
        shares = abs(shares)
        return float(shares * price * self.commission_bps / 1e4
                     + shares * self.commission_per_share)


@dataclass(frozen=True)
class Trade:
    """A single closed round-trip trade."""

    side: str                     # "long" | "short"
    entry_time: Optional[pd.Timestamp]
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: float
    size: float
    stop: float
    target: float
    pnl: float                    # net of all costs, in account currency
    return_pct: float             # pnl as % of entry notional
    r_multiple: float             # net pnl in units of initial risk
    exit_reason: str              # "stop" | "target" | "time" | "end"
    bars_held: int

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.side.upper()} {self.entry_price:.2f}->{self.exit_price:.2f} "
            f"({self.exit_reason}, {self.r_multiple:+.2f}R, {self.pnl:+,.2f})"
        )


@dataclass(frozen=True)
class BacktestResult:
    """Immutable backtest output: trades, equity curve and headline stats."""

    trades: tuple[Trade, ...]
    equity_curve: pd.Series       # mark-to-market equity, indexed by time
    initial_equity: float
    final_equity: float

    # headline stats
    total_return_pct: float
    n_trades: int
    win_rate: float               # 0..1
    profit_factor: float          # gross profit / gross loss (inf if no losses)
    max_drawdown_pct: float       # negative number (e.g. -12.3)
    expectancy: float             # mean net pnl per trade
    avg_r: float                  # mean R-multiple per trade
    sharpe: float                 # annualised, from bar-to-bar equity returns

    symbol: Optional[str] = None
    interval: Optional[str] = None
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def trades_dataframe(self) -> pd.DataFrame:
        """Return the trade blotter as a :class:`pandas.DataFrame`."""
        if not self.trades:
            return pd.DataFrame(
                columns=["side", "entry_time", "exit_time", "entry_price",
                         "exit_price", "size", "pnl", "return_pct", "r_multiple",
                         "exit_reason", "bars_held"]
            )
        return pd.DataFrame([
            {
                "side": t.side,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4),
                "size": t.size,
                "pnl": round(t.pnl, 2),
                "return_pct": round(t.return_pct, 3),
                "r_multiple": round(t.r_multiple, 3),
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            }
            for t in self.trades
        ])

    def as_dict(self) -> dict:
        """Flat, JSON-serialisable stats summary (no equity series / trades)."""
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "initial_equity": self.initial_equity,
            "final_equity": round(self.final_equity, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": (None if self.profit_factor == float("inf")
                              else round(self.profit_factor, 3)),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "expectancy": round(self.expectancy, 2),
            "avg_r": round(self.avg_r, 3),
            "sharpe": round(self.sharpe, 3),
        }

    def summary(self) -> str:
        sym = self.symbol or "data"
        itv = f" {self.interval}" if self.interval else ""
        pf = "∞" if self.profit_factor == float("inf") else f"{self.profit_factor:.2f}"
        lines = [
            f"Backtest — {sym}{itv}  ({len(self.equity_curve)} bars)",
            "-" * 52,
            f"  Equity         : {self.initial_equity:,.0f} -> {self.final_equity:,.0f}"
            f"   ({self.total_return_pct:+.2f}%)",
            f"  Trades         : {self.n_trades}  "
            f"(win rate {self.win_rate * 100:.1f}%)",
            f"  Profit factor  : {pf}",
            f"  Expectancy     : {self.expectancy:+,.2f} / trade "
            f"({self.avg_r:+.2f}R avg)",
            f"  Max drawdown   : {self.max_drawdown_pct:.2f}%",
            f"  Sharpe (ann.)  : {self.sharpe:.2f}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()
