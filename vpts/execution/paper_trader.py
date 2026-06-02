"""Forward paper-walk — survivorship-free evidence gathering (not an edge).

The eleven-fitted-model study (`RESEARCH.md`) found **no survivorship-robust,
tradeable edge** in this data, and the binding constraint is the data itself
(survivorship-biased, free, daily). The one kind of evidence the historical study
*cannot* produce is **survivorship-free** evidence — so this module collects it the
only honest way: forward, in paper, with no look-ahead.

Each run, for a watchlist:

1. **Decide** on data **≤ the as-of date** (`SignalGenerator.analyze`) — strictly no
   look-ahead; a decision is logged the moment it is made.
2. **Record** any actionable call as an *open* paper order (append-only JSONL ledger,
   idempotent per ``(symbol, decision_date)``).
3. **Resolve** previously-open orders against the bars that have *since* arrived:
   fill at the **next bar's open** (real market-order slippage, not the limit price),
   then **first-touch** stop / first-target, with a **time-stop** at ``max_hold``.

This is **paper only** — it never places an order or moves money. It will not
manufacture an edge the backtest says is absent; its value is an honest,
survivorship-free track record accumulated bar by bar.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from vpts.signals.generator import SignalGenerator
from vpts.signals.models import TradeSignal

Loader = Callable[[str], pd.DataFrame]


@dataclass
class PaperOrder:
    """One logged paper decision and (once enough bars arrive) its first-touch outcome."""

    symbol: str
    decision_date: str                 # ISO date of the decision bar (features use data ≤ here)
    side: int                          # +1 long, -1 short
    stop: float
    target: float
    max_hold: int
    status: str = "open"               # open | win | loss | timeout
    fill_date: Optional[str] = None
    fill_price: Optional[float] = None
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    holding_bars: Optional[int] = None
    r_multiple: Optional[float] = None
    setup_quality: float = 0.0
    bias_score: float = 0.0

    @property
    def key(self) -> tuple[str, str]:
        return (self.symbol, self.decision_date)

    @property
    def resolved(self) -> bool:
        return self.status != "open"


def build_order(signal: TradeSignal, *, max_hold: int) -> Optional[PaperOrder]:
    """Turn an actionable ``TradeSignal`` into an open ``PaperOrder`` (None if not tradeable)."""
    if not signal.is_actionable or signal.stop is None or not signal.targets:
        return None
    if signal.timestamp is None:
        raise ValueError("signal has no timestamp — cannot date the decision (no-look-ahead audit).")
    return PaperOrder(
        symbol=signal.symbol or "?",
        decision_date=pd.Timestamp(signal.timestamp).date().isoformat(),
        side=signal.direction,
        stop=float(signal.stop),
        target=float(signal.targets[0]),     # resolve against the first (nearest) target
        max_hold=int(max_hold),
        setup_quality=float(signal.setup_quality),
        bias_score=float(signal.bias_score),
    )


def resolve_order(order: PaperOrder, df: pd.DataFrame) -> PaperOrder:
    """First-touch resolution against bars after the decision (idempotent once resolved).

    Fills at the **next bar's open**; a long stops on ``low ≤ stop`` and targets on
    ``high ≥ target`` (short mirrored); stop wins a same-bar tie (conservative). If the
    full ``max_hold`` window elapses untouched it is a **timeout** at that bar's close.
    Returns the order unchanged if it is already resolved or lacks the bars to decide yet.
    """
    if order.resolved:
        return order
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("resolve_order needs a DatetimeIndex.")
    d = pd.Timestamp(order.decision_date)
    if d not in df.index:
        return order
    pos = int(df.index.get_loc(d))
    fill_pos = pos + 1
    if fill_pos >= len(df):                      # no fill bar has printed yet
        return order

    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    fill = float(df["Open"].to_numpy(float)[fill_pos])
    horizon_end = fill_pos + order.max_hold - 1   # time-stop bar: hold ≤ max_hold bars inclusive
    scan_end = min(horizon_end, len(df) - 1)
    fully_elapsed = horizon_end <= len(df) - 1

    status, xpos, xprice = "open", None, None
    for j in range(fill_pos, scan_end + 1):
        hit_stop = low[j] <= order.stop if order.side == 1 else high[j] >= order.stop
        hit_target = high[j] >= order.target if order.side == 1 else low[j] <= order.target
        if hit_stop:                             # stop assumed first on a tie
            status, xpos, xprice = "loss", j, order.stop
            break
        if hit_target:
            status, xpos, xprice = "win", j, order.target
            break
    if status == "open":
        if not fully_elapsed:
            return order                         # await more bars
        status, xpos, xprice = "timeout", horizon_end, float(close[horizon_end])

    risk = abs(fill - order.stop)
    order.status = status
    order.fill_date = df.index[fill_pos].date().isoformat()
    order.fill_price = fill
    order.exit_date = df.index[xpos].date().isoformat()
    order.exit_price = float(xprice)
    order.holding_bars = int(xpos - fill_pos + 1)
    order.r_multiple = float((xprice - fill) * order.side / risk) if risk > 1e-12 else float("nan")
    return order


class PaperLedger:
    """Append-only JSONL store of paper orders, keyed (upserted) by ``(symbol, decision_date)``."""

    def __init__(self, orders: Optional[list[PaperOrder]] = None) -> None:
        self.orders: list[PaperOrder] = list(orders or [])

    @classmethod
    def load(cls, path: str | Path) -> "PaperLedger":
        p = Path(path)
        if not p.exists():
            return cls([])
        orders = [PaperOrder(**json.loads(line)) for line in p.read_text().splitlines() if line.strip()]
        return cls(orders)

    def save(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(json.dumps(asdict(o)) for o in self.orders) + "\n")

    def upsert(self, order: PaperOrder) -> bool:
        """Add a new order; return False if one already exists for its key (idempotent)."""
        if any(o.key == order.key for o in self.orders):
            return False
        self.orders.append(order)
        return True

    def open_orders(self) -> list[PaperOrder]:
        return [o for o in self.orders if not o.resolved]

    def summary(self) -> dict:
        """Track record over **resolved** orders (the honest, forward, survivorship-free tally).

        ``pct_profitable`` is the real win rate (R > 0) — *not* the target-hit rate, since a
        trade can exit profitably on the time-stop without ever touching its target. The
        exit-type counts (``target_hits``/``stops``/``timeouts``) are reported alongside.
        """
        done = [o for o in self.orders if o.resolved]
        rs = np.array([o.r_multiple for o in done if o.r_multiple is not None and np.isfinite(o.r_multiple)])
        return {
            "open": len(self.open_orders()),
            "resolved": len(done),
            "pct_profitable": float((rs > 0).mean()) if rs.size else float("nan"),   # R > 0
            "avg_R": float(rs.mean()) if rs.size else float("nan"),                   # expectancy / bet
            "total_R": float(rs.sum()) if rs.size else 0.0,
            "target_hits": sum(o.status == "win" for o in done),
            "stops": sum(o.status == "loss" for o in done),
            "timeouts": sum(o.status == "timeout" for o in done),
        }


def run_paper_walk(
    load: Loader,
    watchlist: list[str],
    as_of,
    ledger: PaperLedger,
    *,
    style: str = "reversion",
    max_hold: int = 20,
    account_equity: float = 10_000.0,
    interval: str = "1d",
) -> dict:
    """Decide+record today's signals (data ≤ ``as_of``), resolve open orders, return a report.

    ``load(symbol) → OHLCV`` is injected (live yfinance in production, synthetic in tests);
    ``as_of`` is the decision date (injectable, so the walk is deterministic and testable).
    Mutates ``ledger`` in place; the caller persists it.
    """
    as_of = pd.Timestamp(as_of)
    gen = SignalGenerator(style=style, account_equity=account_equity)
    new_orders, errors = [], []
    for sym in watchlist:
        try:
            df = load(sym)
            past = df.loc[:as_of]                      # strictly no look-ahead at decision time
            if len(past) < 2:
                continue
            signal = gen.analyze(past, symbol=sym, interval=interval)
            order = build_order(signal, max_hold=max_hold)
            if order is not None and ledger.upsert(order):
                new_orders.append(order)
            for o in ledger.open_orders():             # resolve prior decisions vs bars since arrived
                if o.symbol == sym:
                    resolve_order(o, df)
        except Exception as exc:                        # noqa: BLE001 - one bad symbol shouldn't halt the walk
            errors.append((sym, str(exc)))
    return {"as_of": as_of.date().isoformat(), "new": len(new_orders),
            "errors": errors, **ledger.summary()}
