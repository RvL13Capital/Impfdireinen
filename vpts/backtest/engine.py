"""Phase 6 — event-driven backtester with strict no-look-ahead.

The engine walks the data bar by bar. At the **close of bar t** it computes the
full stack (profile → quiet → patterns → confluence → signal) from a *rolling
window ending at t*, and — if flat and the signal is actionable — enters at the
**open of bar t+1** (cost-adjusted). Open positions are then managed bar by bar
against their stop / target(s) (with overnight-gap handling), with an optional
time stop. Only data up to the decision bar is ever used, so there is no
look-ahead.

Everything upstream is reused unchanged; position sizing is fixed-fractional on
the *current* equity and the actual entry-to-stop distance.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from vpts.backtest.models import BacktestResult, CostModel, Trade
from vpts.profile.calculator import VolumeProfileCalculator
from vpts.regime.indicators import ensure_ohlcv
from vpts.regime.patterns import VolumePatternDetector
from vpts.regime.quiet import QuietPhaseDetector
from vpts.scoring.scorer import ConfluenceScorer
from vpts.signals.generator import SignalGenerator
from vpts.signals.models import SignalAction

# Rough bars-per-year by interval, for annualising the Sharpe ratio.
_PERIODS_PER_YEAR = {
    "1d": 252, "1wk": 52, "1mo": 12,
    "1h": 252 * 7, "60m": 252 * 7, "30m": 252 * 13,
    "15m": 252 * 26, "5m": 252 * 78, "2m": 252 * 195, "1m": 252 * 390,
}


class _OpenPosition:
    """Mutable bookkeeping for the single in-flight position."""

    __slots__ = ("side", "entry_time", "entry_price", "size", "stop",
                 "targets", "entry_index")

    def __init__(self, side, entry_time, entry_price, size, stop, targets, entry_index):
        self.side = side
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.size = size
        self.stop = stop
        self.targets = targets
        self.entry_index = entry_index


class Backtester:
    """Walk-forward backtest of the Quiet-Volume signal stack.

    Parameters
    ----------
    lookback:
        Rolling window length (bars) used to compute the profile/regime/signal at
        each decision bar.
    initial_equity:
        Starting account equity.
    cost_model:
        :class:`~vpts.backtest.models.CostModel` for slippage / spread / commission.
    signal_generator, scorer, profile_calculator, quiet_detector, pattern_detector:
        Optional pre-configured components; sensible defaults are built otherwise
        (the profile defaults to ``bin_mode="auto"``).
    max_hold_bars:
        Optional time stop — force-exit after this many bars in a trade.
    recompute_stride:
        Recompute the (flat-state) signal every *stride* bars (default ``1`` =
        every bar). Larger values speed up long runs.
    """

    def __init__(
        self,
        lookback: int = 120,
        initial_equity: float = 10_000.0,
        cost_model: Optional[CostModel] = None,
        signal_generator: Optional[SignalGenerator] = None,
        scorer: Optional[ConfluenceScorer] = None,
        profile_calculator: Optional[VolumeProfileCalculator] = None,
        quiet_detector: Optional[QuietPhaseDetector] = None,
        pattern_detector: Optional[VolumePatternDetector] = None,
        max_hold_bars: Optional[int] = None,
        recompute_stride: int = 1,
    ) -> None:
        if lookback < 30:
            raise ValueError("lookback must be >= 30 (detectors need history).")
        if initial_equity <= 0:
            raise ValueError("initial_equity must be > 0.")
        if max_hold_bars is not None and max_hold_bars < 1:
            raise ValueError("max_hold_bars must be >= 1 or None.")
        if recompute_stride < 1:
            raise ValueError("recompute_stride must be >= 1.")

        self.lookback = int(lookback)
        self.initial_equity = float(initial_equity)
        self.cost_model = cost_model or CostModel()
        self.signal_generator = signal_generator or SignalGenerator()
        self.scorer = scorer or ConfluenceScorer()
        self.profile_calculator = profile_calculator or VolumeProfileCalculator(
            bin_mode="auto"
        )
        self.quiet_detector = quiet_detector or QuietPhaseDetector()
        self.pattern_detector = pattern_detector or VolumePatternDetector()
        self.max_hold_bars = max_hold_bars
        self.recompute_stride = int(recompute_stride)

    # ------------------------------------------------------------------ #
    def run(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> BacktestResult:
        """Run the backtest over an OHLCV frame and return a :class:`BacktestResult`."""
        ensure_ohlcv(df, min_bars=self.lookback + 2)
        opens = df["Open"].to_numpy(float)
        highs = df["High"].to_numpy(float)
        lows = df["Low"].to_numpy(float)
        closes = df["Close"].to_numpy(float)
        index = df.index
        n = len(df)

        cash = self.initial_equity
        position: Optional[_OpenPosition] = None
        pending = None              # actionable signal decided at the previous bar
        trades: list[Trade] = []
        equity_times: list = []
        equity_values: list[float] = []

        # Decisions can first be made once a full window exists (bar lookback-1);
        # the earliest possible entry is therefore at bar `lookback`.
        for i in range(self.lookback - 1, n):
            bar_open, bar_high = opens[i], highs[i]
            bar_low, bar_close = lows[i], closes[i]

            # 1) Enter at THIS bar's open if flat with a pending signal.
            if position is None and pending is not None:
                position = self._try_open(pending, bar_open, cash, index[i], i)
                pending = None

            # 2) Manage the open position on this bar.
            if position is not None:
                # The entry bar only sees intrabar high/low (no gap vs its own open).
                allow_gap = i > position.entry_index
                exit_price, reason = self._check_exit(
                    position, bar_open, bar_high, bar_low, bar_close, i, allow_gap
                )
                if exit_price is not None:
                    cash, trade = self._close(position, exit_price, index[i], i, cash,
                                              reason=reason)
                    trades.append(trade)
                    position = None

            # 3) If flat, decide a signal from data THROUGH this bar (for entry next bar).
            if position is None and pending is None:
                if (i - (self.lookback - 1)) % self.recompute_stride == 0:
                    pending = self._signal(df.iloc[i - self.lookback + 1 : i + 1],
                                           cash, symbol, interval)

            # 4) Mark-to-market equity at this bar's close.
            mtm = cash
            if position is not None:
                d = 1 if position.side == "long" else -1
                mtm += d * (bar_close - position.entry_price) * position.size
            equity_times.append(index[i])
            equity_values.append(mtm)

        # Force-close any position still open at the final close.
        if position is not None:
            cash, trade = self._close(position, float(closes[-1]), index[-1],
                                      n - 1, cash, reason="end")
            trades.append(trade)
            equity_values[-1] = cash

        equity_curve = pd.Series(equity_values, index=pd.Index(equity_times),
                                 name="equity")
        return self._build_result(trades, equity_curve, cash, symbol, interval)

    # ------------------------------------------------------------------ #
    def _try_open(self, pending, bar_open, cash, ts, i):
        """Open a position at *bar_open*, or return ``None`` if it is invalid.

        Two guards make fills realistic and cash-constrained:

        * **Gap through the stop** — if the cost-adjusted open has already gapped
          to or through the signal's stop, the setup is invalidated: we neither
          enter nor fabricate a same-bar fill at the stale stop.
        * **No leverage** — size is fixed-fractional *and* capped so the position's
          notional never exceeds available cash, so a losing trade can't drive
          equity negative.
        """
        side = 1 if pending.action == SignalAction.LONG else -1
        entry = self.cost_model.fill_price(bar_open, side)
        if (side == 1 and entry <= pending.stop) or (side == -1 and entry >= pending.stop):
            return None
        risk = abs(entry - pending.stop)
        if risk <= 0:
            return None
        size = np.floor((cash * self.signal_generator.risk_fraction) / risk)
        size = min(size, np.floor(cash / entry))   # cap notional at cash (no leverage)
        if size <= 0:
            return None
        return _OpenPosition(
            "long" if side == 1 else "short", ts, float(entry), float(size),
            pending.stop, tuple(pending.targets), i,
        )

    def _signal(self, window, equity, symbol, interval):
        """Full stack on the rolling window; returns an actionable signal or None."""
        try:
            profile = self.profile_calculator.calculate(window, symbol, interval)
            quiet = self.quiet_detector.detect(window, symbol, interval)
            patterns = self.pattern_detector.detect(window, profile=profile,
                                                    symbol=symbol, interval=interval)
            score = self.scorer.score(window, profile, quiet, patterns,
                                     symbol=symbol, interval=interval)
            signal = self.signal_generator.from_score(score, profile,
                                                      account_equity=equity)
        except (ValueError, ZeroDivisionError):
            return None
        return signal if signal.is_actionable else None

    def _check_exit(self, pos, o, h, l, c, i, allow_gap):
        """Return ``(exit_price, reason)`` if the bar exits the position, else (None, None)."""
        target = pos.targets[0] if pos.targets else None
        if pos.side == "long":
            if allow_gap and o <= pos.stop:
                return self.cost_model.fill_price(o, -1), "stop"
            if allow_gap and target is not None and o >= target:
                return self.cost_model.fill_price(o, -1), "target"
            if l <= pos.stop:                                   # stop assumed first
                return self.cost_model.fill_price(pos.stop, -1), "stop"
            if target is not None and h >= target:
                return self.cost_model.fill_price(target, -1), "target"
        else:  # short
            if allow_gap and o >= pos.stop:
                return self.cost_model.fill_price(o, 1), "stop"
            if allow_gap and target is not None and o <= target:
                return self.cost_model.fill_price(o, 1), "target"
            if h >= pos.stop:
                return self.cost_model.fill_price(pos.stop, 1), "stop"
            if target is not None and l <= target:
                return self.cost_model.fill_price(target, 1), "target"
        if self.max_hold_bars is not None and (i - pos.entry_index) >= self.max_hold_bars:
            side = -1 if pos.side == "long" else 1
            return self.cost_model.fill_price(c, side), "time"
        return None, None

    def _close(self, pos, exit_price, exit_time, exit_index, cash, reason=None):
        """Realise a trade, apply commissions, update cash, and return (cash, Trade)."""
        d = 1 if pos.side == "long" else -1
        gross = d * (exit_price - pos.entry_price) * pos.size
        commissions = (self.cost_model.commission(pos.entry_price, pos.size)
                       + self.cost_model.commission(exit_price, pos.size))
        pnl = gross - commissions
        cash += pnl
        notional = pos.entry_price * pos.size
        risk_dollar = abs(pos.entry_price - pos.stop) * pos.size
        trade = Trade(
            side=pos.side,
            entry_time=pos.entry_time if isinstance(pos.entry_time, pd.Timestamp) else None,
            exit_time=exit_time if isinstance(exit_time, pd.Timestamp) else None,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            stop=pos.stop,
            target=pos.targets[0] if pos.targets else float("nan"),
            pnl=pnl,
            return_pct=(pnl / notional * 100.0) if notional else 0.0,
            r_multiple=(pnl / risk_dollar) if risk_dollar else 0.0,
            exit_reason=reason or "exit",
            bars_held=exit_index - pos.entry_index,
        )
        return cash, trade

    # ------------------------------------------------------------------ #
    def _build_result(self, trades, equity_curve, final_equity, symbol, interval):
        n_trades = len(trades)
        pnls = np.array([t.pnl for t in trades], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        win_rate = float((pnls > 0).mean()) if n_trades else 0.0
        profit_factor = (gross_profit / gross_loss if gross_loss > 0
                         else (float("inf") if gross_profit > 0 else 0.0))
        expectancy = float(pnls.mean()) if n_trades else 0.0
        avg_r = float(np.mean([t.r_multiple for t in trades])) if n_trades else 0.0
        total_return_pct = (final_equity / self.initial_equity - 1.0) * 100.0

        # Max drawdown on the mark-to-market equity curve.
        if len(equity_curve):
            running_max = equity_curve.cummax()
            drawdown = (equity_curve / running_max - 1.0)
            max_dd = float(drawdown.min() * 100.0)
        else:
            max_dd = 0.0

        sharpe = self._sharpe(equity_curve, interval)

        return BacktestResult(
            trades=tuple(trades),
            equity_curve=equity_curve,
            initial_equity=self.initial_equity,
            final_equity=float(final_equity),
            total_return_pct=float(total_return_pct),
            n_trades=n_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd,
            expectancy=expectancy,
            avg_r=avg_r,
            sharpe=sharpe,
            symbol=symbol,
            interval=interval,
            extra={"cost_model": vars(self.cost_model), "lookback": self.lookback},
        )

    @staticmethod
    def _sharpe(equity_curve: pd.Series, interval: Optional[str]) -> float:
        if len(equity_curve) < 3:
            return 0.0
        rets = equity_curve.pct_change().dropna()
        sd = float(rets.std())
        if sd <= 0 or not np.isfinite(sd):
            return 0.0
        ppy = _PERIODS_PER_YEAR.get((interval or "1d").lower(), 252)
        return float(rets.mean() / sd * np.sqrt(ppy))
