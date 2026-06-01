"""Phase 2 — Quiet-Phase Detector.

Identifies low-energy, "coiled-spring" market conditions — the regime this whole
system is built to exploit — by blending three self-normalising signals:

1. **Low volatility**  — ATR ranked against its own recent history.
2. **Declining / dry volume** — a volume moving average ranked against history.
3. **Range compression** — Bollinger Bandwidth ranked against history.

Each signal is turned into a trailing **percentile rank**, so the readings are
comparable across instruments and timeframes (no hand-tuned absolute thresholds).
A low percentile means "quiet relative to normal for this market". The three are
combined into a single ``quiet_score`` in ``0..100`` and an ``is_quiet`` flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from vpts.regime.indicators import (
    atr,
    bollinger_bandwidth,
    ensure_ohlcv,
    rolling_percentile_rank,
    rolling_slope,
)


@dataclass(frozen=True)
class QuietState:
    """Immutable snapshot of the quiet-phase reading for a single bar."""

    timestamp: Optional[pd.Timestamp]
    quiet_score: float          # 0..100
    is_quiet: bool
    score_volatility: float     # 0..1 sub-score (1 = very quiet)
    score_volume: float
    score_range: float
    atr_pctile: float           # 0..1 (low = quiet)
    volume_pctile: float
    range_pctile: float
    volume_falling: bool
    bars_in_state: int          # consecutive quiet bars up to & incl. this one
    explanation: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.explanation


@dataclass(frozen=True)
class QuietPhaseResult:
    """Per-bar quiet-phase analysis plus the latest-bar :class:`QuietState`."""

    frame: pd.DataFrame
    latest: QuietState
    quiet_threshold: float
    symbol: Optional[str] = None
    interval: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the full per-bar analysis frame."""
        return self.frame

    @property
    def is_quiet(self) -> bool:
        """Convenience: is the most recent bar a quiet phase?"""
        return self.latest.is_quiet

    def quiet_segments(self) -> list[dict]:
        """Return contiguous quiet stretches as ``{start, end, length}`` dicts."""
        flag = self.frame["is_quiet"].to_numpy(dtype=bool)
        index = self.frame.index
        segments: list[dict] = []
        i, n = 0, len(flag)
        while i < n:
            if flag[i]:
                j = i
                while j + 1 < n and flag[j + 1]:
                    j += 1
                segments.append(
                    {"start": index[i], "end": index[j], "length": j - i + 1}
                )
                i = j + 1
            else:
                i += 1
        return segments

    def summary(self) -> str:
        sym = self.symbol or "data"
        itv = f" {self.interval}" if self.interval else ""
        n_quiet = int(self.frame["is_quiet"].sum())
        segs = self.quiet_segments()
        lines = [
            f"Quiet-Phase Analysis — {sym}{itv}  ({len(self.frame)} bars)",
            "-" * 52,
            f"  Latest score   : {self.latest.quiet_score:.0f}/100 "
            f"(threshold {self.quiet_threshold:.0f}) -> "
            f"{'QUIET' if self.latest.is_quiet else 'ACTIVE'}",
            f"  Sub-scores     : volatility {self.latest.score_volatility:.2f}, "
            f"volume {self.latest.score_volume:.2f}, "
            f"range {self.latest.score_range:.2f}",
            f"  Quiet bars     : {n_quiet}/{len(self.frame)} "
            f"in {len(segs)} segment(s); current run {self.latest.bars_in_state}",
            f"  Read           : {self.latest.explanation}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


class QuietPhaseDetector:
    """Detect quiet (low-volatility, low-volume, compressed) market phases.

    Parameters
    ----------
    atr_period:
        Look-back for ATR (volatility).
    volume_window:
        Window for the volume moving average that represents "current" volume.
    range_window:
        Window for the Bollinger Bandwidth (range compression).
    baseline_window:
        Trailing window used to percentile-rank each signal against its own
        history. Larger = a longer notion of "normal".
    weights:
        ``(volatility, volume, range)`` weights for the composite score
        (auto-normalised; default equal).
    quiet_threshold:
        Score (0..100) at or above which a bar is flagged ``is_quiet``.
    atr_method:
        ``"wilder"`` or ``"sma"`` (see :func:`~vpts.regime.indicators.atr`).
    min_bars:
        Minimum number of bars required.
    """

    def __init__(
        self,
        atr_period: int = 14,
        volume_window: int = 20,
        range_window: int = 20,
        baseline_window: int = 100,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        quiet_threshold: float = 60.0,
        atr_method: str = "wilder",
        min_bars: int = 30,
    ) -> None:
        for name, val in (
            ("atr_period", atr_period),
            ("volume_window", volume_window),
            ("range_window", range_window),
            ("baseline_window", baseline_window),
            ("min_bars", min_bars),
        ):
            if int(val) < 2:
                raise ValueError(f"{name} must be >= 2.")
        if len(weights) != 3 or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("weights must be three non-negative numbers, not all 0.")
        if not 0.0 < quiet_threshold < 100.0:
            raise ValueError("quiet_threshold must be in (0, 100).")

        self.atr_period = int(atr_period)
        self.volume_window = int(volume_window)
        self.range_window = int(range_window)
        self.baseline_window = int(baseline_window)
        total = float(sum(weights))
        self.weights = tuple(float(w) / total for w in weights)
        self.quiet_threshold = float(quiet_threshold)
        self.atr_method = atr_method
        self.min_bars = int(min_bars)

    # ------------------------------------------------------------------ #
    def detect(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> QuietPhaseResult:
        """Run the quiet-phase analysis over an OHLCV frame."""
        ensure_ohlcv(df, min_bars=self.min_bars)
        high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]

        atr_series = atr(high, low, close, self.atr_period, self.atr_method)
        vol_ma = volume.rolling(
            self.volume_window, min_periods=max(2, self.volume_window // 2)
        ).mean()
        bandwidth = bollinger_bandwidth(close, self.range_window)

        # Low percentile = quiet, so the sub-score is (1 - percentile_rank).
        atr_pct = rolling_percentile_rank(atr_series, self.baseline_window)
        vol_pct = rolling_percentile_rank(vol_ma, self.baseline_window)
        rng_pct = rolling_percentile_rank(bandwidth, self.baseline_window)
        s_vol = 1.0 - atr_pct
        s_vlm = 1.0 - vol_pct
        s_rng = 1.0 - rng_pct

        w_vol, w_vlm, w_rng = self.weights
        score = 100.0 * (w_vol * s_vol + w_vlm * s_vlm + w_rng * s_rng)
        is_quiet = (score >= self.quiet_threshold) & score.notna()

        volume_falling = rolling_slope(vol_ma, self.volume_window) < 0

        frame = pd.DataFrame(
            {
                "atr": atr_series,
                "atr_pctile": atr_pct,
                "volume_ma": vol_ma,
                "volume_pctile": vol_pct,
                "bandwidth": bandwidth,
                "range_pctile": rng_pct,
                "score_volatility": s_vol,
                "score_volume": s_vlm,
                "score_range": s_rng,
                "quiet_score": score,
                "is_quiet": is_quiet,
                "volume_falling": volume_falling,
            }
        )

        # Consecutive quiet-bar run length ending at each bar.
        q = is_quiet.astype(int)
        run = q.groupby((q == 0).cumsum()).cumsum()
        frame["bars_in_state"] = run

        latest = self._build_state(frame)
        return QuietPhaseResult(
            frame=frame,
            latest=latest,
            quiet_threshold=self.quiet_threshold,
            symbol=symbol,
            interval=interval,
            extra={
                "baseline_window": self.baseline_window,
                "weights": self.weights,
            },
        )

    # ------------------------------------------------------------------ #
    def _build_state(self, frame: pd.DataFrame) -> QuietState:
        """Turn the last row of *frame* into an explained :class:`QuietState`."""
        row = frame.iloc[-1]
        score = float(row["quiet_score"]) if pd.notna(row["quiet_score"]) else 0.0
        is_quiet = bool(row["is_quiet"])
        falling = bool(row["volume_falling"])
        explanation = self._explain(row, score, is_quiet, falling)
        ts = frame.index[-1] if isinstance(frame.index, pd.DatetimeIndex) else None
        return QuietState(
            timestamp=ts,
            quiet_score=score,
            is_quiet=is_quiet,
            score_volatility=float(row["score_volatility"]),
            score_volume=float(row["score_volume"]),
            score_range=float(row["score_range"]),
            atr_pctile=float(row["atr_pctile"]),
            volume_pctile=float(row["volume_pctile"]),
            range_pctile=float(row["range_pctile"]),
            volume_falling=falling,
            bars_in_state=int(row["bars_in_state"]),
            explanation=explanation,
        )

    @staticmethod
    def _explain(
        row: pd.Series, score: float, is_quiet: bool, falling: bool
    ) -> str:
        """Build a natural-language reading of the latest bar."""
        reasons: list[str] = []
        if pd.notna(row["atr_pctile"]) and row["score_volatility"] >= 0.6:
            reasons.append(
                f"volatility in the bottom {row['atr_pctile'] * 100:.0f}% of its range"
            )
        if pd.notna(row["volume_pctile"]) and row["score_volume"] >= 0.6:
            trend = "and falling" if falling else "and flat"
            reasons.append(
                f"volume in the bottom {row['volume_pctile'] * 100:.0f}% {trend}"
            )
        if pd.notna(row["range_pctile"]) and row["score_range"] >= 0.6:
            reasons.append(
                f"price range compressed (bottom {row['range_pctile'] * 100:.0f}%)"
            )

        if is_quiet:
            body = "; ".join(reasons) if reasons else "all energy gauges subdued"
            return f"Quiet phase (score {score:.0f}/100): {body}."
        if not reasons:
            return (
                f"Active / normal (score {score:.0f}/100): volatility, volume and "
                "range near typical levels."
            )
        return (
            f"Mixed (score {score:.0f}/100): "
            + "; ".join(reasons)
            + " — but not yet below the quiet threshold."
        )
