"""Phase 2 — Volume Pattern Detector.

Recognises four classic, institution-revealing volume behaviours and explains
each in plain language. Where a Phase-1 :class:`~vpts.profile.models.VolumeProfile`
is supplied, events are *anchored* to profile levels (POC / VAH / VAL / HVN / LVN)
— e.g. "climax **at the POC**" — which is exactly where these patterns matter most.

Patterns
--------
* **Volume Dry-up** — sustained below-average volume; supply thinning, often a
  pre-breakout coil.
* **Accumulation** — tight, flat range on below-average volume; quiet absorption.
* **Volume Divergence** — price trend not confirmed by volume (e.g. price up while
  volume fades → a rally without participation).
* **Volume Climax** — an extreme volume spike on a wide-range bar; potential
  exhaustion / reversal, most significant at a key level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from vpts.profile.models import VolumeProfile
from vpts.regime.indicators import atr, ensure_ohlcv, rolling_slope


class VolumePatternType(str, Enum):
    """Enumeration of recognised volume patterns."""

    DRY_UP = "dry_up"
    ACCUMULATION = "accumulation"
    DIVERGENCE = "divergence"
    CLIMAX = "climax"


@dataclass(frozen=True)
class VolumePattern:
    """A single detected volume event."""

    type: VolumePatternType
    direction: str               # "bullish" | "bearish" | "neutral"
    timestamp: Optional[pd.Timestamp]
    index_pos: int               # positional index of the anchor bar
    price: float
    volume: float
    strength: float              # pattern-specific intensity (>= 0)
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    n_bars: int
    at_level: Optional[str]      # nearby profile level, e.g. "POC 204.52"
    explanation: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.explanation


@dataclass(frozen=True)
class VolumePatternResult:
    """All detected patterns plus the per-bar metric frame."""

    patterns: tuple[VolumePattern, ...]
    frame: pd.DataFrame
    symbol: Optional[str] = None
    interval: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        return self.frame

    def of_type(self, kind: VolumePatternType) -> list[VolumePattern]:
        """Return patterns of a given :class:`VolumePatternType`."""
        return [p for p in self.patterns if p.type == kind]

    def recent(self, n: int = 5) -> list[VolumePattern]:
        """Return the *n* most recent patterns (latest anchor bar first)."""
        return sorted(self.patterns, key=lambda p: p.index_pos, reverse=True)[:n]

    @property
    def latest(self) -> Optional[VolumePattern]:
        """The most recently anchored pattern, if any."""
        return max(self.patterns, key=lambda p: p.index_pos, default=None)

    def summary(self) -> str:
        sym = self.symbol or "data"
        itv = f" {self.interval}" if self.interval else ""
        counts = {t: len(self.of_type(t)) for t in VolumePatternType}
        lines = [
            f"Volume Patterns — {sym}{itv}  ({len(self.frame)} bars)",
            "-" * 52,
            "  Counts         : "
            + ", ".join(f"{t.value}={counts[t]}" for t in VolumePatternType),
        ]
        if self.patterns:
            lines.append("  Most recent    :")
            for p in self.recent(3):
                lines.append(f"    • {p.explanation}")
        else:
            lines.append("  Most recent    : (none detected)")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


class VolumePatternDetector:
    """Detect Dry-up / Accumulation / Divergence / Climax volume patterns.

    Parameters
    ----------
    volume_ma_window:
        Window for the reference volume moving average.
    baseline_window:
        Longer window used as the "normal" volume/compression reference for
        accumulation.
    dryup_factor:
        A bar is "dry" when its volume is ``<= dryup_factor × volume_MA``.
    dryup_min_run:
        Minimum consecutive dry bars before a dry-up event is emitted.
    climax_volume_factor:
        Volume multiple of the MA needed to qualify as a climax (``>=``).
    climax_range_factor:
        True-range multiple of ATR needed (wide bar) to qualify as a climax.
    atr_period:
        ATR look-back (for range filters and level tolerance).
    divergence_window:
        Window over which price- and volume-trend slopes are compared.
    divergence_eps:
        Minimum absolute normalised slope (per bar) to count as a real trend.
    level_tolerance_atr:
        A pattern is "at" a profile level when within this many ATRs of it.
    min_bars:
        Minimum number of bars required.
    """

    def __init__(
        self,
        volume_ma_window: int = 20,
        baseline_window: int = 60,
        dryup_factor: float = 0.5,
        dryup_min_run: int = 3,
        climax_volume_factor: float = 2.5,
        climax_range_factor: float = 1.5,
        atr_period: int = 14,
        divergence_window: int = 20,
        divergence_eps: float = 5e-4,
        divergence_min_run: int = 3,
        accumulation_flat_atr: float = 2.0,
        accumulation_min_run: int = 5,
        level_tolerance_atr: float = 0.5,
        min_bars: int = 30,
    ) -> None:
        if not 0.0 < dryup_factor < 1.0:
            raise ValueError("dryup_factor must be in (0, 1).")
        if climax_volume_factor <= 1.0:
            raise ValueError("climax_volume_factor must be > 1.")
        if climax_range_factor <= 0.0:
            raise ValueError("climax_range_factor must be > 0.")
        if divergence_eps < 0.0:
            raise ValueError("divergence_eps must be >= 0.")
        if accumulation_flat_atr <= 0.0:
            raise ValueError("accumulation_flat_atr must be > 0.")
        if level_tolerance_atr <= 0.0:
            raise ValueError("level_tolerance_atr must be > 0.")
        for name, val in (
            ("volume_ma_window", volume_ma_window),
            ("baseline_window", baseline_window),
            ("dryup_min_run", dryup_min_run),
            ("atr_period", atr_period),
            ("divergence_window", divergence_window),
            ("min_bars", min_bars),
        ):
            if int(val) < 2:
                raise ValueError(f"{name} must be >= 2.")
        for name, val in (
            ("divergence_min_run", divergence_min_run),
            ("accumulation_min_run", accumulation_min_run),
        ):
            if int(val) < 1:
                raise ValueError(f"{name} must be >= 1.")

        self.volume_ma_window = int(volume_ma_window)
        self.baseline_window = int(baseline_window)
        self.dryup_factor = float(dryup_factor)
        self.dryup_min_run = int(dryup_min_run)
        self.climax_volume_factor = float(climax_volume_factor)
        self.climax_range_factor = float(climax_range_factor)
        self.atr_period = int(atr_period)
        self.divergence_window = int(divergence_window)
        self.divergence_eps = float(divergence_eps)
        self.divergence_min_run = int(divergence_min_run)
        self.accumulation_flat_atr = float(accumulation_flat_atr)
        self.accumulation_min_run = int(accumulation_min_run)
        self.level_tolerance_atr = float(level_tolerance_atr)
        self.min_bars = int(min_bars)

    # ------------------------------------------------------------------ #
    def detect(
        self,
        df: pd.DataFrame,
        profile: Optional[VolumeProfile] = None,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> VolumePatternResult:
        """Detect volume patterns, optionally anchored to a *profile*."""
        ensure_ohlcv(df, min_bars=self.min_bars)
        open_, high = df.get("Open", df["Close"]), df["High"]
        low, close, volume = df["Low"], df["Close"], df["Volume"]

        vol_ma = volume.rolling(
            self.volume_ma_window, min_periods=max(2, self.volume_ma_window // 2)
        ).mean()
        # Climax = spike vs the *recent* average; dry-up = sustained low vs the
        # *longer-term* baseline (a contraction relative to normal activity).
        vol_ratio = volume / vol_ma.replace(0.0, np.nan)
        vol_baseline = volume.rolling(
            self.baseline_window, min_periods=max(2, self.baseline_window // 2)
        ).mean()
        dry_ratio = vol_ma / vol_baseline.replace(0.0, np.nan)
        atr_series = atr(high, low, close, self.atr_period)
        tr_over_atr = (high - low) / atr_series.replace(0.0, np.nan)

        frame = pd.DataFrame(
            {
                "volume": volume,
                "volume_ma": vol_ma,
                "volume_ratio": vol_ratio,
                "dry_ratio": dry_ratio,
                "atr": atr_series,
            }
        )

        patterns: list[VolumePattern] = []
        patterns += self._detect_dryup(frame, dry_ratio, close, atr_series, profile)
        patterns += self._detect_climax(
            frame, vol_ratio, tr_over_atr, open_, close, volume, atr_series, profile
        )
        patterns += self._detect_divergence(frame, close, volume, atr_series, profile)
        patterns += self._detect_accumulation(
            frame, high, low, close, volume, vol_ma, atr_series, profile
        )

        patterns.sort(key=lambda p: p.index_pos)
        return VolumePatternResult(
            patterns=tuple(patterns),
            frame=frame,
            symbol=symbol,
            interval=interval,
            extra={"has_profile": profile is not None},
        )

    # ------------------------------------------------------------------ #
    # Individual detectors
    # ------------------------------------------------------------------ #
    def _detect_dryup(
        self,
        frame: pd.DataFrame,
        dry_ratio: pd.Series,
        close: pd.Series,
        atr_series: pd.Series,
        profile: Optional[VolumeProfile],
    ) -> list[VolumePattern]:
        flag = (dry_ratio <= self.dryup_factor).fillna(False)
        frame["dry_up"] = flag
        events: list[VolumePattern] = []
        for start_i, end_i in _runs(flag):
            if end_i - start_i + 1 < self.dryup_min_run:
                continue
            # Anchor at the driest bar of the run.
            seg = dry_ratio.iloc[start_i : end_i + 1]
            anchor = start_i + int(np.argmin(seg.to_numpy()))
            n_bars = end_i - start_i + 1
            ratio = float(dry_ratio.iloc[anchor])
            price = float(close.iloc[anchor])
            level = _nearest_level(price, profile, self.level_tolerance_atr * float(atr_series.iloc[anchor]))
            expl = (
                f"Volume dry-up: {n_bars} bars with volume "
                f"≤{self.dryup_factor:.0%} of its longer-term baseline "
                f"(low {ratio:.0%}) — supply thinning, watch for expansion"
                + (f" near {level}" if level else "")
                + "."
            )
            events.append(
                self._make(
                    VolumePatternType.DRY_UP, "neutral", frame.index, anchor,
                    price, float(frame["volume"].iloc[anchor]),
                    strength=float(1.0 - ratio), start_i=start_i, end_i=end_i,
                    at_level=level, explanation=expl,
                )
            )
        return events

    def _detect_climax(
        self,
        frame: pd.DataFrame,
        vol_ratio: pd.Series,
        tr_over_atr: pd.Series,
        open_: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        atr_series: pd.Series,
        profile: Optional[VolumeProfile],
    ) -> list[VolumePattern]:
        flag = (
            (vol_ratio >= self.climax_volume_factor)
            & (tr_over_atr >= self.climax_range_factor)
        ).fillna(False)
        frame["climax"] = flag
        events: list[VolumePattern] = []
        for start_i, end_i in _runs(flag):
            seg = vol_ratio.iloc[start_i : end_i + 1]
            anchor = start_i + int(np.argmax(seg.to_numpy()))
            ratio = float(vol_ratio.iloc[anchor])
            up = float(close.iloc[anchor]) >= float(open_.iloc[anchor])
            direction = "bearish" if up else "bullish"
            kind = "buying" if up else "selling"
            price = float(close.iloc[anchor])
            level = _nearest_level(price, profile, self.level_tolerance_atr * float(atr_series.iloc[anchor]))
            expl = (
                f"Volume climax: {ratio:.1f}× average volume on a wide bar "
                f"({kind} climax)"
                + (f" at {level}" if level else "")
                + " — potential exhaustion / reversal."
            )
            events.append(
                self._make(
                    VolumePatternType.CLIMAX, direction, frame.index, anchor,
                    price, float(volume.iloc[anchor]), strength=ratio,
                    start_i=start_i, end_i=end_i, at_level=level, explanation=expl,
                )
            )
        return events

    def _detect_divergence(
        self,
        frame: pd.DataFrame,
        close: pd.Series,
        volume: pd.Series,
        atr_series: pd.Series,
        profile: Optional[VolumeProfile],
    ) -> list[VolumePattern]:
        w = self.divergence_window
        price_slope = rolling_slope(close, w) / close.rolling(w).mean()
        vol_slope = rolling_slope(volume, w) / volume.rolling(w).mean().replace(0.0, np.nan)
        eps = self.divergence_eps
        bearish = (price_slope > eps) & (vol_slope < -eps)
        bullish = (price_slope < -eps) & (vol_slope < -eps)
        frame["bearish_divergence"] = bearish.fillna(False)
        frame["bullish_divergence"] = bullish.fillna(False)

        events: list[VolumePattern] = []
        for flag, direction in ((bearish.fillna(False), "bearish"),
                                (bullish.fillna(False), "bullish")):
            for start_i, end_i in _runs(flag):
                if end_i - start_i + 1 < self.divergence_min_run:
                    continue
                anchor = end_i  # most recent bar of the run
                price = float(close.iloc[anchor])
                if direction == "bearish":
                    msg = "price rising while volume fades — rally lacks participation"
                else:
                    msg = "price falling on shrinking volume — selling pressure waning"
                expl = f"{direction.title()} volume divergence over {end_i - start_i + 1} bars: {msg}."
                events.append(
                    self._make(
                        VolumePatternType.DIVERGENCE, direction, frame.index, anchor,
                        price, float(volume.iloc[anchor]),
                        strength=float(abs(vol_slope.iloc[anchor]) / max(eps, 1e-9)),
                        start_i=start_i, end_i=end_i, at_level=None, explanation=expl,
                    )
                )
        return events

    def _detect_accumulation(
        self,
        frame: pd.DataFrame,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        vol_ma: pd.Series,
        atr_series: pd.Series,
        profile: Optional[VolumeProfile],
    ) -> list[VolumePattern]:
        w = self.volume_ma_window
        # Flat: net drift over the window is small relative to typical bar range.
        net_move = (close - close.shift(w)).abs()
        flat = net_move < (self.accumulation_flat_atr * atr_series)
        # Compressed: the window's high-low range is tighter than its own norm.
        bw = (high.rolling(w).max() - low.rolling(w).min()) / close
        compressed = bw < bw.rolling(self.baseline_window).median()
        # Below-average volume vs the longer baseline.
        vol_base = volume.rolling(self.baseline_window).mean()
        below_vol = vol_ma < vol_base
        depth = (1.0 - vol_ma / vol_base.replace(0.0, np.nan)).clip(0.0, 1.0)
        flag = (flat & compressed & below_vol).fillna(False)
        frame["accumulation"] = flag

        events: list[VolumePattern] = []
        for start_i, end_i in _runs(flag):
            if end_i - start_i + 1 < self.accumulation_min_run:
                continue
            anchor = end_i
            price = float(close.iloc[anchor])
            level = _nearest_level(price, profile, self.level_tolerance_atr * float(atr_series.iloc[anchor]))
            strength = float(depth.iloc[anchor]) if pd.notna(depth.iloc[anchor]) else 0.5
            expl = (
                f"Possible accumulation: {end_i - start_i + 1} bars coiling in a "
                "tight, flat range on below-average volume"
                + (f" near {level}" if level else "")
                + "."
            )
            events.append(
                self._make(
                    VolumePatternType.ACCUMULATION, "bullish", frame.index, anchor,
                    price, float(volume.iloc[anchor]),
                    strength=strength, start_i=start_i, end_i=end_i,
                    at_level=level, explanation=expl,
                )
            )
        return events

    # ------------------------------------------------------------------ #
    @staticmethod
    def _make(
        kind: VolumePatternType,
        direction: str,
        index: pd.Index,
        anchor: int,
        price: float,
        volume: float,
        *,
        strength: float,
        start_i: int,
        end_i: int,
        at_level: Optional[str],
        explanation: str,
    ) -> VolumePattern:
        is_dt = isinstance(index, pd.DatetimeIndex)
        return VolumePattern(
            type=kind,
            direction=direction,
            timestamp=index[anchor] if is_dt else None,
            index_pos=anchor,
            price=price,
            volume=volume,
            strength=float(strength),
            start=index[start_i] if is_dt else None,
            end=index[end_i] if is_dt else None,
            n_bars=end_i - start_i + 1,
            at_level=at_level,
            explanation=explanation,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _runs(flag: pd.Series) -> list[tuple[int, int]]:
    """Return ``(start, end)`` positional index pairs for contiguous True runs."""
    vals = flag.to_numpy(dtype=bool)
    runs: list[tuple[int, int]] = []
    i, n = 0, len(vals)
    while i < n:
        if vals[i]:
            j = i
            while j + 1 < n and vals[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _nearest_level(
    price: float, profile: Optional[VolumeProfile], tol: float
) -> Optional[str]:
    """Return ``"<NAME> <price>"`` of the nearest profile level within *tol*."""
    if profile is None or not np.isfinite(tol) or tol <= 0:
        return None
    candidates: list[tuple[str, float]] = [
        ("POC", profile.poc),
        ("VAH", profile.vah),
        ("VAL", profile.val),
    ]
    candidates += [("HVN", n.price) for n in profile.hvn]
    candidates += [("LVN", n.price) for n in profile.lvn]

    best_name: Optional[str] = None
    best_price = 0.0
    best_dist = tol
    # Strict `<` so ties favour the earlier, more significant level
    # (POC > VAH > VAL > HVN > LVN) rather than the last one checked.
    for name, level in candidates:
        dist = abs(price - level)
        if dist < best_dist:
            best_name, best_price, best_dist = name, level, dist
    if best_name is None:
        return None
    return f"{best_name} {best_price:.2f}"
