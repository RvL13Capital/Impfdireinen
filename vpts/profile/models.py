"""Immutable result objects produced by the Volume Profile Calculator.

Keeping the *result* separate from the *calculator* means downstream phases
(quiet-phase detection, confluence scoring, signal generation, plotting) can
consume a stable, well-documented data contract without caring how the numbers
were produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolumeNode:
    """A single notable price level in the volume distribution.

    Attributes
    ----------
    price:
        Bin-center price of the node.
    volume:
        Total volume traded in that bin (raw, in shares/contracts).
    volume_pct:
        Volume of the node as a percentage of the profile's total volume.
    kind:
        Either ``"HVN"`` (High Volume Node) or ``"LVN"`` (Low Volume Node).
    """

    price: float
    volume: float
    volume_pct: float
    kind: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.kind} @ {self.price:.4f} ({self.volume_pct:.1f}% of vol)"


@dataclass(frozen=True)
class VolumeProfile:
    """The full result of a volume-profile computation.

    All price levels (:attr:`poc`, :attr:`vah`, :attr:`val`) are reported as
    **bin-center prices** so they are directly comparable to each other and to
    HVN/LVN node prices. The raw building blocks (:attr:`bin_edges`,
    :attr:`bin_centers`, :attr:`volume_distribution`) are exposed so any other
    convention (e.g. value-area band edges) can be derived downstream.

    Notes
    -----
    Because free OHLCV data has no intra-bar tick detail, the volume that
    occurred *within* each bar is approximated (see
    :class:`~vpts.profile.calculator.VolumeProfileCalculator`). The profile is
    therefore an *estimate* of where volume traded, not an exact footprint.
    """

    # --- core levels (bin-center prices) ---
    poc: float
    """Point of Control — price level with the most traded volume."""
    vah: float
    """Value Area High — upper bound of the value area."""
    val: float
    """Value Area Low — lower bound of the value area."""

    # --- value-area accounting ---
    poc_volume: float
    value_area_volume: float
    value_area_pct_target: float
    value_area_pct_actual: float

    # --- nodes ---
    hvn: tuple[VolumeNode, ...]
    lvn: tuple[VolumeNode, ...]

    # --- raw building blocks ---
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    volume_distribution: np.ndarray

    # --- bookkeeping / metadata ---
    total_volume: float
    bin_size: float
    num_bins: int
    price_low: float
    price_high: float
    distribution_method: str
    n_bars: int
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    symbol: Optional[str] = None
    interval: Optional[str] = None
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Derived convenience properties
    # ------------------------------------------------------------------ #
    @property
    def value_area_width(self) -> float:
        """Price width of the value area (VAH - VAL)."""
        return float(self.vah - self.val)

    @property
    def profile_range(self) -> float:
        """Full price range covered by the profile (high - low)."""
        return float(self.price_high - self.price_low)

    @property
    def poc_index(self) -> int:
        """Index of the POC bin within :attr:`volume_distribution`."""
        return int(np.argmax(self.volume_distribution))

    # ------------------------------------------------------------------ #
    # Lookups useful for later phases (scoring / signals)
    # ------------------------------------------------------------------ #
    def is_in_value_area(self, price: float) -> bool:
        """Return ``True`` if *price* lies within ``[VAL, VAH]`` (inclusive)."""
        return bool(self.val <= price <= self.vah)

    def location(self, price: float) -> str:
        """Classify *price* relative to the value area.

        Returns one of ``"above_value"``, ``"in_value"`` or ``"below_value"``.
        """
        if price > self.vah:
            return "above_value"
        if price < self.val:
            return "below_value"
        return "in_value"

    def nearest_node(
        self, price: float, kind: Optional[str] = None
    ) -> Optional[VolumeNode]:
        """Return the HVN/LVN closest to *price*.

        Parameters
        ----------
        price:
            Reference price.
        kind:
            ``"HVN"`` or ``"LVN"`` to restrict the search; ``None`` searches
            both. Returns ``None`` if no matching node exists.
        """
        nodes: list[VolumeNode] = []
        if kind in (None, "HVN"):
            nodes.extend(self.hvn)
        if kind in (None, "LVN"):
            nodes.extend(self.lvn)
        if not nodes:
            return None
        return min(nodes, key=lambda n: abs(n.price - price))

    # ------------------------------------------------------------------ #
    # Export helpers
    # ------------------------------------------------------------------ #
    def to_dataframe(self) -> pd.DataFrame:
        """Return the histogram as a tidy, plot-ready :class:`pandas.DataFrame`.

        Indexed by bin-center price, with the per-bin volume, its percentage of
        total volume, and boolean flags marking the POC, value-area membership,
        and HVN/LVN bins. This is the table the Phase 5 dashboard will plot.
        """
        total = self.total_volume if self.total_volume > 0 else np.nan
        hvn_prices = {round(n.price, 10) for n in self.hvn}
        lvn_prices = {round(n.price, 10) for n in self.lvn}

        df = pd.DataFrame(
            {
                "price": self.bin_centers,
                "volume": self.volume_distribution,
                "volume_pct": self.volume_distribution / total * 100.0,
            }
        )
        df["in_value_area"] = (df["price"] >= self.val) & (df["price"] <= self.vah)
        df["is_poc"] = np.isclose(df["price"], self.poc)
        df["is_hvn"] = df["price"].round(10).isin(hvn_prices)
        df["is_lvn"] = df["price"].round(10).isin(lvn_prices)
        return df.set_index("price")

    def as_dict(self) -> dict:
        """Return a flat, JSON-serialisable summary (no numpy arrays)."""
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "start": None if self.start is None else str(self.start),
            "end": None if self.end is None else str(self.end),
            "n_bars": self.n_bars,
            "poc": self.poc,
            "vah": self.vah,
            "val": self.val,
            "value_area_pct_target": self.value_area_pct_target,
            "value_area_pct_actual": self.value_area_pct_actual,
            "total_volume": self.total_volume,
            "num_bins": self.num_bins,
            "bin_size": self.bin_size,
            "price_low": self.price_low,
            "price_high": self.price_high,
            "distribution_method": self.distribution_method,
            "bin_mode": self.extra.get("bin_mode"),
            "hvn": [n.price for n in self.hvn],
            "lvn": [n.price for n in self.lvn],
        }

    def summary(self) -> str:
        """Return a human-readable multi-line summary string."""
        sym = self.symbol or "data"
        itv = f" {self.interval}" if self.interval else ""
        lines = [
            f"Volume Profile — {sym}{itv}  ({self.n_bars} bars)",
            "-" * 52,
            f"  Range          : {self.price_low:.4f} → {self.price_high:.4f}"
            f"  ({self.num_bins} bins, size {self.bin_size:.4f})",
            f"  POC            : {self.poc:.4f}"
            f"   ({self.poc_volume / max(self.total_volume, 1e-9) * 100:.1f}% of vol)",
            f"  Value Area     : VAL {self.val:.4f}  …  VAH {self.vah:.4f}"
            f"   (width {self.value_area_width:.4f})",
            f"  VA coverage    : {self.value_area_pct_actual * 100:.1f}% "
            f"(target {self.value_area_pct_target * 100:.0f}%)",
            f"  Total volume   : {self.total_volume:,.0f}",
        ]
        if self.extra.get("bin_mode") == "auto":
            lines.insert(
                3,
                f"  Binning        : auto — ATR({self.extra.get('atr_period')})="
                f"{self.extra.get('atr', float('nan')):.4f}, "
                f"target bin ≈ {self.extra.get('target_bin_width', float('nan')):.4f}",
            )
        if self.hvn:
            hvn_str = ", ".join(f"{n.price:.4f}" for n in self.hvn)
            lines.append(f"  HVN ({len(self.hvn)})       : {hvn_str}")
        if self.lvn:
            lvn_str = ", ".join(f"{n.price:.4f}" for n in self.lvn)
            lines.append(f"  LVN ({len(self.lvn)})       : {lvn_str}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()
