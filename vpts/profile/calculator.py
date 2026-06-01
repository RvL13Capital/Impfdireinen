"""Phase 1 — Volume Profile Calculator.

Computes a price-by-volume distribution from OHLCV data and extracts the
classic volume-profile levels used to read institutional activity:

* **POC**  – Point of Control (price with the most traded volume)
* **VAH / VAL** – Value Area High / Low (the band holding ~70% of volume)
* **HVN** – High Volume Nodes (acceptance / likely support-resistance)
* **LVN** – Low Volume Nodes (rejection / fast-move "air pockets")

Design notes
------------
* The calculator is intentionally **decoupled from data fetching**: it accepts a
  plain OHLCV :class:`pandas.DataFrame`, so it can be unit-tested with synthetic
  data and reused on any free data source.
* Free OHLCV data has no intra-bar ticks, so volume *within* a bar must be
  approximated. Two methods are provided:

  ``"uniform"`` (default)
      Spread each bar's volume evenly across its ``[Low, High]`` range,
      proportional to how much of each price bin the bar overlaps. This is the
      most faithful OHLC approximation and conserves total volume exactly.
  ``"typical"``
      Dump each bar's whole volume into the single bin containing its typical
      price ``(High + Low + Close) / 3``. Faster and closer to many charting
      packages, but coarser.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from vpts.profile.models import VolumeNode, VolumeProfile

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ("High", "Low", "Close", "Volume")
_VALID_METHODS = ("uniform", "typical")


class VolumeProfileCalculator:
    """Compute a :class:`~vpts.profile.models.VolumeProfile` from OHLCV data.

    Parameters
    ----------
    num_bins:
        Number of price bins (rows) in the profile. Ignored when *bin_size* is
        given. More bins = finer resolution but noisier nodes.
    bin_size:
        Fixed price increment per bin (e.g. a tick size). If provided, the bin
        count is derived from the data's price range and *num_bins* is ignored.
    value_area_pct:
        Fraction of total volume the value area must contain (default ``0.70``).
    distribution:
        ``"uniform"`` or ``"typical"`` (see module docstring).
    hvn_prominence:
        Minimum peak prominence for HVN/LVN detection, expressed as a fraction
        of the largest bin's volume (default ``0.10`` = 10%).
    node_min_distance:
        Minimum separation between detected nodes, in bins (default ``2``).
    smoothing_sigma:
        Gaussian smoothing applied to the histogram **for node detection only**
        (POC / value area always use the raw histogram). ``0`` disables it.
    max_nodes:
        Cap on how many HVNs and LVNs to report, ranked by strength.
    """

    def __init__(
        self,
        num_bins: int = 100,
        bin_size: Optional[float] = None,
        value_area_pct: float = 0.70,
        distribution: str = "uniform",
        hvn_prominence: float = 0.10,
        node_min_distance: int = 2,
        smoothing_sigma: float = 1.0,
        max_nodes: int = 5,
    ) -> None:
        if bin_size is not None:
            if bin_size <= 0:
                raise ValueError("bin_size must be positive.")
        elif num_bins < 2:
            raise ValueError("num_bins must be >= 2.")
        if not 0.0 < value_area_pct < 1.0:
            raise ValueError("value_area_pct must be in the open interval (0, 1).")
        if distribution not in _VALID_METHODS:
            raise ValueError(
                f"distribution must be one of {_VALID_METHODS}, got {distribution!r}."
            )
        if not 0.0 <= hvn_prominence <= 1.0:
            raise ValueError("hvn_prominence must be in [0, 1].")
        if node_min_distance < 1:
            raise ValueError("node_min_distance must be >= 1.")
        if smoothing_sigma < 0:
            raise ValueError("smoothing_sigma must be >= 0.")
        if max_nodes < 0:
            raise ValueError("max_nodes must be >= 0.")

        self.num_bins = int(num_bins)
        self.bin_size = bin_size
        self.value_area_pct = float(value_area_pct)
        self.distribution = distribution
        self.hvn_prominence = float(hvn_prominence)
        self.node_min_distance = int(node_min_distance)
        self.smoothing_sigma = float(smoothing_sigma)
        self.max_nodes = int(max_nodes)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def calculate(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> VolumeProfile:
        """Build a volume profile from an OHLCV frame.

        Parameters
        ----------
        df:
            DataFrame containing at least ``High``, ``Low``, ``Close`` and
            ``Volume`` columns. A :class:`~pandas.DatetimeIndex` is recorded as
            metadata when present but is not required.
        symbol, interval:
            Optional labels stored on the result for reporting/plotting.

        Returns
        -------
        VolumeProfile
            Immutable result holding POC, VAH/VAL, HVN/LVN and the raw
            histogram.

        Raises
        ------
        ValueError
            If required columns are missing, the frame is empty, or the data
            carries no positive volume.
        """
        high, low, close, volume = self._validate_and_extract(df)

        price_low = float(np.min(low))
        price_high = float(np.max(high))
        edges, centers, bin_size = self._build_bins(price_low, price_high)

        hist = self._distribute_volume(high, low, close, volume, edges, bin_size)
        total_volume = float(hist.sum())
        if total_volume <= 0:
            # Should not happen (we already checked volume > 0) but guard anyway.
            raise ValueError("Computed volume distribution is empty.")

        poc_index = int(np.argmax(hist))
        poc = float(centers[poc_index])
        poc_volume = float(hist[poc_index])

        low_idx, high_idx, va_volume = self._value_area(hist, poc_index, total_volume)
        val = float(centers[low_idx])
        vah = float(centers[high_idx])

        hvn, lvn = self._detect_nodes(hist, centers, poc_index, total_volume)

        start = end = None
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index):
            start, end = df.index[0], df.index[-1]

        return VolumeProfile(
            poc=poc,
            vah=vah,
            val=val,
            poc_volume=poc_volume,
            value_area_volume=va_volume,
            value_area_pct_target=self.value_area_pct,
            value_area_pct_actual=va_volume / total_volume,
            hvn=tuple(hvn),
            lvn=tuple(lvn),
            bin_edges=edges,
            bin_centers=centers,
            volume_distribution=hist,
            total_volume=total_volume,
            bin_size=bin_size,
            num_bins=len(centers),
            price_low=price_low,
            price_high=price_high,
            distribution_method=self.distribution,
            n_bars=int(len(close)),
            start=start,
            end=end,
            symbol=symbol,
            interval=interval,
        )

    # ------------------------------------------------------------------ #
    # Internal steps
    # ------------------------------------------------------------------ #
    def _validate_and_extract(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate columns and return clean float arrays (H, L, C, V)."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("`df` must be a pandas DataFrame.")
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing required column(s): {missing}. "
                f"Expected at least {list(_REQUIRED_COLUMNS)}."
            )
        if df.empty:
            raise ValueError("DataFrame is empty; cannot build a volume profile.")

        high = df["High"].to_numpy(dtype=float)
        low = df["Low"].to_numpy(dtype=float)
        close = df["Close"].to_numpy(dtype=float)
        volume = df["Volume"].to_numpy(dtype=float)

        # Drop bars with non-finite OHLC or volume, or non-positive volume.
        valid = (
            np.isfinite(high)
            & np.isfinite(low)
            & np.isfinite(close)
            & np.isfinite(volume)
            & (volume > 0)
            & (high >= low)
        )
        n_dropped = int((~valid).sum())
        if n_dropped:
            logger.debug("Dropping %d invalid/zero-volume bar(s).", n_dropped)
        high, low, close, volume = high[valid], low[valid], close[valid], volume[valid]

        if volume.size == 0 or volume.sum() <= 0:
            raise ValueError(
                "No positive volume found. Volume Profile is undefined for "
                "instruments without volume data (e.g. cash indices like "
                "'^GDAXI'). Use a tradable proxy with real volume — an ETF "
                "(e.g. 'SPY', 'EWG') or a futures contract."
            )
        return high, low, close, volume

    def _build_bins(
        self, price_low: float, price_high: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return ``(edges, centers, bin_size)`` for the price range."""
        if price_high <= price_low:
            # Degenerate range (all bars at one price): make a tiny symmetric box
            # so the profile is still well-defined and POC == that single price.
            pad = abs(price_low) * 1e-6 + 1e-9
            price_low, price_high = price_low - pad, price_high + pad

        if self.bin_size is not None:
            n = int(np.ceil((price_high - price_low) / self.bin_size))
            n = max(n, 1)
            edges = price_low + self.bin_size * np.arange(n + 1)
            # Guarantee the top edge covers price_high.
            if edges[-1] < price_high:
                edges = np.append(edges, edges[-1] + self.bin_size)
            bin_size = float(self.bin_size)
        else:
            edges = np.linspace(price_low, price_high, self.num_bins + 1)
            bin_size = float(edges[1] - edges[0])

        centers = (edges[:-1] + edges[1:]) / 2.0
        return edges, centers, bin_size

    def _distribute_volume(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        edges: np.ndarray,
        bin_size: float,
    ) -> np.ndarray:
        """Distribute per-bar volume across price bins → histogram array."""
        n_bins = len(edges) - 1
        hist = np.zeros(n_bins, dtype=float)

        if self.distribution == "typical":
            typical = (high + low + close) / 3.0
            # np.digitize returns 1..len(edges); shift to 0-based bin indices.
            idx = np.clip(np.digitize(typical, edges) - 1, 0, n_bins - 1)
            np.add.at(hist, idx, volume)
            return hist

        # --- "uniform": spread each bar over its [Low, High] proportional to
        # the overlap with each bin. Vectorised across bins, looped over bars. ---
        bottoms = edges[:-1]
        tops = edges[1:]
        price_low = edges[0]
        for lo, hi, vol in zip(low, high, volume):
            if hi <= lo:
                # Zero-range bar: assign all volume to the containing bin.
                k = int(np.clip((lo - price_low) / bin_size, 0, n_bins - 1))
                hist[k] += vol
                continue
            overlap = np.minimum(hi, tops) - np.maximum(lo, bottoms)
            np.clip(overlap, 0.0, None, out=overlap)
            # overlap.sum() == (hi - lo) since the range is within the edges,
            # so dividing by the span conserves the bar's total volume.
            hist += (overlap / (hi - lo)) * vol
        return hist

    def _value_area(
        self, hist: np.ndarray, poc_index: int, total_volume: float
    ) -> tuple[int, int, float]:
        """Expand around the POC until the value-area target is reached.

        Implements the classic Market-Profile rule: starting at the POC, repeatedly
        compare the volume of the next **two** rows above the value area against
        the next two rows below it, and annex whichever pair is heavier, until the
        accumulated volume reaches ``value_area_pct`` of the total.

        Returns
        -------
        (low_index, high_index, value_area_volume)
        """
        n = len(hist)
        target = self.value_area_pct * total_volume
        acc = float(hist[poc_index])
        low_idx = high_idx = poc_index

        while acc < target and (low_idx > 0 or high_idx < n - 1):
            up = [i for i in (high_idx + 1, high_idx + 2) if i < n]
            dn = [i for i in (low_idx - 1, low_idx - 2) if i >= 0]
            up_vol = float(hist[up].sum()) if up else -1.0
            dn_vol = float(hist[dn].sum()) if dn else -1.0

            # Prefer the heavier side; if one side is exhausted, take the other.
            take_up = bool(up) and (not dn or up_vol >= dn_vol)
            if take_up:
                acc += up_vol
                high_idx = up[-1]
            elif dn:
                acc += dn_vol
                low_idx = dn[-1]  # dn is [low-1, low-2]; last element is the lowest
            else:  # pragma: no cover - both sides exhausted (loop guard handles it)
                break

        return low_idx, high_idx, acc

    def _detect_nodes(
        self,
        hist: np.ndarray,
        centers: np.ndarray,
        poc_index: int,
        total_volume: float,
    ) -> tuple[list[VolumeNode], list[VolumeNode]]:
        """Detect High/Low Volume Nodes as peaks/valleys of the distribution."""
        n = len(hist)
        if n < 3 or self.max_nodes == 0:
            # Too few bins for meaningful peak detection — POC is the only node.
            poc_node = VolumeNode(
                price=float(centers[poc_index]),
                volume=float(hist[poc_index]),
                volume_pct=float(hist[poc_index] / total_volume * 100.0),
                kind="HVN",
            )
            return ([poc_node] if self.max_nodes else []), []

        signal = (
            gaussian_filter1d(hist, self.smoothing_sigma)
            if self.smoothing_sigma > 0
            else hist.astype(float)
        )
        peak_max = float(signal.max())
        if peak_max <= 0:
            return [], []
        prominence = self.hvn_prominence * peak_max

        # --- HVNs: peaks of the distribution (always include the POC) ---
        hvn_idx, _ = find_peaks(
            signal, prominence=prominence, distance=self.node_min_distance
        )
        hvn_set = set(int(i) for i in hvn_idx)
        hvn_set.add(poc_index)  # POC is, by definition, the strongest HVN.
        hvn_nodes = self._build_nodes(hvn_set, hist, centers, total_volume, "HVN")
        hvn_nodes.sort(key=lambda node: node.volume, reverse=True)  # strongest first

        # --- LVNs: valleys (peaks of the inverted distribution) ---
        lvn_nodes: list[VolumeNode] = []
        if n >= 5:
            valley_idx, _ = find_peaks(
                peak_max - signal,
                prominence=prominence,
                distance=self.node_min_distance,
            )
            lvn_nodes = self._build_nodes(
                set(int(i) for i in valley_idx), hist, centers, total_volume, "LVN"
            )
            lvn_nodes.sort(key=lambda node: node.volume)  # emptiest first

        return hvn_nodes[: self.max_nodes], lvn_nodes[: self.max_nodes]

    @staticmethod
    def _build_nodes(
        indices: set[int],
        hist: np.ndarray,
        centers: np.ndarray,
        total_volume: float,
        kind: str,
    ) -> list[VolumeNode]:
        """Turn a set of bin indices into :class:`VolumeNode` objects."""
        return [
            VolumeNode(
                price=float(centers[i]),
                volume=float(hist[i]),
                volume_pct=float(hist[i] / total_volume * 100.0),
                kind=kind,
            )
            for i in sorted(indices)
        ]
