"""Structural analytics — turning a static profile into quantifiable signals.

Pure, dependency-light functions (numpy + scipy.stats/signal) implementing the
retail-focused microstructure math:

* **Synthetic delta** — Close Location Value × volume, an OHLC estimate of
  aggressive buy/sell pressure (no level-2 data needed), mapped onto the profile.
* **Profile shape** — volume-weighted skewness / kurtosis, double-distribution
  detection, and a P / b / B / D shape classification.
* **Institutional footprints** — volume *ledges* (cliff-like drops between
  adjacent bins) and *poor highs* (blunt, unfinished auctions).
* **Time-decayed gravity** — an EMA-decayed profile whose POC reveals cost-basis
  migration vs the lifetime POC.

Everything here operates on a *single* window's data + its
:class:`~vpts.profile.models.VolumeProfile`; the rolling/time-series features
(VACR z-score, POC-migration slope) are assembled by
:mod:`vpts.structure.dataset`, which walks bars with no look-ahead.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from vpts.profile.models import VolumeProfile

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Synthetic Cumulative Volume Delta (CVD) — order-flow estimate from OHLC
# --------------------------------------------------------------------------- #
def close_location_value(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """Close Location Value ∈ [-1, +1] per bar.

    ``CLV = ((C - L) - (H - C)) / (H - L)`` — +1 when the bar closes on its high
    (aggressive buying), -1 on its low. Zero-range bars yield 0 (no information).
    """
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    close = np.asarray(close, float)
    rng = high - low
    clv = np.zeros_like(rng)
    nz = rng > _EPS
    clv[nz] = ((close[nz] - low[nz]) - (high[nz] - close[nz])) / rng[nz]
    return np.clip(clv, -1.0, 1.0)


def _distribute_uniform(
    per_bar: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Spread a per-bar quantity across price bins ∝ overlap with ``[low, high]``.

    Mirrors :meth:`VolumeProfileCalculator._distribute_volume` (uniform method)
    but works for *signed* quantities (e.g. synthetic delta), so the result
    aligns bin-for-bin with the profile's volume histogram.
    """
    n_bins = len(edges) - 1
    out = np.zeros(n_bins, dtype=float)
    bottoms, tops = edges[:-1], edges[1:]
    price_low = edges[0]
    span = float(edges[-1] - edges[0])
    bin_size = span / n_bins if n_bins else 1.0
    for lo, hi, val in zip(low, high, per_bar):
        if hi - lo <= _EPS:
            k = int(np.clip((lo - price_low) / max(bin_size, _EPS), 0, n_bins - 1))
            out[k] += val
            continue
        overlap = np.minimum(hi, tops) - np.maximum(lo, bottoms)
        np.clip(overlap, 0.0, None, out=overlap)
        out += (overlap / (hi - lo)) * val
    return out


def synthetic_delta_profile(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Signed-volume (``volume × CLV``) histogram aligned to ``edges``."""
    delta = np.asarray(volume, float) * close_location_value(high, low, close)
    return _distribute_uniform(delta, np.asarray(low, float), np.asarray(high, float), edges)


def synthetic_delta_stats(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    profile: VolumeProfile,
) -> tuple[float, float]:
    """Return ``(net_delta_frac, poc_delta_frac)`` for a window.

    ``net_delta_frac`` is total signed volume / total volume ∈ [-1, 1] (overall
    buy/sell dominance). ``poc_delta_frac`` is the signed-volume fraction *at the
    POC bin* — strongly positive ⇒ passive accumulation defending fair value
    (a "buy the dip" tell); strongly negative ⇒ distribution.
    """
    edges = profile.bin_edges
    delta_hist = synthetic_delta_profile(high, low, close, volume, edges)
    vol_hist = _distribute_uniform(np.asarray(volume, float),
                                   np.asarray(low, float), np.asarray(high, float), edges)
    total = float(vol_hist.sum())
    net = float(delta_hist.sum()) / total if total > _EPS else 0.0
    k = int(np.argmin(np.abs(profile.bin_centers - profile.poc)))
    poc_frac = float(delta_hist[k] / vol_hist[k]) if vol_hist[k] > _EPS else 0.0
    return float(np.clip(net, -1, 1)), float(np.clip(poc_frac, -1, 1))


# --------------------------------------------------------------------------- #
# Profile shape — volume-weighted moments & topology
# --------------------------------------------------------------------------- #
def weighted_moments(centers: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, float]:
    """Volume-weighted ``(mean, std, skewness, kurtosis)`` of the distribution.

    ``centers`` are bin prices, ``weights`` the per-bin volume. Skewness is the
    standardized 3rd moment (negative ⇒ long tail toward *low* prices, i.e. a
    top-heavy "P"); kurtosis is Pearson's (normal ≈ 3). Degenerate inputs return
    a neutral ``(mean, 0, 0, 3)``.
    """
    w = np.asarray(weights, float)
    x = np.asarray(centers, float)
    tot = float(w.sum())
    if tot <= _EPS:
        return float(x.mean() if x.size else 0.0), 0.0, 0.0, 3.0
    p = w / tot
    mean = float((p * x).sum())
    var = float((p * (x - mean) ** 2).sum())
    if var <= _EPS:
        return mean, 0.0, 0.0, 3.0
    sd = float(np.sqrt(var))
    skew = float((p * ((x - mean) / sd) ** 3).sum())
    kurt = float((p * ((x - mean) / sd) ** 4).sum())
    return mean, sd, skew, kurt


def poc_location(profile: VolumeProfile) -> float:
    """POC position within the high-low range ∈ [0, 1] (1 = at the high)."""
    rng = profile.price_high - profile.price_low
    if rng <= _EPS:
        return 0.5
    return float(np.clip((profile.poc - profile.price_low) / rng, 0.0, 1.0))


def distribution_peaks(
    profile: VolumeProfile, prominence_frac: float = 0.15, sigma: float = 1.0
) -> tuple[np.ndarray, int]:
    """Prominent peaks of the (smoothed) volume distribution → ``(idx, count)``."""
    hist = profile.volume_distribution.astype(float)
    if hist.size < 3:
        return np.array([], dtype=int), 0
    sig = gaussian_filter1d(hist, sigma) if sigma > 0 else hist
    peak_max = float(sig.max())
    if peak_max <= _EPS:
        return np.array([], dtype=int), 0
    idx, _ = find_peaks(sig, prominence=prominence_frac * peak_max, distance=2)
    return idx, int(idx.size)


def double_distribution(
    profile: VolumeProfile, valley_frac: float = 0.20, **kw
) -> tuple[bool, float]:
    """Detect a two-peak (B-shape) profile with a deep LVN valley between peaks.

    Returns ``(is_double, valley_ratio)`` where ``valley_ratio`` is the lowest
    bin between the two tallest peaks as a fraction of the *mean* bin volume.
    Bimodal iff exactly two prominent peaks and that valley < ``valley_frac``.
    """
    idx, n = distribution_peaks(profile, **kw)
    hist = profile.volume_distribution.astype(float)
    mean_bin = float(hist.mean()) if hist.size else 0.0
    if n < 2 or mean_bin <= _EPS:
        return False, 1.0
    order = idx[np.argsort(hist[idx])[::-1]]
    a, b = sorted(order[:2].tolist())
    valley = float(hist[a : b + 1].min())
    ratio = valley / mean_bin
    return bool(n == 2 and ratio < valley_frac), float(ratio)


# Shape-class codes (small ints so they slot into a numeric feature matrix).
SHAPE_UNKNOWN, SHAPE_P, SHAPE_b, SHAPE_D, SHAPE_B = 0, 1, 2, 3, 4
SHAPE_NAMES = {0: "unknown", 1: "P", 2: "b", 3: "D", 4: "B"}


def classify_shape(
    profile: VolumeProfile,
    skew: Optional[float] = None,
    *,
    skew_thresh: float = 0.5,
    top_frac: float = 0.70,
    bottom_frac: float = 0.30,
) -> int:
    """Classify the profile topology into P / b / B / D (see module docstring).

    Order: a clean double-distribution wins (B); else a top-heavy left-skew is a
    **P** (accumulation/short-squeeze) and a bottom-heavy right-skew a **b**
    (distribution/capitulation); a roughly symmetric single peak is **D** (range).
    """
    if skew is None:
        _, _, skew, _ = weighted_moments(profile.bin_centers, profile.volume_distribution)
    is_double, _ = double_distribution(profile)
    if is_double:
        return SHAPE_B
    loc = poc_location(profile)
    if skew < -skew_thresh and loc > top_frac:
        return SHAPE_P
    if skew > skew_thresh and loc < bottom_frac:
        return SHAPE_b
    if abs(skew) < skew_thresh:
        return SHAPE_D
    return SHAPE_UNKNOWN


# --------------------------------------------------------------------------- #
# Institutional footprints — ledges & poor highs
# --------------------------------------------------------------------------- #
def detect_ledges(profile: VolumeProfile, z: float = 3.0) -> list[int]:
    """Bin indices where volume drops/jumps > ``z`` σ between adjacent bins.

    A ledge is a cliff-like step in the histogram — passive limit-order walls that
    make the highest-conviction stop-loss locations. Returns the *upper* index of
    each qualifying adjacent pair.
    """
    hist = profile.volume_distribution.astype(float)
    if hist.size < 4:
        return []
    diffs = np.diff(hist)
    sd = float(diffs.std())
    if sd <= _EPS:
        return []
    return [int(i + 1) for i in np.flatnonzero(np.abs(diffs) > z * sd)]


def poor_high(profile: VolumeProfile, top_frac: float = 0.02, vol_frac: float = 0.50) -> bool:
    """True if the auction is *unfinished* at the top (blunt, not tapering).

    Looks at the top ``top_frac`` of price bins; flags a poor high if their mean
    volume exceeds ``vol_frac`` × the average bin volume (the profile should
    taper to ~0 at a *finished* high). Poor highs are magnetically revisited.
    """
    hist = profile.volume_distribution.astype(float)
    n = hist.size
    if n < 5:
        return False
    k = max(1, int(round(top_frac * n)))
    avg = float(hist.mean())
    if avg <= _EPS:
        return False
    return bool(float(hist[-k:].mean()) > vol_frac * avg)


def poor_low(profile: VolumeProfile, top_frac: float = 0.02, vol_frac: float = 0.50) -> bool:
    """Symmetric counterpart of :func:`poor_high` at the bottom of the range."""
    hist = profile.volume_distribution.astype(float)
    n = hist.size
    if n < 5:
        return False
    k = max(1, int(round(top_frac * n)))
    avg = float(hist.mean())
    if avg <= _EPS:
        return False
    return bool(float(hist[:k].mean()) > vol_frac * avg)


# --------------------------------------------------------------------------- #
# Time-decayed volume gravity — cost-basis migration
# --------------------------------------------------------------------------- #
def decayed_poc(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    profile: VolumeProfile,
    halflife: float = 21.0,
) -> float:
    """POC of an EMA-decayed profile (recent bars weighted more heavily).

    Each bar's volume is scaled by ``0.5 ** (age / halflife)`` (age 0 = most
    recent), the decayed histogram is rebuilt on the profile's bins, and its POC
    price is returned. Compared with the lifetime POC this reveals whether recent
    trade is migrating to higher/lower fair value.
    """
    volume = np.asarray(volume, float)
    n = volume.size
    if n == 0:
        return float(profile.poc)
    age = np.arange(n - 1, -1, -1, dtype=float)         # last bar -> age 0
    w = 0.5 ** (age / max(halflife, _EPS))
    hist = _distribute_uniform(volume * w, np.asarray(low, float),
                               np.asarray(high, float), profile.bin_edges)
    if hist.sum() <= _EPS:
        return float(profile.poc)
    return float(profile.bin_centers[int(np.argmax(hist))])


def value_area_compression_ratio(profile: VolumeProfile, ref_price: Optional[float] = None) -> float:
    """Value-area width as a fraction of price: ``(VAH - VAL) / ref_price``.

    The raw "quietness" measure; :mod:`vpts.structure.dataset` z-scores it over a
    rolling window to flag a "coiled spring" (the true quiet trigger).
    """
    ref = ref_price if (ref_price and ref_price > _EPS) else profile.poc
    if ref <= _EPS:
        return 0.0
    return float(profile.value_area_width / ref)
