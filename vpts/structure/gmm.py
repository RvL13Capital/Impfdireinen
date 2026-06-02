"""Parametric EM decomposition of the volume profile (1-D Gaussian mixture).

The heuristic decomposition in :mod:`vpts.structure.analytics` finds peaks on a
*smoothed* histogram (``distribution_peaks`` / ``double_distribution``) — a fast
but non-parametric reading. This module instead fits a **1-D Gaussian mixture by
weighted Expectation-Maximization**, treating each price bin as one observation
weighted by its volume. It recovers the *hidden POCs* (component means), their
widths, and mixing proportions, and from them derives a small set of **scale-free**
structural features:

* ``gmm_n_modes``       — BIC-selected component count (1/2/3): is the auction multi-modal?
* ``gmm_separation``    — ``(μ_hi − μ_lo) / pooled_σ``: how distinct the two hidden POCs are.
* ``gmm_weight_imbalance`` — ``π_hi − π_lo``: which hidden node holds more volume.
* ``gmm_antimode_depth`` — depth of the density valley (LVN) between the two modes ∈ [0, 1].
* ``gmm_dist_nearest``  — distance from price to the nearest hidden POC (price sitting on a node?).
* ``gmm_price_zone``    — which side of the antimode (LVN transition zone) price is on.
* ``gmm_gravity``       — signed gap from price to the *dominant* hidden POC (fair-value pull).

Everything is computed in **normalized price space** ``u = (p − low)/(high − low) ∈ [0,1]``
so the features are comparable across instruments. The result is a
:class:`~vpts.ml.models.FactorDataset` that feeds the very same CPCV factor harness
as the heuristic features — so the honest question ("does the principled
decomposition carry out-of-sample signal, and does it survive survivorship?") can
be asked on equal footing. **No new dependency: pure numpy.**
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from vpts.profile.calculator import VolumeProfileCalculator
from vpts.profile.models import VolumeProfile
from vpts.regime.indicators import ensure_ohlcv
from vpts.ml.models import FactorDataset

_EPS = 1e-12

# The numeric feature matrix fed to the CPCV factor harness (see module docstring).
GMM_FEATURES: tuple[str, ...] = (
    "gmm_n_modes",
    "gmm_separation",
    "gmm_weight_imbalance",
    "gmm_antimode_depth",
    "gmm_dist_nearest",
    "gmm_price_zone",
    "gmm_gravity",
)


@dataclass(frozen=True)
class GMMFit:
    """A fitted 1-D Gaussian mixture (means sorted ascending, in [0,1] price space)."""

    means: np.ndarray       # (k,) component means, ascending
    sigmas: np.ndarray      # (k,) component std-devs
    weights: np.ndarray     # (k,) mixing proportions, sum to 1
    k: int
    loglik: float           # per-observation weighted log-likelihood
    bic: float              # Bayesian information criterion (lower = better)


# --------------------------------------------------------------------------- #
# Weighted moment / quantile helpers (numpy, deterministic)
# --------------------------------------------------------------------------- #
def _weighted_var(x: np.ndarray, wn: np.ndarray) -> float:
    mean = float((wn * x).sum())
    return float((wn * (x - mean) ** 2).sum())


def _weighted_quantiles(x: np.ndarray, wn: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Weighted quantiles of already-ascending ``x`` with normalized weights ``wn``."""
    cum = np.cumsum(wn) - 0.5 * wn          # midpoint rule
    return np.interp(qs, cum, x)


def _normal_pdf(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


# --------------------------------------------------------------------------- #
# The EM fit
# --------------------------------------------------------------------------- #
def fit_gmm_1d(
    x: np.ndarray,
    w: np.ndarray,
    k: int,
    *,
    var_floor: float = 1e-5,
    n_eff: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> GMMFit:
    """Fit a weighted 1-D ``k``-component Gaussian mixture by EM.

    ``x`` are observation locations (ascending), ``w`` their non-negative weights
    (need not sum to 1). Initialisation is **deterministic** (component means at
    weighted quantiles), so the fit — and every feature derived from it — is fully
    reproducible. ``n_eff`` is the honest sample size used for the BIC penalty
    (default: the number of observations); pass the bar count so smoothing/binning
    cannot inflate it. Variances are floored at ``var_floor`` to stop a component
    collapsing onto a single bin.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    tot = float(w.sum())
    ne = float(n_eff) if n_eff is not None else float(max(x.size, 1))

    # Degenerate: no volume, too few points, or asking for more comps than points.
    nz = int(np.count_nonzero(w > _EPS))
    if tot <= _EPS or x.size < k or nz < k:
        mu = float((w * x).sum() / tot) if tot > _EPS else float(x.mean() if x.size else 0.5)
        ll = 0.0
        return GMMFit(np.array([mu]), np.array([np.sqrt(max(var_floor, _EPS))]),
                      np.array([1.0]), 1, ll, _bic(ll, 1, ne))

    wn = w / tot
    # Deterministic init: means at weighted quantiles; split if they coincide.
    mu = _weighted_quantiles(x, wn, (np.arange(k) + 0.5) / k).astype(float)
    if k > 1 and float(mu.max() - mu.min()) < 1e-3:
        lo, hi = float(x.min()), float(x.max())
        mu = np.linspace(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), k)
    var = np.full(k, max(var_floor, _weighted_var(x, wn) / k), dtype=float)
    pi = np.full(k, 1.0 / k, dtype=float)

    ll_prev = -np.inf
    ll = 0.0
    for _ in range(max_iter):
        comp = pi[None, :] * _normal_pdf(x[:, None], mu[None, :], var[None, :])   # (n, k)
        dens = comp.sum(axis=1) + _EPS
        resp = comp / dens[:, None]
        ll = float((wn * np.log(dens)).sum())
        Nk = (wn[:, None] * resp).sum(axis=0) + _EPS                              # (k,)
        mu = (wn[:, None] * resp * x[:, None]).sum(axis=0) / Nk
        var = np.maximum((wn[:, None] * resp * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / Nk,
                         var_floor)
        pi = Nk / float(Nk.sum())
        if ll - ll_prev < tol:
            break
        ll_prev = ll

    order = np.argsort(mu)
    return GMMFit(mu[order], np.sqrt(var[order]), pi[order], k, ll, _bic(ll, k, ne))


def _bic(loglik_per_obs: float, k: int, n_eff: float) -> float:
    """BIC from the *per-observation* weighted log-likelihood and honest sample size."""
    total_ll = loglik_per_obs * n_eff
    n_params = 3 * k - 1                      # k means + k vars + (k-1) free weights
    return float(-2.0 * total_ll + n_params * np.log(max(n_eff, 2.0)))


# --------------------------------------------------------------------------- #
# Profile → features
# --------------------------------------------------------------------------- #
def _neutral() -> np.ndarray:
    """Unimodal / degenerate reading: one mode, no separation, no valley."""
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)


def gmm_feature_vector(profile: VolumeProfile, close_price: float, *, k_max: int = 3) -> np.ndarray:
    """Decompose ``profile`` by EM and return the :data:`GMM_FEATURES` row.

    BIC over ``k ∈ {1..k_max}`` selects the mode count; the *geometry* features
    (separation, antimode, zone, gravity) read the two-component fit, which on a
    truly unimodal profile collapses to near-zero separation — the correct
    "no structure" answer. All values are in normalized ``[0,1]`` price space.
    """
    hist = np.asarray(profile.volume_distribution, dtype=float)
    centers = np.asarray(profile.bin_centers, dtype=float)
    lo, hi = float(profile.price_low), float(profile.price_high)
    rng = hi - lo
    if hist.size < 4 or rng <= _EPS or hist.sum() <= _EPS or int(np.count_nonzero(hist > _EPS)) < 2:
        return _neutral()

    u = (centers - lo) / rng                                   # normalized bin prices ∈ [0,1]
    uc = float(np.clip((close_price - lo) / rng, -0.5, 1.5))   # normalized close
    n_eff = float(max(profile.n_bars, 2))
    var_floor = max((1.0 / (2.0 * max(profile.num_bins, 1))) ** 2, 1e-6)

    # --- BIC model selection for the mode count ---
    fits = []
    for k in range(1, max(1, k_max) + 1):
        if k > u.size:
            break
        fits.append(fit_gmm_1d(u, hist, k, var_floor=var_floor, n_eff=n_eff))
    if not fits:
        return _neutral()
    best = min(fits, key=lambda f: f.bic)
    n_modes = float(best.k)

    # --- geometry from the two-component fit (μ ascending) ---
    two = next((f for f in fits if f.k == 2), None)
    if two is None:
        return np.array([n_modes, 0.0, 0.0, 0.0,
                         abs(uc - float(fits[0].means[0])), 0.0,
                         uc - float(fits[0].means[0])], dtype=float)

    mu_lo, mu_hi = float(two.means[0]), float(two.means[1])
    s_lo, s_hi = float(two.sigmas[0]), float(two.sigmas[1])
    p_lo, p_hi = float(two.weights[0]), float(two.weights[1])
    pooled_sigma = float(np.sqrt(p_lo * s_lo ** 2 + p_hi * s_hi ** 2 + _EPS))
    separation = (mu_hi - mu_lo) / pooled_sigma if pooled_sigma > _EPS else 0.0
    weight_imbalance = p_hi - p_lo                              # >0 ⇒ upper node heavier

    # antimode: lowest mixture density between the two means
    if mu_hi - mu_lo > 1e-4:
        grid = np.linspace(mu_lo, mu_hi, 64)
        dens = (two.weights[None, :] * _normal_pdf(grid[:, None], two.means[None, :],
                                                   (two.sigmas[None, :] ** 2))).sum(axis=1)
        j = int(np.argmin(dens))
        u_anti = float(grid[j])
        modal = (two.weights * _normal_pdf(two.means, two.means, two.sigmas ** 2))
        antimode_depth = float(np.clip(1.0 - dens[j] / (float(modal.min()) + _EPS), 0.0, 1.0))
        price_zone = float(np.sign(uc - u_anti)) if antimode_depth > 0.02 else 0.0
    else:
        antimode_depth = 0.0
        price_zone = 0.0

    dist_nearest = min(abs(uc - mu_lo), abs(uc - mu_hi))
    mu_dom = mu_hi if p_hi >= p_lo else mu_lo
    gravity = uc - mu_dom

    return np.array([n_modes, separation, weight_imbalance, antimode_depth,
                     dist_nearest, price_zone, gravity], dtype=float)


# --------------------------------------------------------------------------- #
# Dataset builder (features → forward return), mirrors build_factor_dataset
# --------------------------------------------------------------------------- #
def build_gmm_dataset(
    df: pd.DataFrame,
    *,
    lookback: int = 120,
    horizon: int = 20,
    stride: int = 3,
    k_max: int = 3,
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    profile_calculator: Optional[VolumeProfileCalculator] = None,
) -> FactorDataset:
    """Walk the bars → (EM-GMM profile features → forward ``horizon``-bar return).

    Features at bar ``t`` use only the trailing ``lookback`` window (data ``≤ t``);
    the label is the strictly-future return ``close[t+horizon]/close[t] − 1``, so
    there is no look-ahead. Returns a :class:`~vpts.ml.models.FactorDataset` ready
    for :func:`~vpts.ml.factor_model.cpcv_factor_eval`.
    """
    ensure_ohlcv(df, min_bars=lookback + horizon + 2)
    pc = profile_calculator or VolumeProfileCalculator(bin_mode="auto")
    close = df["Close"].to_numpy(float)
    n = len(df)

    feats: list[np.ndarray] = []
    ys: list[float] = []
    ts: list = []
    for t in range(lookback - 1, n - horizon, max(1, stride)):
        window = df.iloc[t - lookback + 1 : t + 1]
        try:
            profile = pc.calculate(window, symbol, interval)
        except (ValueError, ZeroDivisionError):
            continue
        vec = gmm_feature_vector(profile, close[t], k_max=k_max)
        if not np.all(np.isfinite(vec)):
            continue
        feats.append(vec)
        ys.append(float(close[t + horizon] / close[t] - 1.0))
        ts.append(df.index[t])

    is_dt = isinstance(df.index, pd.DatetimeIndex) and bool(ts)
    return FactorDataset(
        X=np.array(feats, dtype=float).reshape(-1, len(GMM_FEATURES)),
        y=np.array(ys, dtype=float),
        baseline=np.zeros(len(ys), dtype=float),     # no hand-weighted baseline for GMM
        feature_names=GMM_FEATURES,
        horizon=horizon,
        stride=max(1, stride),
        timestamps=pd.DatetimeIndex(ts) if is_dt else None,
        symbol=symbol,
    )
