"""Immutable objects for the learned factor-weight model + its CPCV evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorDataset:
    """Supervised dataset: confluence-factor features → forward returns.

    Each row is one decision bar. ``X`` holds the four (weight-free) confluence
    factors — value-area, key-level and pattern *signed strengths* plus the quiet
    *strength* — built from data **up to** that bar; ``y`` is the forward
    ``horizon``-bar return (strictly future), so there is no look-ahead. ``baseline``
    is the current hand-weighted ``bias_score`` for an apples-to-apples comparison.
    """

    X: np.ndarray
    y: np.ndarray
    baseline: np.ndarray
    feature_names: tuple[str, ...]
    horizon: int
    stride: int
    timestamps: Optional[pd.DatetimeIndex] = None
    symbol: Optional[str] = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def purge_samples(self) -> int:
        """Label-horizon expressed in *samples* (for CPCV purging)."""
        return int(np.ceil(self.horizon / max(self.stride, 1)))


@dataclass(frozen=True)
class FactorCVResult:
    """Out-of-sample evaluation of learned factor weights across CPCV folds."""

    feature_names: tuple[str, ...]
    mean_weights: tuple[float, ...]       # standardized ridge weights, averaged over folds
    oos_ic_mean: float
    oos_ic_median: float
    oos_ic_std: float
    pct_folds_positive_ic: float
    oos_dir_accuracy: float               # 0..1
    oos_ls_return_pct: float              # mean per-sample long/short return, %
    baseline_ic_mean: float               # hand-weighted bias_score IC (OOS)
    n_paths: int
    n_samples: int
    fold_ics: tuple[float, ...]
    alpha: float = 1.0
    symbol: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_samples": self.n_samples,
            "n_paths": self.n_paths,
            "alpha": self.alpha,
            "weights": {n: round(w, 4) for n, w in zip(self.feature_names, self.mean_weights)},
            "oos_ic_mean": round(self.oos_ic_mean, 4),
            "oos_ic_median": round(self.oos_ic_median, 4),
            "oos_ic_std": round(self.oos_ic_std, 4),
            "pct_folds_positive_ic": round(self.pct_folds_positive_ic, 1),
            "oos_dir_accuracy": round(self.oos_dir_accuracy, 4),
            "oos_ls_return_pct": round(self.oos_ls_return_pct, 4),
            "baseline_ic_mean": round(self.baseline_ic_mean, 4),
        }

    def summary(self) -> str:
        sym = self.symbol or "data"
        w = ", ".join(f"{n} {v:+.2f}" for n, v in zip(self.feature_names, self.mean_weights))
        verdict = (
            "no OOS edge (IC ~ 0)" if abs(self.oos_ic_mean) < 0.02
            else ("weak positive OOS signal" if self.oos_ic_mean > 0
                  else "negative OOS (anti-predictive)")
        )
        return "\n".join([
            f"Factor-weight CPCV — {sym}  "
            f"({self.n_samples} samples, {self.n_paths} folds, ridge α={self.alpha})",
            "-" * 56,
            f"  Learned weights : {w}   (standardized)",
            f"  OOS IC          : mean {self.oos_ic_mean:+.3f}  "
            f"median {self.oos_ic_median:+.3f}  σ {self.oos_ic_std:.3f}  "
            f"({self.pct_folds_positive_ic:.0f}% folds > 0)",
            f"  OOS dir. acc.   : {self.oos_dir_accuracy * 100:.1f}%",
            f"  OOS L/S return  : {self.oos_ls_return_pct:+.3f}% / sample",
            f"  Baseline IC     : {self.baseline_ic_mean:+.3f}  (hand-set weights)",
            f"  Verdict         : {verdict}",
        ])

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


@dataclass(frozen=True)
class MetaDataset:
    """Meta-labeling dataset: primary side + triple-barrier win/loss + features.

    ``meta_label`` is 1 iff the primary-side bet won its triple barrier;
    ``realized_return`` is the bet's first-touch return; ``side`` is the primary
    direction. The secondary (meta) model learns ``P(win)`` from ``X``.
    """

    X: np.ndarray
    meta_label: np.ndarray          # {0,1}
    side: np.ndarray                # {+1,-1}
    realized_return: np.ndarray
    feature_names: tuple[str, ...]
    horizon: int
    stride: int
    timestamps: Optional[pd.DatetimeIndex] = None
    symbol: Optional[str] = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def purge_samples(self) -> int:
        return int(np.ceil(self.horizon / max(self.stride, 1)))

    @property
    def base_win_rate(self) -> float:
        return float(self.meta_label.mean()) if len(self) else float("nan")


@dataclass(frozen=True)
class MetaCVResult:
    """OOS comparison of primary-only vs meta-filtered trades across CPCV folds."""

    n_paths: int
    n_samples: int
    base_win_rate: float
    oos_precision_mean: float        # win rate of the meta-TAKEN trades
    oos_auc_mean: float              # meta-classifier discrimination
    primary_return_mean: float       # mean realised return per primary trade
    meta_return_mean: float          # mean realised return per meta-taken trade
    return_improvement_mean: float   # meta - primary
    pct_folds_meta_beats_primary: float
    avg_fraction_taken: float
    threshold: float
    fold_improvements: tuple[float, ...]
    symbol: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol, "n_samples": self.n_samples, "n_paths": self.n_paths,
            "threshold": self.threshold,
            "base_win_rate": round(self.base_win_rate, 4),
            "oos_precision_mean": round(self.oos_precision_mean, 4),
            "oos_auc_mean": round(self.oos_auc_mean, 4),
            "primary_return_mean": round(self.primary_return_mean, 5),
            "meta_return_mean": round(self.meta_return_mean, 5),
            "return_improvement_mean": round(self.return_improvement_mean, 5),
            "pct_folds_meta_beats_primary": round(self.pct_folds_meta_beats_primary, 1),
            "avg_fraction_taken": round(self.avg_fraction_taken, 3),
        }

    def summary(self) -> str:
        sym = self.symbol or "data"
        helps = (self.oos_precision_mean > self.base_win_rate + 0.01
                 and self.return_improvement_mean > 0 and self.oos_auc_mean > 0.52)
        verdict = ("meta-labeling ADDS value (better precision + OOS return)" if helps
                   else "meta-labeling does NOT help (≈ no discrimination)")
        return "\n".join([
            f"Meta-labeling CPCV — {sym}  "
            f"({self.n_samples} events, {self.n_paths} folds, threshold {self.threshold:.2f})",
            "-" * 56,
            f"  Base win rate   : {self.base_win_rate * 100:.1f}%  (all primary signals)",
            f"  Meta precision  : {self.oos_precision_mean * 100:.1f}%  (taken trades, OOS)",
            f"  Meta AUC        : {self.oos_auc_mean:.3f}  (0.5 = no skill)",
            f"  Return/trade    : primary {self.primary_return_mean * 100:+.3f}%  ->  "
            f"meta {self.meta_return_mean * 100:+.3f}%  "
            f"(Δ {self.return_improvement_mean * 100:+.3f}%)",
            f"  Selectivity     : takes {self.avg_fraction_taken * 100:.0f}% of signals; "
            f"meta beats primary in {self.pct_folds_meta_beats_primary:.0f}% of folds",
            f"  Verdict         : {verdict}",
        ])

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


@dataclass(frozen=True)
class MetaPermutationResult:
    """Label-permutation significance test for a meta-labeling evaluation.

    The null shuffles each row's ``(label, return)`` against its features, so any
    feature→outcome relationship is destroyed; the p-value is the fraction of
    permutations whose statistic is at least as extreme as the real one.
    """

    real_auc: float
    null_auc_mean: float
    p_value_auc: float
    real_improvement: float
    null_improvement_mean: float
    p_value_improvement: float
    n_permutations: int
    symbol: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_permutations": self.n_permutations,
            "real_auc": round(self.real_auc, 4),
            "null_auc_mean": round(self.null_auc_mean, 4),
            "p_value_auc": round(self.p_value_auc, 4),
            "real_improvement": round(self.real_improvement, 5),
            "null_improvement_mean": round(self.null_improvement_mean, 5),
            "p_value_improvement": round(self.p_value_improvement, 4),
        }

    def summary(self) -> str:
        sym = self.symbol or "data"
        sig = "SIGNIFICANT" if self.p_value_auc < 0.05 else "not significant"
        return "\n".join([
            f"Permutation test — {sym}  ({self.n_permutations} shuffles)",
            "-" * 56,
            f"  AUC          : real {self.real_auc:.3f}  vs null {self.null_auc_mean:.3f}"
            f"   -> p = {self.p_value_auc:.3f}  ({sig})",
            f"  Return Δ     : real {self.real_improvement * 100:+.3f}%  vs null "
            f"{self.null_improvement_mean * 100:+.3f}%   -> p = {self.p_value_improvement:.3f}",
        ])

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()



@dataclass(frozen=True)
class CrossSectionalPanel:
    """A date×name panel of cross-sectionally **ranked** factors → forward returns.

    Unlike :class:`FactorDataset` (per-name time-series samples pooled together),
    every row here is one ``(rebalance date, name)`` cell whose features have been
    ranked *across the universe on that date* (market-neutral, scale-free). ``y`` is
    the strictly-future forward ``horizon``-day return. ``date_id`` groups rows that
    share a rebalance date — the unit CPCV purges over (purge = horizon in dates).
    """

    X: np.ndarray                  # (n_rows, n_feat) cross-sectional ranks, centred ~[-0.5, 0.5]
    y: np.ndarray                  # (n_rows,) forward horizon-day return
    date_id: np.ndarray            # (n_rows,) int in [0, n_dates)
    feature_names: tuple[str, ...]
    horizon: int
    rebalance: int                 # trading days between sampled cross-sections
    n_dates: int
    symbols: tuple[str, ...]
    dates: Optional[pd.DatetimeIndex] = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_names(self) -> int:
        return len(self.symbols)

    @property
    def purge_dates(self) -> int:
        """Label horizon expressed in *rebalance dates* (for CPCV purging)."""
        return int(np.ceil(self.horizon / max(self.rebalance, 1)))


@dataclass(frozen=True)
class CrossSectionalICResult:
    """OOS evaluation of a cross-sectional rank-factor model.

    The headline ``combined_oos_ic_mean`` is the mean per-date Spearman IC of a
    purged-CPCV ridge combination of the ranked factors; ``single_factor_ic`` is
    each raw factor's model-free pooled rank IC (no fitting) for context.
    """

    feature_names: tuple[str, ...]
    single_factor_ic: tuple[float, ...]   # model-free pooled daily rank-IC per feature
    mean_weights: tuple[float, ...]       # ridge weights on ranked factors (OOS-averaged)
    combined_oos_ic_mean: float
    combined_oos_ic_median: float
    combined_oos_ic_std: float
    pct_dates_positive_ic: float
    n_dates_oos: int                      # number of test dates scored
    n_rows: int
    n_names: int
    horizon: int
    rebalance: int
    alpha: float = 1.0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_names": self.n_names, "n_rows": self.n_rows,
            "horizon": self.horizon, "rebalance": self.rebalance, "alpha": self.alpha,
            "single_factor_ic": {n: round(v, 4) for n, v
                                 in zip(self.feature_names, self.single_factor_ic)},
            "weights": {n: round(w, 4) for n, w
                        in zip(self.feature_names, self.mean_weights)},
            "combined_oos_ic_mean": round(self.combined_oos_ic_mean, 4),
            "combined_oos_ic_median": round(self.combined_oos_ic_median, 4),
            "combined_oos_ic_std": round(self.combined_oos_ic_std, 4),
            "pct_dates_positive_ic": round(self.pct_dates_positive_ic, 1),
            "n_dates_oos": self.n_dates_oos,
        }

    def summary(self) -> str:
        sf = ", ".join(f"{n} {v:+.3f}" for n, v in
                       zip(self.feature_names, self.single_factor_ic))
        w = ", ".join(f"{n} {v:+.2f}" for n, v in
                      zip(self.feature_names, self.mean_weights))
        verdict = (
            "no OOS cross-sectional edge (IC ~ 0)" if abs(self.combined_oos_ic_mean) < 0.01
            else ("positive OOS cross-sectional signal" if self.combined_oos_ic_mean > 0
                  else "negative OOS (anti-predictive)")
        )
        return "\n".join([
            f"Cross-sectional rank IC — {self.n_names} names, "
            f"{self.n_rows} rows, {self.n_dates_oos} OOS dates "
            f"(horizon {self.horizon}d, rebal {self.rebalance}d, ridge α={self.alpha})",
            "-" * 60,
            f"  Single-factor IC : {sf}   (model-free, pooled)",
            f"  Ridge weights    : {w}",
            f"  Combined OOS IC  : mean {self.combined_oos_ic_mean:+.3f}  "
            f"median {self.combined_oos_ic_median:+.3f}  σ {self.combined_oos_ic_std:.3f}  "
            f"({self.pct_dates_positive_ic:.0f}% dates > 0)",
            f"  Verdict          : {verdict}",
        ])

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


@dataclass(frozen=True)
class FactorBucketResult:
    """Conviction-bucketed OOS returns: only bet the signal tails, stay **flat** in the middle.

    Pools out-of-sample predictions, sorts them into ``n_buckets`` equal-count
    quantiles, and reports the mean forward return per bucket. A tradeable rule
    goes **long** the top bucket, **short** the bottom, and **flat** the rest — so
    the ``long_short_spread`` (drift-neutral alpha) is compared against the naive
    always-in-market ``always_on_ls`` to show whether the signal's value lives in
    the conviction tails. Returns are gross unless a ``cost_bps`` round-trip is set.
    """

    n_buckets: int
    bucket_returns_pct: tuple[float, ...]   # mean forward return per signal quantile, low→high, %
    top_return_pct: float                   # long leg = top bucket forward return, %
    bottom_return_pct: float                # bottom bucket forward return, %
    long_short_spread_pct: float            # top − bottom = drift-neutral alpha / bet, %
    spearman: float                         # rank corr of bucket index vs return (monotonicity)
    monotonic: bool                         # spearman > 0
    frac_in_market: float                   # fraction of bars holding a position (long or short)
    always_on_ls_pct: float                 # sign()-based always-in-market L/S, for contrast, %
    cost_bps_roundtrip: float
    long_only_net_pct: float                # top bucket − 1 round-trip (alpha + market drift)
    long_short_net_pct: float               # spread − 2 round-trips (drift-neutral)
    horizon: int
    n_samples: int
    symbol: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_buckets": self.n_buckets,
            "bucket_returns_pct": [round(b, 4) for b in self.bucket_returns_pct],
            "long_short_spread_pct": round(self.long_short_spread_pct, 4),
            "spearman": round(self.spearman, 3),
            "monotonic": self.monotonic,
            "frac_in_market": round(self.frac_in_market, 3),
            "always_on_ls_pct": round(self.always_on_ls_pct, 4),
            "long_only_net_pct": round(self.long_only_net_pct, 4),
            "long_short_net_pct": round(self.long_short_net_pct, 4),
        }

    def summary(self) -> str:
        sym = self.symbol or "data"
        curve = " ".join(f"{b:+.2f}" for b in self.bucket_returns_pct)
        return "\n".join([
            f"Conviction buckets — {sym}  "
            f"({self.n_buckets} quantiles, {self.horizon}-bar fwd return, {self.n_samples} samples)",
            "-" * 56,
            f"  Bucket returns  : {curve}  % (low→high signal)",
            f"  Monotonic       : {'yes' if self.monotonic else 'NO'} (spearman {self.spearman:+.2f})",
            f"  Always-on L/S   : {self.always_on_ls_pct:+.3f}% / bet  (in market 100% of the time)",
            f"  Tails-only L/S  : {self.long_short_spread_pct:+.3f}% / bet  "
            f"(in market {self.frac_in_market * 100:.0f}% of the time)",
            f"  Net @ {self.cost_bps_roundtrip:.0f}bps  : long-only {self.long_only_net_pct:+.3f}%  "
            f"| long/short {self.long_short_net_pct:+.3f}%  per bet",
        ])

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


@dataclass(frozen=True)
class FactorPermutationResult:
    """Label-permutation significance test for a factor-IC evaluation."""

    real_ic: float
    null_ic_mean: float
    p_value: float
    n_permutations: int
    symbol: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_permutations": self.n_permutations,
            "real_ic": round(self.real_ic, 4),
            "null_ic_mean": round(self.null_ic_mean, 4),
            "p_value": round(self.p_value, 4),
        }

    def summary(self) -> str:
        sym = self.symbol or "data"
        sig = "SIGNIFICANT" if self.p_value < 0.05 else "not significant"
        return (f"Factor permutation test — {sym}  ({self.n_permutations} shuffles)\n"
                + "-" * 56
                + f"\n  OOS IC: real {self.real_ic:+.3f}  vs null {self.null_ic_mean:+.3f}"
                + f"   -> p = {self.p_value:.3f}  ({sig})")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()
