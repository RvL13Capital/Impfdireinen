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

