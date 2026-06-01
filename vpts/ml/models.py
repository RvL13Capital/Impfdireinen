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
