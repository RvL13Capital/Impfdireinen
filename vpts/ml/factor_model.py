"""A learned factor-weight model, evaluated out-of-sample with purged CPCV.

This is the first *fitted* model in the system. Instead of the hand-set confluence
weights, a **ridge regression** learns weights on the four confluence factors to
predict the forward ``horizon``-bar return — and, crucially, it is evaluated with
:class:`~vpts.validation.cpcv.CombinatorialPurgedCV` (purge = label horizon,
embargo) so the score is honestly out-of-sample.

Ridge is implemented in plain numpy (no new dependency): the learned, standardized
weights are directly interpretable as factor importances, and the headline metric
is the distribution of **out-of-sample IC** (correlation between prediction and
realised forward return) across CPCV folds, alongside the hand-weighted baseline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from vpts.profile.calculator import VolumeProfileCalculator
from vpts.regime.indicators import ensure_ohlcv
from vpts.regime.patterns import VolumePatternDetector
from vpts.regime.quiet import QuietPhaseDetector
from vpts.scoring.scorer import ConfluenceScorer
from vpts.ml.models import FactorCVResult, FactorDataset
from vpts.validation.cpcv import CombinatorialPurgedCV

_FEATURES = ("value_area", "key_level", "quiet", "patterns")


class RidgeFactorModel:
    """Ridge regression with train-only standardisation (numpy closed form).

    ``fit`` standardises features using the training data, centres the target,
    and solves ``(XᵀX + αI) w = Xᵀy``. ``weights_`` are the standardised factor
    weights (comparable importances).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")
        self.alpha = float(alpha)
        self.mu_: Optional[np.ndarray] = None
        self.sd_: Optional[np.ndarray] = None
        self.ybar_: float = 0.0
        self.weights_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeFactorModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        xs = (X - self.mu_) / self.sd_
        self.ybar_ = float(y.mean())
        yc = y - self.ybar_
        d = xs.shape[1]
        self.weights_ = np.linalg.solve(xs.T @ xs + self.alpha * np.eye(d), xs.T @ yc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("model is not fitted")
        xs = (np.asarray(X, dtype=float) - self.mu_) / self.sd_
        return xs @ self.weights_ + self.ybar_

    def signal(self, X: np.ndarray) -> np.ndarray:
        """The centred (directional) prediction = ``standardized_X · weights``."""
        if self.weights_ is None:
            raise RuntimeError("model is not fitted")
        xs = (np.asarray(X, dtype=float) - self.mu_) / self.sd_
        return xs @ self.weights_


# --------------------------------------------------------------------------- #
def build_factor_dataset(
    df: pd.DataFrame,
    *,
    lookback: int = 120,
    horizon: int = 20,
    stride: int = 1,
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    profile_calculator: Optional[VolumeProfileCalculator] = None,
    quiet_detector: Optional[QuietPhaseDetector] = None,
    pattern_detector: Optional[VolumePatternDetector] = None,
    scorer: Optional[ConfluenceScorer] = None,
) -> FactorDataset:
    """Walk the bars and build (factor features → forward return) samples.

    Features at bar ``t`` use only data ``<= t`` (the rolling window); the label is
    the forward ``horizon``-bar return ``close[t+horizon]/close[t] - 1`` — strictly
    future, so there is no look-ahead.
    """
    ensure_ohlcv(df, min_bars=lookback + horizon + 2)
    pc = profile_calculator or VolumeProfileCalculator(bin_mode="auto")
    qd = quiet_detector or QuietPhaseDetector()
    pat = pattern_detector or VolumePatternDetector()
    sc = scorer or ConfluenceScorer()
    closes = df["Close"].to_numpy(float)
    n = len(df)

    feats: list[list[float]] = []
    ys: list[float] = []
    base: list[float] = []
    ts: list = []
    for t in range(lookback - 1, n - horizon, max(1, stride)):
        window = df.iloc[t - lookback + 1 : t + 1]
        try:
            profile = pc.calculate(window, symbol, interval)
            quiet = qd.detect(window, symbol, interval)
            patterns = pat.detect(window, profile=profile, symbol=symbol, interval=interval)
            score = sc.score(window, profile, quiet, patterns, symbol=symbol, interval=interval)
        except (ValueError, ZeroDivisionError):
            continue
        comps = {c.name: c for c in score.components}
        feats.append([
            comps["value_area"].strength * comps["value_area"].direction,
            comps["key_level"].strength * comps["key_level"].direction,
            comps["quiet"].strength,                       # non-directional regime factor
            comps["patterns"].strength * comps["patterns"].direction,
        ])
        ys.append(float(closes[t + horizon] / closes[t] - 1.0))
        base.append(float(score.bias_score))
        ts.append(df.index[t])

    is_dt = isinstance(df.index, pd.DatetimeIndex) and ts
    return FactorDataset(
        X=np.array(feats, dtype=float).reshape(-1, len(_FEATURES)),
        y=np.array(ys, dtype=float),
        baseline=np.array(base, dtype=float),
        feature_names=_FEATURES,
        horizon=horizon,
        stride=max(1, stride),
        timestamps=pd.DatetimeIndex(ts) if is_dt else None,
        symbol=symbol,
    )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cpcv_factor_eval(
    dataset: FactorDataset,
    cv: Optional[CombinatorialPurgedCV] = None,
    alpha: float = 1.0,
) -> FactorCVResult:
    """Fit ridge factor weights on each CPCV train fold; score OOS on the test fold.

    Reports the distribution of out-of-sample IC (prediction vs realised forward
    return), directional accuracy, a costless long/short return, the averaged
    learned weights, and the hand-weighted baseline IC for comparison.
    """
    X, y, base = dataset.X, dataset.y, dataset.baseline
    m = len(dataset)
    cv = cv or CombinatorialPurgedCV(
        n_groups=6, n_test_groups=2, purge=dataset.purge_samples, embargo_pct=0.01
    )

    fold_ics: list[float] = []
    baseline_ics: list[float] = []
    accs: list[float] = []
    lss: list[float] = []
    weights: list[np.ndarray] = []
    for split in cv.split(m):
        tr, te = split.train_idx, split.test_idx
        if tr.size < max(10, X.shape[1] + 2) or te.size < 3:
            continue
        model = RidgeFactorModel(alpha).fit(X[tr], y[tr])
        sig = model.signal(X[te])          # directional (centred) prediction
        yt = y[te]
        ic = _corr(sig, yt)
        if np.isfinite(ic):
            fold_ics.append(ic)
            weights.append(model.weights_)
            accs.append(float(np.mean(np.sign(sig) == np.sign(yt))))
            lss.append(float(np.mean(np.sign(sig) * yt)))
            b_ic = _corr(base[te], yt)
            if np.isfinite(b_ic):
                baseline_ics.append(b_ic)

    if not fold_ics:
        raise ValueError(
            "CPCV produced no usable folds for factor evaluation — increase the "
            "dataset size or reduce n_groups."
        )
    ics = np.array(fold_ics, dtype=float)
    w_mean = np.mean(np.vstack(weights), axis=0)
    return FactorCVResult(
        feature_names=dataset.feature_names,
        mean_weights=tuple(round(float(x), 4) for x in w_mean),
        oos_ic_mean=float(ics.mean()),
        oos_ic_median=float(np.median(ics)),
        oos_ic_std=float(ics.std()),
        pct_folds_positive_ic=float((ics > 0).mean() * 100.0),
        oos_dir_accuracy=float(np.mean(accs)),
        oos_ls_return_pct=float(np.mean(lss) * 100.0),
        baseline_ic_mean=float(np.mean(baseline_ics)) if baseline_ics else float("nan"),
        n_paths=int(ics.size),
        n_samples=m,
        fold_ics=tuple(round(float(x), 4) for x in ics),
        alpha=float(alpha),
        symbol=dataset.symbol,
    )


def permutation_test_factor(
    dataset: FactorDataset,
    cv: Optional[CombinatorialPurgedCV] = None,
    n_permutations: int = 200,
    alpha: float = 1.0,
    seed: int = 0,
) -> "FactorPermutationResult":
    """Label-shuffle significance test for the out-of-sample factor IC.

    Re-runs :func:`cpcv_factor_eval` with the targets ``y`` shuffled against the
    features (same CV splits throughout); the p-value is the fraction of
    permutations whose OOS IC is at least the real one.
    """
    from vpts.ml.models import FactorPermutationResult

    m = len(dataset)
    cv = cv or CombinatorialPurgedCV(
        n_groups=6, n_test_groups=2, purge=dataset.purge_samples, embargo_pct=0.01)
    real = cpcv_factor_eval(dataset, cv, alpha)
    rng = np.random.default_rng(seed)

    null: list[float] = []
    for _ in range(n_permutations):
        perm = rng.permutation(m)
        shuffled = FactorDataset(
            X=dataset.X, y=dataset.y[perm], baseline=dataset.baseline,
            feature_names=dataset.feature_names, horizon=dataset.horizon,
            stride=dataset.stride, symbol=dataset.symbol)
        try:
            r = cpcv_factor_eval(shuffled, cv, alpha)
        except ValueError:
            continue
        if np.isfinite(r.oos_ic_mean):
            null.append(r.oos_ic_mean)

    arr = np.array(null, dtype=float)
    p = float((np.sum(arr >= real.oos_ic_mean) + 1) / (arr.size + 1))
    return FactorPermutationResult(
        real_ic=real.oos_ic_mean,
        null_ic_mean=float(arr.mean()) if arr.size else float("nan"),
        p_value=p, n_permutations=int(arr.size), symbol=dataset.symbol)
