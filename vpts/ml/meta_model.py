"""Secondary (meta) model + its purged CPCV evaluation.

A small numpy **logistic regression** learns ``P(the primary bet wins)`` from the
confluence meta-features, and is evaluated out-of-sample with the purged+embargoed
CPCV. The honest question: does filtering the primary signals by the meta-model's
probability improve precision and per-trade return *out of sample*?
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from vpts.ml.models import MetaCVResult, MetaDataset
from vpts.validation.cpcv import CombinatorialPurgedCV


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class LogisticMetaModel:
    """L2-regularised logistic regression (numpy, train-only standardisation)."""

    def __init__(self, l2: float = 1.0, lr: float = 0.5, iters: int = 800) -> None:
        if l2 < 0:
            raise ValueError("l2 must be >= 0.")
        self.l2 = float(l2)
        self.lr = float(lr)
        self.iters = int(iters)
        self.mu_: Optional[np.ndarray] = None
        self.sd_: Optional[np.ndarray] = None
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticMetaModel":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        xs = (X - self.mu_) / self.sd_
        n, d = xs.shape
        # Degenerate (single-class) target -> constant base-rate predictor.
        p = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        if y.std() == 0:
            self.w_ = np.zeros(d)
            self.b_ = float(np.log(p / (1 - p)))
            return self
        self.w_ = np.zeros(d)
        self.b_ = float(np.log(p / (1 - p)))
        for _ in range(self.iters):
            pred = _sigmoid(xs @ self.w_ + self.b_)
            err = pred - y
            self.w_ -= self.lr * (xs.T @ err / n + self.l2 * self.w_ / n)
            self.b_ -= self.lr * float(err.mean())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("model is not fitted")
        xs = (np.asarray(X, float) - self.mu_) / self.sd_
        return _sigmoid(xs @ self.w_ + self.b_)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank AUC = P(score | y=1 > score | y=0), ties counted as 0.5."""
    pos = score[y == 1]
    neg = score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def cpcv_meta_eval(
    dataset: MetaDataset,
    cv: Optional[CombinatorialPurgedCV] = None,
    threshold: float = 0.5,
    l2: float = 1.0,
    cost_bps: float = 0.0,
    select_top: Optional[float] = None,
) -> MetaCVResult:
    """Fit the meta-model per CPCV train fold; compare OOS primary vs meta-filtered.

    For each purged test fold: predict ``P(win)`` and take a subset of the primary
    signals, comparing the realised return / win rate of the taken trades against
    taking *all* primary signals. Trades are selected either by an **absolute**
    probability (``p >= threshold``) or, when ``select_top`` is set, by a
    **relative** rule that takes the top ``select_top`` fraction of ratings within
    the fold (e.g. ``0.2`` = act on the best-rated 20% of setups, flat otherwise) —
    the realistic "rate setups and trade only the best" rule, robust to a weak
    signal whose probabilities cluster near the base rate. ``cost_bps`` is a
    per-trade round-trip cost subtracted from every realised return.
    """
    if select_top is not None and not (0.0 < select_top <= 1.0):
        raise ValueError("select_top must be in (0, 1].")
    X = dataset.X
    y = dataset.meta_label
    ret = dataset.realized_return
    m = len(dataset)
    cost = cost_bps / 1e4
    cv = cv or CombinatorialPurgedCV(
        n_groups=6, n_test_groups=2, purge=dataset.purge_samples, embargo_pct=0.01)

    precisions: list[float] = []
    aucs: list[float] = []
    prim_rets: list[float] = []
    meta_rets: list[float] = []
    fracs: list[float] = []
    improvements: list[float] = []
    for split in cv.split(m):
        tr, te = split.train_idx, split.test_idx
        if tr.size < max(20, X.shape[1] + 2) or te.size < 5:
            continue
        model = LogisticMetaModel(l2=l2).fit(X[tr], y[tr])
        p = model.predict_proba(X[te])
        yte = y[te]
        rte = ret[te] - cost                 # net of per-trade round-trip cost
        if select_top is not None:           # relative: trade the best-rated fraction
            taken = p >= np.quantile(p, 1.0 - select_top)
        else:
            taken = p >= threshold
        prim_ret = float(rte.mean())
        if taken.sum() > 0:
            meta_ret = float(rte[taken].mean())
            precision = float(yte[taken].mean())
        else:
            meta_ret = 0.0
            precision = float("nan")
        prim_rets.append(prim_ret)
        meta_rets.append(meta_ret)
        improvements.append(meta_ret - prim_ret)
        fracs.append(float(taken.mean()))
        if np.isfinite(precision):
            precisions.append(precision)
        a = _auc(yte, p)
        if np.isfinite(a):
            aucs.append(a)

    if not improvements:
        raise ValueError(
            "CPCV produced no usable folds for meta evaluation — increase the "
            "dataset size or reduce n_groups.")
    imp = np.array(improvements, dtype=float)
    return MetaCVResult(
        n_paths=int(imp.size),
        n_samples=m,
        base_win_rate=dataset.base_win_rate,
        oos_precision_mean=float(np.mean(precisions)) if precisions else float("nan"),
        oos_auc_mean=float(np.mean(aucs)) if aucs else float("nan"),
        primary_return_mean=float(np.mean(prim_rets)),
        meta_return_mean=float(np.mean(meta_rets)),
        return_improvement_mean=float(imp.mean()),
        pct_folds_meta_beats_primary=float((imp > 0).mean() * 100.0),
        avg_fraction_taken=float(np.mean(fracs)),
        threshold=float(threshold),
        fold_improvements=tuple(round(float(x), 5) for x in imp),
        symbol=dataset.symbol,
        extra={"cost_bps": float(cost_bps)},
    )


def permutation_test_meta(
    dataset: MetaDataset,
    cv: Optional[CombinatorialPurgedCV] = None,
    n_permutations: int = 200,
    threshold: float = 0.55,
    l2: float = 1.0,
    cost_bps: float = 0.0,
    seed: int = 0,
    select_top: Optional[float] = None,
) -> "MetaPermutationResult":
    """Label-permutation significance test for a meta evaluation.

    Re-runs :func:`cpcv_meta_eval` on the real dataset and on *n_permutations*
    copies whose ``(label, return)`` rows are shuffled against the features
    (destroying any feature→outcome relationship), using the *same* CV splits
    throughout. The p-values are the fraction of permutations whose AUC /
    return-improvement is at least as large as the real one.
    """
    from vpts.ml.models import MetaPermutationResult

    m = len(dataset)
    cv = cv or CombinatorialPurgedCV(
        n_groups=6, n_test_groups=2, purge=dataset.purge_samples, embargo_pct=0.01)
    real = cpcv_meta_eval(dataset, cv, threshold, l2, cost_bps, select_top)
    rng = np.random.default_rng(seed)

    null_auc: list[float] = []
    null_imp: list[float] = []
    for _ in range(n_permutations):
        perm = rng.permutation(m)
        shuffled = MetaDataset(
            X=dataset.X, meta_label=dataset.meta_label[perm], side=dataset.side,
            realized_return=dataset.realized_return[perm],
            feature_names=dataset.feature_names, horizon=dataset.horizon,
            stride=dataset.stride, symbol=dataset.symbol)
        try:
            r = cpcv_meta_eval(shuffled, cv, threshold, l2, cost_bps, select_top)
        except ValueError:
            continue
        if np.isfinite(r.oos_auc_mean):
            null_auc.append(r.oos_auc_mean)
        null_imp.append(r.return_improvement_mean)

    auc_arr = np.array(null_auc, dtype=float)
    imp_arr = np.array(null_imp, dtype=float)
    p_auc = float((np.sum(auc_arr >= real.oos_auc_mean) + 1) / (auc_arr.size + 1))
    p_imp = float((np.sum(imp_arr >= real.return_improvement_mean) + 1) / (imp_arr.size + 1))
    return MetaPermutationResult(
        real_auc=real.oos_auc_mean,
        null_auc_mean=float(auc_arr.mean()) if auc_arr.size else float("nan"),
        p_value_auc=p_auc,
        real_improvement=real.return_improvement_mean,
        null_improvement_mean=float(imp_arr.mean()) if imp_arr.size else float("nan"),
        p_value_improvement=p_imp,
        n_permutations=int(max(auc_arr.size, imp_arr.size)),
        symbol=dataset.symbol,
    )
