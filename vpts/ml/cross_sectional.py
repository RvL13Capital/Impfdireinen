"""Cross-sectional rank factors — the standard equity-alpha construction.

Everything before this scored each name on its *own* time series. This instead
ranks names **against each other on the same day**: classic 1-month reversal,
12-1 momentum, 60-day volatility (the low-vol anomaly) and a volume-trend proxy,
each turned into a market-neutral cross-sectional rank. The bet is relative —
"which names will out/under-perform the cross-section" — and the score is the
per-date **rank IC** (Spearman of prediction vs forward return across names),
combined out-of-sample with the same purged CPCV (purging over rebalance dates).

All features at bar ``t`` use only data ``<= t``; the label is the strictly-future
forward return. Ranking is contemporaneous across names (no look-ahead).
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from vpts.ml.factor_model import RidgeFactorModel
from vpts.ml.models import CrossSectionalICResult, CrossSectionalPanel
from vpts.validation.cpcv import CombinatorialPurgedCV

CROSS_SECTIONAL_FEATURES = ("mom_21", "mom_252_21", "vol_60", "vol_trend")


# --------------------------------------------------------------------------- #
# rank helpers (numpy; average ranks for tie-safe Spearman)
# --------------------------------------------------------------------------- #
def _avg_rank(x: np.ndarray) -> np.ndarray:
    """Average ranks 1..n (ties share their mean rank)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _rank_centered(x: np.ndarray) -> np.ndarray:
    """Centred cross-sectional rank in [-0.5, 0.5] (market-neutral, scale-free)."""
    n = np.asarray(x).size
    if n < 2:
        return np.zeros(n, dtype=float)
    return (_avg_rank(x) - 1.0) / (n - 1.0) - 0.5


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra, rb = _avg_rank(a), _avg_rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------- #
def _name_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per-name factor columns (all <= t) + strictly-future forward return."""
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["mom_21"] = close / close.shift(21) - 1.0                 # 1-month (reversal sign)
    out["mom_252_21"] = close.shift(21) / close.shift(252) - 1.0  # 12-1 momentum
    out["vol_60"] = close.pct_change().rolling(60).std()          # realised volatility
    out["vol_trend"] = volume.rolling(20).mean() / volume.rolling(60).mean() - 1.0
    out["fwd"] = close.shift(-horizon) / close - 1.0              # label (future)
    return out


def build_cross_sectional_panel(
    frames: Mapping[str, pd.DataFrame],
    *,
    horizon: int = 20,
    rebalance: int = 5,
    min_names: int = 5,
) -> CrossSectionalPanel:
    """Assemble a date×name panel of cross-sectionally ranked factors → forward return.

    On every ``rebalance``-th trading date that has at least ``min_names`` fully
    warmed-up names, each factor is ranked across that day's names (centred to
    ``[-0.5, 0.5]``) and stored with the name's forward ``horizon``-day return.
    """
    if len(frames) < min_names:
        raise ValueError(f"need >= {min_names} names for a cross-section; got {len(frames)}.")
    feats = CROSS_SECTIONAL_FEATURES
    per_name = {sym: _name_features(df, horizon) for sym, df in frames.items()}
    symbols = tuple(sorted(per_name))

    # Wide (date × name) matrices, one per feature plus the forward return.
    wide = {f: pd.concat({s: per_name[s][f] for s in symbols}, axis=1)[list(symbols)]
            for f in (*feats, "fwd")}
    all_dates = wide["fwd"].index
    sampled = all_dates[::max(1, rebalance)]

    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    d_rows: list[int] = []
    kept_dates: list = []
    fmat = np.column_stack([wide[f].reindex(sampled).to_numpy(float) for f in feats])
    fmat = fmat.reshape(len(sampled), len(feats), len(symbols))     # (date, feat, name)
    ymat = wide["fwd"].reindex(sampled).to_numpy(float)             # (date, name)

    for di in range(len(sampled)):
        feat_d = fmat[di]                 # (n_feat, n_names)
        y_d = ymat[di]                    # (n_names,)
        valid = np.isfinite(y_d) & np.all(np.isfinite(feat_d), axis=0)
        if int(valid.sum()) < min_names:
            continue
        ranked = np.column_stack([_rank_centered(feat_d[j, valid]) for j in range(len(feats))])
        date_id = len(kept_dates)
        for r in range(int(valid.sum())):
            X_rows.append([float(v) for v in ranked[r]])
            y_rows.append(float(y_d[valid][r]))
            d_rows.append(date_id)
        kept_dates.append(sampled[di])

    if not kept_dates:
        raise ValueError("no cross-sections met the min_names / warm-up requirement.")
    is_dt = isinstance(all_dates, pd.DatetimeIndex)
    return CrossSectionalPanel(
        X=np.array(X_rows, dtype=float).reshape(-1, len(feats)),
        y=np.array(y_rows, dtype=float),
        date_id=np.array(d_rows, dtype=int),
        feature_names=feats,
        horizon=horizon,
        rebalance=max(1, rebalance),
        n_dates=len(kept_dates),
        symbols=symbols,
        dates=pd.DatetimeIndex(kept_dates) if is_dt else None,
    )


# --------------------------------------------------------------------------- #
def _date_groups(date_id: np.ndarray, n_dates: int) -> list[np.ndarray]:
    """Row indices for each date_id (groups[d] = rows on date d)."""
    order = np.argsort(date_id, kind="mergesort")
    sid = date_id[order]
    bounds = np.flatnonzero(np.diff(sid)) + 1
    chunks = np.split(order, bounds)
    groups: list[np.ndarray] = [np.empty(0, dtype=int)] * n_dates
    for ch in chunks:
        if ch.size:
            groups[int(date_id[ch[0]])] = ch
    return groups


def cross_sectional_ic_eval(
    panel: CrossSectionalPanel,
    cv: Optional[CombinatorialPurgedCV] = None,
    alpha: float = 1.0,
) -> CrossSectionalICResult:
    """Purged-CPCV out-of-sample evaluation of a ridge cross-sectional rank model.

    Reports each raw factor's model-free pooled rank IC, then the distribution of
    per-date OOS rank IC for a ridge combination fit on the train dates and scored
    on the purged/embargoed test dates.
    """
    X, y, date_id = panel.X, panel.y, panel.date_id
    groups = _date_groups(date_id, panel.n_dates)
    cv = cv or CombinatorialPurgedCV(
        n_groups=6, n_test_groups=2, purge=panel.purge_dates, embargo_pct=0.01)

    # Model-free single-factor IC: per-date Spearman of each ranked factor vs fwd.
    single: list[float] = []
    for j in range(X.shape[1]):
        ics = [_spearman(X[g, j], y[g]) for g in groups if g.size >= 3]
        ics = [v for v in ics if np.isfinite(v)]
        single.append(float(np.mean(ics)) if ics else float("nan"))

    # Combined OOS: fit on train dates, score per test date.
    oos_ic: list[float] = []
    weights: list[np.ndarray] = []
    for split in cv.split(panel.n_dates):
        tr_dates, te_dates = split.train_idx, split.test_idx
        tr_rows = np.concatenate([groups[d] for d in tr_dates]) if tr_dates.size else np.empty(0, int)
        if tr_rows.size < max(20, X.shape[1] + 2):
            continue
        model = RidgeFactorModel(alpha).fit(X[tr_rows], y[tr_rows])
        weights.append(model.weights_)
        for d in te_dates:
            g = groups[d]
            if g.size < 3:
                continue
            ic = _spearman(model.signal(X[g]), y[g])
            if np.isfinite(ic):
                oos_ic.append(ic)

    if not oos_ic:
        raise ValueError(
            "CPCV produced no usable OOS dates for the cross-sectional model — "
            "increase the universe/history or reduce n_groups.")
    arr = np.array(oos_ic, dtype=float)
    w_mean = np.mean(np.vstack(weights), axis=0) if weights else np.full(X.shape[1], np.nan)
    return CrossSectionalICResult(
        feature_names=panel.feature_names,
        single_factor_ic=tuple(round(float(v), 4) for v in single),
        mean_weights=tuple(round(float(v), 4) for v in w_mean),
        combined_oos_ic_mean=float(arr.mean()),
        combined_oos_ic_median=float(np.median(arr)),
        combined_oos_ic_std=float(arr.std()),
        pct_dates_positive_ic=float((arr > 0).mean() * 100.0),
        n_dates_oos=int(arr.size),
        n_rows=len(panel),
        n_names=panel.n_names,
        horizon=panel.horizon,
        rebalance=panel.rebalance,
        alpha=float(alpha),
    )


def permutation_test_cross_sectional(
    panel: CrossSectionalPanel,
    cv: Optional[CombinatorialPurgedCV] = None,
    n_permutations: int = 200,
    alpha: float = 1.0,
    seed: int = 0,
) -> "FactorPermutationResult":
    """Within-date label-shuffle significance test for the combined OOS rank IC.

    Each permutation shuffles the forward returns **among the names on the same
    date** (destroying the feature→return link while preserving the cross-sectional
    structure), then re-runs :func:`cross_sectional_ic_eval`. The p-value is the
    fraction of shuffles whose combined OOS IC matches or beats the real one.
    """
    from vpts.ml.models import FactorPermutationResult

    cv = cv or CombinatorialPurgedCV(
        n_groups=6, n_test_groups=2, purge=panel.purge_dates, embargo_pct=0.01)
    real = cross_sectional_ic_eval(panel, cv, alpha)
    groups = _date_groups(panel.date_id, panel.n_dates)
    rng = np.random.default_rng(seed)

    null: list[float] = []
    for _ in range(n_permutations):
        y_perm = panel.y.copy()
        for g in groups:                       # shuffle within each date's names
            if g.size > 1:
                y_perm[g] = panel.y[g][rng.permutation(g.size)]
        shuffled = CrossSectionalPanel(
            X=panel.X, y=y_perm, date_id=panel.date_id,
            feature_names=panel.feature_names, horizon=panel.horizon,
            rebalance=panel.rebalance, n_dates=panel.n_dates, symbols=panel.symbols)
        try:
            r = cross_sectional_ic_eval(shuffled, cv, alpha)
        except ValueError:
            continue
        if np.isfinite(r.combined_oos_ic_mean):
            null.append(r.combined_oos_ic_mean)

    arr = np.array(null, dtype=float)
    p = float((np.sum(arr >= real.combined_oos_ic_mean) + 1) / (arr.size + 1))
    return FactorPermutationResult(
        real_ic=real.combined_oos_ic_mean,
        null_ic_mean=float(arr.mean()) if arr.size else float("nan"),
        p_value=p, n_permutations=int(arr.size), symbol="cross-section")
