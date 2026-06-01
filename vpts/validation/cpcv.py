"""Combinatorial Purged Cross-Validation (CPCV) with purging + embargo.

Implements the CV scheme from López de Prado, *Advances in Financial Machine
Learning*, and a thin evaluator that wraps the Phase-6 :class:`~vpts.backtest.engine.Backtester`.

Two pieces:

* :meth:`CombinatorialPurgedCV.split` — the rigorous, data-agnostic **splitter**:
  partitions the timeline into ``n_groups`` contiguous groups, takes every
  combination of ``n_test_groups`` as the test set, and **purges** train
  observations whose label window overlaps a test block plus an **embargo** of
  train observations immediately after each test block. This is the reusable
  machinery you need the moment any parameter is fit on the train folds.

* :meth:`CombinatorialPurgedCV.backtest_paths` — runs the (fixed-parameter)
  backtester out-of-sample on each group with a real ``lookback`` warm-up, then
  aggregates the combinations into a **distribution** of OOS performance.

Honest scope
------------
The vpts strategy has **no fitted parameters**, so here CPCV quantifies the
*robustness/dispersion* of OOS performance across recombined held-out periods —
it is **not** protection against parameter-selection overfitting (there is no
selection step) and it does **not** fix survivorship bias in the data. The
splitter is provided precisely so that, when you do start selecting parameters
(weights, style, ML factors), the evaluation stays purged and embargoed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from vpts.backtest.engine import Backtester
from vpts.validation.models import CPCVResult, GroupResult


@dataclass(frozen=True)
class PurgedSplit:
    """One CPCV split: integer-position train/test indices and the test groups."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    test_groups: tuple[int, ...]


class CombinatorialPurgedCV:
    """Combinatorial purged CV splitter (+ a backtester evaluator).

    Parameters
    ----------
    n_groups:
        Number of contiguous groups the timeline is partitioned into.
    n_test_groups:
        How many groups form each test set (``1 < n_test_groups < n_groups``).
        Produces ``C(n_groups, n_test_groups)`` splits.
    purge:
        Label-horizon in bars: train observations within ``purge`` bars *before*
        a test block are removed (their forward label would overlap the test).
    embargo_pct:
        Fraction of the sample embargoed *after* each test block (train
        observations there are dropped to break serial-correlation leakage).
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge: int = 0,
        embargo_pct: float = 0.0,
    ) -> None:
        if n_groups < 2:
            raise ValueError("n_groups must be >= 2.")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("n_test_groups must satisfy 1 <= n_test_groups < n_groups.")
        if purge < 0:
            raise ValueError("purge must be >= 0.")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1).")
        self.n_groups = int(n_groups)
        self.n_test_groups = int(n_test_groups)
        self.purge = int(purge)
        self.embargo_pct = float(embargo_pct)

    def n_splits(self) -> int:
        return math.comb(self.n_groups, self.n_test_groups)

    # ------------------------------------------------------------------ #
    def split(self, n_samples: int) -> Iterator[PurgedSplit]:
        """Yield purged + embargoed train/test index splits for *n_samples*."""
        if n_samples < self.n_groups:
            raise ValueError("n_samples must be >= n_groups.")
        groups = np.array_split(np.arange(n_samples), self.n_groups)
        embargo_bars = math.ceil(self.embargo_pct * n_samples)

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.sort(np.concatenate([groups[g] for g in combo]))
            train_groups = [g for g in range(self.n_groups) if g not in combo]
            train_idx = (
                np.sort(np.concatenate([groups[g] for g in train_groups]))
                if train_groups else np.array([], dtype=int)
            )
            train_idx = self._purge_embargo(
                train_idx, test_idx, n_samples, self.purge, embargo_bars
            )
            yield PurgedSplit(train_idx=train_idx, test_idx=test_idx, test_groups=combo)

    @staticmethod
    def _purge_embargo(
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        n: int,
        purge: int,
        embargo_bars: int,
    ) -> np.ndarray:
        """Drop train indices that purge (pre-block) or embargo (post-block) require."""
        if train_idx.size == 0:
            return train_idx
        drop: set[int] = set()
        for a, b in _contiguous_blocks(test_idx):
            if purge:
                drop.update(range(max(0, a - purge), a))            # purge before block
            if embargo_bars:
                drop.update(range(b + 1, min(n, b + 1 + embargo_bars)))  # embargo after
        if not drop:
            return train_idx
        return train_idx[~np.isin(train_idx, list(drop))]

    # ------------------------------------------------------------------ #
    def backtest_paths(
        self,
        df: pd.DataFrame,
        backtester: Backtester,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> CPCVResult:
        """Run *backtester* out-of-sample per group, aggregate combinations.

        Each group is backtested on a contiguous slice that includes a real
        ``backtester.lookback``-bar warm-up taken from the immediately preceding
        data (groups without enough preceding history are skipped). Combinations
        of ``n_test_groups`` groups are then aggregated into the OOS distribution.
        """
        n = len(df)
        groups = np.array_split(np.arange(n), self.n_groups)
        lookback = backtester.lookback

        group_results: list[GroupResult] = []
        for gi, g in enumerate(groups):
            a, b = int(g[0]), int(g[-1])
            start = a - lookback
            if start < 0 or (b + 1 - start) < lookback + 2:
                group_results.append(GroupResult(
                    gi, df.index[a], df.index[b], 0, float("nan"),
                    float("nan"), float("nan"), skipped=True))
                continue
            res = backtester.run(df.iloc[start : b + 1], symbol=symbol, interval=interval)
            group_results.append(GroupResult(
                gi, df.index[a], df.index[b], res.n_trades,
                res.total_return_pct, res.sharpe, res.max_drawdown_pct))

        # Combinatorial paths: mean across the k groups of each combination.
        path_returns: list[float] = []
        path_sharpes: list[float] = []
        for combo in combinations(range(self.n_groups), self.n_test_groups):
            members = [group_results[c] for c in combo]
            if any(m.skipped for m in members):       # every path = k real OOS groups
                continue
            path_returns.append(float(np.mean([m.return_pct for m in members])))
            sh = [m.sharpe for m in members if np.isfinite(m.sharpe)]
            path_sharpes.append(float(np.mean(sh)) if sh else float("nan"))

        return self._build_result(group_results, path_returns, path_sharpes, symbol)

    def _build_result(self, group_results, path_returns, path_sharpes, symbol):
        ret = np.array(path_returns, dtype=float)
        shp = np.array([s for s in path_sharpes if np.isfinite(s)], dtype=float)
        if ret.size == 0:
            raise ValueError(
                "CPCV produced no usable paths — too few bars per group for the "
                "backtester's lookback. Increase data length or reduce n_groups/lookback."
            )
        return CPCVResult(
            group_results=tuple(group_results),
            n_paths=int(ret.size),
            n_test_groups=self.n_test_groups,
            return_mean=float(ret.mean()),
            return_median=float(np.median(ret)),
            return_std=float(ret.std()),
            return_min=float(ret.min()),
            return_max=float(ret.max()),
            pct_paths_profitable=float((ret > 0).mean() * 100.0),
            sharpe_mean=float(shp.mean()) if shp.size else float("nan"),
            sharpe_median=float(np.median(shp)) if shp.size else float("nan"),
            path_returns=tuple(round(float(x), 4) for x in ret),
            path_sharpes=tuple(round(float(x), 4) for x in path_sharpes),
            symbol=symbol,
            extra={"purge": self.purge, "embargo_pct": self.embargo_pct},
        )


def _contiguous_blocks(indices: np.ndarray) -> list[tuple[int, int]]:
    """Collapse a sorted integer index array into ``(start, end)`` contiguous runs."""
    if len(indices) == 0:
        return []
    arr = np.sort(np.asarray(indices))
    blocks = []
    start = prev = int(arr[0])
    for x in arr[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
        else:
            blocks.append((start, prev))
            start = prev = x
    blocks.append((start, prev))
    return blocks
