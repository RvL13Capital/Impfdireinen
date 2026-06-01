"""Immutable results for combinatorial purged cross-validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupResult:
    """Out-of-sample backtest of a single contiguous group (CV fold)."""

    group: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    n_trades: int
    return_pct: float
    sharpe: float
    max_drawdown_pct: float
    skipped: bool = False


@dataclass(frozen=True)
class CPCVResult:
    """Distribution of out-of-sample performance across combinatorial paths.

    Each *path* is one combination of ``n_test_groups`` held-out groups; its
    metric is the equal-weight mean across those groups' OOS results. The spread
    of these paths is the honest read on how stable the edge is across recombined
    out-of-sample periods (it is a *robustness* estimate, not a forward guarantee).
    """

    group_results: tuple[GroupResult, ...]
    n_paths: int
    n_test_groups: int
    return_mean: float
    return_median: float
    return_std: float
    return_min: float
    return_max: float
    pct_paths_profitable: float
    sharpe_mean: float
    sharpe_median: float
    path_returns: tuple[float, ...]
    path_sharpes: tuple[float, ...]
    symbol: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def n_groups(self) -> int:
        return len(self.group_results)

    @property
    def n_usable_groups(self) -> int:
        return sum(1 for g in self.group_results if not g.skipped)

    # ------------------------------------------------------------------ #
    def groups_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"group": g.group, "start": g.start, "end": g.end,
             "n_trades": g.n_trades, "return_%": round(g.return_pct, 2),
             "sharpe": round(g.sharpe, 2), "maxDD_%": round(g.max_drawdown_pct, 2),
             "skipped": g.skipped}
            for g in self.group_results
        ])

    def paths_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"mean_return_pct": self.path_returns,
                             "mean_sharpe": self.path_sharpes})

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_groups": self.n_groups,
            "n_usable_groups": self.n_usable_groups,
            "n_test_groups": self.n_test_groups,
            "n_paths": self.n_paths,
            "return_mean": round(self.return_mean, 3),
            "return_median": round(self.return_median, 3),
            "return_std": round(self.return_std, 3),
            "return_min": round(self.return_min, 3),
            "return_max": round(self.return_max, 3),
            "pct_paths_profitable": round(self.pct_paths_profitable, 1),
            "sharpe_mean": round(self.sharpe_mean, 3),
            "sharpe_median": round(self.sharpe_median, 3),
        }

    def summary(self) -> str:
        sym = self.symbol or "data"
        lines = [
            f"CPCV — {sym}  (N={self.n_groups} groups, k={self.n_test_groups} test, "
            f"{self.n_paths} paths)",
            "-" * 56,
            f"  Usable groups   : {self.n_usable_groups}/{self.n_groups}",
            f"  OOS return/path : mean {self.return_mean:+.2f}%  "
            f"median {self.return_median:+.2f}%  "
            f"[{self.return_min:+.2f}%, {self.return_max:+.2f}%]  σ {self.return_std:.2f}",
            f"  Paths profitable: {self.pct_paths_profitable:.0f}%",
            f"  OOS Sharpe/path : mean {self.sharpe_mean:.2f}  "
            f"median {self.sharpe_median:.2f}",
        ]
        per_group = ", ".join(
            f"g{g.group} {g.return_pct:+.1f}%" for g in self.group_results if not g.skipped
        )
        if per_group:
            lines.append(f"  Per-group OOS   : {per_group}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()
