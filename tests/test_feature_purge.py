"""Tests for the Experiment-13 feature-clustering helper (examples/feature_purge.py).

The clustering is checked on hand-built correlation matrices with a known block
structure; the threshold behaviour is checked at the boundary.

    python tests/test_feature_purge.py
    pytest tests/test_feature_purge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from feature_purge import cluster_features  # noqa: E402


def _corr(rho: np.ndarray, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rho, index=cols, columns=cols)


def test_cluster_groups_two_correlated_blocks() -> None:
    cols = ["a", "b", "c", "d"]
    rho = np.array([
        [1.00, 0.90, 0.10, 0.10],
        [0.90, 1.00, 0.10, 0.10],
        [0.10, 0.10, 1.00, 0.85],
        [0.10, 0.10, 0.85, 1.00],
    ])
    labels = cluster_features(_corr(rho, cols), threshold=0.7)
    assert labels["a"] == labels["b"]      # the two strongly-correlated pairs each merge
    assert labels["c"] == labels["d"]
    assert labels["a"] != labels["c"]      # but the blocks stay apart
    assert len(set(labels.values())) == 2


def test_cluster_threshold_boundary() -> None:
    cols = ["a", "b"]
    corr = _corr(np.array([[1.0, 0.5], [0.5, 1.0]]), cols)
    assert len(set(cluster_features(corr, threshold=0.7).values())) == 2   # 0.5 < 0.7 → separate
    assert len(set(cluster_features(corr, threshold=0.4).values())) == 1   # 0.5 ≥ 0.4 → merge


def test_cluster_handles_negative_correlation() -> None:
    # |ρ| drives clustering: a strong *negative* correlation still merges.
    cols = ["a", "b", "c"]
    rho = np.array([[1.0, -0.92, 0.05], [-0.92, 1.0, 0.05], [0.05, 0.05, 1.0]])
    labels = cluster_features(_corr(rho, cols), threshold=0.7)
    assert labels["a"] == labels["b"] and labels["a"] != labels["c"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
