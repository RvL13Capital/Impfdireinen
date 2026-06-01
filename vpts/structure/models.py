"""Immutable structural-feature row + the canonical feature ordering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

# The numeric feature matrix fed to the (validated) CPCV harness. Categorical
# shape is one-hot encoded (is_P/is_b/is_B; D and "unknown" are the baseline) so
# it is honest for the linear ridge model as well as for trees.
STRUCTURAL_FEATURES: tuple[str, ...] = (
    "delta_net",              # net synthetic delta (buy/sell dominance) [-1, 1]
    "delta_poc",              # synthetic delta AT the POC [-1, 1]
    "skew",                   # volume-weighted profile skewness
    "kurtosis",               # volume-weighted profile kurtosis (Pearson)
    "poc_loc",                # POC location in the H-L range [0, 1]
    "vacr_z",                 # value-area compression z-score (the "quiet" trigger)
    "poc_slope",              # POC-migration slope, fractional drift / step
    "cost_basis_migration",   # (decayed POC - lifetime POC) / ATR
    "n_ledges",               # count of >3σ volume cliffs (institutional walls)
    "poor_high",              # 1 if the top of the auction is unfinished/blunt
    "is_P",                   # one-hot P-shape (accumulation / squeeze)
    "is_b",                   # one-hot b-shape (distribution / capitulation)
    "is_B",                   # one-hot B-shape (double distribution / trend shift)
)


@dataclass(frozen=True)
class StructuralFeatures:
    """One decision bar's structural-microstructure features (all data ≤ t)."""

    delta_net: float
    delta_poc: float
    skew: float
    kurtosis: float
    poc_loc: float
    vacr_z: float
    poc_slope: float
    cost_basis_migration: float
    n_ledges: float
    poor_high: float
    is_P: float
    is_b: float
    is_B: float
    # --- metadata (not part of the feature vector) ---
    shape_class: int = 0
    poc: float = float("nan")
    vacr: float = float("nan")
    timestamp: Optional[pd.Timestamp] = None

    feature_names: ClassVar[tuple[str, ...]] = STRUCTURAL_FEATURES

    def to_vector(self) -> np.ndarray:
        """Return the feature row in :data:`STRUCTURAL_FEATURES` order."""
        return np.array([getattr(self, n) for n in STRUCTURAL_FEATURES], dtype=float)
