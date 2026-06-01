"""Structural analytics — synthetic delta, profile shape, footprints, decay.

Transforms a static Volume Profile into quantifiable, no-look-ahead time-series
features (:mod:`vpts.structure.analytics`) and assembles them into a
:class:`~vpts.ml.models.FactorDataset` (:func:`build_structural_dataset`) that
plugs straight into the validated purged-CPCV harness.
"""
from __future__ import annotations

from vpts.structure.analytics import (
    SHAPE_NAMES,
    classify_shape,
    close_location_value,
    decayed_poc,
    detect_ledges,
    double_distribution,
    poor_high,
    poor_low,
    synthetic_delta_profile,
    synthetic_delta_stats,
    value_area_compression_ratio,
    weighted_moments,
)
from vpts.structure.dataset import build_structural_dataset, build_structural_meta_dataset
from vpts.structure.models import STRUCTURAL_FEATURES, StructuralFeatures

__all__ = [
    "build_structural_dataset",
    "build_structural_meta_dataset",
    "STRUCTURAL_FEATURES",
    "StructuralFeatures",
    "SHAPE_NAMES",
    "classify_shape",
    "close_location_value",
    "decayed_poc",
    "detect_ledges",
    "double_distribution",
    "poor_high",
    "poor_low",
    "synthetic_delta_profile",
    "synthetic_delta_stats",
    "value_area_compression_ratio",
    "weighted_moments",
]
