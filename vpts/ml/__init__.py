"""Learned models — factor weights and triple-barrier meta-labeling.

All models are evaluated out-of-sample with the purged CPCV from
:mod:`vpts.validation`.
"""
from __future__ import annotations

from vpts.ml.factor_model import (
    RidgeFactorModel,
    build_factor_dataset,
    cpcv_factor_eval,
    permutation_test_factor,
)
from vpts.ml.features import ENRICHED_FEATURES, build_enriched_factor_dataset
from vpts.ml.labeling import build_meta_dataset, triple_barrier_labels
from vpts.ml.meta_model import (
    LogisticMetaModel,
    cpcv_meta_eval,
    permutation_test_meta,
)
from vpts.ml.models import (
    FactorCVResult,
    FactorDataset,
    FactorPermutationResult,
    MetaCVResult,
    MetaDataset,
    MetaPermutationResult,
)

__all__ = [
    # factor weights
    "build_factor_dataset",
    "build_enriched_factor_dataset",
    "ENRICHED_FEATURES",
    "RidgeFactorModel",
    "cpcv_factor_eval",
    "permutation_test_factor",
    "FactorDataset",
    "FactorCVResult",
    "FactorPermutationResult",
    # triple-barrier meta-labeling
    "triple_barrier_labels",
    "build_meta_dataset",
    "LogisticMetaModel",
    "cpcv_meta_eval",
    "permutation_test_meta",
    "MetaDataset",
    "MetaCVResult",
    "MetaPermutationResult",
]
