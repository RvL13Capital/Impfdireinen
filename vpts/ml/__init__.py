"""Learned models — factor weights and triple-barrier meta-labeling.

All models are evaluated out-of-sample with the purged CPCV from
:mod:`vpts.validation`.
"""
from __future__ import annotations

from vpts.ml.factor_model import (
    RidgeFactorModel,
    build_factor_dataset,
    cpcv_factor_eval,
)
from vpts.ml.labeling import build_meta_dataset, triple_barrier_labels
from vpts.ml.meta_model import LogisticMetaModel, cpcv_meta_eval
from vpts.ml.models import (
    FactorCVResult,
    FactorDataset,
    MetaCVResult,
    MetaDataset,
)

__all__ = [
    # factor weights
    "build_factor_dataset",
    "RidgeFactorModel",
    "cpcv_factor_eval",
    "FactorDataset",
    "FactorCVResult",
    # triple-barrier meta-labeling
    "triple_barrier_labels",
    "build_meta_dataset",
    "LogisticMetaModel",
    "cpcv_meta_eval",
    "MetaDataset",
    "MetaCVResult",
]
