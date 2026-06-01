"""Learned models — factor-weight regression evaluated with purged CPCV.

* :func:`~vpts.ml.factor_model.build_factor_dataset` — confluence factors -> forward return.
* :class:`~vpts.ml.factor_model.RidgeFactorModel` — numpy ridge that learns factor weights.
* :func:`~vpts.ml.factor_model.cpcv_factor_eval` — honest out-of-sample evaluation.
* :class:`~vpts.ml.models.FactorDataset` / :class:`~vpts.ml.models.FactorCVResult`.
"""
from __future__ import annotations

from vpts.ml.factor_model import (
    RidgeFactorModel,
    build_factor_dataset,
    cpcv_factor_eval,
)
from vpts.ml.models import FactorCVResult, FactorDataset

__all__ = [
    "build_factor_dataset",
    "RidgeFactorModel",
    "cpcv_factor_eval",
    "FactorDataset",
    "FactorCVResult",
]
