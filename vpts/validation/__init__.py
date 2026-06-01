"""Validation — Combinatorial Purged Cross-Validation (CPCV) with embargo.

* :class:`~vpts.validation.cpcv.CombinatorialPurgedCV` — purged + embargoed
  combinatorial splitter and a backtester evaluator.
* :class:`~vpts.validation.models.CPCVResult` / :class:`~vpts.validation.models.GroupResult`.
"""
from __future__ import annotations

from vpts.validation.cpcv import CombinatorialPurgedCV, PurgedSplit
from vpts.validation.models import CPCVResult, GroupResult

__all__ = ["CombinatorialPurgedCV", "PurgedSplit", "CPCVResult", "GroupResult"]
