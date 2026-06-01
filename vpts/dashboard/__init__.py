"""Phase 5 — Streamlit + Plotly dashboard.

The pure figure builders in :mod:`vpts.dashboard.charts` are import-safe (they
only need plotly) and unit-tested. The Streamlit shell lives in
:mod:`vpts.dashboard.app` and is launched with::

    streamlit run vpts/dashboard/app.py

``streamlit``/``plotly`` are optional extras, so they are **not** imported at the
``vpts`` package root — only here, on demand.
"""
from __future__ import annotations

from vpts.dashboard import charts

__all__ = ["charts"]
