"""Phase 5 — pure Plotly figure builders for the dashboard.

These functions are deliberately **decoupled from Streamlit**: each takes the
already-computed Phase 1–4 objects and returns a :class:`plotly.graph_objects.Figure`.
That keeps them unit-testable offline (no browser, no Streamlit runtime) and lets
the ``app.py`` shell stay thin.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vpts.profile.models import VolumeProfile
from vpts.regime.patterns import VolumePatternResult, VolumePatternType
from vpts.regime.quiet import QuietPhaseResult
from vpts.scoring.models import ConfluenceScore
from vpts.signals.models import SignalAction, TradeSignal

# --- palette (dark, trader-friendly) --------------------------------------- #
TEMPLATE = "plotly_dark"
BULL = "#26a69a"
BEAR = "#ef5350"
NEUT = "#9e9e9e"
POC_C = "#ffca28"
VA_FILL = "rgba(92, 107, 192, 0.12)"
VA_BAR = "#5c6bc0"
OFF_BAR = "#37474f"
ENTRY_C = "#26c6da"
STOP_C = "#ef5350"
TARGET_C = "#66bb6a"
PAPER = "#0e1117"

_DIR_COLOR = {1: BULL, -1: BEAR, 0: NEUT}
_PATTERN_SYMBOL = {
    VolumePatternType.CLIMAX: "x",
    VolumePatternType.DRY_UP: "circle",
    VolumePatternType.ACCUMULATION: "triangle-up",
    VolumePatternType.DIVERGENCE: "diamond",
}


def _dt_index(index: pd.Index):
    """Plain ``datetime`` array for a DatetimeIndex (kaleido/orjson-safe), else as-is."""
    return index.to_pydatetime() if isinstance(index, pd.DatetimeIndex) else np.asarray(index)


def _dt_scalar(value):
    """Coerce a pandas ``Timestamp`` to ``datetime`` (orjson-safe), else pass through."""
    return value.to_pydatetime() if isinstance(value, pd.Timestamp) else value


def _base_layout(fig: go.Figure, height: int, title: Optional[str] = None) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        height=height,
        margin=dict(l=48, r=16, t=48 if title else 16, b=32),
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=15)))
    return fig


def price_profile_figure(
    df: pd.DataFrame,
    profile: VolumeProfile,
    signal: Optional[TradeSignal] = None,
    patterns: Optional[VolumePatternResult] = None,
    max_bars: int = 400,
    height: int = 640,
) -> go.Figure:
    """Candlesticks + the volume-profile histogram, with levels/markers/signal.

    Left panel: price candlesticks with POC/VAH/VAL lines, the value-area band,
    pattern markers and (if actionable) entry/stop/target lines. Right panel:
    the horizontal volume-by-price profile, POC and value-area bars highlighted.
    """
    view = df.tail(max_bars)
    view_x = _dt_index(view.index)
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        column_widths=[0.78, 0.22], horizontal_spacing=0.012,
    )

    # --- price candles (col 1) --------------------------------------- #
    fig.add_trace(
        go.Candlestick(
            x=view_x, open=view["Open"], high=view["High"],
            low=view["Low"], close=view["Close"], name="price",
            increasing_line_color=BULL, decreasing_line_color=BEAR,
            showlegend=False,
        ),
        row=1, col=1,
    )

    # --- volume profile (col 2) -------------------------------------- #
    tdf = profile.to_dataframe()
    colors = [
        POC_C if is_poc else (VA_BAR if in_va else OFF_BAR)
        for is_poc, in_va in zip(tdf["is_poc"], tdf["in_value_area"])
    ]
    fig.add_trace(
        go.Bar(
            x=tdf["volume"], y=tdf.index, orientation="h",
            marker_color=colors, name="volume profile", showlegend=False,
            hovertemplate="price %{y:.2f}<br>volume %{x:,.0f}<extra></extra>",
        ),
        row=1, col=2,
    )

    # --- value-area band + key levels on the price panel ------------- #
    fig.add_hrect(y0=profile.val, y1=profile.vah, fillcolor=VA_FILL,
                  line_width=0, row=1, col=1)
    for level, name, color in (
        (profile.poc, "POC", POC_C),
        (profile.vah, "VAH", "#90a4ae"),
        (profile.val, "VAL", "#90a4ae"),
    ):
        fig.add_hline(
            y=level, line=dict(color=color, width=1, dash="dot"),
            annotation_text=f"{name} {level:.2f}",
            annotation_position="top left",
            annotation_font_size=10, row=1, col=1,
        )

    # --- pattern markers --------------------------------------------- #
    if patterns is not None and patterns.patterns:
        for ptype in VolumePatternType:
            evs = patterns.of_type(ptype)
            evs = [e for e in evs if 0 <= e.index_pos < len(df)]
            if not evs:
                continue
            xs = [_dt_scalar(df.index[e.index_pos]) for e in evs]
            ys = [e.price for e in evs]
            cols = [_DIR_COLOR[1 if e.direction == "bullish"
                               else -1 if e.direction == "bearish" else 0]
                    for e in evs]
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="markers", name=ptype.value,
                    marker=dict(symbol=_PATTERN_SYMBOL[ptype], size=10,
                                color=cols, line=dict(width=1, color="#eceff1")),
                    text=[e.explanation for e in evs],
                    hovertemplate="%{text}<extra></extra>",
                ),
                row=1, col=1,
            )

    # --- signal levels ----------------------------------------------- #
    if signal is not None and signal.is_actionable:
        fig.add_hline(y=signal.entry, line=dict(color=ENTRY_C, width=1.5),
                      annotation_text=f"{signal.action.value.upper()} entry "
                                      f"{signal.entry:.2f}",
                      annotation_position="bottom left",
                      annotation_font_size=10, row=1, col=1)
        fig.add_hline(y=signal.stop, line=dict(color=STOP_C, width=1.2, dash="dash"),
                      annotation_text=f"stop {signal.stop:.2f}",
                      annotation_position="bottom left",
                      annotation_font_size=10, row=1, col=1)
        for i, tgt in enumerate(signal.targets, 1):
            fig.add_hline(y=tgt, line=dict(color=TARGET_C, width=1.0, dash="dot"),
                          annotation_text=f"T{i} {tgt:.2f}",
                          annotation_position="bottom left",
                          annotation_font_size=10, row=1, col=1)

    title = f"{profile.symbol or 'data'} {profile.interval or ''} — Price & Volume Profile"
    fig = _base_layout(fig, height, title)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(title_text="volume", row=1, col=2)
    return fig


def quiet_phase_figure(
    quiet_result: QuietPhaseResult, height: int = 250
) -> go.Figure:
    """Quiet-score line over time with the threshold and shaded quiet segments."""
    fr = quiet_result.frame
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_dt_index(fr.index), y=fr["quiet_score"], name="quiet score",
        line=dict(color="#42a5f5", width=1.6), fill="tozeroy",
        fillcolor="rgba(66,165,245,0.10)",
    ))
    fig.add_hline(
        y=quiet_result.quiet_threshold,
        line=dict(color="#ffa726", width=1, dash="dash"),
        annotation_text=f"quiet ≥ {quiet_result.quiet_threshold:.0f}",
        annotation_position="top left", annotation_font_size=10,
    )
    for seg in quiet_result.quiet_segments():
        fig.add_vrect(x0=_dt_scalar(seg["start"]), x1=_dt_scalar(seg["end"]),
                      fillcolor="rgba(38,166,154,0.14)", line_width=0)
    fig = _base_layout(fig, height, "Quiet-phase score")
    fig.update_yaxes(range=[0, 100])
    return fig


def confluence_gauge_figure(score: ConfluenceScore, height: int = 280) -> go.Figure:
    """A 0–100 setup-quality gauge, coloured by directional bias."""
    color = {"bullish": BULL, "bearish": BEAR, "neutral": NEUT}[score.bias]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score.setup_quality,
        number={"suffix": "/100", "font": {"size": 30}},
        title={"text": f"Setup Quality<br><span style='font-size:0.8em'>"
                       f"{score.bias.upper()}  (bias {score.bias_score:+.0f})</span>"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#263238"},
                {"range": [40, 70], "color": "#37474f"},
                {"range": [70, 100], "color": "#455a64"},
            ],
        },
    ))
    return _base_layout(fig, height)


def component_figure(score: ConfluenceScore, height: int = 280) -> go.Figure:
    """Horizontal bars of each component's weighted strength, coloured by bias."""
    comps = sorted(score.components, key=lambda c: c.weighted_strength)
    fig = go.Figure(go.Bar(
        x=[c.weighted_strength for c in comps],
        y=[c.name for c in comps],
        orientation="h",
        marker_color=[_DIR_COLOR[c.direction] for c in comps],
        text=[f"{c.strength:.2f} {c.bias_label}" for c in comps],
        textposition="outside",
        hovertext=[c.reason for c in comps],
        hovertemplate="%{y}: %{hovertext}<extra></extra>",
        showlegend=False,
    ))
    fig = _base_layout(fig, height, "Confluence components (weighted strength)")
    fig.update_xaxes(range=[0, max(1.0, max((c.weighted_strength for c in comps),
                                            default=1.0)) * 1.25])
    return fig
