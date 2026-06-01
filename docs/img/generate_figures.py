"""Generate the committed documentation figures (dark theme, GitHub-friendly PNGs).

Every number here is transcribed from RESEARCH.md (the validation log) so the
figures stay reproducible and in sync with the writeup:

    python docs/img/generate_figures.py      # writes *.png into docs/img/

Requires plotly + kaleido (already in requirements for the dashboard).
"""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

OUT = Path(__file__).resolve().parent
BG = "#0e1117"
BLUE, RED, GREY, AMBER, GREEN = "#42a5f5", "#ef5350", "#78909c", "#ffa726", "#26a69a"


def _base(fig: go.Figure, title: str, h: int = 460) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                      title=dict(text=title, font=dict(size=17)), height=h,
                      font=dict(size=13), margin=dict(l=10, r=20, t=54, b=44),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def _save(fig: go.Figure, name: str, w: int = 1040, h: int = 460) -> None:
    fig.write_image(str(OUT / name), width=w, height=h, scale=2)
    print("wrote", name)


# 1) The arc scorecard — how far each experiment climbed toward a tradeable edge.
def arc_scorecard() -> None:
    exps = [  # (label, gates climbed 0-5, headline)
        ("1 · Walk-forward backtest",          1.0, "+14.5% — drift / survivorship"),
        ("2 · Factor ridge (confluence)",      0.4, "OOS IC ≈ 0"),
        ("3 · Meta-labeling (confluence)",     2.0, "AUC sig (survivors) → survivorship"),
        ("4 · Enriched per-name features",     0.4, "no OOS edge"),
        ("5 · Cross-sectional ranks",          1.2, "near-miss, underpowered"),
        ("6 · Structural features (8 names)",  2.0, "IC +0.10, p<0.01"),
        ("7 · Structural × 88 names",          3.0, "IC +0.035, p=0.005"),
        ("8 · Structural + survivorship",      3.0, "graceful decay → p 0.47"),
        ("9 · Decomposition + cost",           3.0, "L/S inverts +0.26→−1.07%/bet"),
        ("10 · Swing rater (selectivity)",     2.6, "lift +0.14%/bet, p=0.005 (surv)"),
        ("11 · Selectivity stress-test",       3.0, "robust 9/9 surv · injected p=0.106"),
    ]
    labels = [e[0] for e in exps][::-1]
    vals = [e[1] for e in exps][::-1]
    heads = [e[2] for e in exps][::-1]
    colors = [RED if v < 1.5 else AMBER for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=colors,
        text=heads, textposition="outside", textfont=dict(size=11), cliponaxis=False))
    fig.add_vline(x=4, line=dict(color=GREEN, width=2, dash="dash"))
    fig.add_annotation(x=4.02, y=labels[-1], yshift=20, text="◀ survivorship-robust:<br>the finish line (never crossed)",
                       showarrow=False, xanchor="left", font=dict(color=GREEN, size=11))
    _base(fig, "The ladder to a tradeable edge — 11 experiments, none cross the line", h=560)
    fig.update_layout(margin=dict(l=10, r=20, t=54, b=58))
    fig.update_xaxes(range=[0, 6.4], tickvals=[1, 2, 3, 4, 5],
                     ticktext=["shows<br>OOS signal", "clears<br>null (surv.)",
                               "robust<br>(widen/params)", "survivorship-<br>robust",
                               "net-of-cost<br>profitable"])
    _save(fig, "arc_scorecard.png", w=1100, h=560)


# 2) Survivorship mirage — conviction-bucket forward return inverts under injection.
def survivorship_inversion() -> None:
    b = [1, 2, 3, 4, 5]
    surv = [1.08, 1.26, 1.15, 1.34, 1.54]
    inj = [-0.23, -0.24, -0.66, -0.86, -1.09]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b, y=surv, mode="lines+markers", name="survivors only",
                             line=dict(color=BLUE, width=3), marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=b, y=inj, mode="lines+markers", name="+ delisted (~31%)",
                             line=dict(color=RED, width=3), marker=dict(size=9)))
    fig.add_hline(y=0, line=dict(color="#888", width=1))
    fig.add_annotation(x=5, y=1.54, text="most-bullish bucket = best (+1.54%)", showarrow=True,
                       arrowhead=2, ay=-28, font=dict(color=BLUE, size=11))
    fig.add_annotation(x=5, y=-1.09, text="most-bullish bucket = WORST (−1.09%)", showarrow=True,
                       arrowhead=2, ay=28, font=dict(color=RED, size=11))
    _base(fig, "Survivorship mirage: the conviction edge inverts off survivors")
    fig.update_xaxes(title="signal quantile (1 = most bearish → 5 = most bullish)", tickvals=b)
    fig.update_yaxes(title="mean 20-bar forward return (%)")
    _save(fig, "survivorship_inversion.png")


# 3) Structural IC decays gracefully (not a cliff) as delisted names are injected.
def structural_ic_sweep() -> None:
    dead = [0, 1, 5, 9]
    ic = [0.041, 0.029, 0.013, 0.001]
    p = [0.005, 0.015, 0.085, 0.473]
    rate = ["0%", "3%", "20%", "31%"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rate, y=ic, name="pooled OOS IC", marker_color=BLUE,
                         text=[f"{v:+.3f}" for v in ic], textposition="outside"))
    fig.add_trace(go.Scatter(x=rate, y=p, name="permutation p", yaxis="y2", mode="lines+markers",
                             line=dict(color=AMBER, width=3), marker=dict(size=9)))
    fig.add_hline(y=0.05, line=dict(color=RED, width=1, dash="dash"), yref="y2")
    fig.add_annotation(xref="paper", x=1, y=0.05, yref="y2", text="p = 0.05", showarrow=False,
                       xanchor="right", yshift=9, font=dict(color=RED, size=10))
    _base(fig, "Structural signal: graceful decay under survivorship injection")
    fig.update_xaxes(title="synthetic delisted names injected (≈ delisting rate)")
    fig.update_yaxes(title="pooled OOS IC", range=[0, 0.05])
    fig.update_layout(yaxis2=dict(title="permutation p-value", overlaying="y", side="right",
                                  range=[0, 0.5], showgrid=False))
    _save(fig, "structural_ic_sweep.png")


# 4) Selectivity robustness grid — positive on survivors across params, but n.s. injected.
def selectivity_grid() -> None:
    cells = ["h=5", "h=10", "h=15", "RR 1.5:1", "RR 2:1", "RR 3:1", "top 10%", "top 20%", "top 30%"]
    surv = [0.088, 0.075, 0.110, 0.116, 0.075, 0.084, 0.094, 0.075, 0.075]
    inj = [-0.007, 0.075, 0.090, -0.007, 0.075, 0.129, 0.056, 0.075, 0.111]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cells, y=surv, name="survivors  (p=0.023)", marker_color=BLUE))
    fig.add_trace(go.Bar(x=cells, y=inj, name="+delisted  (p=0.106, n.s.)", marker_color=RED))
    fig.add_hline(y=0, line=dict(color="#888", width=1))
    _base(fig, "Selectivity lift: robust on survivors (9/9), but DIP-carried & n.s. injected")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="parameter grid (vary one around horizon 10 / R:R 2:1 / top 20%)")
    fig.update_yaxes(title="expectancy lift (% / trade)")
    _save(fig, "selectivity_grid.png")


# 5) XGBoost memorizes in-sample but is sub-0.5 OOS — the overfitting trap.
def xgboost_overfit() -> None:
    models = ["logistic", "XGBoost"]
    ins = [0.689, 0.943]
    oos = [0.529, 0.496]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=models, y=ins, name="in-sample AUC", marker_color=RED,
                         text=[f"{v:.3f}" for v in ins], textposition="outside"))
    fig.add_trace(go.Bar(x=models, y=oos, name="out-of-sample AUC", marker_color=BLUE,
                         text=[f"{v:.3f}" for v in oos], textposition="outside"))
    fig.add_hline(y=0.5, line=dict(color="#888", width=1, dash="dash"),
                  annotation_text="0.5 = no skill", annotation_position="bottom right")
    _base(fig, "MFE/MAE classifier: XGBoost memorizes in-sample (0.943) but is 0.496 OOS")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="AUC", range=[0.45, 1.0])
    _save(fig, "xgboost_overfit.png", w=900, h=440)


if __name__ == "__main__":
    arc_scorecard()
    survivorship_inversion()
    structural_ic_sweep()
    selectivity_grid()
    xgboost_overfit()
    print("done ->", OUT)
