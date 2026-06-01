"""Phase 5 — Streamlit dashboard shell for the Quiet-Volume system.

Launch it (locally or on the free Streamlit Community Cloud / Colab)::

    streamlit run vpts/dashboard/app.py

All heavy lifting lives in the Phase 1–4 modules and in the pure figure builders
in :mod:`vpts.dashboard.charts`; this file is only the thin interactive shell, so
importing it has no side effects until :func:`main` is called (under ``streamlit
run`` the module's ``__name__`` is ``"__main__"``).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run vpts/dashboard/app.py` from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from vpts import (  # noqa: E402
    ConfluenceScorer,
    DataFetchError,
    MarketDataFetcher,
    QuietPhaseDetector,
    SignalGenerator,
    VolumePatternDetector,
    VolumeProfileCalculator,
)
from vpts.dashboard import charts  # noqa: E402

WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "SPY", "QQQ"]
PERIODS = ["3mo", "6mo", "1y", "2y", "5y", "max"]
INTERVALS = ["1d", "1h", "30m", "15m", "5m", "1wk"]


# --------------------------------------------------------------------------- #
# Cached data + compute
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=3600)
def load_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    return MarketDataFetcher().fetch(symbol, period=period, interval=interval)


def run_pipeline(df: pd.DataFrame, symbol: str, interval: str, cfg: dict):
    """Run profile → quiet → patterns → confluence → signal with *cfg*."""
    profile = VolumeProfileCalculator(
        bin_mode=cfg["bin_mode"], num_bins=cfg["num_bins"],
        value_area_pct=cfg["value_area_pct"],
    ).calculate(df, symbol=symbol, interval=interval)
    quiet = QuietPhaseDetector().detect(df, symbol=symbol, interval=interval)
    patterns = VolumePatternDetector().detect(
        df, profile=profile, symbol=symbol, interval=interval
    )
    score = ConfluenceScorer().score(df, profile, quiet, patterns,
                                    symbol=symbol, interval=interval)
    signal = SignalGenerator(
        style=cfg["style"], risk_fraction=cfg["risk_pct"] / 100.0,
        account_equity=cfg["equity"],
    ).from_score(score, profile)
    return profile, quiet, patterns, score, signal


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def deep_dive(symbol: str, period: str, interval: str, cfg: dict) -> None:
    try:
        df = load_data(symbol, period, interval)
    except DataFetchError as exc:
        st.error(f"Could not load **{symbol}**: {exc}")
        return

    profile, quiet, patterns, score, signal = run_pipeline(df, symbol, interval, cfg)

    last = float(df["Close"].iloc[-1])
    top = st.columns(4)
    top[0].metric(symbol, f"{last:,.2f}")
    top[1].metric("Setup quality", f"{score.setup_quality:.0f}/100")
    top[2].metric("Bias", score.bias.upper(), f"{score.bias_score:+.0f}")
    top[3].metric("Signal", signal.action.value.upper())

    st.plotly_chart(
        charts.price_profile_figure(df, profile, signal=signal, patterns=patterns),
        use_container_width=True,
    )

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(charts.confluence_gauge_figure(score), use_container_width=True)
    with right:
        st.plotly_chart(charts.component_figure(score), use_container_width=True)

    st.plotly_chart(charts.quiet_phase_figure(quiet), use_container_width=True)

    st.subheader("Trade signal")
    if signal.is_actionable:
        cols = st.columns(5)
        cols[0].metric("Action", signal.action.value.upper())
        cols[1].metric("Entry", f"{signal.entry:,.2f}")
        cols[2].metric("Stop", f"{signal.stop:,.2f}")
        cols[3].metric("R:R", f"{signal.risk_reward_ratio:.2f}")
        cols[4].metric("Size", f"{signal.suggested_size:g}")
    st.code(signal.explain(), language="text")

    with st.expander("Confluence rationale & components"):
        st.write(score.rationale)
        st.dataframe(pd.DataFrame(score.breakdown()).T, use_container_width=True)

    if patterns.patterns:
        with st.expander(f"Recent volume patterns ({len(patterns.patterns)})"):
            st.dataframe(
                pd.DataFrame([
                    {"type": p.type.value, "when": p.timestamp, "dir": p.direction,
                     "price": round(p.price, 2), "at_level": p.at_level,
                     "why": p.explanation}
                    for p in patterns.recent(12)
                ]),
                use_container_width=True,
            )


def scanner(period: str, interval: str, cfg: dict) -> None:
    st.caption("Ranks a fixed watchlist by confluence setup quality.")
    rows = []
    bar = st.progress(0.0)
    for i, sym in enumerate(WATCHLIST, 1):
        try:
            df = load_data(sym, period, interval)
            _, quiet, _, score, signal = run_pipeline(df, sym, interval, cfg)
            rows.append({
                "symbol": sym,
                "price": round(float(df["Close"].iloc[-1]), 2),
                "quality": round(score.setup_quality, 1),
                "bias": score.bias,
                "bias_score": round(score.bias_score, 1),
                "quiet": round(quiet.latest.quiet_score, 0),
                "signal": signal.action.value.upper(),
                "R:R": signal.risk_reward_ratio,
            })
        except DataFetchError as exc:
            rows.append({"symbol": sym, "price": None, "quality": None,
                         "bias": f"error: {exc}", "signal": "—"})
        bar.progress(i / len(WATCHLIST))
    bar.empty()

    table = pd.DataFrame(rows)
    if "quality" in table:
        table = table.sort_values("quality", ascending=False, na_position="last")
    st.dataframe(table, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Quiet-Volume", page_icon="📊", layout="wide")
    st.title("📊 Quiet-Volume — Volume Profile & Quiet-Phase System")
    st.caption("Free · explainable · built for low-volatility phases")

    with st.sidebar:
        st.header("Controls")
        symbol = st.text_input("Ticker", value="AAPL").strip().upper()
        period = st.selectbox("Period", PERIODS, index=PERIODS.index("1y"))
        interval = st.selectbox("Interval", INTERVALS, index=0)
        style = st.radio("Signal style", ["reversion", "breakout"], horizontal=True)
        with st.expander("Profile & risk settings"):
            bin_mode = st.selectbox("Bin mode", ["auto", "fixed"], index=0)  # auto default
            num_bins = st.slider("Bins (fixed mode)", 20, 300, 100, step=10)
            value_area_pct = st.slider("Value area %", 0.50, 0.90, 0.70, step=0.01)
            risk_pct = st.slider("Risk per trade (%)", 0.25, 5.0, 1.0, step=0.25)
            equity = st.number_input("Account equity", min_value=100.0,
                                     value=10_000.0, step=500.0)

    cfg = dict(bin_mode=bin_mode, num_bins=num_bins, value_area_pct=value_area_pct,
               style=style, risk_pct=risk_pct, equity=equity)

    deep_tab, scan_tab = st.tabs(["🔍 Deep dive", "📡 Scanner"])
    with deep_tab:
        if symbol:
            deep_dive(symbol, period, interval, cfg)
        else:
            st.info("Enter a ticker in the sidebar to begin.")
    with scan_tab:
        scanner(period, interval, cfg)


if __name__ == "__main__":
    main()
