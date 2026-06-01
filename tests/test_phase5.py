"""Phase 5 test-suite — dashboard Plotly figure builders.

Tests the **pure** figure builders offline (no browser / Streamlit runtime) and
checks that the Streamlit app module imports without side effects. Requires
``plotly`` and ``streamlit`` (optional extras).

    python tests/test_phase5.py
    pytest tests/test_phase5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402

from vpts import (  # noqa: E402
    ConfluenceScorer,
    QuietPhaseDetector,
    SignalGenerator,
    VolumePatternDetector,
    VolumeProfileCalculator,
)
from vpts.dashboard import charts  # noqa: E402


# --------------------------------------------------------------------------- #
def regime_df(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_w, n_q, n_e = 60, 45, 45
    cw = 100 + np.cumsum(rng.normal(0, 0.4, n_w))
    cq = cw[-1] + np.cumsum(rng.normal(0.0, 0.06, n_q))
    ce = cq[-1] + np.cumsum(rng.normal(0.3, 1.0, n_e))
    close = np.concatenate([cw, cq, ce])
    ranges = np.concatenate([np.full(n_w, 1.0), np.full(n_q, 0.2), np.full(n_e, 1.8)])
    high, low = close + ranges / 2, close - ranges / 2
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = np.concatenate([
        rng.integers(2500, 3500, n_w).astype(float),
        rng.integers(700, 1100, n_q).astype(float),
        rng.integers(3000, 5000, n_e).astype(float),
    ])
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def _pipeline(df):
    # bin_mode="auto" — showcase the ATR-adaptive binning, as in the dashboard.
    profile = VolumeProfileCalculator(bin_mode="auto").calculate(df, symbol="SYN", interval="1d")
    quiet = QuietPhaseDetector().detect(df, symbol="SYN", interval="1d")
    patterns = VolumePatternDetector().detect(df, profile=profile, symbol="SYN")
    score = ConfluenceScorer().score(df, profile, quiet, patterns, symbol="SYN", interval="1d")
    signal = SignalGenerator(min_quality=30).from_score(score, profile)
    return profile, quiet, patterns, score, signal


# --------------------------------------------------------------------------- #
def test_price_profile_figure_minimal() -> None:
    df = regime_df()
    profile, *_ = _pipeline(df)
    fig = charts.price_profile_figure(df, profile)
    assert isinstance(fig, go.Figure)
    kinds = {type(t).__name__ for t in fig.data}
    assert "Candlestick" in kinds and "Bar" in kinds   # price + volume profile


def test_price_profile_figure_with_signal_and_patterns() -> None:
    df = regime_df()
    profile, quiet, patterns, score, signal = _pipeline(df)
    fig = charts.price_profile_figure(df, profile, signal=signal, patterns=patterns)
    assert isinstance(fig, go.Figure)
    # Pattern marker traces add Scatter traces on top of candles + profile bar.
    assert len(fig.data) >= 2
    assert any(isinstance(t, go.Scatter) for t in fig.data)


def test_quiet_phase_figure() -> None:
    df = regime_df()
    _, quiet, *_ = _pipeline(df)
    fig = charts.quiet_phase_figure(quiet)
    assert isinstance(fig, go.Figure)
    assert any(isinstance(t, go.Scatter) for t in fig.data)
    assert fig.layout.yaxis.range == (0, 100)


def test_confluence_gauge_figure() -> None:
    df = regime_df()
    *_, score, _ = _pipeline(df)
    fig = charts.confluence_gauge_figure(score)
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Indicator)
    assert fig.data[0].value == score.setup_quality


def test_component_figure_has_all_components() -> None:
    df = regime_df()
    *_, score, _ = _pipeline(df)
    fig = charts.component_figure(score)
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Bar)
    assert len(fig.data[0].y) == len(score.components) == 4


def test_figures_render_to_dict() -> None:
    # to_dict / to_json must succeed (catches malformed figures).
    df = regime_df()
    profile, quiet, patterns, score, signal = _pipeline(df)
    for fig in (
        charts.price_profile_figure(df, profile, signal=signal, patterns=patterns),
        charts.quiet_phase_figure(quiet),
        charts.confluence_gauge_figure(score),
        charts.component_figure(score),
    ):
        assert isinstance(fig.to_dict(), dict)


def test_app_module_imports_without_side_effects() -> None:
    # Importing the Streamlit shell must not execute the app (logic is in main()).
    import vpts.dashboard.app as app
    assert callable(app.main)
    assert app.WATCHLIST and isinstance(app.WATCHLIST, list)


def test_app_runs_headless_end_to_end() -> None:
    """Run the whole Streamlit app headlessly (data monkeypatched to synthetic).

    This is the closest we get to "does the dashboard actually render?" without a
    browser or live network: Streamlit's AppTest executes the real script and
    captures any uncaught exception.
    """
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # pragma: no cover - streamlit testing unavailable
        print("      (skipped: streamlit.testing not available)")
        return

    import vpts.data.fetcher as fetchmod

    df = regime_df()
    original = fetchmod.MarketDataFetcher.fetch
    fetchmod.MarketDataFetcher.fetch = (
        lambda self, symbol, period="6mo", interval="1d", **kw: df.copy()
    )
    try:
        app_path = Path(__file__).resolve().parents[1] / "vpts" / "dashboard" / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=90)
        at.run()
    finally:
        fetchmod.MarketDataFetcher.fetch = original

    assert not at.exception, f"app raised: {at.exception}"
    assert len(at.tabs) >= 2          # deep-dive + scanner tabs rendered
    assert at.title                   # header rendered


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} Phase-5 tests …\n")
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"  ✗ {t.__name__}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__} (error)\n      {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  ✓ {t.__name__}")
    print(f"\n{passed} passed, {failed} failed.")
    if not failed:
        print("\nFigure builders OK. Launch the dashboard with:")
        print("    streamlit run vpts/dashboard/app.py")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
