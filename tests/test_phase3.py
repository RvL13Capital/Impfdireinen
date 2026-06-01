"""Phase 3 test-suite — Confluence & Scoring Engine.

Fully offline and deterministic.

    python tests/test_phase3.py     # pretty PASS/FAIL report + sample score
    pytest tests/test_phase3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts import (  # noqa: E402
    ConfluenceScore,
    ConfluenceScorer,
    QuietPhaseDetector,
    VolumePatternDetector,
    VolumeProfileCalculator,
)


# --------------------------------------------------------------------------- #
# Fixtures (self-contained: warm-up → quiet coil → expansion with a climax)
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
    ci = n_w + n_q + 20
    vol[ci] *= 6.0
    high[ci], low[ci] = close[ci] + 4.5, close[ci] - 4.5
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def _score(df: pd.DataFrame, **kw) -> ConfluenceScore:
    return ConfluenceScorer(**kw).analyze(df, symbol="SYN", interval="1d")


# --------------------------------------------------------------------------- #
# Core invariants
# --------------------------------------------------------------------------- #
def test_bounds_and_bias_invariant() -> None:
    for df in (regime_df().iloc[:105], regime_df(), regime_df(9)):
        s = _score(df)
        assert 0.0 <= s.setup_quality <= 100.0
        assert -100.0 <= s.bias_score <= 100.0
        # Directional conviction can never exceed the quality of the evidence.
        assert abs(s.bias_score) <= s.setup_quality + 1e-6


def test_bias_label_matches_score() -> None:
    band = 10.0
    for df in (regime_df().iloc[:105], regime_df()):
        s = _score(df, neutral_band=band)
        if s.bias_score > band:
            assert s.bias == "bullish"
        elif s.bias_score < -band:
            assert s.bias == "bearish"
        else:
            assert s.bias == "neutral"


def test_components_and_breakdown_shape() -> None:
    s = _score(regime_df())
    names = [c.name for c in s.components]
    assert names == ["value_area", "key_level", "quiet", "patterns"]
    bd = s.breakdown()
    assert set(bd) == set(names)
    for v in bd.values():
        assert set(v) == {"weight", "strength", "direction", "reason"}
    assert "Confluence" in s.summary()
    # as_dict must be JSON-serialisable.
    json.dumps(s.as_dict())


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #
def test_quiet_component_tracks_quiet_score() -> None:
    df = regime_df().iloc[:105]
    profile = VolumeProfileCalculator().calculate(df)
    quiet = QuietPhaseDetector().detect(df)
    patterns = VolumePatternDetector().detect(df, profile=profile)
    s = ConfluenceScorer().score(df, profile, quiet, patterns)
    quiet_comp = next(c for c in s.components if c.name == "quiet")
    assert np.isclose(quiet_comp.strength, quiet.latest.quiet_score / 100.0)
    assert quiet_comp.direction == 0  # quiet is a non-directional amplifier


def test_directional_response_coil_vs_expansion() -> None:
    coil = _score(regime_df().iloc[:105])      # quiet, near value low
    expansion = _score(regime_df())            # stretched above value, bearish div
    assert coil.bias_score > expansion.bias_score
    assert coil.bias_score > 0                  # leans bullish
    assert expansion.bias == "bearish"


def test_weights_change_the_score() -> None:
    df = regime_df().iloc[:105]
    default = _score(df)
    no_quiet = _score(df, weights={"quiet": 0.0})
    # Quiet is the dominant component in the coil, so dropping it must matter.
    assert not np.isclose(default.setup_quality, no_quiet.setup_quality)


def test_quiet_only_weighting_is_neutral() -> None:
    # With only the (non-directional) quiet component weighted, bias must vanish.
    s = _score(
        regime_df(),
        weights={"value_area": 0.0, "key_level": 0.0, "quiet": 1.0, "patterns": 0.0},
    )
    assert s.bias == "neutral"
    assert abs(s.bias_score) < 1e-6


def test_analyze_matches_explicit_score() -> None:
    df = regime_df().iloc[:105]
    profile = VolumeProfileCalculator().calculate(df)
    quiet = QuietPhaseDetector().detect(df)
    patterns = VolumePatternDetector().detect(df, profile=profile)
    explicit = ConfluenceScorer().score(df, profile, quiet, patterns)
    auto = ConfluenceScorer().analyze(df)
    assert np.isclose(explicit.setup_quality, auto.setup_quality)
    assert np.isclose(explicit.bias_score, auto.bias_score)


def test_determinism() -> None:
    df = regime_df()
    assert _score(df).as_dict() == _score(df).as_dict()


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_invalid_config() -> None:
    for kwargs in (
        {"weights": {"bogus": 1.0}},
        {"weights": {"quiet": -1.0}},
        {"weights": {"value_area": 0.0, "key_level": 0.0, "quiet": 0.0, "patterns": 0.0}},
        {"neutral_band": 100.0},
        {"pattern_recency": 0},
    ):
        try:
            ConfluenceScorer(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_score_min_bars_guard() -> None:
    df = regime_df().iloc[:1]
    profile_df = regime_df()
    profile = VolumeProfileCalculator().calculate(profile_df)
    quiet = QuietPhaseDetector().detect(profile_df)
    patterns = VolumePatternDetector().detect(profile_df, profile=profile)
    try:
        ConfluenceScorer().score(df, profile, quiet, patterns)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for too-few bars")


# --------------------------------------------------------------------------- #
# Manual runner
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    import logging

    logging.getLogger("vpts").setLevel(logging.ERROR)
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    print(f"Running {len(tests)} Phase-3 tests …\n")
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

    print("\n" + "=" * 56)
    print("Sample confluence score (end of the quiet coil):")
    print("=" * 56)
    print(_score(regime_df().iloc[:105]).summary())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
