"""Tests for vpts.data.crypto — bar parsing + backward pagination/dedup (offline, no network).

The live HTTP call is exercised only in `examples/crypto_realvol.py`; here a fake
`requests`-style session is injected so the parse contract and the to_ts pagination /
de-dup / sort logic are checked deterministically with no network.

    python tests/test_crypto.py
    pytest tests/test_crypto.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpts.data.fetcher import DataFetchError  # noqa: E402
from vpts.data.crypto import _parse_bars, fetch_crypto_ohlcv  # noqa: E402


def _bar(ts: int, close: float, vbuy: float = 10.0, vsell: float = 8.0) -> dict:
    return {"TIMESTAMP": ts, "OPEN": close, "HIGH": close + 1, "LOW": close - 1,
            "CLOSE": close, "VOLUME": vbuy + vsell, "VOLUME_BUY": vbuy, "VOLUME_SELL": vsell}


class _Resp:
    def __init__(self, payload): self._p = payload
    def json(self): return self._p


class _Session:
    """Returns canned pages by call order; records the to_ts cursors it was given."""
    def __init__(self, pages): self.pages, self.cursors = pages, []
    def get(self, url, params=None, timeout=None):
        self.cursors.append((params or {}).get("to_ts"))
        i = len(self.cursors) - 1
        return _Resp({"Data": self.pages[i] if i < len(self.pages) else []})


# --------------------------------------------------------------------------- #
def test_parse_bars_contract() -> None:
    df = _parse_bars([_bar(200, 101.0, 12.0, 9.0), _bar(100, 99.0, 5.0, 7.0)])
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume", "vbuy", "vsell"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.index) == sorted(df.index)                  # sorted ascending by time
    assert float(df["Close"].iloc[0]) == 99.0 and float(df["vbuy"].iloc[1]) == 12.0
    # real buy/sell preserved (the whole point) and Volume = their sum here
    assert float(df["vsell"].iloc[0]) == 7.0


def test_parse_bars_empty() -> None:
    df = _parse_bars([])
    assert df.empty and "vbuy" in df.columns


def test_fetch_paginates_backwards_and_sorts() -> None:
    # page 1 (newest) then an older page, then exhausted → merged, de-duped, sorted.
    s = _Session([[_bar(300, 30), _bar(400, 40)],          # call 1 (to_ts=None)
                  [_bar(100, 10), _bar(200, 20)],          # call 2 (to_ts=299)
                  []])                                      # call 3 → stop
    df = fetch_crypto_ohlcv("BTC-USDT", limit=150, sleep=0, session=s)
    assert [int(t.timestamp()) for t in df.index] == [100, 200, 300, 400]
    assert s.cursors[0] is None and s.cursors[1] == 299      # cursor advanced to oldest-1


def test_fetch_dedupes_and_stops_without_progress() -> None:
    # same page every call → de-dup to 2 rows and stop (no backward progress).
    s = _Session([[_bar(100, 10), _bar(200, 20)]] * 5)
    df = fetch_crypto_ohlcv("BTC-USDT", limit=999, sleep=0, session=s)
    assert len(df) == 2 and len(s.cursors) <= 2


def test_fetch_raises_on_empty() -> None:
    with pytest.raises(DataFetchError):
        fetch_crypto_ohlcv("NOPE-USDT", limit=10, sleep=0, session=_Session([[]]))


def test_fetch_rejects_bad_frequency() -> None:
    with pytest.raises(ValueError):
        fetch_crypto_ohlcv("BTC-USDT", frequency="weeks", session=_Session([[]]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
