"""Free, keyless crypto OHLCV **+ real order-flow** fetcher (CCData spot API).

The equity path (`MarketDataFetcher`, yfinance) gives only daily *total* volume. This
gives a single venue's **real aggressor-side buy/sell volume** (`VOLUME_BUY` /
`VOLUME_SELL`) — complete for that order book — so experiments can use *real* order
flow instead of the synthetic close-location-value proxy. For a continuously-listed
major it is also a **survivorship-light** universe (no delisting confound for the
names you pick). Keyless (CCData free tier), paginated over the 100-rows/call limit.

    from vpts.data.crypto import fetch_crypto_ohlcv
    df = fetch_crypto_ohlcv("BTC-USDT", market="binance", limit=2000)   # OHLCV + vbuy/vsell

Honest scope: single-venue volume (no consolidated crypto tape exists); pick the venue
you would actually trade. Free tier is rate-limited — be polite (the default sleep is).
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from vpts.data.fetcher import DataFetchError

_FREQ = {"days": "days", "hours": "hours", "minutes": "minutes"}
_BASE = "https://data-api.cryptocompare.com/spot/v1/historical/{freq}"


def _parse_bars(rows: list[dict]) -> pd.DataFrame:
    """CCData spot bars → OHLCV + real buy/sell volume DataFrame (pure, offline-testable).

    Keeps the standard OHLCV contract (so the bars drop straight into the existing
    profile/structure/backtest pipeline) plus ``vbuy``/``vsell`` (base-asset aggressor
    volume). De-duplicated and sorted by time; a DatetimeIndex from the unix timestamp.
    """
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "vbuy", "vsell"])
    df = pd.DataFrame(
        [{"Open": float(b["OPEN"]), "High": float(b["HIGH"]), "Low": float(b["LOW"]),
          "Close": float(b["CLOSE"]), "Volume": float(b["VOLUME"]),
          "vbuy": float(b["VOLUME_BUY"]), "vsell": float(b["VOLUME_SELL"])} for b in rows],
        index=pd.to_datetime([int(b["TIMESTAMP"]) for b in rows], unit="s"),
    )
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_crypto_ohlcv(
    instrument: str,
    *,
    market: str = "binance",
    limit: int = 2000,
    frequency: str = "days",
    to_ts: Optional[int] = None,
    timeout: float = 30.0,
    sleep: float = 0.15,
    session=None,
) -> pd.DataFrame:
    """Fetch up to ``limit`` bars of OHLCV + real buy/sell volume for ``instrument``.

    Paginates the 100-row/call CCData spot endpoint *backwards* via ``to_ts`` until
    ``limit`` bars (or history) is exhausted. ``frequency`` ∈ {days, hours, minutes}.
    ``session`` lets callers pass a pooled ``requests.Session`` (and tests inject a fake).
    Raises :class:`~vpts.data.fetcher.DataFetchError` on a network error or empty result.
    """
    if frequency not in _FREQ:
        raise ValueError(f"frequency must be one of {tuple(_FREQ)}, got {frequency!r}.")
    if limit < 1:
        raise ValueError("limit must be >= 1.")
    import requests  # bundled via yfinance; imported lazily so the module stays import-light
    get = (session or requests).get
    url = _BASE.format(freq=_FREQ[frequency])

    bars: dict[int, dict] = {}
    cursor = to_ts
    for _ in range(max(1, -(-limit // 100)) + 1):       # ceil(limit/100), +1 slack
        params = {"market": market, "instrument": instrument, "limit": 100}
        if cursor:
            params["to_ts"] = cursor
        try:
            data = get(url, params=params, timeout=timeout).json().get("Data") or []
        except Exception as exc:                         # noqa: BLE001 - normalise to DataFetchError
            raise DataFetchError(f"{instrument} @ {market}: {exc}") from exc
        if not data:
            break
        for b in data:
            bars[int(b["TIMESTAMP"])] = b
        oldest = min(int(b["TIMESTAMP"]) for b in data)
        if cursor is not None and oldest >= cursor:      # no backward progress → stop
            break
        cursor = oldest - 1
        if len(bars) >= limit:
            break
        time.sleep(sleep)

    if not bars:
        raise DataFetchError(f"{instrument} @ {market}: no data returned.")
    rows = sorted(bars.values(), key=lambda b: int(b["TIMESTAMP"]))[-limit:]
    return _parse_bars(rows)
