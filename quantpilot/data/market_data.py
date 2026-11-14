"""Alpaca market data access (historical daily bars + latest trades).

Data-feed notes (shown in the dashboard):
- Free Alpaca plans get real-time data from the IEX exchange only (~2-3% of US volume),
  so latest prices/volumes can differ from the consolidated (SIP) tape.
- Historical SIP data is available on the free plan only for bars older than 15 minutes.
  We therefore request historical bars from SIP with ``end = now - 16 min`` and fall back to IEX.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from quantpilot.config import Settings

log = logging.getLogger("quantpilot.data")


class MarketData:
    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        if client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            client = StockHistoricalDataClient(settings.api_key.get(), settings.secret_key.get())
        self.client = client
        self.last_feed_used: str | None = None

    def _feed(self, name: str):
        from alpaca.data.enums import DataFeed
        return DataFeed.SIP if name == "sip" else DataFeed.IEX

    def daily_bars(self, symbols: list[str], days: int = 1500, end: datetime | None = None) -> dict[str, pd.DataFrame]:
        """Split/dividend adjusted daily OHLCV, one DataFrame per symbol indexed by NY date."""
        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        end = end or (datetime.now(timezone.utc) - timedelta(minutes=16))
        start = end - timedelta(days=days)
        feeds = [self.settings.historical_feed] + (["iex"] if self.settings.historical_feed != "iex" else [])
        last_exc: Exception | None = None
        for feed in feeds:
            try:
                req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start, end=end,
                                       adjustment=Adjustment.ALL, feed=self._feed(feed))
                df = self.client.get_stock_bars(req).df
                self.last_feed_used = feed
                return split_bars(df)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("Tarihsel veri alınamadı (feed=%s): %s", feed, exc)
        raise RuntimeError(f"Tarihsel veri alınamadı: {last_exc}")

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        from alpaca.data.requests import StockLatestTradeRequest
        req = StockLatestTradeRequest(symbol_or_symbols=symbols, feed=self._feed(self.settings.data_feed))
        res = self.client.get_stock_latest_trade(req)
        return {s: float(t.price) for s, t in res.items()}


def split_bars(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out
    if isinstance(df.index, pd.MultiIndex):
        for sym, g in df.groupby(level=0):
            out[sym] = _normalize(g.droplevel(0))
    else:
        out["?"] = _normalize(df)
    return out


def _normalize(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    idx = pd.to_datetime(g.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York")
    g.index = pd.DatetimeIndex(idx.date, name="date")
    g = g[~g.index.duplicated(keep="last")].sort_index()
    return g[["open", "high", "low", "close", "volume"]].astype(float)


def completed_daily_bars(df: pd.DataFrame, now_ny: datetime, market_open_today: bool) -> pd.DataFrame:
    """Drop today's still-forming daily bar so live signals use only completed sessions
    (identical information set to the backtest: signal at close t, act during session t+1)."""
    if df.empty:
        return df
    today = now_ny.date()
    if market_open_today and df.index[-1].date() >= today:
        return df.iloc[:-1]
    return df
