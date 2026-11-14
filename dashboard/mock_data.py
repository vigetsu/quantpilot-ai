"""Clearly labeled DEMO / MOCK data used ONLY when Alpaca API credentials are not yet configured.

NEVER written into the real SQLite tables, and NEVER presented as real paper fills.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def demo_bars(symbols: list[str] | None = None, n: int = 550) -> dict[str, pd.DataFrame]:
    symbols = symbols or ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "QQQ"]
    out: dict[str, pd.DataFrame] = {}
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
    for k, s in enumerate(symbols):
        rng = np.random.default_rng(100 + k)
        drift = 0.0005 if s != "TSLA" else 0.0002
        vol = 0.012 if s == "QQQ" else 0.018
        rets = rng.normal(drift, vol, n)
        close = (150 + k * 25) * np.exp(np.cumsum(rets))
        open_ = close * (1 + rng.normal(0, 0.003, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
        volume = rng.integers(5_000_000, 40_000_000, n).astype(float)
        out[s] = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
    return out


def demo_bundle() -> dict:
    bars = demo_bars()
    dates = bars["QQQ"].index[-120:]
    rng = np.random.default_rng(7)
    eq = 100_000 * np.exp(np.cumsum(rng.normal(0.0004, 0.006, len(dates))))
    qqq = 100_000 * bars["QQQ"]["close"].loc[dates] / bars["QQQ"]["close"].loc[dates[0]]
    positions = [
        {"symbol": "AAPL", "qty": 35, "avg_entry_price": 212.40, "current_price": 221.80,
         "market_value": 35 * 221.80, "unrealized_pl": 35 * (221.80 - 212.40),
         "unrealized_plpc": 221.80 / 212.40 - 1, "unrealized_intraday_pl": 42.0},
        {"symbol": "MSFT", "qty": 20, "avg_entry_price": 418.10, "current_price": 414.50,
         "market_value": 20 * 414.50, "unrealized_pl": 20 * (414.50 - 418.10),
         "unrealized_plpc": 414.50 / 418.10 - 1, "unrealized_intraday_pl": -28.0},
    ]
    orders = [
        {"id": "DEMO-1", "client_order_id": "DEMO-qp-NVDA-buy", "symbol": "NVDA", "side": "buy", "qty": 30,
         "filled_qty": 30, "order_type": "market", "status": "filled", "submitted_at": str(dates[-25].date()),
         "filled_at": str(dates[-25].date()), "requested_price": 124.0, "filled_avg_price": 124.12,
         "reason": "[DEMO] EMA20 > EMA50, RSI 56.2, ML=0.61", "origin": "demo"},
        {"id": "DEMO-2", "client_order_id": "DEMO-qp-NVDA-sell", "symbol": "NVDA", "side": "sell", "qty": 30,
         "filled_qty": 30, "order_type": "market", "status": "filled", "submitted_at": str(dates[-10].date()),
         "filled_at": str(dates[-10].date()), "requested_price": 131.5, "filled_avg_price": 131.40,
         "reason": "[DEMO] Trend dönüşü EMA20 < EMA50", "origin": "demo"},
        {"id": "DEMO-3", "client_order_id": "DEMO-qp-AAPL-buy", "symbol": "AAPL", "side": "buy", "qty": 35,
         "filled_qty": 35, "order_type": "market", "status": "filled", "submitted_at": str(dates[-8].date()),
         "filled_at": str(dates[-8].date()), "requested_price": 212.3, "filled_avg_price": 212.40,
         "reason": "[DEMO] EMA20 > EMA50, RSI 52.0, ML=0.58", "origin": "demo"},
        {"id": "DEMO-4", "client_order_id": "DEMO-qp-MSFT-buy", "symbol": "MSFT", "side": "buy", "qty": 20,
         "filled_qty": 20, "order_type": "market", "status": "filled", "submitted_at": str(dates[-4].date()),
         "filled_at": str(dates[-4].date()), "requested_price": 418.0, "filled_avg_price": 418.10,
         "reason": "[DEMO] EMA20 > EMA50, RSI 49.4, ML=0.57", "origin": "demo"},
        {"id": "DEMO-5", "client_order_id": "DEMO-qp-AAPL-stop", "symbol": "AAPL", "side": "sell", "qty": 35,
         "filled_qty": 0, "order_type": "stop", "status": "new", "submitted_at": str(dates[-8].date()),
         "filled_at": None, "requested_price": 204.10, "filled_avg_price": None,
         "reason": "[DEMO] Koruyucu stop (ATR) — gerçekleşmemiş", "origin": "demo"},
    ]
    mv = sum(p["market_value"] for p in positions)
    equity = float(eq[-1])
    account = {"equity": equity, "cash": equity - mv, "buying_power": equity - mv,
               "last_equity": float(eq[-2]), "long_market_value": mv, "status": "DEMO (Anahtar Yok)"}
    return {"account": account, "positions": positions, "orders": orders,
            "equity_series": pd.Series(eq, index=dates), "qqq_series": pd.Series(qqq.values, index=dates),
            "bars": bars}
