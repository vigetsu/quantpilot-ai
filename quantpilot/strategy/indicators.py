"""Technical indicators. All are causal: value at t uses only data up to and including t."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out = out.where(avg_loss != 0, 100.0)
    return out.where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
                     axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def historical_volatility(close: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    return np.log(close).diff().rolling(window).std() * np.sqrt(periods_per_year)


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def compute_indicators(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    p = params or {}
    ef, es = int(p.get("ema_fast", 20)), int(p.get("ema_slow", 50))
    out = df.copy()
    c = out["close"]
    out["ema_fast"] = ema(c, ef)
    out["ema_slow"] = ema(c, es)
    out["rsi"] = rsi(c, int(p.get("rsi_period", 14)))
    out = out.join(macd(c))
    out["atr"] = atr(out, int(p.get("atr_period", 14)))
    out["hv20"] = historical_volatility(c, 20)
    out["vol_sma20"] = out["volume"].rolling(20).mean()
    out["vol_ratio"] = out["volume"] / out["vol_sma20"]
    out["obv"] = obv(out)
    out["obv_slope"] = out["obv"].diff(10) / out["vol_sma20"].replace(0, np.nan) / 10
    out["mom_10"] = c.pct_change(10)
    out["mom_20"] = c.pct_change(20)
    out["ret_1"] = c.pct_change(1)
    out["ret_5"] = c.pct_change(5)
    out["trend_strength"] = (out["ema_fast"] - out["ema_slow"]) / out["atr"]
    out["dist_ema_fast"] = c / out["ema_fast"] - 1
    out["high_52w_dist"] = c / c.rolling(252, min_periods=60).max() - 1
    return out
