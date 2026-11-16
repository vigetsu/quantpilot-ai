import numpy as np
import pandas as pd

from conftest import synthetic_bars

from quantpilot.db import DEFAULT_PARAMS
from quantpilot.strategy.indicators import atr, compute_indicators, ema, macd, rsi
from quantpilot.strategy.rules import evaluate


def test_ema_matches_manual():
    s = pd.Series([1, 2, 3, 4, 5, 6], dtype=float)
    e = ema(s, 3)
    alpha = 2 / 4
    manual = [1.0]
    for x in s[1:]:
        manual.append(alpha * x + (1 - alpha) * manual[-1])
    assert np.isnan(e.iloc[0]) and np.isnan(e.iloc[1])
    assert np.allclose(e.iloc[2:], manual[2:])


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert rsi(up, 14).iloc[-1] == 100.0
    down = pd.Series(np.arange(40, 1, -1, dtype=float))
    assert rsi(down, 14).iloc[-1] < 1e-9
    r = rsi(synthetic_bars()["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_wilder_known_value():
    # Alternating +2 / -1 changes with Wilder EWM (alpha=1/14) -> exact steady-state on odd step is 65.0
    deltas = [0.0] + [2.0 if i % 2 == 0 else -1.0 for i in range(200)]
    closes = pd.Series(100 + np.cumsum(deltas))
    val = rsi(closes, 14).iloc[-1]
    assert abs(val - 65.0) < 0.01


def test_atr_constant_range():
    df = pd.DataFrame({"high": [11.0] * 30, "low": [9.0] * 30, "close": [10.0] * 30, "open": [10.0] * 30})
    assert abs(atr(df, 14).iloc[-1] - 2.0) < 1e-9


def test_macd_columns():
    m = macd(synthetic_bars()["close"])
    assert set(m.columns) == {"macd", "macd_signal", "macd_hist"}
    assert np.allclose(m["macd_hist"].dropna(), (m["macd"] - m["macd_signal"]).dropna())


def test_indicators_are_causal():
    df = synthetic_bars(400)
    full = compute_indicators(df)
    cut = compute_indicators(df.iloc[:300])
    cols = ["ema_fast", "ema_slow", "rsi", "macd_hist", "atr", "hv20", "vol_ratio", "mom_20"]
    pd.testing.assert_frame_equal(full[cols].iloc[:300], cut[cols])


P = dict(DEFAULT_PARAMS)


def row(**kw):
    base = dict(ema_fast=105, ema_slow=100, rsi=55, atr=2.0, macd_hist=0.1, vol_ratio=1.0)
    base.update(kw)
    return base


def test_entry_buy():
    assert evaluate(row(), P, False, 0.6).action == "BUY"


def test_entry_blocked_by_rsi():
    assert evaluate(row(rsi=75), P, False, 0.6).action == "HOLD"
    assert evaluate(row(rsi=40), P, False, 0.6).action == "HOLD"


def test_entry_blocked_by_trend():
    assert evaluate(row(ema_fast=99), P, False, 0.6).action == "HOLD"


def test_entry_blocked_by_ml():
    pm = dict(P, use_ml=True)  # ML filter is opt-in (off by default)
    assert evaluate(row(), pm, False, 0.55).action == "HOLD"  # must be strictly > 0.55
    assert evaluate(row(), pm, False, None).action == "HOLD"  # no model => no entry


def test_ml_can_be_disabled():
    assert evaluate(row(), dict(P, use_ml=False), False, None).action == "BUY"


def test_exit_on_trend_reversal():
    assert evaluate(row(ema_fast=99), P, True, 0.6).action == "SELL"


def test_hold_position():
    assert evaluate(row(), P, True, 0.6).action == "HOLD"


def test_conflicts_detected():
    d = evaluate(row(macd_hist=-0.5), P, False, 0.6)
    assert d.conflicts


def test_nan_holds():
    assert evaluate(row(rsi=float("nan")), P, False, 0.9).action == "HOLD"
