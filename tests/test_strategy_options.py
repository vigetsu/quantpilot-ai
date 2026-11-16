"""Tests for the optional regime / momentum / chandelier rules (shared by live engine and backtest)."""
import numpy as np
import pandas as pd

from conftest import synthetic_bars
from quantpilot.backtest.engine import run_backtest
from quantpilot.db import DEFAULT_PARAMS, Database
from quantpilot.strategy.indicators import compute_indicators, market_regime
from quantpilot.strategy.rules import evaluate, rank_score

P = dict(DEFAULT_PARAMS)


def row(**kw):
    base = dict(ema_fast=105, ema_slow=100, rsi=55, atr=2.0, macd_hist=0.1, vol_ratio=1.0, close=110.0,
                chandelier=100.0, mom_rank=0.10)
    base.update(kw)
    return base


def test_ml_is_off_by_default():
    assert DEFAULT_PARAMS["use_ml"] is False
    assert evaluate(row(), P, False, None).action == "BUY"


def test_regime_filter_blocks_entries_unless_risk_on():
    p = dict(P, use_regime_filter=True)
    assert evaluate(row(regime_ok=True), p, False, None).action == "BUY"
    assert evaluate(row(regime_ok=False), p, False, None).action == "HOLD"
    assert evaluate(row(regime_ok=None), p, False, None).action == "HOLD"          # unknown => fail safe
    assert evaluate(row(regime_ok=float("nan")), p, False, None).action == "HOLD"
    # exits are never blocked by the regime filter
    assert evaluate(row(regime_ok=False, ema_fast=95), p, True, None).action == "SELL"


def test_momentum_filter():
    p = dict(P, momentum_filter=True, min_momentum=0.0)
    assert evaluate(row(mom_rank=0.05), p, False, None).action == "BUY"
    assert evaluate(row(mom_rank=-0.05), p, False, None).action == "HOLD"
    assert evaluate(row(mom_rank=float("nan")), p, False, None).action == "HOLD"


def test_chandelier_exit():
    p = dict(P, exit_chandelier=True)
    assert evaluate(row(close=99.0, chandelier=100.0), p, True, None).action == "SELL"
    assert evaluate(row(close=101.0, chandelier=100.0), p, True, None).action == "HOLD"
    # disabled => same bar is held
    assert evaluate(row(close=99.0, chandelier=100.0), P, True, None).action == "HOLD"


def test_ema_cross_exit_can_be_disabled():
    p = dict(P, exit_on_ema_cross=False)
    assert evaluate(row(ema_fast=95), p, True, None).action == "HOLD"


def test_rank_score():
    assert rank_score(0.7, row(mom_rank=0.01), P) == 0.7
    assert rank_score(None, row(mom_rank=0.3), dict(P, rank_by="momentum")) == 0.3
    assert rank_score(None, row(mom_rank=float("nan")), dict(P, rank_by="momentum")) < -1e8


def test_momentum_indicator_is_causal():
    df = synthetic_bars(400, seed=3)
    full = compute_indicators(df, P)
    cut = compute_indicators(df.iloc[:300], P)
    pd.testing.assert_series_equal(full["mom_rank"].iloc[:300], cut["mom_rank"], check_names=False)
    pd.testing.assert_series_equal(full["chandelier"].iloc[:300], cut["chandelier"], check_names=False)
    i = 250
    exp = df["close"].iloc[i - 21] / df["close"].iloc[i - 21 - 126] - 1
    assert np.isclose(full["mom_rank"].iloc[i], exp)


def test_market_regime():
    up = pd.DataFrame({"close": np.linspace(100, 200, 260)}, index=pd.bdate_range("2024-01-01", periods=260))
    reg = market_regime(up, 200)
    assert reg.iloc[:199].isna().all() and bool(reg.iloc[-1]) is True
    down = pd.DataFrame({"close": np.linspace(200, 100, 260)}, index=up.index)
    assert bool(market_regime(down, 200).iloc[-1]) is False


def test_backtest_regime_filter_blocks_all_entries_in_bear_market():
    bars = {"AAA": synthetic_bars(500, seed=1, drift=0.002)}
    bench = pd.DataFrame({"close": np.linspace(300, 100, 500)}, index=bars["AAA"].index)  # QQQ always falling
    p = dict(P, use_regime_filter=True)
    res = run_backtest(bars, p, 100_000, benchmark=bench)
    assert res.trades == []
    res2 = run_backtest(bars, P, 100_000, benchmark=bench)
    assert len(res2.trades) > 0


def test_set_params_validates_rank_by(tmp_path):
    db = Database(tmp_path / "x.db")
    assert db.set_params({"rank_by": "momentum"})["rank_by"] == "momentum"
    assert db.set_params({"rank_by": "bogus"})["rank_by"] == "momentum"


def test_live_engine_applies_regime_filter(settings, broker, fake_client, db, monkeypatch):
    from datetime import datetime, time as dtime
    from types import SimpleNamespace
    import quantpilot.engine as eng_mod
    from quantpilot.data.universe import validate_watchlist
    from quantpilot.engine import TradingEngine
    from quantpilot.scheduler.calendar import NY, get_session

    now = datetime(2026, 6, 1, 11, 0, tzinfo=NY)
    fake_client.get_clock = lambda: SimpleNamespace(timestamp=now, is_open=True, next_open=now, next_close=now)
    fake_client.get_calendar = lambda filters=None: [SimpleNamespace(date=now.date(), open=dtime(9, 30), close=dtime(16, 0))]
    monkeypatch.setattr(eng_mod, "get_session", lambda brk: get_session(brk, now))
    fake_client.get_asset = lambda sym: SimpleNamespace(symbol=sym, exchange="NASDAQ", status="active", tradable=True,
                                                        asset_class="us_equity", fractionable=True, name=sym)
    for w in list(db.watchlist()):
        if w["symbol"] != "AAPL":
            db.remove_symbol(w["symbol"])
    validate_watchlist(broker, db)
    db.set_state("bot_running", True)
    db.set_state("auto_trading", True)
    aapl = synthetic_bars(300, seed=7, drift=0.01)
    falling_qqq = pd.DataFrame({"open": np.linspace(300, 100, 300), "high": np.linspace(301, 101, 300),
                                "low": np.linspace(299, 99, 300), "close": np.linspace(300, 100, 300),
                                "volume": 1e6}, index=aapl.index)
    fetched = []

    def daily_bars(syms, days=500):
        fetched.append(list(syms))
        return {"AAPL": aapl, "QQQ": falling_qqq}
    md = SimpleNamespace(daily_bars=daily_bars, latest_prices=lambda syms: {"AAPL": float(aapl["close"].iloc[-1])},
                         last_feed_used="sip")
    db.set_params({"use_ml": False, "rsi_entry_min": 0, "rsi_entry_max": 100, "use_regime_filter": True})
    eng = TradingEngine(settings, db, broker, md, None)
    monkeypatch.setattr(eng, "gate", lambda params: (True, []))
    res = eng.run_cycle()
    assert "QQQ" in fetched[-1]
    assert db.get_state("market_regime") == "risk_off"
    assert res["orders"] == [] and fake_client.submitted == []
    assert [s["symbol"] for s in res["signals"]] == ["AAPL"]  # QQQ itself is not traded
    db.set_params({"use_regime_filter": False})
    assert eng.run_cycle()["orders"][0]["side"] == "BUY"
