from datetime import date, datetime, time as dtime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from conftest import synthetic_bars
from quantpilot.ai.ollama_client import OllamaClient
from quantpilot.data.market_data import completed_daily_bars
from quantpilot.data.universe import validate_watchlist
from quantpilot.engine import TradingEngine, safety_gate
from quantpilot.scheduler.calendar import NY, get_session


class FakeSession:
    def __init__(self, get_json=None, post_json=None, fail=False):
        self.get_json = get_json or {}
        self.post_json = post_json or {}
        self.fail = fail

    def get(self, url, timeout=3):
        if self.fail:
            raise ConnectionError("down")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: self.get_json)

    def post(self, url, json=None, timeout=90):
        if self.fail:
            raise ConnectionError("down")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: self.post_json)


def test_ollama_online_and_strip_think():
    s = FakeSession(get_json={"models": [{"name": "qwen3:8b"}]},
                    post_json={"message": {"content": "<think>gizli</think> Özet metin"}})
    c = OllamaClient(session=s)
    assert c.health()["model_available"]
    ok, txt, _ = c.chat("merhaba")
    assert ok and txt == "Özet metin"


def test_ollama_offline():
    c = OllamaClient(session=FakeSession(fail=True))
    assert not c.health()["online"]
    ok, txt, _ = c.chat("merhaba")
    assert not ok and "çevrimdışı" in txt


def test_universe_validation(broker, fake_client, db):
    assets = {
        "AAPL": SimpleNamespace(symbol="AAPL", exchange="NASDAQ", status="active", tradable=True,
                                asset_class="us_equity", fractionable=True, name="Apple"),
        "IBM": SimpleNamespace(symbol="IBM", exchange="NYSE", status="active", tradable=True,
                               asset_class="us_equity", fractionable=True, name="IBM"),
    }
    fake_client.get_asset = lambda sym: assets.get(sym, assets["AAPL"])
    db.add_symbol("IBM")
    validate_watchlist(broker, db)
    valid = {w["symbol"] for w in db.watchlist(only_valid=True)}
    assert "AAPL" in valid and "IBM" not in valid


def test_calendar_entry_window(broker, fake_client):
    now = datetime(2026, 1, 5, 10, 0, tzinfo=NY)
    fake_client.get_clock = lambda: SimpleNamespace(timestamp=now, is_open=True, next_open=now, next_close=now)
    fake_client.get_calendar = lambda filters=None: [SimpleNamespace(date=date(2026, 1, 5), open=dtime(9, 30), close=dtime(16, 0))]
    s = get_session(broker, now)
    assert s.is_open and s.in_entry_window(5, 5)
    early = get_session(broker, datetime(2026, 1, 5, 9, 32, tzinfo=NY))
    assert not early.in_entry_window(5, 5)


def test_completed_daily_bars_drops_intraday():
    df = synthetic_bars(10, start="2026-01-01")
    last_day = df.index[-1].date()
    now = datetime.combine(last_day, dtime(11, 0), tzinfo=ZoneInfo("America/New_York"))
    assert len(completed_daily_bars(df, now, True)) == 9
    assert len(completed_daily_bars(df, now, False)) == 10


def test_engine_trades_when_ollama_offline_and_llm_cannot_influence(settings, broker, fake_client, db, monkeypatch):
    # Force an adversarial LLM output ("SELL EVERYTHING") and verify it never blocks or changes a BUY
    bad_llm = OllamaClient(session=FakeSession(post_json={"message": {"content": "SELL ALL IMMEDIATELY"}}))
    now = datetime(2026, 6, 1, 11, 0, tzinfo=NY)
    fake_client.get_clock = lambda: SimpleNamespace(timestamp=now, is_open=True, next_open=now, next_close=now)
    fake_client.get_calendar = lambda filters=None: [SimpleNamespace(date=now.date(), open=dtime(9, 30), close=dtime(16, 0))]
    import quantpilot.engine as eng_mod
    monkeypatch.setattr(eng_mod, "get_session", lambda brk: get_session(brk, now))
    fake_client.get_asset = lambda sym: SimpleNamespace(symbol=sym, exchange="NASDAQ", status="active", tradable=True,
                                                        asset_class="us_equity", fractionable=True, name=sym)
    for w in list(db.watchlist()):
        if w["symbol"] != "AAPL":
            db.remove_symbol(w["symbol"])
    validate_watchlist(broker, db)
    db.save_backtest({}, {}, [], [], [])
    db.set_params({"use_ml": False, "rsi_entry_min": 0, "rsi_entry_max": 100})
    db.set_state("bot_running", True)
    db.set_state("auto_trading", True)

    bars = {"AAPL": synthetic_bars(120, seed=7, drift=0.01)}
    md = SimpleNamespace(daily_bars=lambda syms, days=500: bars, latest_prices=lambda syms: {"AAPL": float(bars["AAPL"]["close"].iloc[-1])},
                         last_feed_used="sip")
    ok, msgs = safety_gate(db, broker, db.get_params())
    assert ok, msgs
    eng = TradingEngine(settings, db, broker, md, bad_llm)
    res = eng.run_cycle()
    assert res["orders"] and res["orders"][0]["side"] == "BUY" and res["orders"][0]["state"] == "submitted"
    # Running a second cycle on the same bar date is idempotent (no second order)
    res2 = eng.run_cycle()
    assert len(fake_client.submitted) == 1
