"""Pre-market / after-hours trading: session phases, limit orders, stale prices, software stops."""
from datetime import date, datetime, time as dtime, timedelta, timezone
from types import SimpleNamespace

import pytest

import quantpilot.engine as eng_mod
from conftest import make_order, synthetic_bars
from quantpilot.data.universe import validate_watchlist
from quantpilot.engine import TradingEngine
from quantpilot.scheduler.calendar import NY, get_session, market_status_text

DAY = date(2026, 6, 1)  # a Monday


def _clock(fake_client, now, close=dtime(16, 0)):
    is_open = datetime.combine(DAY, dtime(9, 30), tzinfo=NY) <= now < datetime.combine(DAY, close, tzinfo=NY)
    fake_client.get_clock = lambda: SimpleNamespace(timestamp=now, is_open=is_open, next_open=now, next_close=now)
    fake_client.get_calendar = lambda filters=None: [SimpleNamespace(date=DAY, open=dtime(9, 30), close=close)]


@pytest.mark.parametrize("hhmm,phase", [((3, 59), "closed"), ((4, 0), "pre"), ((9, 29), "pre"), ((9, 30), "regular"),
                                        ((15, 59), "regular"), ((16, 0), "post"), ((19, 59), "post"), ((20, 0), "closed")])
def test_session_phases(broker, fake_client, hhmm, phase):
    now = datetime.combine(DAY, dtime(*hhmm), tzinfo=NY)
    _clock(fake_client, now)
    assert get_session(broker, now).phase == phase


def test_early_close_day_after_hours_end_at_17(broker, fake_client):
    for hhmm, phase in (((13, 30), "post"), ((16, 59), "post"), ((17, 0), "closed")):
        now = datetime.combine(DAY, dtime(*hhmm), tzinfo=NY)
        _clock(fake_client, now, close=dtime(13, 0))
        assert get_session(broker, now).phase == phase


def test_status_text_shows_extended_phases(broker, fake_client):
    now = datetime.combine(DAY, dtime(8, 0), tzinfo=NY)
    _clock(fake_client, now)
    assert market_status_text(get_session(broker, now)).startswith("SEANS ÖNCESİ - normal seansa 1 sa 30 dk")


def _setup(settings, broker, fake_client, db, monkeypatch, now, trade_age_min=1.0, params=None):
    _clock(fake_client, now)
    monkeypatch.setattr(eng_mod, "get_session", lambda brk: get_session(brk, now))
    fake_client.get_asset = lambda sym: SimpleNamespace(symbol=sym, exchange="NASDAQ", status="active", tradable=True,
                                                        asset_class="us_equity", fractionable=True, name=sym)
    for w in list(db.watchlist()):
        if w["symbol"] != "AAPL":
            db.remove_symbol(w["symbol"])
    validate_watchlist(broker, db)
    db.set_state("bot_running", True)
    db.set_state("auto_trading", True)
    db.set_params({"use_ml": False, "rsi_entry_min": 0, "rsi_entry_max": 100, **(params or {})})
    bars = {"AAPL": synthetic_bars(150, seed=7, drift=0.01, start="2025-12-01")}
    last = float(bars["AAPL"]["close"].iloc[-1])
    ts = now.astimezone(timezone.utc) - timedelta(minutes=trade_age_min)
    md = SimpleNamespace(daily_bars=lambda syms, days=500: bars, latest_prices=lambda syms: {"AAPL": last},
                         latest_trades=lambda syms: {"AAPL": (last, ts)}, last_feed_used="sip")
    eng = TradingEngine(settings, db, broker, md, None)
    monkeypatch.setattr(eng, "gate", lambda p: (True, []))
    return eng, last


def test_premarket_buy_is_extended_hours_limit_order(settings, broker, fake_client, db, monkeypatch):
    now = datetime.combine(DAY, dtime(7, 0), tzinfo=NY)
    eng, last = _setup(settings, broker, fake_client, db, monkeypatch, now)
    res = eng.run_cycle()
    assert res["phase"] == "pre" and res["orders"][0]["state"] == "submitted"
    req = fake_client.submitted[0]
    assert req.extended_hours is True and req.type.value == "limit" and req.time_in_force.value == "day"
    assert req.limit_price == pytest.approx(round(last * 1.003, 2))
    assert db.get_state("market_phase") == "pre"


def test_extended_hours_can_be_disabled(settings, broker, fake_client, db, monkeypatch):
    now = datetime.combine(DAY, dtime(17, 0), tzinfo=NY)
    eng, _ = _setup(settings, broker, fake_client, db, monkeypatch, now, params={"extended_hours_trading": False})
    assert eng.run_cycle() == {"skipped": "market_closed", "phase": "post"}
    assert fake_client.submitted == []


def test_stale_price_blocks_extended_order(settings, broker, fake_client, db, monkeypatch):
    now = datetime.combine(DAY, dtime(5, 0), tzinfo=NY)
    eng, _ = _setup(settings, broker, fake_client, db, monkeypatch, now, trade_age_min=600)
    res = eng.run_cycle()
    assert res["orders"] == [] and fake_client.submitted == []
    sig = db.recent_signals(limit=1)[0]
    assert "çok eski" in sig["risk_notes"]


def test_regular_session_still_uses_market_orders(settings, broker, fake_client, db, monkeypatch):
    now = datetime.combine(DAY, dtime(11, 0), tzinfo=NY)
    eng, _ = _setup(settings, broker, fake_client, db, monkeypatch, now)
    eng.run_cycle()
    assert fake_client.submitted[0].type.value == "market" and not fake_client.submitted[0].extended_hours


def test_after_hours_software_stop_sells_with_limit(settings, broker, fake_client, db, monkeypatch):
    now = datetime.combine(DAY, dtime(17, 0), tzinfo=NY)
    eng, last = _setup(settings, broker, fake_client, db, monkeypatch, now,
                       params={"exit_on_ema_cross": False, "exit_rsi_max": 0.0, "exit_chandelier": False})
    # an existing QuantPilot position whose planned stop is ABOVE the current price, plus its resting GTC stop
    buy = make_order(id="b1", client_order_id="qp-AAPL-buy-20260520", side="buy", qty="10", filled_qty="10",
                     status="filled", filled_avg_price=str(last * 1.2), submitted_at="2026-05-21T14:00:00+00:00")
    stop = make_order(id="s1", client_order_id="qp-AAPL-buy-20260520-stop-1", side="sell", qty="10",
                      order_type="stop", type="stop", status="new", stop_price=str(last * 1.05),
                      submitted_at="2026-05-21T14:01:00+00:00", time_in_force="gtc")
    fake_client.orders = {"qp-AAPL-buy-20260520": buy, "qp-AAPL-buy-20260520-stop-1": stop}
    fake_client.positions = [SimpleNamespace(symbol="AAPL", qty="10", side="long", avg_entry_price=str(last * 1.2),
                                             current_price=str(last), market_value=str(last * 10),
                                             unrealized_pl="-100", unrealized_plpc="-0.1",
                                             unrealized_intraday_pl="0", asset_class="us_equity")]
    db.create_intent(client_order_id="qp-AAPL-buy-20260520", symbol="AAPL", side="buy", qty=10,
                     requested_price=last * 1.2, planned_stop=last * 1.05, reason="test")
    def get_orders(filter=None):  # like Alpaca: status=open returns only live, non-canceled orders
        want_open = filter is not None and str(getattr(filter.status, "value", filter.status)) == "open"
        return [o for o in fake_client.orders.values()
                if o.id not in fake_client.canceled and (not want_open or o.status in ("new", "accepted"))]
    fake_client.get_orders = get_orders
    res = eng.run_cycle()
    assert "s1" in fake_client.canceled
    sells = [r for r in fake_client.submitted if r.side.value == "sell"]
    assert len(sells) == 1 and sells[0].type.value == "limit" and sells[0].extended_hours is True
    assert sells[0].limit_price == pytest.approx(round(last * 0.997, 2))
    assert any("yazılımsal stop" in str(o) for o in res["orders"])
