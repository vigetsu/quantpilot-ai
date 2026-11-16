"""Dashboard 1-share test orders."""
from datetime import date, datetime, time as dtime, timedelta, timezone
from types import SimpleNamespace

from quantpilot.data.universe import validate_watchlist
from quantpilot.execution.manual import test_order as place_test_order
from quantpilot.scheduler.calendar import NY

DAY = date(2026, 6, 1)


def _prep(broker, fake_client, db, hhmm):
    now = datetime.combine(DAY, dtime(*hhmm), tzinfo=NY)
    is_open = dtime(9, 30) <= dtime(*hhmm) < dtime(16, 0)
    fake_client.get_clock = lambda: SimpleNamespace(timestamp=now, is_open=is_open, next_open=now, next_close=now)
    fake_client.get_calendar = lambda filters=None: [SimpleNamespace(date=DAY, open=dtime(9, 30), close=dtime(16, 0))]
    fake_client.get_asset = lambda sym: SimpleNamespace(symbol=sym, exchange="NASDAQ", status="active", tradable=True,
                                                        asset_class="us_equity", fractionable=True, name=sym)
    validate_watchlist(broker, db)
    md = SimpleNamespace(latest_trades=lambda syms: {"AAPL": (200.0, now.astimezone(timezone.utc) - timedelta(minutes=3))})
    return now, md


def test_regular_session_buy_is_market_order(broker, fake_client, db):
    now, md = _prep(broker, fake_client, db, (11, 0))
    r = place_test_order(broker, md, db, "AAPL", "buy", now=now)
    assert r["ok"], r
    req = fake_client.submitted[0]
    assert req.type.value == "market" and req.qty == 1 and req.client_order_id.startswith("qptest-AAPL-buy-")
    # stored as a non-QuantPilot order: no bot cooldown / protective stop
    assert db.orders()[0]["origin"] != "quantpilot"


def test_premarket_buy_is_extended_limit(broker, fake_client, db):
    now, md = _prep(broker, fake_client, db, (7, 0))
    r = place_test_order(broker, md, db, "AAPL", "buy", now=now)
    assert r["ok"], r
    req = fake_client.submitted[0]
    assert req.type.value == "limit" and req.extended_hours is True and req.limit_price == 202.0
    assert "seans öncesi" in r["message"]


def test_closed_market_refused(broker, fake_client, db):
    now, md = _prep(broker, fake_client, db, (22, 0))
    r = place_test_order(broker, md, db, "AAPL", "buy", now=now)
    assert not r["ok"] and "kapalı" in r["message"] and fake_client.submitted == []


def test_sell_requires_position(broker, fake_client, db):
    now, md = _prep(broker, fake_client, db, (11, 0))
    r = place_test_order(broker, md, db, "AAPL", "sell", now=now)
    assert not r["ok"] and "Açığa satış" in r["message"] and fake_client.submitted == []
    fake_client.positions = [SimpleNamespace(symbol="AAPL", qty="1", side="long", avg_entry_price="200",
                                             current_price="201", market_value="201", unrealized_pl="1",
                                             unrealized_plpc="0.005", unrealized_intraday_pl="1", asset_class="us_equity")]
    r = place_test_order(broker, md, db, "AAPL", "sell", now=now)
    assert r["ok"] and fake_client.submitted[0].side.value == "sell"


def test_emergency_stop_and_unknown_symbol_refused(broker, fake_client, db):
    now, md = _prep(broker, fake_client, db, (11, 0))
    assert not place_test_order(broker, md, db, "ZZZZ", "buy", now=now)["ok"]
    db.set_state("emergency_stop", True)
    r = place_test_order(broker, md, db, "AAPL", "buy", now=now)
    assert not r["ok"] and "ACİL" in r["message"] and fake_client.submitted == []
