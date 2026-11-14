from types import SimpleNamespace

from conftest import FakeAPIError, make_order

from quantpilot.analytics.performance import fifo_realized, trade_stats
from quantpilot.broker.sync import Synchronizer


def test_account_and_positions_sync(broker, fake_client, db):
    fake_client.positions = [SimpleNamespace(symbol="AAPL", qty="10", side="long", avg_entry_price="150",
                                             current_price="160", market_value="1600", unrealized_pl="100",
                                             unrealized_plpc="0.0667", unrealized_intraday_pl="5", asset_class="us_equity")]
    fake_client.orders = {
        "a": make_order(id="o1", client_order_id="a", status="filled", filled_qty="10", filled_avg_price="150",
                        filled_at="2026-01-02T15:01:00+00:00"),
        "b": make_order(id="o2", client_order_id="b", status="partially_filled", qty="10", filled_qty="4",
                        filled_avg_price="151"),
        "c": make_order(id="o3", client_order_id="c", status="rejected"),
    }
    res = Synchronizer(broker, db).sync_all()
    assert res["account"]["equity"] == 100000.0
    assert db.positions()[0]["symbol"] == "AAPL"
    statuses = {o["id"]: o["status"] for o in db.orders()}
    assert statuses == {"o1": "filled", "o2": "partially_filled", "o3": "rejected"}
    assert db.get_state("last_sync")


def test_status_update_overwrites(broker, fake_client, db):
    fake_client.orders = {"a": make_order(id="o1", client_order_id="a", status="accepted")}
    s = Synchronizer(broker, db)
    s.sync_all()
    fake_client.orders["a"] = make_order(id="o1", client_order_id="a", status="filled", filled_qty="10",
                                         filled_avg_price="100")
    s.sync_all()
    o = db.orders()[0]
    assert o["status"] == "filled" and o["filled_qty"] == 10


def test_fifo_only_counts_fills():
    orders = [
        {"id": "1", "symbol": "X", "side": "buy", "filled_qty": 10, "filled_avg_price": 100, "filled_at": "1"},
        {"id": "2", "symbol": "X", "side": "buy", "filled_qty": 0, "filled_avg_price": None, "filled_at": None},  # unfilled
        {"id": "3", "symbol": "X", "side": "sell", "filled_qty": 4, "filled_avg_price": 110, "filled_at": "2"},  # partial
        {"id": "4", "symbol": "X", "side": "sell", "filled_qty": 6, "filled_avg_price": 95, "filled_at": "3"},
    ]
    trades, per = fifo_realized(orders)
    assert per["3"] == 40 and per["4"] == -30
    st = trade_stats(trades)
    assert st["n_trades"] == 2 and st["win_rate"] == 0.5
    assert abs(st["profit_factor"] - 40 / 30) < 1e-9


def test_reconcile_unknown_intent_found(broker, fake_client, db):
    db.create_intent(client_order_id="qp-AAPL-buy-20260101", symbol="AAPL", side="buy", qty=10)
    db.update_intent("qp-AAPL-buy-20260101", "unknown")
    fake_client.orders["qp-AAPL-buy-20260101"] = make_order(id="z1")
    Synchronizer(broker, db).reconcile_intents()
    assert db.get_intent("qp-AAPL-buy-20260101")["state"] == "submitted"


def test_reconcile_unknown_intent_missing(broker, db):
    db.create_intent(client_order_id="qp-MSFT-buy-20260101", symbol="MSFT", side="buy", qty=1)
    Synchronizer(broker, db).reconcile_intents(unknown_grace_seconds=0)
    assert db.get_intent("qp-MSFT-buy-20260101")["state"] == "not_found"


def test_read_retry_on_429(broker, fake_client):
    calls = {"n": 0}
    acct = fake_client.account

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeAPIError("rate limit", 429)
        return acct
    fake_client.get_account = flaky
    assert broker.get_account()["equity"] == 100000.0
    assert calls["n"] == 3
