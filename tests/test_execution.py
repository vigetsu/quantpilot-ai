from conftest import FakeAPIError, make_order

from quantpilot.execution.executor import OrderExecutor, make_client_order_id
from quantpilot.risk.engine import RiskDecision


def ex(broker, db):
    return OrderExecutor(broker, db, sleep=lambda s: None)


def test_unapproved_never_submits(broker, fake_client, db):
    r = ex(broker, db).execute(RiskDecision(False, 10), "AAPL", "buy", "qp-AAPL-buy-1", 100, "x")
    assert r["state"] == "blocked" and fake_client.submitted == []


def test_non_decision_object_rejected(broker, fake_client, db):
    # e.g. an LLM response dict cannot masquerade as a risk decision
    r = ex(broker, db).execute({"approved": True, "qty": 10}, "AAPL", "buy", "qp-AAPL-buy-1", 100, "x")
    assert r["state"] == "blocked" and fake_client.submitted == []


def test_submit_and_idempotency(broker, fake_client, db):
    e = ex(broker, db)
    coid = make_client_order_id("AAPL", "buy", "2026-01-05")
    r1 = e.execute(RiskDecision(True, 10, 95.0), "AAPL", "buy", coid, 100, "trend")
    assert r1.submitted and len(fake_client.submitted) == 1
    assert fake_client.submitted[0].time_in_force.value == "day"
    r2 = e.execute(RiskDecision(True, 10, 95.0), "AAPL", "buy", coid, 100, "trend")
    assert r2["state"] == "duplicate" and len(fake_client.submitted) == 1
    o = db.orders()[0]
    assert o["origin"] == "quantpilot" and o["reason"] == "trend" and o["planned_stop"] == 95.0


def test_existing_remote_order_not_resubmitted(broker, fake_client, db):
    fake_client.orders["qp-AAPL-buy-20260105"] = make_order(client_order_id="qp-AAPL-buy-20260105")
    r = ex(broker, db).execute(RiskDecision(True, 10, 95.0), "AAPL", "buy", "qp-AAPL-buy-20260105", 100, "x")
    assert r["state"] == "duplicate" and fake_client.submitted == []


def test_rejected_order_recorded_not_retried(broker, fake_client, db):
    fake_client.submit_error = FakeAPIError("insufficient buying power", 403)
    e = ex(broker, db)
    r = e.execute(RiskDecision(True, 10, 95.0), "AAPL", "buy", "qp-AAPL-buy-x", 100, "x")
    assert r["state"] == "rejected"
    assert db.get_intent("qp-AAPL-buy-x")["state"] == "rejected"
    fake_client.submit_error = None
    r2 = e.execute(RiskDecision(True, 10, 95.0), "AAPL", "buy", "qp-AAPL-buy-x", 100, "x")
    assert r2["state"] == "duplicate" and len(fake_client.submitted) == 1


def test_timeout_is_looked_up_not_resubmitted(broker, fake_client, db):
    fake_client.submit_error = TimeoutError("read timeout")
    r = ex(broker, db).execute(RiskDecision(True, 5, 95.0), "AAPL", "buy", "qp-AAPL-buy-t", 100, "x")
    assert r["state"] == "unknown"
    assert len(fake_client.submitted) == 1
    assert db.get_intent("qp-AAPL-buy-t")["state"] == "unknown"


def test_timeout_but_order_exists(broker, fake_client, db):
    def submit(order_data):
        fake_client.orders[order_data.client_order_id] = make_order(id="late", client_order_id=order_data.client_order_id)
        raise ConnectionError("dropped")
    fake_client.submit_order = submit
    r = ex(broker, db).execute(RiskDecision(True, 5, 95.0), "AAPL", "buy", "qp-AAPL-buy-c", 100, "x")
    assert r["state"] == "submitted" and db.get_intent("qp-AAPL-buy-c")["alpaca_order_id"] == "late"


def test_rate_limit_on_submit_is_unknown(broker, fake_client, db):
    fake_client.submit_error = FakeAPIError("too many requests", 429)
    r = ex(broker, db).execute(RiskDecision(True, 5, 95.0), "AAPL", "buy", "qp-AAPL-buy-r", 100, "x")
    assert r["state"] == "unknown" and len(fake_client.submitted) == 1


def test_protective_stop_placed_once(broker, fake_client, db):
    e = ex(broker, db)
    e.execute(RiskDecision(True, 10, 95.0), "AAPL", "buy", "qp-AAPL-buy-20260105", 100, "x")
    db.upsert_order({"id": "id-1", "client_order_id": "qp-AAPL-buy-20260105", "status": "filled",
                     "filled_qty": 10, "filled_avg_price": 100.2})
    pos = [{"symbol": "AAPL", "qty": 10}]
    acts = e.ensure_protective_stops(pos, [])
    assert len(acts) == 1
    stop_req = fake_client.submitted[-1]
    assert stop_req.stop_price == 95.0 and stop_req.side.value == "sell" and stop_req.time_in_force.value == "gtc"
    open_orders = [{"symbol": "AAPL", "side": "sell", "order_type": "stop"}]
    assert e.ensure_protective_stops(pos, open_orders) == []
    assert len(fake_client.submitted) == 2


def test_no_stop_for_external_positions(broker, fake_client, db):
    assert ex(broker, db).ensure_protective_stops([{"symbol": "TSLA", "qty": 3}], []) == []
    assert fake_client.submitted == []


def test_emergency_stop(broker, fake_client, db):
    db.set_state("auto_trading", True)
    msg = ex(broker, db).emergency_stop()
    assert db.flag("emergency_stop") and not db.flag("auto_trading")
    assert "ACİL" in msg
