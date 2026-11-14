from datetime import datetime, timedelta, timezone

from quantpilot.db import DEFAULT_PARAMS, Database
from quantpilot.risk.engine import PortfolioState, RiskContext, check_entry, check_exit, position_size
from quantpilot.risk.selftest import run_self_test

P = dict(DEFAULT_PARAMS)


def ctx(**kw):
    b = dict(auto_trading=True, emergency_stop=False, bot_running=True, market_open=True, in_entry_window=True,
             asset_ok=True, safety_gate_ok=True)
    b.update(kw)
    return RiskContext(**b)


def st(**kw):
    b = dict(equity=100_000.0, cash=100_000.0, last_equity=100_000.0, positions={}, open_orders=[])
    b.update(kw)
    return PortfolioState(**b)


def test_risk_per_trade_binding():
    # stop dist = 2*5 = 10 => risk budget 1000 => 100 shares; 10% cap at $50 => 200 shares
    qty, _ = position_size(100_000, 100_000, 50.0, 5.0, 0, 0, P)
    assert qty == 100


def test_position_cap_binding():
    # stop dist tiny => risk allows many; 10% cap at $100 => 100 shares
    qty, _ = position_size(100_000, 100_000, 100.0, 0.1, 0, 0, P)
    assert qty == 100


def test_exposure_cap():
    qty, _ = position_size(100_000, 100_000, 100.0, 0.1, 45_000, 0, P)
    assert qty == 50


def test_no_leverage_cash_cap():
    qty, _ = position_size(100_000, 1_000, 100.0, 0.1, 0, 0, P)
    assert qty == 9  # 1000*0.98/100


def test_hard_limits_cannot_be_loosened():
    loose = dict(P, max_position_pct=0.5, max_exposure_pct=1.0, max_risk_per_trade_pct=0.1)
    qty, _ = position_size(100_000, 100_000, 100.0, 0.1, 0, 0, loose)
    assert qty == 100


def test_db_clamps_limits(tmp_path):
    db = Database(tmp_path / "x.db")
    p = db.set_params({"max_position_pct": 0.9, "max_daily_drawdown_pct": 0.5, "rsi_entry_min": 40})
    assert p["max_position_pct"] == 0.10 and p["max_daily_drawdown_pct"] == 0.03 and p["rsi_entry_min"] == 40


def test_entry_approved_and_stop():
    d = check_entry("AAPL", 100.0, 2.0, st(), ctx(), P)
    assert d.approved and d.stop_price == 96.0
    assert d.qty * (100 - 96) <= 1000


def test_daily_drawdown_blocks():
    assert not check_entry("AAPL", 100, 2, st(equity=96_999, cash=96_999), ctx(), P).approved
    assert check_entry("AAPL", 100, 2, st(equity=97_500, cash=97_500), ctx(), P).approved


def test_duplicate_position_blocks():
    s = st(positions={"AAPL": {"qty": 10, "market_value": 1000}})
    assert not check_entry("AAPL", 100, 2, s, ctx(), P).approved


def test_pending_order_blocks():
    s = st(open_orders=[{"symbol": "AAPL", "side": "buy", "qty": 5}])
    assert not check_entry("AAPL", 100, 2, s, ctx(), P).approved


def test_cooldown_blocks():
    now = datetime.now(timezone.utc)
    s = st(last_trade_time={"AAPL": now - timedelta(minutes=30)})
    assert not check_entry("AAPL", 100, 2, s, ctx(now=now), P).approved
    s2 = st(last_trade_time={"AAPL": now - timedelta(days=2)})
    assert check_entry("AAPL", 100, 2, s2, ctx(now=now), P).approved


def test_all_gates():
    for kw in ({"emergency_stop": True}, {"bot_running": False}, {"auto_trading": False}, {"market_open": False},
               {"in_entry_window": False}, {"asset_ok": False}, {"safety_gate_ok": False}):
        assert not check_entry("AAPL", 100, 2, st(), ctx(**kw), P).approved, kw


def test_exit_rules():
    s = st(positions={"AAPL": {"qty": 7.0, "market_value": 700}})
    d = check_exit("AAPL", s, ctx())
    assert d.approved and d.qty == 7
    assert not check_exit("MSFT", s, ctx()).approved  # no short selling
    assert not check_exit("AAPL", s, ctx(emergency_stop=True)).approved
    s2 = st(positions={"AAPL": {"qty": 7.0}}, open_orders=[{"symbol": "AAPL", "side": "sell", "order_type": "market"}])
    assert not check_exit("AAPL", s2, ctx()).approved


def test_selftest_passes():
    ok, msgs = run_self_test()
    assert ok, msgs
