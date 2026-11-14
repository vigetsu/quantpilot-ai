"""Automated safety self-test of the risk engine, run before auto paper trading can be enabled."""
from __future__ import annotations

from quantpilot.db import DEFAULT_PARAMS
from quantpilot.risk.engine import PortfolioState, RiskContext, check_entry, check_exit


def _ctx(**kw) -> RiskContext:
    base = dict(auto_trading=True, emergency_stop=False, bot_running=True, market_open=True, in_entry_window=True,
                asset_ok=True, safety_gate_ok=True)
    base.update(kw)
    return RiskContext(**base)


def _state(**kw) -> PortfolioState:
    base = dict(equity=100_000.0, cash=100_000.0, last_equity=100_000.0, positions={}, open_orders=[])
    base.update(kw)
    return PortfolioState(**base)


def run_self_test(params: dict | None = None) -> tuple[bool, list[str]]:
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    results: list[tuple[str, bool]] = []

    d = check_entry("AAPL", 100.0, 2.0, _state(), _ctx(), p)
    results.append(("Normal koşulda giriş onaylanır", d.approved))
    results.append(("Pozisyon %10 özsermayeyi aşmaz", d.qty * 100.0 <= 100_000 * 0.10 + 1e-6))
    results.append(("Planlanan risk %1'i aşmaz", d.qty * (100.0 - (d.stop_price or 0)) <= 1_000 + 1e-6))
    results.append(("Acil durdurma girişi engeller", not check_entry("AAPL", 100, 2, _state(), _ctx(emergency_stop=True), p).approved))
    results.append(("Piyasa kapalıyken giriş yok", not check_entry("AAPL", 100, 2, _state(), _ctx(market_open=False), p).approved))
    results.append(("Otomatik emir kapalıyken giriş yok", not check_entry("AAPL", 100, 2, _state(), _ctx(auto_trading=False), p).approved))
    full = {f"S{i}": {"qty": 100, "market_value": 10_000.0} for i in range(5)}
    results.append(("%50 maruziyet limiti uygulanır", not check_entry("AAPL", 100, 2, _state(positions=full, cash=50_000), _ctx(), p).approved))
    results.append(("%3 günlük düşüşte giriş durur", not check_entry("AAPL", 100, 2, _state(equity=96_900, cash=96_900), _ctx(), p).approved))
    results.append(("Bekleyen emirde çift emir yok", not check_entry("AAPL", 100, 2, _state(open_orders=[{"symbol": "AAPL", "side": "buy", "qty": 10}]), _ctx(), p).approved))
    results.append(("Kaldıraç yok (nakit sınırı)", check_entry("AAPL", 100, 0.5, _state(cash=500.0), _ctx(), p).qty <= 4))
    results.append(("Pozisyon yokken satış yok (açığa satış yok)", not check_exit("AAPL", _state(), _ctx()).approved))
    msgs = [("✅ " if ok else "❌ ") + name for name, ok in results]
    return all(ok for _, ok in results), msgs
