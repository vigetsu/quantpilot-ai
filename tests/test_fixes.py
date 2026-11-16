"""Regression tests for the cloud/dashboard fixes."""
from datetime import datetime, time as dtime
from types import SimpleNamespace

from conftest import synthetic_bars
from quantpilot.ai.ollama_client import BUILTIN_LABEL, OllamaClient, _builtin_analysis, signal_prompt
from quantpilot.commands import process_commands
from quantpilot.data.universe import validate_watchlist
from quantpilot.engine import TradingEngine
from quantpilot.ml.model import resolve_model_path
from quantpilot.scheduler.calendar import NY, get_session


def _noop(*a, **k):
    return None


def test_emergency_stop_command_is_processed(settings, broker, db):
    eng = TradingEngine(settings, db, broker, SimpleNamespace(), None)
    db.set_state("auto_trading", True)
    db.enqueue_command("emergency_stop", {"close_positions": False})
    db.enqueue_command("pause")
    process_commands(db, eng, _noop, _noop, _noop, _noop)
    assert db.flag("emergency_stop") and not db.flag("auto_trading") and not db.flag("bot_running")
    assert db.pending_commands() == []


def test_windows_model_path_resolves_on_linux(tmp_path):
    (tmp_path / "v1.joblib").write_bytes(b"x")
    win = r"C:\Users\someone\project\models\v1.joblib"
    assert resolve_model_path(win, tmp_path) == tmp_path / "v1.joblib"
    assert resolve_model_path(r"C:\x\missing.joblib", tmp_path) is None
    assert resolve_model_path(None, tmp_path) is None


def test_builtin_fallback_is_labeled_not_ai():
    c = OllamaClient(base_url="http://127.0.0.1:9")  # network is blocked in tests -> template fallback
    ok, txt, _ = c.chat(signal_prompt("AAPL", "HOLD", {"close": 100.0}, None, [], [], []))
    assert ok and c.source_label() == BUILTIN_LABEL
    assert c.health()["mode"] == "builtin" and "AI değil" in c.health()["message"]


def test_builtin_template_survives_missing_values():
    txt = _builtin_analysis(signal_prompt("AAPL", "HOLD", {"close": None, "ema_fast": None, "rsi": float("nan")},
                                          None, [], [], []))
    assert "AAPL" in txt and "bilinmeyen fiyat" in txt


def test_buy_candidates_ranked_by_ml_probability(settings, broker, fake_client, db, monkeypatch):
    """With room for only ONE position, the highest-probability BUY must get it (as in the backtest)."""
    now = datetime(2026, 6, 1, 11, 0, tzinfo=NY)
    fake_client.get_clock = lambda: SimpleNamespace(timestamp=now, is_open=True, next_open=now, next_close=now)
    fake_client.get_calendar = lambda filters=None: [SimpleNamespace(date=now.date(), open=dtime(9, 30), close=dtime(16, 0))]
    import quantpilot.engine as eng_mod
    monkeypatch.setattr(eng_mod, "get_session", lambda brk: get_session(brk, now))
    fake_client.get_asset = lambda sym: SimpleNamespace(symbol=sym, exchange="NASDAQ", status="active", tradable=True,
                                                        asset_class="us_equity", fractionable=True, name=sym)
    for w in list(db.watchlist()):
        if w["symbol"] not in ("AAPL", "MSFT"):
            db.remove_symbol(w["symbol"])
    validate_watchlist(broker, db)
    db.save_backtest({}, {}, [], [], [])
    db.set_params({"use_ml": True, "rsi_entry_min": 0, "rsi_entry_max": 100, "ml_prob_threshold": 0.5,
                   "max_exposure_pct": 0.10})
    db.set_state("bot_running", True)
    db.set_state("auto_trading", True)

    bars = {s: synthetic_bars(120, seed=7, drift=0.01) for s in ("AAPL", "MSFT")}
    md = SimpleNamespace(daily_bars=lambda syms, days=500: bars,
                         latest_prices=lambda syms: {s: float(bars[s]["close"].iloc[-1]) for s in syms},
                         last_feed_used="sip")
    probs = {"AAPL": 0.60, "MSFT": 0.90}
    monkeypatch.setattr(eng_mod, "load_active_predictor",
                        lambda db_: SimpleNamespace(version="t", predict_latest=lambda b, p: probs))
    eng = TradingEngine(settings, db, broker, md, None)
    monkeypatch.setattr(eng, "gate", lambda params: (True, []))
    res = eng.run_cycle()
    buys = [o for o in res["orders"] if o.get("side") == "BUY" and o.get("state") == "submitted"]
    assert [o["symbol"] for o in buys] == ["MSFT"], res


def test_code_fingerprint_matches_loaded_code():
    import quantpilot
    assert quantpilot.LOADED_FINGERPRINT == quantpilot.code_fingerprint()
    assert len(quantpilot.LOADED_FINGERPRINT) == 12
