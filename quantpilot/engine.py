"""Trading engine: one evaluation cycle = sync -> data -> indicators -> ML -> rules -> RISK -> execute.

The Ollama LLM is called AFTER decisions are final, asynchronously, for explanations only.
"""
from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from quantpilot.ai.ollama_client import OllamaClient, portfolio_prompt, signal_prompt
from quantpilot.broker.sync import OPEN_STATUSES, Synchronizer
from quantpilot.data.market_data import completed_daily_bars
from quantpilot.db import Database, utcnow
from quantpilot.execution.executor import OrderExecutor, make_client_order_id
from quantpilot.ml.model import load_active_predictor, resolve_model_path
from quantpilot.risk.engine import PortfolioState, RiskContext, check_entry, check_exit
from quantpilot.risk.selftest import run_self_test
from quantpilot.scheduler.calendar import get_session
from quantpilot.strategy.indicators import compute_indicators, market_regime
from quantpilot.strategy.rules import evaluate, rank_score

log = logging.getLogger("quantpilot.engine")
REGIME_SYMBOL = "QQQ"
IND_KEYS = ["close", "ema_fast", "ema_slow", "rsi", "macd", "macd_signal", "macd_hist", "atr", "hv20", "vol_ratio",
            "mom_10", "mom_20", "trend_strength"]


def safety_gate(db: Database, broker, params: dict) -> tuple[bool, list[str]]:
    """Checks that must ALL pass before automatic paper orders are permitted."""
    msgs, ok = [], True
    try:
        acct = broker.get_account()
        good = acct["status"] in ("ACTIVE", "AccountStatus.ACTIVE") and not acct["trading_blocked"] \
            and not acct["account_blocked"]
        msgs.append(("✅" if good else "❌") + f" Alpaca PAPER hesabı erişilebilir (durum={acct['status']})")
        ok &= good
    except Exception as exc:  # noqa: BLE001
        msgs.append(f"❌ Alpaca bağlantısı başarısız: {exc}")
        ok = False
    has_bt = bool(db.backtests(limit=1))
    msgs.append(("✅" if has_bt else "❌") + " En az bir backtest çalıştırıldı")
    ok &= has_bt
    if params.get("use_ml", True):
        rec = db.active_model()
        has_model = rec is not None and resolve_model_path(rec["path"]) is not None
        msgs.append(("✅" if has_model else "❌") + " Aktif ML modeli mevcut (dosya bulundu)")
        ok &= has_model
    st_ok, st_msgs = run_self_test(params)
    msgs.append(("✅" if st_ok else "❌") + " Risk motoru öz-testi")
    ok &= st_ok
    valid = db.watchlist(only_valid=True)
    msgs.append(("✅" if valid else "❌") + f" Doğrulanmış NASDAQ hisse sayısı: {len(valid)}")
    ok &= bool(valid)
    return bool(ok), msgs


class TradingEngine:
    def __init__(self, settings, db: Database, broker, market_data, ollama: OllamaClient | None = None):
        self.settings = settings
        self.db = db
        self.broker = broker
        self.md = market_data
        self.sync = Synchronizer(broker, db)
        self.exec = OrderExecutor(broker, db)
        self.ollama = ollama
        self._ai_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai")
        self._ai_done: set[tuple] = set()
        self._gate_cache: tuple[float, bool, list[str]] | None = None

    def gate(self, params: dict) -> tuple[bool, list[str]]:
        now = time.monotonic()
        if self._gate_cache and now - self._gate_cache[0] < 600:
            return self._gate_cache[1], self._gate_cache[2]
        ok, msgs = safety_gate(self.db, self.broker, params)
        self._gate_cache = (now, ok, msgs)
        self.db.set_state("safety_gate", json.dumps({"ok": ok, "checks": msgs}, ensure_ascii=False))
        return ok, msgs

    def _last_trade_times(self) -> dict[str, datetime]:
        out: dict[str, datetime] = {}
        for o in self.db.orders(limit=2000):
            if o.get("origin") != "quantpilot" or o.get("status") in ("rejected",) or o.get("order_type") == "stop":
                continue
            ts = o.get("submitted_at")
            if not ts:
                continue
            t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if o["symbol"] not in out or t > out[o["symbol"]]:
                out[o["symbol"]] = t
        return out

    def run_cycle(self, force_eval: bool = False) -> dict:
        params = self.db.get_params()
        session = get_session(self.broker)
        phase = session.phase
        # extended = pre-market / after-hours trading with limit orders (opt-out in the dashboard)
        extended = bool(params.get("extended_hours_trading", False)) and phase in ("pre", "post")
        tradable = session.is_open or extended
        self.db.set_state("market_open", session.is_open)
        self.db.set_state("market_phase", phase)
        if not tradable and not force_eval:
            return {"skipped": "market_closed", "phase": phase}
        summary = {"ts": utcnow(), "signals": [], "orders": [], "phase": phase}

        sync_res = self.sync.sync_all()
        account = sync_res["account"]
        symbols = [w["symbol"] for w in self.db.watchlist(only_valid=True)]
        if not symbols:
            log.warning("Doğrulanmış hisse yok - döngü atlandı")
            return {"skipped": "no_symbols"}

        use_regime = bool(params.get("use_regime_filter", False))
        fetch = list(dict.fromkeys(symbols + ([REGIME_SYMBOL] if use_regime else [])))
        raw = self.md.daily_bars(fetch, days=500)
        self.db.set_state("last_data_update", utcnow())
        self.db.set_state("active_feed", f"tarihsel={self.md.last_feed_used}, anlık={self.settings.data_feed}")
        # Today's daily bar is still forming during the regular session and does not exist yet (or is
        # partial) in pre-market; after the close it is complete and may be used (signal at close t).
        drop_today = session.is_open or phase == "pre"
        bars = {s: completed_daily_bars(df, session.now_ny, drop_today) for s, df in raw.items()}
        regime_ok = None  # unknown => the rule blocks new entries (fail safe)
        if use_regime:
            qb = bars.get(REGIME_SYMBOL)
            if qb is not None and len(qb):
                reg = market_regime(qb, int(params.get("regime_sma", 200))).dropna()
                if len(reg):
                    regime_ok = bool(reg.iloc[-1])
            self.db.set_state("market_regime", "unknown" if regime_ok is None else ("risk_on" if regime_ok else "risk_off"))
            if REGIME_SYMBOL not in symbols:
                bars.pop(REGIME_SYMBOL, None)
        stale: dict[str, str] = {}
        try:
            if extended:
                # Trades are sparse outside regular hours: only use a last trade that is recent enough.
                max_age = float(params.get("extended_max_price_age_min", 15))
                now_utc = session.now_ny.astimezone(timezone.utc)
                prices = {}
                for s, (px, ts) in self.md.latest_trades(symbols).items():
                    age = (now_utc - ts).total_seconds() / 60 if ts else None
                    if age is not None and age <= max_age:
                        prices[s] = px
                    else:
                        stale[s] = "bilinmiyor" if age is None else f"{age:.0f} dk"
            else:
                prices = self.md.latest_prices(symbols)
        except Exception as exc:  # noqa: BLE001
            log.warning("Anlık fiyat alınamadı - yeni emir verilmeyecek: %s", exc)
            prices = {}
        buf = float(params.get("extended_limit_buffer_pct", 0.003))

        predictor = load_active_predictor(self.db) if params.get("use_ml", True) else None
        probs = predictor.predict_latest(bars, params) if predictor else {}

        positions = {p["symbol"]: p for p in self.db.positions()}
        open_orders = [o for o in self.db.orders(limit=1000) if o["status"] in OPEN_STATUSES]
        state = PortfolioState(equity=account["equity"], cash=account["cash"], last_equity=account["last_equity"],
                               positions=positions, open_orders=open_orders, last_trade_time=self._last_trade_times())
        gate_ok, _ = self.gate(params)
        valid_set = set(symbols)

        # Pass 1: evaluate every symbol. Pass 2: act - exits/holds first, then BUY candidates
        # ranked by ML probability (highest first), exactly like the backtester, so that when
        # exposure/cash is limited the strongest signals get capital (not alphabetical order).
        evaluated = []
        for sym in symbols:
            df = bars.get(sym)
            if df is None or len(df) < 60:
                continue
            ind = compute_indicators(df, params)
            row = ind.iloc[-1].to_dict()
            if use_regime:
                row["regime_ok"] = regime_ok
            bar_date = str(ind.index[-1].date()) if hasattr(ind.index[-1], "date") else str(ind.index[-1])
            prob = probs.get(sym)
            dec = evaluate(row, params, sym in positions and positions[sym]["qty"] > 0, prob)
            evaluated.append((sym, row, bar_date, prob, dec))
        # same priority function as the backtester (ML probability or 6-1 month momentum)
        evaluated.sort(key=lambda e: (e[4].action == "BUY", -rank_score(e[3], e[1], params)))

        for sym, row, bar_date, prob, dec in evaluated:
            price = prices.get(sym)
            ctx = RiskContext(auto_trading=self.db.flag("auto_trading"), emergency_stop=self.db.flag("emergency_stop"),
                              bot_running=self.db.flag("bot_running"), market_open=tradable,
                              in_entry_window=(True if extended else
                                               session.in_entry_window(int(params["entry_skip_open_minutes"]),
                                                                       int(params["entry_skip_close_minutes"]))),
                              asset_ok=sym in valid_set, safety_gate_ok=gate_ok,
                              now=session.now_ny.astimezone(timezone.utc))
            rd, coid, result = None, None, None
            rd_notes: list[str] = []
            no_px = (f"Seans dışı: son işlem fiyatı çok eski ({stale[sym]}) - emir verilmedi" if sym in stale
                     else "Anlık fiyat yok - emir verilmedi")
            if dec.action == "BUY":
                if price is None:
                    rd_notes = [no_px]
                    rd = None
                else:
                    rd = check_entry(sym, price, float(row["atr"]), state, ctx, params)
                    coid = make_client_order_id(sym, "buy", bar_date)
            elif dec.action == "SELL":
                if extended and price is None:
                    rd_notes = [no_px]  # a limit sell needs a fresh reference price
                else:
                    rd = check_exit(sym, state, ctx)
                    coid = make_client_order_id(sym, "sell", bar_date)
            if rd is not None and rd.approved and extended:
                rd.notes.append(f"Seans dışı ({'seans öncesi' if phase == 'pre' else 'seans sonrası'}): "
                                f"limit emir, son fiyat ±%{buf * 100:.1f}")
            indicators = {k: (None if row.get(k) is None or (isinstance(row.get(k), float) and math.isnan(row[k]))
                              else float(row[k])) for k in IND_KEYS}
            sig = {"symbol": sym, "bar_date": bar_date, "action": dec.action, "price": price or row["close"],
                   "indicators": indicators, "ml_prob": prob, "model_version": predictor.version if predictor else None,
                   "reasons": dec.reasons + [f"Çelişki: {c}" for c in dec.conflicts],
                   "risk_approved": None if rd is None else int(rd.approved),
                   "risk_notes": (rd.notes if rd else rd_notes),
                   "qty": rd.qty if rd else None, "client_order_id": coid if rd and rd.approved else None}
            signal_id = self.db.add_signal(sig)
            summary["signals"].append({"symbol": sym, "action": dec.action, "ml_prob": prob,
                                       "approved": bool(rd and rd.approved)})

            if rd is not None and rd.approved:
                reason = "; ".join(dec.reasons)
                if dec.action == "SELL":
                    if not self.exec.cancel_symbol_orders(sym, open_orders):
                        log.warning("%s açık emirleri iptal edilemedi - satış ertelendi", sym)
                        continue
                    result = self.exec.execute(rd, sym, "sell", coid, price, reason, signal_id,
                                               limit_price=price * (1 - buf) if extended else None)
                else:
                    result = self.exec.execute(rd, sym, "buy", coid, price, reason, signal_id,
                                               limit_price=price * (1 + buf) if extended else None)
                    if result.get("state") in ("submitted", "unknown"):
                        state.open_orders.append({"symbol": sym, "side": "buy", "qty": rd.qty, "ref_price": price})
                summary["orders"].append({"symbol": sym, "side": dec.action, "state": result.get("state"),
                                          "message": result.get("message")})
            self._queue_ai(signal_id, sym, bar_date, dec, prob, indicators, sig["risk_notes"])

        if params.get("place_stop_orders", True) and self.db.flag("auto_trading") and not self.db.flag("emergency_stop"):
            self.sync.sync_all()
            open_now = [o for o in self.db.orders(limit=1000) if o["status"] in OPEN_STATUSES]
            acts = []
            if extended:
                # Broker stop orders do not trigger outside regular hours -> enforce the stop ourselves.
                # Runs BEFORE placing missing GTC stops, so it never races a stop it did not see.
                key = f"{session.now_ny:%Y%m%d}{phase}"
                acts += self.exec.software_stops(self.db.positions(), open_now, prices, key, buf)
                if acts:
                    self.sync.sync_all()
                    open_now = [o for o in self.db.orders(limit=1000) if o["status"] in OPEN_STATUSES]
            acts += self.exec.ensure_protective_stops(self.db.positions(), open_now)
            if acts:
                summary["orders"].extend({"stop": a} for a in acts)
        self._queue_portfolio_ai(account, summary)
        self.db.set_state("last_evaluation", utcnow())
        self.db.set_state("last_cycle_summary", json.dumps(summary, ensure_ascii=False, default=str))
        return summary

    # ------------------------------------------------------------------ AI
    def _queue_ai(self, signal_id, sym, bar_date, dec, prob, indicators, risk_notes):
        if not self.ollama:
            return
        key = (sym, bar_date, dec.action)
        if key in self._ai_done:
            return
        self._ai_done.add(key)

        def job():
            ok, text, lat = self.ollama.chat(signal_prompt(sym, dec.action, indicators, prob, dec.reasons,
                                                           dec.conflicts, risk_notes))
            self.db.add_ai_analysis("signal", sym, self.ollama.source_label(), ok, text, lat)
            if ok:
                self.db.set_signal_ai(signal_id, text)
            self.db.set_state("ollama_status", "ok" if ok else "offline")
        self._ai_pool.submit(job)

    def _queue_portfolio_ai(self, account, summary):
        if not self.ollama:
            return
        key = ("__portfolio__", datetime.now(timezone.utc).strftime("%Y-%m-%d"), datetime.now(timezone.utc).hour)
        if key in self._ai_done:
            return
        self._ai_done.add(key)
        data = {"ozsermaye": account["equity"], "nakit": account["cash"],
                "gunluk_degisim": account["equity"] - (account["last_equity"] or account["equity"]),
                "pozisyonlar": [{k: p[k] for k in ("symbol", "qty", "unrealized_pl")} for p in self.db.positions()],
                "sinyaller": summary["signals"]}

        def job():
            ok, text, lat = self.ollama.chat(portfolio_prompt(data))
            self.db.add_ai_analysis("portfolio", None, self.ollama.source_label(), ok, text, lat)
            self.db.set_state("ollama_status", "ok" if ok else "offline")
        self._ai_pool.submit(job)
