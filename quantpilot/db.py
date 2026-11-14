"""SQLite persistence layer (WAL mode, safe for dashboard + worker processes)."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD"]

# Configurable strategy + risk parameters (editable from the dashboard).
DEFAULT_PARAMS: dict[str, Any] = {
    # strategy
    "ema_fast": 20,
    "ema_slow": 50,
    "rsi_period": 14,
    "rsi_entry_min": 45.0,
    "rsi_entry_max": 70.0,
    "ml_prob_threshold": 0.55,
    "use_ml": True,
    "atr_period": 14,
    "atr_stop_mult": 2.0,
    "exit_rsi_max": 80.0,           # exit if RSI above this (overextended); 0 disables
    "ml_exit_prob": 0.40,           # exit if ML prob drops below this; 0 disables
    "label_horizon_days": 5,
    # risk
    "max_position_pct": 0.10,
    "max_exposure_pct": 0.50,
    "max_risk_per_trade_pct": 0.01,
    "max_daily_drawdown_pct": 0.03,
    "cooldown_minutes": 1440,
    "cash_buffer_pct": 0.02,
    "entry_skip_open_minutes": 5,
    "entry_skip_close_minutes": 5,
    "place_stop_orders": True,
    # scheduling
    "strategy_interval_minutes": 15,
    "sync_interval_seconds": 60,
    # backtest
    "slippage_bps": 5.0,
    "commission_per_share": 0.0,
}

# Hard ceilings: dashboard edits can only make limits STRICTER than these.
HARD_LIMITS = {
    "max_position_pct": 0.10,
    "max_exposure_pct": 0.50,
    "max_risk_per_trade_pct": 0.01,
    "max_daily_drawdown_pct": 0.03,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT);
CREATE TABLE IF NOT EXISTS bot_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT, command TEXT NOT NULL, payload TEXT,
    created_at TEXT, processed_at TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS watchlist (
    symbol TEXT PRIMARY KEY, added_at TEXT, is_valid INTEGER DEFAULT 0,
    exchange TEXT, note TEXT, validated_at TEXT);
CREATE TABLE IF NOT EXISTS account_snapshots (
    ts TEXT PRIMARY KEY, equity REAL, cash REAL, buying_power REAL, last_equity REAL,
    portfolio_value REAL, long_market_value REAL, status TEXT);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY, client_order_id TEXT UNIQUE, symbol TEXT, side TEXT, qty REAL,
    filled_qty REAL, order_type TEXT, time_in_force TEXT, status TEXT, submitted_at TEXT,
    filled_at TEXT, updated_at TEXT, filled_avg_price REAL, requested_price REAL,
    stop_price REAL, limit_price REAL, planned_stop REAL, reason TEXT, signal_id INTEGER,
    origin TEXT DEFAULT 'external');
CREATE INDEX IF NOT EXISTS ix_orders_symbol ON orders(symbol);
CREATE TABLE IF NOT EXISTS order_intents (
    client_order_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, qty REAL, order_type TEXT,
    requested_price REAL, planned_stop REAL, reason TEXT, signal_id INTEGER,
    created_at TEXT, state TEXT, alpaca_order_id TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY, qty REAL, avg_entry_price REAL, current_price REAL,
    market_value REAL, unrealized_pl REAL, unrealized_plpc REAL, unrealized_intraday_pl REAL,
    updated_at TEXT);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, bar_date TEXT, symbol TEXT, action TEXT,
    price REAL, indicators TEXT, ml_prob REAL, model_version TEXT, reasons TEXT,
    risk_approved INTEGER, risk_notes TEXT, qty REAL, client_order_id TEXT, ai_summary TEXT);
CREATE INDEX IF NOT EXISTS ix_signals_ts ON signals(ts);
CREATE TABLE IF NOT EXISTS ai_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, scope TEXT, symbol TEXT,
    model TEXT, ok INTEGER, content TEXT, latency_s REAL);
CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, params TEXT, metrics TEXT,
    equity TEXT, trades TEXT, benchmark TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS model_registry (
    version TEXT PRIMARY KEY, created_at TEXT, path TEXT, params TEXT, metrics TEXT,
    train_start TEXT, train_end TEXT, data_hash TEXT, is_active INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, level TEXT, component TEXT, message TEXT);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._seed()

    # ------------------------------------------------------------------ core
    @contextmanager
    def tx(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def query(self, sql: str, params: Iterable = ()) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return [dict(r) for r in cur.fetchall()]

    def execute(self, sql: str, params: Iterable = ()) -> int:
        with self.tx() as c:
            cur = c.execute(sql, tuple(params))
            return cur.lastrowid

    def _seed(self) -> None:
        if not self.query("SELECT 1 FROM watchlist LIMIT 1"):
            for s in DEFAULT_WATCHLIST:
                self.execute("INSERT OR IGNORE INTO watchlist(symbol, added_at) VALUES (?,?)", (s, utcnow()))
        defaults = {"bot_running": "false", "auto_trading": "false", "emergency_stop": "false"}
        for k, v in defaults.items():
            self.execute("INSERT OR IGNORE INTO bot_state(key, value, updated_at) VALUES (?,?,?)", (k, v, utcnow()))

    # -------------------------------------------------------------- settings
    def get_params(self) -> dict[str, Any]:
        params = dict(DEFAULT_PARAMS)
        for r in self.query("SELECT key, value FROM settings"):
            if r["key"] in params:
                params[r["key"]] = json.loads(r["value"])
        return params

    def set_params(self, updates: dict[str, Any]) -> dict[str, Any]:
        clean = {}
        for k, v in updates.items():
            if k not in DEFAULT_PARAMS:
                continue
            default = DEFAULT_PARAMS[k]
            if isinstance(default, bool):
                v = bool(v)
            elif isinstance(default, int):
                v = int(v)
            elif isinstance(default, float):
                v = float(v)
            if k in HARD_LIMITS:
                v = max(0.0, min(v, HARD_LIMITS[k]))
            clean[k] = v
        with self.tx() as c:
            for k, v in clean.items():
                c.execute("INSERT OR REPLACE INTO settings(key, value, updated_at) VALUES (?,?,?)",
                          (k, json.dumps(v), utcnow()))
        return self.get_params()

    # ------------------------------------------------------------- bot state
    def get_state(self, key: str, default: str | None = None) -> str | None:
        r = self.query("SELECT value FROM bot_state WHERE key=?", (key,))
        return r[0]["value"] if r else default

    def get_state_row(self, key: str) -> dict | None:
        r = self.query("SELECT * FROM bot_state WHERE key=?", (key,))
        return r[0] if r else None

    def set_state(self, key: str, value: Any) -> None:
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif not isinstance(value, str):
            value = json.dumps(value, default=str)
        self.execute("INSERT OR REPLACE INTO bot_state(key, value, updated_at) VALUES (?,?,?)",
                     (key, value, utcnow()))

    def flag(self, key: str) -> bool:
        return (self.get_state(key, "false") or "false").lower() == "true"

    # -------------------------------------------------------------- commands
    def enqueue_command(self, command: str, payload: dict | None = None) -> int:
        return self.execute("INSERT INTO commands(command, payload, created_at) VALUES (?,?,?)",
                            (command, json.dumps(payload or {}), utcnow()))

    def pending_commands(self) -> list[dict]:
        return self.query("SELECT * FROM commands WHERE processed_at IS NULL ORDER BY id")

    def complete_command(self, cid: int, result: str) -> None:
        self.execute("UPDATE commands SET processed_at=?, result=? WHERE id=?", (utcnow(), result, cid))

    # -------------------------------------------------------------- watchlist
    def watchlist(self, only_valid: bool = False) -> list[dict]:
        sql = "SELECT * FROM watchlist" + (" WHERE is_valid=1" if only_valid else "") + " ORDER BY symbol"
        return self.query(sql)

    def add_symbol(self, symbol: str) -> None:
        self.execute("INSERT OR IGNORE INTO watchlist(symbol, added_at) VALUES (?,?)", (symbol.upper().strip(), utcnow()))

    def remove_symbol(self, symbol: str) -> None:
        self.execute("DELETE FROM watchlist WHERE symbol=?", (symbol.upper().strip(),))

    def set_symbol_validation(self, symbol: str, is_valid: bool, exchange: str | None, note: str) -> None:
        self.execute("UPDATE watchlist SET is_valid=?, exchange=?, note=?, validated_at=? WHERE symbol=?",
                     (1 if is_valid else 0, exchange, note, utcnow(), symbol))

    # ---------------------------------------------------------------- events
    def add_event(self, level: str, component: str, message: str) -> None:
        self.execute("INSERT INTO events(ts, level, component, message) VALUES (?,?,?,?)",
                     (utcnow(), level, component, message[:2000]))

    def recent_events(self, limit: int = 200) -> list[dict]:
        return self.query("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))

    # ---------------------------------------------------------------- orders
    ORDER_COLS = ("id", "client_order_id", "symbol", "side", "qty", "filled_qty", "order_type",
                  "time_in_force", "status", "submitted_at", "filled_at", "updated_at",
                  "filled_avg_price", "stop_price", "limit_price")

    def upsert_order(self, o: dict) -> None:
        """Insert/update broker-reported fields; preserves local fields (reason, planned_stop...)."""
        cols = [c for c in self.ORDER_COLS if c in o]
        with self.tx() as c:
            existing = c.execute("SELECT id FROM orders WHERE id=?", (o["id"],)).fetchone()
            if existing:
                sets = ", ".join(f"{k}=?" for k in cols if k != "id")
                c.execute(f"UPDATE orders SET {sets} WHERE id=?", [o[k] for k in cols if k != "id"] + [o["id"]])
            else:
                intent = c.execute("SELECT * FROM order_intents WHERE client_order_id=?",
                                   (o.get("client_order_id"),)).fetchone()
                extra = {}
                if intent:
                    extra = {"requested_price": intent["requested_price"], "planned_stop": intent["planned_stop"],
                             "reason": intent["reason"], "signal_id": intent["signal_id"], "origin": "quantpilot"}
                elif str(o.get("client_order_id", "")).startswith("qp-"):
                    extra = {"origin": "quantpilot"}
                allc = cols + list(extra.keys())
                vals = [o[k] for k in cols] + list(extra.values())
                c.execute(f"INSERT INTO orders({', '.join(allc)}) VALUES ({', '.join('?' * len(allc))})", vals)

    def orders(self, limit: int = 500, symbol: str | None = None) -> list[dict]:
        if symbol:
            return self.query("SELECT * FROM orders WHERE symbol=? ORDER BY submitted_at DESC LIMIT ?", (symbol, limit))
        return self.query("SELECT * FROM orders ORDER BY submitted_at DESC LIMIT ?", (limit,))

    # --------------------------------------------------------- order intents
    def get_intent(self, client_order_id: str) -> dict | None:
        r = self.query("SELECT * FROM order_intents WHERE client_order_id=?", (client_order_id,))
        return r[0] if r else None

    def create_intent(self, **kw) -> bool:
        """Returns False if an intent with this client_order_id already exists (idempotency)."""
        try:
            self.execute(
                "INSERT INTO order_intents(client_order_id, symbol, side, qty, order_type, requested_price,"
                " planned_stop, reason, signal_id, created_at, state) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (kw["client_order_id"], kw["symbol"], kw["side"], kw["qty"], kw.get("order_type", "market"),
                 kw.get("requested_price"), kw.get("planned_stop"), kw.get("reason"), kw.get("signal_id"),
                 utcnow(), "pending_submit"))
            return True
        except sqlite3.IntegrityError:
            return False

    def update_intent(self, client_order_id: str, state: str, alpaca_order_id: str | None = None,
                      error: str | None = None) -> None:
        self.execute("UPDATE order_intents SET state=?, alpaca_order_id=COALESCE(?, alpaca_order_id), error=? "
                     "WHERE client_order_id=?", (state, alpaca_order_id, error, client_order_id))

    def unresolved_intents(self) -> list[dict]:
        return self.query("SELECT * FROM order_intents WHERE state IN ('pending_submit','unknown')")

    # ------------------------------------------------------------- positions
    def replace_positions(self, positions: list[dict]) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM positions")
            for p in positions:
                c.execute("INSERT INTO positions(symbol, qty, avg_entry_price, current_price, market_value,"
                          " unrealized_pl, unrealized_plpc, unrealized_intraday_pl, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                          (p["symbol"], p["qty"], p["avg_entry_price"], p["current_price"], p["market_value"],
                           p["unrealized_pl"], p["unrealized_plpc"], p.get("unrealized_intraday_pl"), utcnow()))

    def positions(self) -> list[dict]:
        return self.query("SELECT * FROM positions ORDER BY market_value DESC")

    def add_account_snapshot(self, a: dict) -> None:
        self.execute("INSERT OR REPLACE INTO account_snapshots(ts, equity, cash, buying_power, last_equity,"
                     " portfolio_value, long_market_value, status) VALUES (?,?,?,?,?,?,?,?)",
                     (utcnow(), a["equity"], a["cash"], a["buying_power"], a["last_equity"],
                      a.get("portfolio_value"), a.get("long_market_value"), a.get("status")))

    def latest_account(self) -> dict | None:
        r = self.query("SELECT * FROM account_snapshots ORDER BY ts DESC LIMIT 1")
        return r[0] if r else None

    def account_history(self) -> list[dict]:
        return self.query("SELECT * FROM account_snapshots ORDER BY ts")

    # --------------------------------------------------------------- signals
    def add_signal(self, s: dict) -> int:
        return self.execute(
            "INSERT INTO signals(ts, bar_date, symbol, action, price, indicators, ml_prob, model_version, reasons,"
            " risk_approved, risk_notes, qty, client_order_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (utcnow(), s.get("bar_date"), s["symbol"], s["action"], s.get("price"),
             json.dumps(s.get("indicators", {}), default=float), s.get("ml_prob"), s.get("model_version"),
             json.dumps(s.get("reasons", []), ensure_ascii=False), s.get("risk_approved"),
             json.dumps(s.get("risk_notes", []), ensure_ascii=False), s.get("qty"), s.get("client_order_id")))

    def set_signal_ai(self, signal_id: int, text: str) -> None:
        self.execute("UPDATE signals SET ai_summary=? WHERE id=?", (text, signal_id))

    def recent_signals(self, limit: int = 100) -> list[dict]:
        return self.query("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,))

    def latest_signals_per_symbol(self) -> list[dict]:
        return self.query("SELECT s.* FROM signals s JOIN (SELECT symbol, MAX(id) mid FROM signals GROUP BY symbol) m"
                          " ON s.id=m.mid ORDER BY s.symbol")

    # ------------------------------------------------------------------ misc
    def add_ai_analysis(self, scope: str, symbol: str | None, model: str, ok: bool, content: str,
                        latency: float) -> int:
        return self.execute("INSERT INTO ai_analyses(ts, scope, symbol, model, ok, content, latency_s)"
                            " VALUES (?,?,?,?,?,?,?)", (utcnow(), scope, symbol, model, 1 if ok else 0, content, latency))

    def recent_ai(self, limit: int = 20) -> list[dict]:
        return self.query("SELECT * FROM ai_analyses ORDER BY id DESC LIMIT ?", (limit,))

    def save_backtest(self, params: dict, metrics: dict, equity: list, trades: list, benchmark: list,
                      notes: str = "") -> int:
        return self.execute("INSERT INTO backtest_runs(ts, params, metrics, equity, trades, benchmark, notes)"
                            " VALUES (?,?,?,?,?,?,?)",
                            (utcnow(), json.dumps(params, default=str), json.dumps(metrics, default=str),
                             json.dumps(equity, default=str), json.dumps(trades, default=str),
                             json.dumps(benchmark, default=str), notes))

    def backtests(self, limit: int = 20) -> list[dict]:
        return self.query("SELECT * FROM backtest_runs ORDER BY id DESC LIMIT ?", (limit,))

    def register_model(self, version: str, path: str, params: dict, metrics: dict, train_start: str,
                       train_end: str, data_hash: str, activate: bool = True) -> None:
        with self.tx() as c:
            if activate:
                c.execute("UPDATE model_registry SET is_active=0")
            c.execute("INSERT OR REPLACE INTO model_registry(version, created_at, path, params, metrics, train_start,"
                      " train_end, data_hash, is_active) VALUES (?,?,?,?,?,?,?,?,?)",
                      (version, utcnow(), path, json.dumps(params, default=str), json.dumps(metrics, default=str),
                       train_start, train_end, data_hash, 1 if activate else 0))

    def active_model(self) -> dict | None:
        r = self.query("SELECT * FROM model_registry WHERE is_active=1 ORDER BY created_at DESC LIMIT 1")
        return r[0] if r else None

    def models(self) -> list[dict]:
        return self.query("SELECT * FROM model_registry ORDER BY created_at DESC")
