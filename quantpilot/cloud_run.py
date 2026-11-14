"""One-shot cloud execution entrypoint (for GitHub Actions / scheduled cloud runners).

When invoked in a stateless/cached cloud environment, this module:
1. Verifies PAPER-ONLY safety guards and connects to the Alpaca Paper account.
2. Synchronizes account, positions, and orders into SQLite.
3. Validates the NASDAQ watchlist if not yet validated.
4. Trains the walk-forward ML model and runs an initial backtest if none exist in the cache.
5. Enables bot_running + auto_trading (when ``QP_CLOUD_AUTO=true``) and runs one full
   strategy + risk + idempotent paper execution cycle.
"""
from __future__ import annotations

import json
import os
import sys

from quantpilot.backtest.runner import run_and_store
from quantpilot.broker.alpaca_client import AlpacaBroker
from quantpilot.config import ConfigError, load_settings
from quantpilot.data.market_data import MarketData
from quantpilot.data.universe import validate_watchlist
from quantpilot.db import Database
from quantpilot.engine import TradingEngine
from quantpilot.logging_setup import setup_logging
from quantpilot.ml.model import ModelTrainer, load_active_predictor


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    try:
        settings = load_settings(require_credentials=True)
    except ConfigError as exc:
        print(f"[HATA] {exc}")
        return 2

    db = Database(settings.db_path)
    log = setup_logging(settings.log_dir, settings.log_level, settings.secret_values(), db, "cloud")
    broker = AlpacaBroker(settings)
    md = MarketData(settings)
    engine = TradingEngine(settings, db, broker, md, ollama=None)

    log.info("Bulut döngüsü başladı (PAPER ONLY). Hesap senkronize ediliyor...")
    engine.sync.sync_all()

    if not db.watchlist(only_valid=True):
        validate_watchlist(broker, db)

    params = db.get_params()
    syms = [w["symbol"] for w in db.watchlist(only_valid=True)]

    need_model = params.get("use_ml", True) and load_active_predictor(db) is None
    need_bt = not db.backtests(limit=1)
    if (need_model or need_bt) and syms:
        log.info("Bulut önyükleme: tarihsel veriler çekiliyor (model=%s, backtest=%s)...", need_model, need_bt)
        bars = md.daily_bars(list(set(syms + ["QQQ"])), days=1100)
        if need_model:
            ModelTrainer(settings.model_dir, db).train_and_register(
                {k: v for k, v in bars.items() if k != "QQQ"}, params
            )
        if need_bt:
            run_and_store(bars, params, db, settings.model_dir, settings.initial_capital, f"alpaca:{md.last_feed_used}")

    if os.environ.get("QP_CLOUD_AUTO", "true").lower() in ("1", "true", "yes"):
        db.set_state("bot_running", True)
        if not db.flag("emergency_stop"):
            db.set_state("auto_trading", True)

    force = os.environ.get("QP_FORCE_EVAL", "false").lower() in ("1", "true", "yes")
    res = engine.run_cycle(force_eval=force)
    print(json.dumps(res, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
