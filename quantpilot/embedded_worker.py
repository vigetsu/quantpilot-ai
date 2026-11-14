"""Embedded background scheduler so Streamlit Community Cloud runs the 24/7 paper worker in-process."""
from __future__ import annotations

import logging
import os
import threading

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from quantpilot.backtest.runner import run_and_store
from quantpilot.data.universe import validate_watchlist
from quantpilot.db import utcnow
from quantpilot.engine import TradingEngine
from quantpilot.ml.model import ModelTrainer, load_active_predictor
from quantpilot.worker import NY_TZ, SingleInstanceLock

log = logging.getLogger("quantpilot.embedded")
_LOCK = threading.Lock()
_STARTED = False
_FILE_LOCK: SingleInstanceLock | None = None


def ensure_embedded_worker(settings, db, broker, md, ollama) -> bool:
    """Start an in-process BackgroundScheduler if no external worker process is already holding worker.lock."""
    global _STARTED, _FILE_LOCK
    if not settings.has_credentials or broker is None or md is None:
        return False
    with _LOCK:
        if _STARTED:
            return True
        fl = SingleInstanceLock(settings.db_path.parent / "worker.lock")
        if not fl.acquire():
            return False  # standalone start_bot.bat is already running
        _FILE_LOCK = fl
        _STARTED = True

    eng = TradingEngine(settings, db, broker, md, ollama)
    cycle_lock = threading.Lock()

    def heartbeat():
        db.set_state("worker_heartbeat", utcnow())
        db.set_state("worker_pid", f"cloud-{os.getpid()}")

    def job_sync():
        try:
            eng.sync.sync_all()
        except Exception as exc:  # noqa: BLE001
            log.warning("Bulut senkronizasyon hatası: %s", exc)

    def job_cycle(force: bool = False):
        if not cycle_lock.acquire(blocking=False):
            return
        try:
            eng.run_cycle(force_eval=force)
        except Exception as exc:  # noqa: BLE001
            log.exception("Bulut strateji döngüsü hatası: %s", exc)
        finally:
            cycle_lock.release()

    def bootstrap():
        try:
            heartbeat()
            job_sync()
            if not db.watchlist(only_valid=True):
                validate_watchlist(broker, db)
            params = db.get_params()
            syms = [w["symbol"] for w in db.watchlist(only_valid=True)]
            need_model = params.get("use_ml", True) and load_active_predictor(db) is None
            need_bt = not db.backtests(limit=1)
            if (need_model or need_bt) and syms:
                bars = md.daily_bars(list(set(syms + ["QQQ"])), days=1100)
                if need_model:
                    ModelTrainer(settings.model_dir, db).train_and_register(
                        {k: v for k, v in bars.items() if k != "QQQ"}, params
                    )
                if need_bt:
                    run_and_store(bars, params, db, settings.model_dir, settings.initial_capital,
                                  f"alpaca:{md.last_feed_used}")
                eng._gate_cache = None
            if db.get_state("cloud_bootstrapped") != "true":
                db.set_state("bot_running", True)
                if not db.flag("emergency_stop"):
                    db.set_state("auto_trading", True)
                db.set_state("cloud_bootstrapped", "true")
            if not db.recent_signals(limit=1):
                job_cycle(force=True)
        except Exception as exc:  # noqa: BLE001
            log.exception("Bulut ilk hazırlık hatası: %s", exc)

    p = db.get_params()
    step = max(1, int(p["strategy_interval_minutes"]))
    sched = BackgroundScheduler(timezone=NY_TZ, job_defaults={"coalesce": True, "max_instances": 1,
                                                              "misfire_grace_time": 120})
    sched.add_job(heartbeat, IntervalTrigger(seconds=30), id="hb")
    sched.add_job(job_sync, IntervalTrigger(seconds=int(p["sync_interval_seconds"])), id="sync")
    sched.add_job(job_cycle, CronTrigger(day_of_week="mon-fri", hour="9-15", minute=f"*/{step}", second=20,
                                         timezone=NY_TZ), id="cycle")
    sched.start()
    threading.Thread(target=bootstrap, daemon=True, name="qp-bootstrap").start()
    return True
