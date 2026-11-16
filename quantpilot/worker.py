"""QuantPilot AI trading worker (runs independently of the dashboard).

Start with: start_bot.bat  (or: python -m quantpilot.worker)
Only ONE worker may run at a time (OS file lock). When this process or Windows stops,
trading stops; open GTC stop orders remain at Alpaca.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from quantpilot.ai.ollama_client import OllamaClient
from quantpilot.broker.alpaca_client import AlpacaBroker
from quantpilot.commands import process_commands
from quantpilot.config import ConfigError, load_settings
from quantpilot.data.market_data import MarketData
from quantpilot.data.universe import validate_watchlist
from quantpilot.db import Database, utcnow
from quantpilot.engine import TradingEngine
from quantpilot.logging_setup import setup_logging
from quantpilot.ml.model import ModelTrainer

log = logging.getLogger("quantpilot.worker")
NY_TZ = "America/New_York"


class SingleInstanceLock:
    """Cross-process exclusive lock on a file (Windows: msvcrt, POSIX: fcntl)."""

    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            self.fh = None
            return False
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(str(os.getpid()))
        self.fh.flush()
        return True

    def release(self) -> None:
        if self.fh:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.fh.seek(0)
                    msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self.fh.close()
            self.fh = None


class Worker:
    def __init__(self):
        self.settings = load_settings(require_credentials=True)
        self.db = Database(self.settings.db_path)
        setup_logging(self.settings.log_dir, self.settings.log_level, self.settings.secret_values(), self.db, "worker")
        self.broker = AlpacaBroker(self.settings)
        self.md = MarketData(self.settings)
        self.ollama = OllamaClient(self.settings.ollama_url, self.settings.ollama_model, self.settings.ollama_timeout)
        self.engine = TradingEngine(self.settings, self.db, self.broker, self.md, self.ollama)
        self.cycle_lock = threading.Lock()
        self.scheduler = BlockingScheduler(timezone=NY_TZ, job_defaults={"coalesce": True, "max_instances": 1,
                                                                         "misfire_grace_time": 120})

    # -------------------------------------------------------------- jobs
    def heartbeat(self):
        self.db.set_state("worker_heartbeat", utcnow())
        self.db.set_state("worker_pid", str(os.getpid()))

    def job_sync(self):
        try:
            self.engine.sync.sync_all()
        except Exception as exc:  # noqa: BLE001
            self.db.set_state("alpaca_status", f"hata: {exc}")
            log.error("Senkronizasyon başarısız (API kesintisi olabilir): %s", exc)

    def job_cycle(self, force: bool = False):
        if not self.cycle_lock.acquire(blocking=False):
            log.info("Önceki döngü hâlâ çalışıyor - atlandı")
            return
        try:
            res = self.engine.run_cycle(force_eval=force)
            log.info("Strateji döngüsü: %s", json.dumps(res, ensure_ascii=False, default=str)[:500])
        except Exception as exc:  # noqa: BLE001
            log.exception("Strateji döngüsü hatası (emir gönderilmedi): %s", exc)
        finally:
            self.cycle_lock.release()

    def job_ollama_health(self):
        h = self.ollama.health()
        self.db.set_state("ollama_status", "ok" if h["online"] and h["model_available"] else "offline")
        self.db.set_state("ollama_message", h["message"])

    def job_validate(self):
        try:
            res = validate_watchlist(self.broker, self.db)
            log.info("Hisse listesi doğrulandı: %s", ", ".join(f"{r['symbol']}={'OK' if r['valid'] else 'X'}" for r in res))
        except Exception as exc:  # noqa: BLE001
            log.error("Hisse doğrulama başarısız: %s", exc)

    def job_retrain(self):
        try:
            params = self.db.get_params()
            syms = [w["symbol"] for w in self.db.watchlist(only_valid=True)]
            if not syms:
                return
            bars = self.md.daily_bars(syms, days=1825)
            res = ModelTrainer(self.settings.model_dir, self.db).train_and_register(bars, params)
            self.engine._gate_cache = None
            log.info("Model yeniden eğitildi: %s", res["version"])
        except Exception as exc:  # noqa: BLE001
            log.exception("Model eğitimi başarısız: %s", exc)

    def job_commands(self):
        process_commands(self.db, self.engine, self.job_cycle, self.job_sync, self.job_validate, self.job_retrain)

    # -------------------------------------------------------------- main
    def startup(self):
        log.info("QuantPilot AI worker başlıyor (PAPER ONLY, pid=%s)", os.getpid())
        self.heartbeat()
        # Reconciliation on restart: Alpaca is the source of truth.
        self.job_sync()
        self.engine.sync.reconcile_intents(unknown_grace_seconds=0)
        self.job_validate()
        self.job_ollama_health()
        params = self.db.get_params()
        if params.get("use_ml", True) and self.db.active_model() is None:
            log.info("Aktif model yok - ilk model eğitiliyor")
            threading.Thread(target=self.job_retrain, daemon=True).start()

    def run(self):
        self.startup()
        p = self.db.get_params()
        s = self.scheduler
        s.add_job(self.heartbeat, IntervalTrigger(seconds=30), id="heartbeat")
        s.add_job(self.job_commands, IntervalTrigger(seconds=3), id="commands")
        s.add_job(self.job_sync, IntervalTrigger(seconds=int(p["sync_interval_seconds"])), id="sync")
        s.add_job(self.job_ollama_health, IntervalTrigger(minutes=2), id="ollama")
        step = max(1, int(p["strategy_interval_minutes"]))
        s.add_job(self.job_cycle, CronTrigger(day_of_week="mon-fri", hour="9-15", minute=f"*/{step}", second=20,
                                              timezone=NY_TZ), id="cycle")
        s.add_job(self.job_validate, CronTrigger(day_of_week="mon-fri", hour=8, minute=30, timezone=NY_TZ), id="validate")
        s.add_job(self.job_retrain, CronTrigger(day_of_week="sat", hour=10, minute=0, timezone=NY_TZ), id="retrain")
        log.info("Zamanlayıcı başladı (America/New_York). Strateji her %s dakikada (yalnızca normal seans).", step)
        s.start()


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
    lock = SingleInstanceLock(settings.db_path.parent / "worker.lock")
    if not lock.acquire():
        print("[HATA] Başka bir QuantPilot worker zaten çalışıyor. Aynı anda yalnızca bir worker çalışabilir.")
        return 3
    try:
        w = Worker()

        def _stop(*_):
            log.info("Worker durduruluyor - otomatik işlem duracak.")
            w.db.set_state("worker_heartbeat", "")
            try:
                w.scheduler.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
        w.run()
    except (KeyboardInterrupt, SystemExit):
        pass
    except ConfigError as exc:
        print(f"[HATA] {exc}")
        return 2
    finally:
        lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
