"""Dashboard -> worker command queue processing, shared by the standalone and embedded workers.

The dashboard enqueues commands in SQLite whenever a worker heartbeat is fresh, so EVERY
worker that writes a heartbeat must also drain this queue (otherwise buttons such as
emergency stop would silently do nothing).
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Callable

from quantpilot.db import Database
from quantpilot.engine import TradingEngine

log = logging.getLogger("quantpilot.commands")


def process_commands(db: Database, engine: TradingEngine, run_cycle: Callable[..., None],
                     sync: Callable[[], None], validate: Callable[[], None], retrain: Callable[[], None]) -> None:
    for cmd in db.pending_commands():
        name, payload = cmd["command"], json.loads(cmd["payload"] or "{}")
        try:
            if name == "start":
                db.set_state("bot_running", True)
                result = "Bot başlatıldı"
            elif name == "pause":
                db.set_state("bot_running", False)
                result = "Bot duraklatıldı"
            elif name == "emergency_stop":
                result = engine.exec.emergency_stop(close_positions=bool(payload.get("close_positions")))
            elif name == "clear_emergency":
                db.set_state("emergency_stop", False)
                result = "Acil durdurma kaldırıldı (otomatik emirler kapalı kalır)"
            elif name == "run_cycle":
                threading.Thread(target=run_cycle, kwargs={"force": True}, daemon=True).start()
                result = "Döngü başlatıldı"
            elif name == "sync":
                sync()
                result = "Senkronize edildi"
            elif name == "validate_watchlist":
                validate()
                result = "Doğrulandı"
            elif name == "retrain":
                threading.Thread(target=retrain, daemon=True).start()
                result = "Eğitim başlatıldı"
            elif name == "refresh_gate":
                engine._gate_cache = None
                result = "Güvenlik kontrolleri yenilenecek"
            else:
                result = f"Bilinmeyen komut: {name}"
        except Exception as exc:  # noqa: BLE001
            result = f"Hata: {exc}"
            log.exception("Komut hatası %s", name)
        db.complete_command(cmd["id"], result)
        log.info("Komut işlendi: %s -> %s", name, result)
