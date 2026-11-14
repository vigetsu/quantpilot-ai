"""Structured JSON logging with secret redaction and DB error mirroring."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

_KEY_PATTERN = re.compile(r"\b(PK|AK|SK)[A-Z0-9]{14,}\b")


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: list[str] | None = None):
        super().__init__()
        self.secrets = [s for s in (secrets or []) if s and len(s) >= 4]

    def redact(self, text: str) -> str:
        for s in self.secrets:
            text = text.replace(s, "***REDACTED***")
        return _KEY_PATTERN.sub("***REDACTED***", text)

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        record.msg = self.redact(msg)
        record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class DbEventHandler(logging.Handler):
    """Mirrors WARNING+ records into the SQLite events table for the dashboard."""

    def __init__(self, db, level=logging.WARNING):
        super().__init__(level)
        self.db = db

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.db.add_event(record.levelname, record.name, record.getMessage())
        except Exception:  # never let logging crash the app
            pass


def setup_logging(log_dir: Path, level: str = "INFO", secrets: list[str] | None = None,
                  db=None, name: str = "quantpilot") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("quantpilot")
    root.setLevel(getattr(logging, level, logging.INFO))
    root.handlers.clear()
    redactor = RedactingFilter(secrets)

    fh = RotatingFileHandler(log_dir / f"{name}.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    fh.addFilter(redactor)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    ch.addFilter(redactor)
    root.addHandler(ch)

    if db is not None:
        dh = DbEventHandler(db)
        dh.addFilter(redactor)
        root.addHandler(dh)
    # keep noisy libs quiet
    for noisy in ("urllib3", "apscheduler", "alpaca"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return root
