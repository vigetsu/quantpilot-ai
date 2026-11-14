"""Synchronise local SQLite state with the actual Alpaca paper account (broker is the source of truth)."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from quantpilot.broker.alpaca_client import AlpacaBroker, BrokerError
from quantpilot.db import Database, utcnow

log = logging.getLogger("quantpilot.sync")

OPEN_STATUSES = {"new", "accepted", "pending_new", "partially_filled", "held", "accepted_for_bidding",
                 "pending_replace", "pending_cancel", "calculated", "done_for_day"}
TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "replaced", "stopped", "suspended"}


class Synchronizer:
    def __init__(self, broker: AlpacaBroker, db: Database):
        self.broker = broker
        self.db = db

    def sync_all(self, order_lookback_days: int = 120) -> dict:
        """Pull account, positions and orders. Raises BrokerError on failure (caller logs)."""
        account = self.broker.get_account()
        self.db.add_account_snapshot(account)
        positions = self.broker.get_positions()
        self.db.replace_positions(positions)
        after = datetime.now(timezone.utc) - timedelta(days=order_lookback_days)
        orders = self.broker.get_orders(status="all", after=after)
        for o in orders:
            self.db.upsert_order(o)
        self.reconcile_intents()
        self.db.set_state("last_sync", utcnow())
        self.db.set_state("alpaca_status", "ok")
        return {"account": account, "positions": positions, "orders": len(orders)}

    def reconcile_intents(self, unknown_grace_seconds: int = 120) -> None:
        """Resolve order intents whose submission outcome was not confirmed (e.g. after a crash/timeout)."""
        for intent in self.db.unresolved_intents():
            coid = intent["client_order_id"]
            try:
                found = self.broker.get_order_by_client_id(coid)
            except BrokerError as exc:
                log.warning("Niyet uzlaştırılamadı %s: %s", coid, exc)
                continue
            if found:
                self.db.upsert_order(found)
                self.db.update_intent(coid, "submitted", alpaca_order_id=found["id"])
                log.info("Uzlaştırma: %s Alpaca'da bulundu (durum=%s)", coid, found["status"])
            else:
                created = datetime.fromisoformat(intent["created_at"])
                if (datetime.now(timezone.utc) - created).total_seconds() > unknown_grace_seconds:
                    # Not at Alpaca: the order never existed. Mark as failed; the deterministic
                    # client_order_id prevents re-submission for the same signal.
                    self.db.update_intent(coid, "not_found", error="Alpaca'da bulunamadı - gönderilmemiş sayıldı")
                    log.warning("Uzlaştırma: %s Alpaca'da yok, gönderilmemiş olarak işaretlendi", coid)

    def open_orders(self) -> list[dict]:
        return [o for o in self.db.orders(limit=1000) if o["status"] in OPEN_STATUSES]
