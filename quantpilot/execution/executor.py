"""Idempotent PAPER order execution.

Guarantees:
1. Deterministic ``client_order_id`` per (symbol, side, signal bar date) -> one order per signal.
2. An *intent* row is persisted BEFORE submission, so a crash mid-submit is recoverable.
3. Failed/ambiguous submissions are never blindly retried; ambiguous ones are looked up
   by ``client_order_id`` (now and on every sync) instead of being resubmitted.
4. Only a RiskDecision(approved=True) can lead to submission.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from quantpilot.broker.alpaca_client import AlpacaBroker, BrokerError, OrderOutcomeUnknown, OrderRejected
from quantpilot.broker.sync import OPEN_STATUSES
from quantpilot.db import Database
from quantpilot.risk.engine import RiskDecision

log = logging.getLogger("quantpilot.execution")


def make_client_order_id(symbol: str, side: str, bar_date: str, suffix: str = "") -> str:
    base = f"qp-{symbol}-{side}-{bar_date.replace('-', '')}"
    return f"{base}-{suffix}" if suffix else base


class ExecutionResult(dict):
    @property
    def submitted(self) -> bool:
        return self.get("state") == "submitted"


class OrderExecutor:
    def __init__(self, broker: AlpacaBroker, db: Database, sleep: Callable[[float], None] = time.sleep):
        self.broker = broker
        self.db = db
        self._sleep = sleep

    def execute(self, decision: RiskDecision, symbol: str, side: str, client_order_id: str,
                requested_price: float | None, reason: str, signal_id: int | None = None,
                limit_price: float | None = None) -> ExecutionResult:
        """Market order in the regular session; with ``limit_price`` an extended-hours DAY limit order."""
        if not isinstance(decision, RiskDecision) or not decision.approved or decision.qty < 1:
            return ExecutionResult(state="blocked", message="Risk motoru onayı yok")
        if side not in ("buy", "sell"):
            return ExecutionResult(state="blocked", message="Geçersiz yön")

        existing = self.db.get_intent(client_order_id)
        if existing:
            return ExecutionResult(state="duplicate", message=f"Bu sinyal için emir zaten var ({existing['state']})",
                                   client_order_id=client_order_id)
        try:
            remote = self.broker.get_order_by_client_id(client_order_id)
        except BrokerError as exc:
            return ExecutionResult(state="blocked", message=f"Ön kontrol başarısız, emir gönderilmedi: {exc}")
        if remote:
            self.db.upsert_order(remote)
            return ExecutionResult(state="duplicate", message="Emir Alpaca'da zaten mevcut", order=remote)

        if not self.db.create_intent(client_order_id=client_order_id, symbol=symbol, side=side, qty=decision.qty,
                                     order_type="limit" if limit_price else "market",
                                     requested_price=requested_price, planned_stop=decision.stop_price,
                                     reason=reason, signal_id=signal_id):
            return ExecutionResult(state="duplicate", message="Eşzamanlı niyet tespit edildi")
        try:
            if limit_price:
                order = self.broker.submit_extended_limit_order(symbol, decision.qty, side, limit_price, client_order_id)
            else:
                order = self.broker.submit_market_order(symbol, decision.qty, side, client_order_id)
        except OrderRejected as exc:
            self.db.update_intent(client_order_id, "rejected", error=str(exc))
            log.error("Emir reddedildi %s %s x%s: %s", side, symbol, decision.qty, exc)
            return ExecutionResult(state="rejected", message=str(exc), client_order_id=client_order_id)
        except OrderOutcomeUnknown as exc:
            self.db.update_intent(client_order_id, "unknown", error=str(exc))
            log.warning("Emir sonucu belirsiz, yeniden GÖNDERİLMEYECEK, sorgulanacak: %s", client_order_id)
            self._sleep(2.0)
            try:
                found = self.broker.get_order_by_client_id(client_order_id)
            except BrokerError:
                found = None
            if found:
                self.db.upsert_order(found)
                self.db.update_intent(client_order_id, "submitted", alpaca_order_id=found["id"])
                return ExecutionResult(state="submitted", order=found, client_order_id=client_order_id)
            return ExecutionResult(state="unknown", message=str(exc), client_order_id=client_order_id)

        self.db.upsert_order(order)
        self.db.update_intent(client_order_id, "submitted", alpaca_order_id=order["id"])
        if order.get("status") == "rejected":
            log.error("Alpaca emri reddedildi: %s", client_order_id)
            return ExecutionResult(state="rejected", order=order, client_order_id=client_order_id)
        log.info("PAPER emir gönderildi: %s %s x%s (durum=%s)", side, symbol, decision.qty, order.get("status"))
        return ExecutionResult(state="submitted", order=order, client_order_id=client_order_id)

    # ------------------------------------------------------------ stops
    def ensure_protective_stops(self, positions: list[dict], open_orders: list[dict]) -> list[str]:
        """Place a GTC stop for each QuantPilot position lacking one. Stops are NOT guaranteed
        to fill at the stop price: once triggered they become market orders."""
        actions = []
        for p in positions:
            qty = int(p.get("qty") or 0)
            if qty < 1:
                continue
            sym = p["symbol"]
            # any open SELL (a stop, or an extended-hours exit/stop limit order) already holds these shares
            if any(o["symbol"] == sym and o.get("side") == "sell" for o in open_orders):
                continue
            buys = [o for o in self.db.orders(symbol=sym, limit=50)
                    if o.get("side") == "buy" and o.get("origin") == "quantpilot" and (o.get("filled_qty") or 0) > 0
                    and o.get("planned_stop") and o.get("status") not in OPEN_STATUSES]
            if not buys:
                continue  # external/manual position: do not touch
            buy = buys[0]
            prefix = f"{buy['client_order_id']}-stop-"
            prior = self.db.query("SELECT * FROM order_intents WHERE client_order_id LIKE ? ORDER BY created_at",
                                  (prefix + "%",))
            if prior and prior[-1]["state"] in ("pending_submit", "unknown"):
                continue  # wait for reconciliation, never double-submit
            if len(prior) >= 5:
                log.error("%s için koruyucu stop 5 kez başarısız oldu - manuel inceleme gerekli", sym)
                continue
            coid = f"{prefix}{len(prior) + 1}"
            stop = float(buy["planned_stop"])
            if not self.db.create_intent(client_order_id=coid, symbol=sym, side="sell", qty=qty, order_type="stop",
                                         requested_price=stop, planned_stop=stop, reason="Koruyucu stop (ATR)"):
                continue
            try:
                o = self.broker.submit_stop_order(sym, qty, stop, coid)
                self.db.upsert_order(o)
                self.db.update_intent(coid, "submitted", alpaca_order_id=o["id"])
                actions.append(f"{sym} stop {stop:.2f} x{qty}")
            except OrderRejected as exc:
                self.db.update_intent(coid, "rejected", error=str(exc))
                log.error("Stop emri reddedildi %s: %s", sym, exc)
            except OrderOutcomeUnknown as exc:
                self.db.update_intent(coid, "unknown", error=str(exc))
        return actions

    def software_stops(self, positions: list[dict], open_orders: list[dict], prices: dict[str, float],
                       session_key: str, buffer_pct: float) -> list[str]:
        """Outside regular hours broker stop orders do not trigger, so check the planned stop ourselves:
        if the (fresh) price is at/below it, cancel the resting GTC stop and sell with an extended-hours limit."""
        actions = []
        for p in positions:
            qty = int(p.get("qty") or 0)
            sym = p["symbol"]
            price = prices.get(sym)
            if qty < 1 or price is None:
                continue
            buys = [o for o in self.db.orders(symbol=sym, limit=50)
                    if o.get("side") == "buy" and o.get("origin") == "quantpilot" and (o.get("filled_qty") or 0) > 0
                    and o.get("planned_stop")]
            if not buys:
                continue  # external/manual position: do not touch
            stop = float(buys[0]["planned_stop"])
            if price > stop:
                continue
            if any(o["symbol"] == sym and o.get("side") == "sell" and o.get("order_type") == "limit" for o in open_orders):
                continue  # an exit limit order is already working
            if not self.cancel_symbol_orders(sym, open_orders):
                log.warning("%s yazılımsal stop: açık emirler iptal edilemedi", sym)
                continue
            from quantpilot.risk.engine import RiskDecision as _RD
            coid = f"{buys[0]['client_order_id']}-xstop-{session_key}"
            res = self.execute(_RD(True, qty, None, ["Yazılımsal stop"]), sym, "sell", coid, price,
                               f"Seans dışı yazılımsal stop: fiyat {price:.2f} <= stop {stop:.2f}",
                               limit_price=price * (1 - buffer_pct))
            actions.append(f"{sym} yazılımsal stop {price:.2f}<={stop:.2f}: {res.get('state')}")
        return actions

    def cancel_symbol_orders(self, symbol: str, open_orders: list[dict], wait_s: float = 6.0) -> bool:
        """Cancel open orders for a symbol (e.g. its stop before a market exit). Returns True when clear."""
        targets = [o for o in open_orders if o["symbol"] == symbol]
        for o in targets:
            try:
                self.broker.cancel_order(o["id"])
            except BrokerError as exc:
                log.warning("İptal başarısız %s: %s", o["id"], exc)
        if not targets:
            return True
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            try:
                still = [o for o in self.broker.get_orders(status="open") if o["symbol"] == symbol]
            except BrokerError:
                still = targets
            for o in still:
                self.db.upsert_order(o)
            if not still:
                return True
            self._sleep(1.0)
        return False

    # -------------------------------------------------------- emergency
    def emergency_stop(self, close_positions: bool = False) -> str:
        self.db.set_state("emergency_stop", True)
        self.db.set_state("auto_trading", False)
        msgs = []
        try:
            n = self.broker.cancel_all_orders()
            msgs.append(f"{n} açık emir iptal edildi")
        except BrokerError as exc:
            msgs.append(f"Emir iptali başarısız: {exc}")
        if close_positions:
            try:
                n = self.broker.close_all_positions()
                msgs.append(f"{n} pozisyon için kapatma emri gönderildi")
            except BrokerError as exc:
                msgs.append(f"Pozisyon kapatma başarısız: {exc}")
        msg = "ACİL DURDURMA: " + "; ".join(msgs)
        log.warning(msg)
        return msg


def open_orders_from_db(db: Database) -> list[dict]:
    return [o for o in db.orders(limit=1000) if o["status"] in OPEN_STATUSES]
