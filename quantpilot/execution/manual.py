"""Manual 1-share TEST orders from the dashboard (PAPER account only, like everything else).

Goes through the same broker wrapper as the bot, so it exercises order submission, extended-hours
limit orders and account sync end to end. Test orders use the ``qptest-`` client_order_id prefix:
they are stored as non-QuantPilot orders, so they never set the bot's cooldown and never get the
bot's protective stops. (If the bot's strategy later decides SELL for that symbol, it will close the
whole position, test share included.)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from quantpilot.broker.alpaca_client import BrokerError, OrderRejected, OrderOutcomeUnknown
from quantpilot.db import Database
from quantpilot.scheduler.calendar import get_session

log = logging.getLogger("quantpilot.manual")
TEST_PREFIX = "qptest-"
TEST_LIMIT_PCT = 0.01  # outside regular hours: limit 1% through the last trade to make a fill likely


def test_order(broker, md, db: Database, symbol: str, side: str, qty: int = 1, now: datetime | None = None) -> dict:
    """Submit a small test order. Returns {"ok": bool, "message": str, "order": dict | None}."""
    side = side.lower()
    if side not in ("buy", "sell") or qty < 1:
        return {"ok": False, "message": "Geçersiz yön veya miktar.", "order": None}
    if db.flag("emergency_stop"):
        return {"ok": False, "message": "ACİL DURDURMA aktif - test emri gönderilmedi.", "order": None}
    valid = {w["symbol"] for w in db.watchlist(only_valid=True)}
    if symbol not in valid:
        return {"ok": False, "message": f"{symbol} doğrulanmış izleme listesinde değil.", "order": None}

    session = get_session(broker, now)
    phase = session.phase
    if phase == "closed":
        nxt = session.next_open.strftime("%d.%m.%Y %H:%M ET") if session.next_open else "?"
        return {"ok": False, "message": f"Piyasa kapalı (seans öncesi/sonrası da değil). Sonraki normal seans: {nxt}.",
                "order": None}

    if side == "sell":
        try:
            held = sum(float(p["qty"]) for p in broker.get_positions() if p["symbol"] == symbol)
        except BrokerError as exc:
            return {"ok": False, "message": f"Pozisyon okunamadı: {exc}", "order": None}
        if held < qty:
            return {"ok": False, "message": f"{symbol} için satılacak {qty} adet yok (eldeki: {held:g}). Açığa satış yapılmaz.",
                    "order": None}

    coid = f"{TEST_PREFIX}{symbol}-{side}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    try:
        if phase == "regular":
            order = broker.submit_market_order(symbol, qty, side, coid)
            how = "piyasa emri (normal seans)"
        else:
            trades = md.latest_trades([symbol])
            if symbol not in trades:
                return {"ok": False, "message": f"{symbol} için son fiyat alınamadı.", "order": None}
            last, ts = trades[symbol]
            limit = last * (1 + TEST_LIMIT_PCT) if side == "buy" else last * (1 - TEST_LIMIT_PCT)
            order = broker.submit_extended_limit_order(symbol, qty, side, limit, coid)
            age = ""
            if ts is not None:
                age = f", son işlem {int((session.now_ny.astimezone(timezone.utc) - ts).total_seconds() // 60)} dk önce"
            how = (f"limit emir ({'seans öncesi' if phase == 'pre' else 'seans sonrası'}), limit ${limit:,.2f} "
                   f"(son fiyat ${last:,.2f}{age})")
    except OrderRejected as exc:
        return {"ok": False, "message": f"Alpaca emri reddetti: {exc}", "order": None}
    except OrderOutcomeUnknown as exc:
        return {"ok": False, "message": f"Emir sonucu belirsiz, birkaç saniye sonra 'Senkronize Et' ile kontrol edin: {exc}",
                "order": None}
    except BrokerError as exc:
        return {"ok": False, "message": f"Alpaca hatası: {exc}", "order": None}

    db.upsert_order(order)
    msg = (f"TEST {'ALIŞ' if side == 'buy' else 'SATIŞ'}: {symbol} x{qty}, {how}. "
           f"Alpaca durumu: {order.get('status')}.")
    db.add_event("INFO", "manual", msg)
    log.info(msg)
    return {"ok": True, "message": msg, "order": order}
