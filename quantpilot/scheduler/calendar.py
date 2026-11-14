"""US market session handling (America/New_York).

Uses Alpaca's clock + calendar endpoints, which already account for weekends, NYSE/NASDAQ
holidays, early closes and daylight-saving transitions. If the broker is unreachable the
session is treated as CLOSED (fail safe: no trading).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
log = logging.getLogger("quantpilot.calendar")


@dataclass
class SessionInfo:
    now_ny: datetime
    is_open: bool
    session_open: datetime | None
    session_close: datetime | None
    next_open: datetime | None
    next_close: datetime | None
    source: str = "alpaca"

    def minutes_since_open(self) -> float | None:
        if not self.session_open:
            return None
        return (self.now_ny - self.session_open).total_seconds() / 60

    def minutes_to_close(self) -> float | None:
        if not self.session_close:
            return None
        return (self.session_close - self.now_ny).total_seconds() / 60

    def in_entry_window(self, skip_open_min: int, skip_close_min: int) -> bool:
        if not self.is_open:
            return False
        so, tc = self.minutes_since_open(), self.minutes_to_close()
        if so is None or tc is None:
            return False
        return so >= skip_open_min and tc >= skip_close_min


def _to_ny(x) -> datetime | None:
    if x is None:
        return None
    if isinstance(x, str):
        x = datetime.fromisoformat(x)
    if x.tzinfo is None:
        x = x.replace(tzinfo=NY)
    return x.astimezone(NY)


def _combine(day, t) -> datetime:
    if isinstance(t, str):
        h, m = t.split(":")[:2]
        t = dtime(int(h), int(m))
    if isinstance(t, datetime):
        return _to_ny(t)
    return datetime.combine(day, t, tzinfo=NY)


def get_session(broker, now: datetime | None = None) -> SessionInfo:
    now_ny = (now or datetime.now(NY)).astimezone(NY)
    try:
        clock = broker.get_clock()
        cal = broker.get_calendar(now_ny.date(), now_ny.date())
    except Exception as exc:  # noqa: BLE001
        log.warning("Piyasa saati alınamadı, güvenli mod (kapalı): %s", exc)
        return SessionInfo(now_ny, False, None, None, None, None, source="unavailable")
    so = sc = None
    if cal:
        d = cal[0]
        day = d["date"] if not isinstance(d["date"], str) else datetime.fromisoformat(d["date"]).date()
        if day == now_ny.date():
            so, sc = _combine(day, d["open"]), _combine(day, d["close"])
    return SessionInfo(now_ny, bool(clock["is_open"]), so, sc, _to_ny(clock["next_open"]), _to_ny(clock["next_close"]))


def market_status_text(s: SessionInfo) -> str:
    if s.source == "unavailable":
        return "Bilinmiyor (Alpaca'ya ulaşılamadı)"
    if s.is_open:
        return f"AÇIK - kapanışa {s.minutes_to_close():.0f} dk"
    nxt = s.next_open.strftime("%d.%m.%Y %H:%M ET") if s.next_open else "?"
    return f"KAPALI - sonraki açılış {nxt}"
