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
PRE_MARKET_START = dtime(4, 0)    # Alpaca extended hours: pre-market 04:00-09:30 ET
AFTER_HOURS_END = dtime(20, 0)    # after-hours 16:00-20:00 ET
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
    pre_open: datetime | None = None      # extended-hours pre-market start (04:00 ET) on a trading day
    post_close: datetime | None = None    # extended-hours after-hours end (20:00 ET; close+4h on early-close days)

    @property
    def phase(self) -> str:
        """'regular' | 'pre' | 'post' | 'closed' (overnight, weekends, holidays or broker unreachable)."""
        if self.is_open:
            return "regular"
        if self.pre_open and self.session_open and self.pre_open <= self.now_ny < self.session_open:
            return "pre"
        if self.session_close and self.post_close and self.session_close <= self.now_ny < self.post_close:
            return "post"
        return "closed"

    def minutes_to(self, t: datetime | None) -> float | None:
        return None if t is None else (t - self.now_ny).total_seconds() / 60

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
    so = sc = pre = post = None
    if cal:
        d = cal[0]
        day = d["date"] if not isinstance(d["date"], str) else datetime.fromisoformat(d["date"]).date()
        if day == now_ny.date():
            so, sc = _combine(day, d["open"]), _combine(day, d["close"])
            pre = datetime.combine(day, PRE_MARKET_START, tzinfo=NY)
            post = min(sc + timedelta(hours=4), datetime.combine(day, AFTER_HOURS_END, tzinfo=NY))
    return SessionInfo(now_ny, bool(clock["is_open"]), so, sc, _to_ny(clock["next_open"]), _to_ny(clock["next_close"]),
                       pre_open=pre, post_close=post)


def _dur(minutes: float | None) -> str:
    if minutes is None:
        return "?"
    m = max(0, int(round(minutes)))
    return f"{m // 60} sa {m % 60} dk" if m >= 60 else f"{m} dk"


def market_status_text(s: SessionInfo) -> str:
    if s.source == "unavailable":
        return "Bilinmiyor (Alpaca'ya ulaşılamadı)"
    ph = s.phase
    if ph == "regular":
        return f"AÇIK (normal seans) - kapanışa {_dur(s.minutes_to_close())}"
    if ph == "pre":
        return f"SEANS ÖNCESİ - normal seansa {_dur(s.minutes_to(s.session_open))}"
    if ph == "post":
        return f"SEANS SONRASI - bitişe {_dur(s.minutes_to(s.post_close))}"
    nxt = s.next_open.strftime("%d.%m.%Y %H:%M ET") if s.next_open else "?"
    return f"KAPALI - sonraki normal seans {nxt}"
