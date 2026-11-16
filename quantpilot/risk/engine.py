"""Risk engine - the FINAL authority on whether an order may be created.

Nothing (strategy, ML, or LLM) can bypass these checks. Pure functions => easily tested,
and the same sizing logic is used by the backtester.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from quantpilot.db import HARD_LIMITS


@dataclass
class PortfolioState:
    equity: float
    cash: float
    last_equity: float                    # equity at previous close (Alpaca account.last_equity)
    positions: dict[str, dict]            # symbol -> {qty, market_value}
    open_orders: list[dict]               # broker open orders
    last_trade_time: dict[str, datetime] = field(default_factory=dict)


@dataclass
class RiskContext:
    auto_trading: bool
    emergency_stop: bool
    bot_running: bool
    market_open: bool
    in_entry_window: bool
    asset_ok: bool
    safety_gate_ok: bool
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RiskDecision:
    approved: bool
    qty: int = 0
    stop_price: float | None = None
    notes: list[str] = field(default_factory=list)


def effective_limits(params: dict) -> dict:
    return {k: min(float(params.get(k, v)), v) for k, v in HARD_LIMITS.items()}


def daily_drawdown(equity: float, last_equity: float) -> float:
    if not last_equity or last_equity <= 0:
        return 0.0
    return equity / last_equity - 1.0


def position_size(equity: float, cash: float, price: float, atr: float, current_exposure: float,
                  pending_buy_value: float, params: dict) -> tuple[int, list[str]]:
    """Whole-share size = min(risk budget, per-stock cap, total exposure cap, available cash)."""
    lim = effective_limits(params)
    notes: list[str] = []
    if price <= 0 or atr is None or not math.isfinite(atr) or atr <= 0:
        return 0, ["Geçersiz fiyat/ATR"]
    stop_dist = float(params.get("atr_stop_mult", 2.0)) * atr
    q_risk = math.floor(equity * lim["max_risk_per_trade_pct"] / stop_dist)
    q_pos = math.floor(equity * lim["max_position_pct"] / price)
    room = equity * lim["max_exposure_pct"] - current_exposure - pending_buy_value
    q_exp = math.floor(max(room, 0.0) / price)
    usable_cash = cash * (1 - float(params.get("cash_buffer_pct", 0.02))) - pending_buy_value
    q_cash = math.floor(max(usable_cash, 0.0) / price)   # cash only => no leverage
    qty = max(0, min(q_risk, q_pos, q_exp, q_cash))
    binding = min((("risk/işlem", q_risk), ("hisse başı limit", q_pos), ("toplam maruziyet", q_exp),
                   ("nakit", q_cash)), key=lambda x: x[1])
    notes.append(f"Boyut {qty} adet (bağlayıcı kısıt: {binding[0]}; risk={q_risk}, poz={q_pos}, "
                 f"maruziyet={q_exp}, nakit={q_cash})")
    return qty, notes


def check_entry(symbol: str, price: float, atr: float, state: PortfolioState, ctx: RiskContext,
                params: dict) -> RiskDecision:
    notes: list[str] = []
    blockers: list[str] = []
    lim = effective_limits(params)

    if ctx.emergency_stop:
        blockers.append("ACİL DURDURMA aktif")
    if not ctx.bot_running:
        blockers.append("Bot duraklatılmış")
    if not ctx.auto_trading:
        blockers.append("Otomatik paper emirleri kapalı")
    if not ctx.safety_gate_ok:
        blockers.append("Güvenlik kontrolleri geçilmedi")
    if not ctx.market_open:
        blockers.append("Piyasa kapalı (yalnızca normal seans)")
    elif not ctx.in_entry_window:
        blockers.append("Seans açılış/kapanış koruma penceresi")
    if not ctx.asset_ok:
        blockers.append("Hisse doğrulanmadı (NASDAQ/aktif/işlem yapılabilir değil)")

    dd = daily_drawdown(state.equity, state.last_equity)
    if dd <= -lim["max_daily_drawdown_pct"]:
        blockers.append(f"Günlük düşüş limiti aşıldı ({dd:.2%} <= -{lim['max_daily_drawdown_pct']:.0%}) - yeni giriş yok")
    if symbol in state.positions and abs(state.positions[symbol].get("qty", 0)) > 0:
        blockers.append("Bu hissede zaten pozisyon var")
    if any(o["symbol"] == symbol for o in state.open_orders if o.get("side") == "buy"):
        blockers.append("Bu hisse için bekleyen alış emri var (çift emir engellendi)")
    last = state.last_trade_time.get(symbol)
    cooldown = timedelta(minutes=int(params.get("cooldown_minutes", 0)))
    if last and ctx.now - last < cooldown:
        blockers.append(f"Bekleme süresi dolmadı (son işlem {last.isoformat(timespec='minutes')})")

    exposure = sum(abs(p.get("market_value") or 0.0) for p in state.positions.values())
    pending = sum((o.get("qty") or 0) * (o.get("ref_price") or price) for o in state.open_orders
                  if o.get("side") == "buy")
    if exposure + pending >= state.equity * lim["max_exposure_pct"]:
        blockers.append(f"Toplam maruziyet limiti dolu ({exposure / max(state.equity, 1):.1%})")

    qty, size_notes = position_size(state.equity, state.cash, price, atr, exposure, pending, params)
    notes.extend(size_notes)
    if qty < 1:
        blockers.append("Hesaplanan miktar < 1 adet")
    stop = round(price - float(params.get("atr_stop_mult", 2.0)) * atr, 2) if atr else None
    if stop is not None and stop <= 0:
        blockers.append("Geçersiz stop fiyatı")

    if blockers:
        return RiskDecision(False, 0, stop, blockers + notes)
    planned_risk = qty * (price - stop)
    notes.append(f"Planlanan risk ${planned_risk:,.0f} ({planned_risk / state.equity:.2%} özsermaye), stop ${stop:.2f}")
    return RiskDecision(True, qty, stop, notes)


def check_exit(symbol: str, state: PortfolioState, ctx: RiskContext) -> RiskDecision:
    """Exits reduce risk, so fewer gates apply - but still never short, never duplicate."""
    blockers = []
    pos = state.positions.get(symbol)
    qty = int(math.floor(abs(pos.get("qty", 0)))) if pos else 0
    if qty < 1:
        blockers.append("Satılacak pozisyon yok (açığa satış yasak)")
    if pos and pos.get("qty", 0) < 0:
        blockers.append("Kısa pozisyon tespit edildi - manuel inceleme gerekli")
    if ctx.emergency_stop:
        blockers.append("ACİL DURDURMA aktif - otomatik emir yok")
    if not ctx.market_open:
        blockers.append("Piyasa kapalı")
    if not ctx.auto_trading or not ctx.bot_running or not ctx.safety_gate_ok:
        blockers.append("Otomatik emirler devre dışı / bot duraklatıldı")
    if any(o["symbol"] == symbol and o.get("side") == "sell" and (o.get("order_type") or "") in ("market", "limit")
           for o in state.open_orders):
        blockers.append("Bekleyen satış emri var (çift emir engellendi)")
    if blockers:
        return RiskDecision(False, 0, None, blockers)
    return RiskDecision(True, qty, None, [f"Tüm pozisyon kapatılacak: {qty} adet"])
