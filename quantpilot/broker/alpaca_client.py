"""Alpaca PAPER trading wrapper.

- Always constructs ``TradingClient(paper=True)``; refuses anything else.
- Read calls retry with backoff on network errors / HTTP 429 / 5xx.
- Order submission is NEVER blindly retried. If the outcome is ambiguous (timeout,
  connection drop), the caller must look the order up by ``client_order_id``.
- Returns plain dicts so the rest of the system (and tests) do not depend on SDK models.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable

from quantpilot.config import ConfigError, Settings, validate_paper_only

log = logging.getLogger("quantpilot.broker")


class BrokerError(RuntimeError):
    pass


class OrderRejected(BrokerError):
    """Broker definitively rejected the order (4xx). Safe: no order exists."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class OrderOutcomeUnknown(BrokerError):
    """Network/timeout during submission: the order MAY exist. Look up by client_order_id."""


def _v(x: Any) -> Any:
    """Enum -> value, UUID -> str, datetime -> iso, numeric strings -> float."""
    if x is None:
        return None
    if hasattr(x, "value"):
        return x.value
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if type(x).__name__ == "UUID":
        return str(x)
    return x


def _f(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def order_to_dict(o: Any) -> dict:
    otype = _v(getattr(o, "order_type", None)) or _v(getattr(o, "type", None))
    return {
        "id": str(_v(o.id)),
        "client_order_id": o.client_order_id,
        "symbol": o.symbol,
        "side": _v(o.side),
        "qty": _f(o.qty),
        "filled_qty": _f(o.filled_qty) or 0.0,
        "order_type": otype,
        "time_in_force": _v(getattr(o, "time_in_force", None)),
        "status": _v(o.status),
        "submitted_at": _v(getattr(o, "submitted_at", None)) or _v(getattr(o, "created_at", None)),
        "filled_at": _v(getattr(o, "filled_at", None)),
        "updated_at": _v(getattr(o, "updated_at", None)),
        "filled_avg_price": _f(getattr(o, "filled_avg_price", None)),
        "stop_price": _f(getattr(o, "stop_price", None)),
        "limit_price": _f(getattr(o, "limit_price", None)),
    }


def is_transient(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None)
    if code is not None:
        return code == 429 or code >= 500
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "network")) or isinstance(exc, (TimeoutError, ConnectionError, OSError))


@dataclass
class AssetInfo:
    symbol: str
    exchange: str | None
    status: str | None
    tradable: bool
    asset_class: str | None
    fractionable: bool
    name: str | None


class AlpacaBroker:
    def __init__(self, settings: Settings, trading_client: Any | None = None, max_retries: int = 3,
                 sleep: Callable[[float], None] = time.sleep):
        validate_paper_only(settings.paper, settings.base_url, settings.api_key.get())
        self.settings = settings
        self.max_retries = max_retries
        self._sleep = sleep
        if trading_client is None:
            if not settings.has_credentials:
                raise ConfigError("Alpaca Paper API anahtarları eksik.")
            from alpaca.trading.client import TradingClient
            trading_client = TradingClient(settings.api_key.get(), settings.secret_key.get(), paper=True)
            base_obj = getattr(trading_client, "_base_url", "")
            base = str(getattr(base_obj, "value", base_obj)).lower()
            if base and "paper-api" not in base:
                raise ConfigError("TradingClient paper uç noktasına bağlanmadı - işlem durduruldu.")
        self.client = trading_client

    # ------------------------------------------------------------ retry core
    def _read(self, fn: Callable, *args, **kwargs):
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if attempt < self.max_retries and is_transient(exc):
                    log.warning("Alpaca okuma hatası (deneme %s/%s): %s", attempt, self.max_retries, exc)
                    self._sleep(delay)
                    delay *= 2
                    continue
                raise BrokerError(f"Alpaca isteği başarısız: {exc}") from exc

    # --------------------------------------------------------------- account
    def get_account(self) -> dict:
        a = self._read(self.client.get_account)
        return {
            "account_number": getattr(a, "account_number", None),
            "status": _v(getattr(a, "status", None)),
            "currency": getattr(a, "currency", "USD"),
            "equity": _f(a.equity),
            "cash": _f(a.cash),
            "buying_power": _f(a.buying_power),
            "last_equity": _f(a.last_equity),
            "portfolio_value": _f(getattr(a, "portfolio_value", None)),
            "long_market_value": _f(getattr(a, "long_market_value", None)),
            "trading_blocked": bool(getattr(a, "trading_blocked", False)),
            "account_blocked": bool(getattr(a, "account_blocked", False)),
            "pattern_day_trader": bool(getattr(a, "pattern_day_trader", False)),
        }

    def get_positions(self) -> list[dict]:
        out = []
        for p in self._read(self.client.get_all_positions):
            out.append({
                "symbol": p.symbol,
                "qty": _f(p.qty),
                "side": _v(getattr(p, "side", "long")),
                "avg_entry_price": _f(p.avg_entry_price),
                "current_price": _f(p.current_price),
                "market_value": _f(p.market_value),
                "unrealized_pl": _f(p.unrealized_pl),
                "unrealized_plpc": _f(p.unrealized_plpc),
                "unrealized_intraday_pl": _f(getattr(p, "unrealized_intraday_pl", None)),
                "asset_class": _v(getattr(p, "asset_class", None)),
            })
        return out

    def get_orders(self, status: str = "all", after: datetime | None = None, limit: int = 500) -> list[dict]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        req = GetOrdersRequest(status=QueryOrderStatus(status), limit=limit, after=after, nested=False)
        return [order_to_dict(o) for o in self._read(self.client.get_orders, filter=req)]

    def get_order_by_client_id(self, client_order_id: str) -> dict | None:
        try:
            o = self.client.get_order_by_client_id(client_order_id)
        except Exception as exc:  # noqa: BLE001
            if getattr(exc, "status_code", None) == 404:
                return None
            raise BrokerError(f"Emir sorgulanamadı ({client_order_id}): {exc}") from exc
        return order_to_dict(o)

    def get_clock(self) -> dict:
        c = self._read(self.client.get_clock)
        return {"timestamp": c.timestamp, "is_open": bool(c.is_open), "next_open": c.next_open, "next_close": c.next_close}

    def get_calendar(self, start: date, end: date) -> list[dict]:
        from alpaca.trading.requests import GetCalendarRequest
        days = self._read(self.client.get_calendar, filters=GetCalendarRequest(start=start, end=end))
        return [{"date": d.date, "open": d.open, "close": d.close} for d in days]

    def get_asset(self, symbol: str) -> AssetInfo:
        a = self._read(self.client.get_asset, symbol)
        return AssetInfo(symbol=a.symbol, exchange=_v(a.exchange), status=_v(a.status), tradable=bool(a.tradable),
                         asset_class=_v(a.asset_class), fractionable=bool(getattr(a, "fractionable", False)),
                         name=getattr(a, "name", None))

    def get_portfolio_history(self, period: str = "1A", timeframe: str = "1D") -> dict:
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        h = self._read(self.client.get_portfolio_history,
                       history_filter=GetPortfolioHistoryRequest(period=period, timeframe=timeframe))
        return {"timestamp": list(h.timestamp or []), "equity": [_f(x) for x in (h.equity or [])],
                "profit_loss": [_f(x) for x in (h.profit_loss or [])], "base_value": _f(h.base_value)}

    # ---------------------------------------------------------------- orders
    def submit_market_order(self, symbol: str, qty: int, side: str, client_order_id: str) -> dict:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        if qty <= 0 or int(qty) != qty:
            raise OrderRejected("Miktar pozitif tam sayı olmalı.")
        req = MarketOrderRequest(symbol=symbol, qty=int(qty), side=OrderSide(side), time_in_force=TimeInForce.DAY,
                                 client_order_id=client_order_id, extended_hours=False)
        return self._submit(req, client_order_id)

    def submit_extended_limit_order(self, symbol: str, qty: int, side: str, limit_price: float,
                                    client_order_id: str) -> dict:
        """DAY limit order eligible for pre-market / after-hours (Alpaca accepts only limit orders there).

        An unfilled order stays live through the rest of the day's sessions and then expires."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        if qty <= 0 or int(qty) != qty:
            raise OrderRejected("Miktar pozitif tam sayı olmalı.")
        px = round(float(limit_price), 2 if limit_price >= 1 else 4)
        if px <= 0:
            raise OrderRejected("Geçersiz limit fiyatı.")
        req = LimitOrderRequest(symbol=symbol, qty=int(qty), side=OrderSide(side), time_in_force=TimeInForce.DAY,
                                limit_price=px, client_order_id=client_order_id, extended_hours=True)
        return self._submit(req, client_order_id)

    def submit_stop_order(self, symbol: str, qty: int, stop_price: float, client_order_id: str) -> dict:
        """Protective SELL stop (GTC). Note: a stop becomes a market order when triggered
        and may fill well below the stop price (gaps, fast markets)."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import StopOrderRequest
        req = StopOrderRequest(symbol=symbol, qty=int(qty), side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                               stop_price=round(float(stop_price), 2), client_order_id=client_order_id)
        return self._submit(req, client_order_id)

    def _submit(self, req: Any, client_order_id: str) -> dict:
        try:
            o = self.client.submit_order(order_data=req)
            return order_to_dict(o)
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "status_code", None)
            if code is not None and 400 <= code < 500 and code != 429:
                raise OrderRejected(f"Alpaca emri reddetti ({code}): {exc}", code) from exc
            # 429 / 5xx / network: we cannot know whether the order was accepted.
            raise OrderOutcomeUnknown(f"Emir sonucu belirsiz ({client_order_id}): {exc}") from exc

    def cancel_order(self, order_id: str) -> None:
        self._read(self.client.cancel_order_by_id, order_id)

    def cancel_all_orders(self) -> int:
        res = self._read(self.client.cancel_orders)
        return len(res or [])

    def close_all_positions(self) -> int:
        res = self._read(self.client.close_all_positions, cancel_orders=True)
        return len(res or [])
