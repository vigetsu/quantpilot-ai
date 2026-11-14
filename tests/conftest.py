import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantpilot.config import Secret, Settings  # noqa: E402
from quantpilot.db import Database  # noqa: E402


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Any real network connection in tests is a failure -> guarantees no real orders."""
    def guard(*a, **k):
        raise RuntimeError("Testlerde ağ erişimi yasak")
    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key=Secret("PKTESTKEY1234567890"), secret_key=Secret("secretvalue1234567890abcd"),
                    db_path=tmp_path / "t.db", log_dir=tmp_path / "logs", model_dir=tmp_path / "models")


class FakeAPIError(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg)
        self.status_code = status_code


def make_order(**kw):
    base = dict(id="11111111-1111-1111-1111-111111111111", client_order_id="qp-AAPL-buy-20260101", symbol="AAPL",
                side="buy", qty="10", filled_qty="0", order_type="market", type="market", time_in_force="day",
                status="accepted", submitted_at="2026-01-02T15:00:00+00:00", created_at=None, filled_at=None,
                updated_at=None, filled_avg_price=None, stop_price=None, limit_price=None)
    base.update(kw)
    return SimpleNamespace(**base)


class FakeTradingClient:
    """Mimics alpaca TradingClient - records submissions, never touches the network."""

    def __init__(self):
        self.submitted = []
        self.orders = {}
        self.submit_error = None
        self.account = SimpleNamespace(account_number="PA123", status="ACTIVE", currency="USD", equity="100000",
                                       cash="100000", buying_power="200000", last_equity="100000",
                                       portfolio_value="100000", long_market_value="0", trading_blocked=False,
                                       account_blocked=False, pattern_day_trader=False)
        self.positions = []
        self.canceled = []

    def get_account(self):
        return self.account

    def get_all_positions(self):
        return self.positions

    def get_orders(self, filter=None):
        return list(self.orders.values())

    def get_order_by_client_id(self, coid):
        if coid in self.orders:
            return self.orders[coid]
        raise FakeAPIError("not found", 404)

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        if self.submit_error:
            raise self.submit_error
        o = make_order(id=f"id-{len(self.submitted)}", client_order_id=order_data.client_order_id,
                       symbol=order_data.symbol, side=order_data.side.value, qty=str(order_data.qty),
                       order_type=order_data.type.value, status="accepted")
        self.orders[order_data.client_order_id] = o
        return o

    def cancel_order_by_id(self, oid):
        self.canceled.append(oid)

    def cancel_orders(self):
        n = len(self.orders)
        return [1] * n

    def close_all_positions(self, cancel_orders=True):
        return [1] * len(self.positions)


@pytest.fixture
def fake_client():
    return FakeTradingClient()


@pytest.fixture
def broker(settings, fake_client):
    from quantpilot.broker.alpaca_client import AlpacaBroker
    return AlpacaBroker(settings, trading_client=fake_client, sleep=lambda s: None)


def synthetic_bars(n=600, seed=0, drift=0.0005, start="2022-01-03"):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.015, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
