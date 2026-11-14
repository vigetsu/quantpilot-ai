"""Performance analytics computed ONLY from actual filled paper orders / real equity series."""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_OBS_FOR_SHARPE = 30


@dataclass
class ClosedTrade:
    symbol: str
    qty: float
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    pnl: float
    exit_order_id: str


def fifo_realized(orders: list[dict]) -> tuple[list[ClosedTrade], dict[str, float]]:
    """Match filled quantities FIFO. Unfilled/canceled/rejected orders contribute nothing.

    Partially filled orders contribute only their ``filled_qty`` at ``filled_avg_price``.
    Returns (closed trades, realized P&L per sell order id).
    """
    fills = [o for o in orders if (o.get("filled_qty") or 0) > 0 and o.get("filled_avg_price")]
    fills.sort(key=lambda o: (o.get("filled_at") or o.get("updated_at") or o.get("submitted_at") or ""))
    lots: dict[str, deque] = defaultdict(deque)
    trades: list[ClosedTrade] = []
    per_order: dict[str, float] = {}
    for o in fills:
        sym, qty, px = o["symbol"], float(o["filled_qty"]), float(o["filled_avg_price"])
        t = o.get("filled_at") or o.get("updated_at") or ""
        if o["side"] == "buy":
            lots[sym].append([qty, px, t])
            continue
        remaining = qty
        realized = 0.0
        while remaining > 1e-9 and lots[sym]:
            lot = lots[sym][0]
            take = min(remaining, lot[0])
            pnl = (px - lot[1]) * take
            realized += pnl
            trades.append(ClosedTrade(sym, take, lot[1], px, lot[2], t, pnl, o["id"]))
            lot[0] -= take
            remaining -= take
            if lot[0] <= 1e-9:
                lots[sym].popleft()
        per_order[o["id"]] = realized
    return trades, per_order


def trade_stats(trades: list) -> dict:
    pnls = [t.pnl if hasattr(t, "pnl") else t["pnl"] for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win, gross_loss = sum(wins), -sum(losses)
    return {
        "n_trades": n,
        "win_rate": (len(wins) / n) if n else None,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else None),
        "total_pnl": float(sum(pnls)),
        "avg_trade": float(np.mean(pnls)) if n else None,
    }


def equity_metrics(equity: pd.Series, periods_per_year: int = 252) -> dict:
    """equity: daily equity series indexed by date."""
    equity = equity.dropna()
    if len(equity) < 2:
        return {"total_return": None, "max_drawdown": None, "sharpe": None, "cagr": None, "n_obs": len(equity),
                "volatility": None}
    rets = equity.pct_change().dropna()
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = max(len(equity) / periods_per_year, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if equity.iloc[0] > 0 and len(equity) > 20 else None
    sharpe = None
    if len(rets) >= MIN_OBS_FOR_SHARPE and rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * math.sqrt(periods_per_year))
    return {
        "total_return": float(total_return),
        "max_drawdown": float(dd.min()),
        "sharpe": sharpe,
        "cagr": float(cagr) if cagr is not None else None,
        "n_obs": int(len(rets)),
        "volatility": float(rets.std() * math.sqrt(periods_per_year)) if len(rets) > 1 else None,
    }


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0
