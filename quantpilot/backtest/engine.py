"""Daily, multi-symbol, long-only event backtester (SIMULATION - never mixed with paper metrics).

Execution assumptions (no look-ahead):
- Signals are computed from the bar at close t (indicators use data <= t, ML probs are
  out-of-sample walk-forward predictions made by models trained only on data before t).
- Orders execute at the OPEN of t+1, adjusted by slippage (bps) + per-share commission.
- Protective stops: if low <= stop, fill at min(open, stop) minus slippage (gap-through fills
  are worse than the stop price, as in reality).
- Same sizing function and limits as the live risk engine; whole shares; cash only (no leverage).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantpilot.analytics.performance import equity_metrics, trade_stats
from quantpilot.risk.engine import effective_limits, position_size
from quantpilot.strategy.indicators import compute_indicators, market_regime
from quantpilot.strategy.rules import evaluate, rank_score


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[dict]
    metrics: dict
    benchmark: pd.Series | None = None
    signals: list[dict] = field(default_factory=list)


def run_backtest(bars: dict[str, pd.DataFrame], params: dict, initial_capital: float = 100_000.0,
                 probs: dict[tuple[str, pd.Timestamp], float] | None = None, start: pd.Timestamp | None = None,
                 end: pd.Timestamp | None = None, benchmark: pd.DataFrame | None = None,
                 test_start: pd.Timestamp | None = None) -> BacktestResult:
    slip = float(params.get("slippage_bps", 5.0)) / 10_000
    comm = float(params.get("commission_per_share", 0.0))
    stop_mult = float(params.get("atr_stop_mult", 2.0))
    lim = effective_limits(params)
    cooldown_days = math.ceil(int(params.get("cooldown_minutes", 0)) / 1440)
    use_ml = bool(params.get("use_ml", True))

    ind = {s: compute_indicators(df, params) for s, df in bars.items() if len(df) > 0}
    all_dates = sorted(set().union(*[set(df.index) for df in ind.values()])) if ind else []
    dates = [pd.Timestamp(d) for d in all_dates]
    if start is not None:
        dates = [d for d in dates if d >= pd.Timestamp(start)]
    if end is not None:
        dates = [d for d in dates if d <= pd.Timestamp(end)]
    for s in ind:
        ind[s].index = pd.DatetimeIndex(ind[s].index)
    if params.get("use_regime_filter", False):
        reg = None
        if benchmark is not None and len(benchmark):
            b = benchmark.copy()
            b.index = pd.DatetimeIndex(b.index)
            reg = market_regime(b, int(params.get("regime_sma", 200)))
        for s in ind:
            # regime at close t uses QQQ data <= t only; acted on at t+1 open like every other signal
            ind[s]["regime_ok"] = reg.reindex(ind[s].index) if reg is not None else np.nan

    cash = initial_capital
    positions: dict[str, dict] = {}
    pending: dict[str, dict] = {}
    last_trade: dict[str, int] = {}
    last_close: dict[str, float] = {}
    trades: list[dict] = []
    eq_points: list[tuple[pd.Timestamp, float]] = []
    prev_equity = initial_capital

    def close_position(sym, px_raw, d, reason):
        nonlocal cash
        pos = positions.pop(sym)
        px = px_raw * (1 - slip)
        cash += pos["qty"] * px - comm * pos["qty"]
        pnl = (px - pos["entry_price"]) * pos["qty"] - comm * pos["qty"] - pos["entry_comm"]
        trades.append({"symbol": sym, "entry_date": str(pos["entry_date"].date()), "exit_date": str(d.date()),
                       "qty": pos["qty"], "entry_price": round(pos["entry_price"], 4), "exit_price": round(px, 4),
                       "pnl": round(pnl, 2), "return_pct": px / pos["entry_price"] - 1, "exit_reason": reason})

    for i, d in enumerate(dates):
        today = {s: df.loc[d] for s, df in ind.items() if d in df.index}
        # 1) execute pending orders at today's open
        for sym, order in list(pending.items()):
            bar = today.get(sym)
            if bar is None:
                continue  # no trading for this symbol today; keep pending one more day
            pending.pop(sym)
            if order["side"] == "sell" and sym in positions:
                close_position(sym, bar["open"], d, order["reason"])
                last_trade[sym] = i
            elif order["side"] == "buy" and sym not in positions:
                px = bar["open"] * (1 + slip)
                qty = min(order["qty"], math.floor(cash / (px + comm))) if px > 0 else 0
                if qty >= 1:
                    cash -= qty * px + comm * qty
                    positions[sym] = {"qty": qty, "entry_price": px, "stop": order["stop"], "entry_date": d,
                                      "entry_comm": comm * qty}
                    last_trade[sym] = i
        # 2) protective stops intraday
        for sym in list(positions):
            bar = today.get(sym)
            if bar is None:
                continue
            stop = positions[sym]["stop"]
            if stop and bar["low"] <= stop:
                close_position(sym, min(bar["open"], stop), d, "Stop-loss")
                last_trade[sym] = i
                pending.pop(sym, None)
        # 3) mark to market
        for sym, bar in today.items():
            last_close[sym] = float(bar["close"])
        exposure = sum(p["qty"] * last_close.get(s, p["entry_price"]) for s, p in positions.items())
        equity = cash + exposure
        eq_points.append((d, equity))
        dd_blocked = prev_equity > 0 and (equity / prev_equity - 1) <= -lim["max_daily_drawdown_pct"]
        prev_equity = equity

        # 4) signals at today's close -> orders for tomorrow's open
        candidates = []
        for sym, bar in today.items():
            prob = probs.get((sym, d)) if probs is not None else None
            if use_ml and probs is None:
                prob = None
            dec = evaluate(bar.to_dict(), params, sym in positions, prob)
            if dec.action == "SELL" and sym in positions:
                pending[sym] = {"side": "sell", "reason": "; ".join(dec.reasons)}
            elif dec.action == "BUY" and sym not in positions and sym not in pending and not dd_blocked:
                if sym in last_trade and (i - last_trade[sym]) < cooldown_days:
                    continue
                candidates.append((rank_score(prob, bar.to_dict(), params), sym, bar))
        pending_value = 0.0
        for prob, sym, bar in sorted(candidates, key=lambda x: -x[0]):
            price, a = float(bar["close"]), float(bar["atr"])
            qty, _ = position_size(equity, cash, price, a, exposure, pending_value, params)
            if qty >= 1:
                pending[sym] = {"side": "buy", "qty": qty, "stop": price - stop_mult * a}
                pending_value += qty * price

    # close open positions at the last close for reporting (marked, flagged as open)
    for sym, pos in positions.items():
        px = last_close.get(sym, pos["entry_price"])
        trades.append({"symbol": sym, "entry_date": str(pos["entry_date"].date()), "exit_date": None,
                       "qty": pos["qty"], "entry_price": round(pos["entry_price"], 4), "exit_price": round(px, 4),
                       "pnl": round((px - pos["entry_price"]) * pos["qty"], 2),
                       "return_pct": px / pos["entry_price"] - 1, "exit_reason": "AÇIK (dönem sonu)"})

    equity_s = pd.Series({d: v for d, v in eq_points}, name="equity", dtype=float)
    closed = [t for t in trades if t["exit_date"]]
    metrics = {"full": {**equity_metrics(equity_s), **trade_stats(closed)}}
    if len(equity_s):
        metrics["full"]["final_equity"] = float(equity_s.iloc[-1])
        metrics["exposure_avg"] = None
    bench_s = None
    if benchmark is not None and len(equity_s):
        b = benchmark["close"].copy()
        b.index = pd.DatetimeIndex(b.index)
        b = b.reindex(equity_s.index).ffill().dropna()
        if len(b):
            bench_s = initial_capital * b / b.iloc[0] * (1 - slip)
            metrics["benchmark_qqq"] = equity_metrics(bench_s)
    if test_start is not None and len(equity_s):
        ts = pd.Timestamp(test_start)
        e_test = equity_s[equity_s.index >= ts]
        t_test = [t for t in closed if pd.Timestamp(t["entry_date"]) >= ts]
        metrics["test"] = {**equity_metrics(e_test), **trade_stats(t_test), "start": str(ts.date())}
        e_train = equity_s[equity_s.index < ts]
        t_train = [t for t in closed if pd.Timestamp(t["entry_date"]) < ts]
        metrics["train"] = {**equity_metrics(e_train), **trade_stats(t_train)}
        if bench_s is not None:
            metrics["benchmark_qqq_test"] = equity_metrics(bench_s[bench_s.index >= ts])
    metrics["assumptions"] = {"execution": "sonraki gün açılış", "slippage_bps": slip * 10_000,
                              "commission_per_share": comm, "stop_fill": "min(açılış, stop) - kayma"}
    return BacktestResult(equity_s, trades, metrics, bench_s)
