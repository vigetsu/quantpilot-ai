"""High-level backtest runner: fetch data, (optionally) build walk-forward OOS ML probabilities, simulate, store."""
from __future__ import annotations

import logging

import pandas as pd

from quantpilot.backtest.engine import BacktestResult, run_backtest
from quantpilot.ml.features import build_panel
from quantpilot.ml.model import ModelTrainer

log = logging.getLogger("quantpilot.backtest")


def oos_probabilities(bars: dict[str, pd.DataFrame], params: dict, model_dir, n_folds: int = 8,
                      test_size_days: int = 63) -> tuple[dict, pd.Timestamp, dict]:
    panel = build_panel(bars, params, with_label=True)
    trainer = ModelTrainer(model_dir)
    metrics, oos_df, folds = trainer.walk_forward(panel, params, "hgb", n_folds=n_folds, test_size_days=test_size_days)
    probs = {(r.symbol, pd.Timestamp(r.date)): float(r.prob) for r in oos_df.itertuples()}
    return probs, folds[0].test_start, metrics


def run_and_store(bars_all: dict[str, pd.DataFrame], params: dict, db, model_dir, initial_capital: float,
                  data_feed: str | None = None) -> tuple[int, BacktestResult]:
    bars = {k: v for k, v in bars_all.items() if k != "QQQ"}
    qqq = bars_all.get("QQQ")
    notes = []
    ml_metrics = None
    if params.get("use_ml", True):
        probs, start, ml_metrics = oos_probabilities(bars, params, model_dir)
        res = run_backtest(bars, params, initial_capital, probs=probs, start=start, benchmark=qqq)
        res.metrics["test"] = dict(res.metrics["full"], start=str(start.date()))
        notes.append(f"ML olasılıkları walk-forward örneklem dışı tahminlerdir; backtest {start.date()} "
                     "tarihinden başlar (tamamı test dönemi).")
    else:
        dates = sorted(set().union(*[set(df.index) for df in bars.values()]))
        start = pd.Timestamp(dates[min(60, len(dates) - 1)])
        test_start = pd.Timestamp(dates[int(len(dates) * 0.7)])
        res = run_backtest(bars, params, initial_capital, probs=None, start=start, benchmark=qqq,
                           test_start=test_start)
        notes.append(f"Kronolojik bölme: eğitim/geliştirme < {test_start.date()} <= test.")
    if ml_metrics:
        res.metrics["ml_walk_forward"] = ml_metrics["oos_overall"]
    if data_feed:
        res.metrics["data_feed"] = data_feed
    equity = [[str(d.date()), round(v, 2)] for d, v in res.equity.items()]
    bench = [[str(d.date()), round(v, 2)] for d, v in res.benchmark.items()] if res.benchmark is not None else []
    run_id = db.save_backtest(params, res.metrics, equity, res.trades, bench, " ".join(notes))
    log.info("Backtest #%s kaydedildi: getiri=%s", run_id, res.metrics["full"].get("total_return"))
    return run_id, res
