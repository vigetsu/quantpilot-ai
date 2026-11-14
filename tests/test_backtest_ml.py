import numpy as np
import pandas as pd

from conftest import synthetic_bars
from quantpilot.backtest.engine import run_backtest
from quantpilot.db import DEFAULT_PARAMS
from quantpilot.ml.features import FEATURES, build_panel, walk_forward_folds
from quantpilot.ml.model import ModelTrainer, Predictor

P = dict(DEFAULT_PARAMS, use_ml=False, cooldown_minutes=0)


def test_no_lookahead_in_backtest():
    """Altering future prices after date T must not alter equity or trades up to T."""
    b1 = {"AAPL": synthetic_bars(300, seed=1)}
    b2 = {"AAPL": b1["AAPL"].copy()}
    cut = b1["AAPL"].index[200]
    b2["AAPL"].loc[b2["AAPL"].index > cut, ["open", "high", "low", "close"]] *= 3.0
    r1 = run_backtest(b1, P, 100_000)
    r2 = run_backtest(b2, P, 100_000)
    pd.testing.assert_series_equal(r1.equity[r1.equity.index <= cut], r2.equity[r2.equity.index <= cut])


def test_slippage_and_commission_reduce_returns():
    bars = {"AAPL": synthetic_bars(300, seed=2, drift=0.002)}
    free = run_backtest(bars, dict(P, slippage_bps=0, commission_per_share=0), 100_000)
    costly = run_backtest(bars, dict(P, slippage_bps=30, commission_per_share=0.05), 100_000)
    if free.trades:
        assert costly.equity.iloc[-1] < free.equity.iloc[-1]


def test_gap_through_stop_fills_at_open():
    df = synthetic_bars(100, seed=3, drift=0.004)
    # force a crash gap on the last day
    df.iloc[-1, df.columns.get_loc("open")] = df["close"].iloc[-2] * 0.5
    df.iloc[-1, df.columns.get_loc("low")] = df["close"].iloc[-2] * 0.49
    df.iloc[-1, df.columns.get_loc("high")] = df["close"].iloc[-2] * 0.52
    df.iloc[-1, df.columns.get_loc("close")] = df["close"].iloc[-2] * 0.5
    res = run_backtest({"AAPL": df}, dict(P, rsi_entry_min=0, rsi_entry_max=100), 100_000)
    stops = [t for t in res.trades if t["exit_reason"] == "Stop-loss"]
    if stops:
        assert stops[-1]["exit_price"] <= df["open"].iloc[-1] + 1e-6


def test_walk_forward_folds_purged():
    bars = {"AAPL": synthetic_bars(500, seed=1), "MSFT": synthetic_bars(500, seed=2)}
    panel = build_panel(bars, DEFAULT_PARAMS).dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    h = int(DEFAULT_PARAMS["label_horizon_days"])
    folds = walk_forward_folds(panel["date"], n_folds=3, test_size_days=40, embargo_days=h)
    assert folds
    uniq = sorted(panel["date"].unique())
    pos = {d: i for i, d in enumerate(uniq)}
    for f in folds:
        # every training row's label uses prices up to date + h sessions -> must be strictly before test_start
        assert pos[f.train_end] + h < pos[f.test_start]
        assert set(f.train_idx).isdisjoint(set(f.test_idx))


def test_ml_reproducible_and_versioned(tmp_path, db):
    bars = {"AAPL": synthetic_bars(520, seed=10), "MSFT": synthetic_bars(520, seed=11)}
    t1 = ModelTrainer(tmp_path / "m1", db).train_and_register(bars, DEFAULT_PARAMS, kind="logreg")
    t2 = ModelTrainer(tmp_path / "m2").train_and_register(bars, DEFAULT_PARAMS, kind="logreg")
    assert abs(t1["metrics"]["oos_overall"]["brier"] - t2["metrics"]["oos_overall"]["brier"]) < 1e-12
    pred = Predictor(t1["path"])
    probs = pred.predict_latest(bars, DEFAULT_PARAMS)
    assert set(probs) == {"AAPL", "MSFT"}
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert db.active_model()["version"] == t1["version"]
