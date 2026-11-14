"""ML features/labels and leakage-free walk-forward validation.

Label at date t: return from the NEXT session's open (t+1, when we could actually trade)
to the close of t+h. Features at t use only data up to t's close. Training sets are
purged so no training label overlaps the test window (embargo = h sessions).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantpilot.strategy.indicators import compute_indicators

FEATURES = ["rsi", "macd_hist_n", "atr_n", "hv20", "vol_ratio", "obv_slope", "mom_10", "mom_20", "ret_1", "ret_5",
            "trend_strength", "dist_ema_fast", "ema_ratio", "high_52w_dist"]


def build_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    ind = compute_indicators(df, params)
    ind["macd_hist_n"] = ind["macd_hist"] / ind["close"]
    ind["atr_n"] = ind["atr"] / ind["close"]
    ind["ema_ratio"] = ind["ema_fast"] / ind["ema_slow"] - 1
    return ind


def add_label(ind: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = ind.copy()
    entry = out["open"].shift(-1)
    exit_ = out["close"].shift(-horizon)
    out["fwd_ret"] = exit_ / entry - 1
    out["label"] = np.where(out["fwd_ret"].notna(), (out["fwd_ret"] > 0).astype(float), np.nan)
    return out


def build_panel(bars: dict[str, pd.DataFrame], params: dict, with_label: bool = True) -> pd.DataFrame:
    frames = []
    h = int(params.get("label_horizon_days", 5))
    for sym, df in bars.items():
        if len(df) < 80:
            continue
        ind = build_features(df, params)
        if with_label:
            ind = add_label(ind, h)
        ind["symbol"] = sym
        frames.append(ind)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index().sort_values(["date", "symbol"]).reset_index(drop=True)
    return panel.replace([np.inf, -np.inf], np.nan)


def data_hash(panel: pd.DataFrame) -> str:
    cols = ["date", "symbol", "close", "volume"]
    return hashlib.sha256(pd.util.hash_pandas_object(panel[cols], index=False).values.tobytes()).hexdigest()[:16]


@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_folds(dates: pd.Series, n_folds: int = 5, test_size_days: int = 63, embargo_days: int = 5,
                       min_train_days: int = 252) -> list[Fold]:
    """Expanding-window, date-based folds. Train dates end ``embargo_days`` sessions before the test start."""
    uniq = np.array(sorted(pd.unique(dates)))
    folds: list[Fold] = []
    total_test = n_folds * test_size_days
    first_test = len(uniq) - total_test
    if first_test - embargo_days < min_train_days:
        n_folds = max(1, (len(uniq) - min_train_days - embargo_days) // test_size_days)
        first_test = len(uniq) - n_folds * test_size_days
    if first_test - embargo_days < 20:
        return []
    d = dates.values
    for k in range(n_folds):
        ts = first_test + k * test_size_days
        te = min(ts + test_size_days, len(uniq))
        train_last = uniq[ts - embargo_days - 1]
        test_first, test_last = uniq[ts], uniq[te - 1]
        tr = np.where(d <= train_last)[0]
        te_idx = np.where((d >= test_first) & (d <= test_last))[0]
        folds.append(Fold(tr, te_idx, pd.Timestamp(train_last), pd.Timestamp(test_first), pd.Timestamp(test_last)))
    return folds
