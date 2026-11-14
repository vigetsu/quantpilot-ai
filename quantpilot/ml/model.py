"""Reproducible, calibrated classifier for P(positive forward return) with walk-forward evaluation."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quantpilot.ml.features import FEATURES, Fold, build_panel, data_hash, walk_forward_folds

log = logging.getLogger("quantpilot.ml")
SEED = 42


def _base_model(kind: str = "hgb"):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=SEED))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
                                          l2_regularization=1.0, random_state=SEED)


def _inner_splits(dates: np.ndarray, embargo: int, n: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged chronological splits inside the training set (for calibration)."""
    uniq = np.array(sorted(pd.unique(dates)))
    size = len(uniq) // (n + 1)
    splits = []
    for k in range(1, n + 1):
        cut = k * size
        if cut - embargo <= 10 or cut >= len(uniq):
            continue
        tr_last = uniq[cut - embargo - 1]
        te_first = uniq[cut]
        te_last = uniq[min(cut + size, len(uniq)) - 1]
        tr = np.where(dates <= tr_last)[0]
        te = np.where((dates >= te_first) & (dates <= te_last))[0]
        if len(tr) > 50 and len(te) > 20:
            splits.append((tr, te))
    return splits


def fit_model(X: pd.DataFrame, y: pd.Series, dates: np.ndarray, embargo: int, kind: str = "hgb"):
    splits = _inner_splits(dates, embargo)
    if len(splits) >= 2:
        model = CalibratedClassifierCV(_base_model(kind), method="sigmoid", cv=splits)
    else:
        model = _base_model(kind)
    model.fit(X, y)
    return model


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = {"n": int(len(y)), "base_rate": float(np.mean(y)) if len(y) else None}
    if len(y) == 0:
        return out
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    out["accuracy"] = float(accuracy_score(y, p > 0.5))
    out["majority_baseline_acc"] = float(max(np.mean(y), 1 - np.mean(y)))
    out["brier"] = float(brier_score_loss(y, pc))
    out["log_loss"] = float(log_loss(y, pc, labels=[0, 1]))
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
    return out


class ModelTrainer:
    def __init__(self, model_dir: Path, db=None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.db = db

    def walk_forward(self, panel: pd.DataFrame, params: dict, kind: str = "hgb", n_folds: int = 5,
                     test_size_days: int = 63) -> tuple[dict, pd.DataFrame, list[Fold]]:
        h = int(params.get("label_horizon_days", 5))
        data = panel.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
        folds = walk_forward_folds(data["date"], n_folds=n_folds, test_size_days=test_size_days, embargo_days=h)
        if not folds:
            raise ValueError("Walk-forward için yeterli geçmiş veri yok")
        oos = []
        fold_metrics = []
        for f in folds:
            tr, te = data.iloc[f.train_idx], data.iloc[f.test_idx]
            m = fit_model(tr[FEATURES], tr["label"].astype(int), tr["date"].values, h, kind)
            p = m.predict_proba(te[FEATURES])[:, 1]
            oos.append(pd.DataFrame({"date": te["date"].values, "symbol": te["symbol"].values, "prob": p,
                                     "label": te["label"].values}))
            fm = _metrics(te["label"].values.astype(int), p)
            fm.update(train_end=str(f.train_end.date()), test_start=str(f.test_start.date()),
                      test_end=str(f.test_end.date()), n_train=int(len(tr)))
            fold_metrics.append(fm)
        oos_df = pd.concat(oos, ignore_index=True)
        overall = _metrics(oos_df["label"].values.astype(int), oos_df["prob"].values)
        # calibration table (reliability) on OOS predictions
        bins = pd.cut(oos_df["prob"], bins=[0, .3, .4, .45, .5, .55, .6, .7, 1.0], include_lowest=True).rename("bin")
        calib = oos_df.groupby(bins, observed=True).agg(mean_prob=("prob", "mean"), freq=("label", "mean"),
                                                         n=("label", "size")).reset_index()
        calib["bin"] = calib["bin"].astype(str)
        metrics = {"kind": kind, "oos_overall": overall, "folds": fold_metrics,
                   "calibration": calib[["bin", "mean_prob", "freq", "n"]].to_dict("records")}
        return metrics, oos_df, folds

    def train_and_register(self, bars: dict[str, pd.DataFrame], params: dict, kind: str = "hgb") -> dict:
        panel = build_panel(bars, params, with_label=True)
        if panel.empty:
            raise ValueError("Model eğitimi için veri yok")
        metrics, oos_df, _ = self.walk_forward(panel, params, kind)
        baseline_metrics = None
        if kind != "logreg":
            try:
                baseline_metrics = self.walk_forward(panel, params, "logreg")[0]["oos_overall"]
            except Exception as exc:  # noqa: BLE001
                log.warning("Baz model değerlendirilemedi: %s", exc)
        metrics["baseline_logreg_oos"] = baseline_metrics

        h = int(params.get("label_horizon_days", 5))
        data = panel.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
        final = fit_model(data[FEATURES], data["label"].astype(int), data["date"].values, h, kind)
        version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
        path = self.model_dir / f"{version}.joblib"
        meta = {"version": version, "features": FEATURES, "kind": kind, "horizon": h, "seed": SEED,
                "train_start": str(pd.Timestamp(data["date"].min()).date()),
                "train_end": str(pd.Timestamp(data["date"].max()).date()), "data_hash": data_hash(panel),
                "symbols": sorted(data["symbol"].unique().tolist()),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        joblib.dump({"model": final, "meta": meta}, path)
        (self.model_dir / f"{version}.json").write_text(json.dumps({"meta": meta, "metrics": metrics}, indent=2,
                                                                   default=str), encoding="utf-8")
        oos_df.to_csv(self.model_dir / f"{version}_oos.csv", index=False)
        if self.db is not None:
            self.db.register_model(version, str(path), {"kind": kind, "horizon": h, "features": FEATURES}, metrics,
                                   meta["train_start"], meta["train_end"], meta["data_hash"])
        log.info("Model eğitildi %s OOS AUC=%s", version, metrics["oos_overall"].get("roc_auc"))
        return {"version": version, "path": str(path), "metrics": metrics, "meta": meta}


class Predictor:
    def __init__(self, path: str | Path):
        obj = joblib.load(path)
        self.model = obj["model"]
        self.meta = obj["meta"]
        self.version = self.meta["version"]

    def predict_latest(self, bars: dict[str, pd.DataFrame], params: dict) -> dict[str, float]:
        """Probability for the latest completed bar of each symbol. NaN features => no prediction."""
        out = {}
        panel = build_panel(bars, params, with_label=False)
        if panel.empty:
            return out
        last = panel.sort_values("date").groupby("symbol").tail(1)
        for _, row in last.iterrows():
            x = row[self.meta["features"]]
            if x.isna().any():
                continue
            out[row["symbol"]] = float(self.model.predict_proba(pd.DataFrame([x.values], columns=self.meta["features"]))[0, 1])
        return out


def load_active_predictor(db) -> Predictor | None:
    rec = db.active_model()
    if not rec or not Path(rec["path"]).exists():
        return None
    try:
        return Predictor(rec["path"])
    except Exception as exc:  # noqa: BLE001
        log.error("Model yüklenemedi: %s", exc)
        return None
