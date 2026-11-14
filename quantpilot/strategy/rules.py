"""Deterministic long-only strategy rules (the source of BUY/SELL/HOLD decisions)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Decision:
    action: str                 # BUY | SELL | HOLD
    reasons: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def evaluate(row: dict, params: dict, has_position: bool, ml_prob: float | None) -> Decision:
    """row: latest COMPLETED bar with indicator columns."""
    ef, es, r = row.get("ema_fast"), row.get("ema_slow"), row.get("rsi")
    if not all(_ok(v) for v in (ef, es, r, row.get("atr"))):
        return Decision("HOLD", ["Yetersiz veri: göstergeler hesaplanamadı"])

    use_ml = bool(params.get("use_ml", True))
    trend_up = ef > es
    conflicts: list[str] = []
    if trend_up and row.get("macd_hist", 0) < 0:
        conflicts.append("Trend yukarı ancak MACD histogramı negatif (momentum zayıflıyor)")
    if not trend_up and row.get("macd_hist", 0) > 0:
        conflicts.append("Trend aşağı ancak MACD histogramı pozitif")
    if _ok(ml_prob) and trend_up and ml_prob < 0.5:
        conflicts.append(f"Trend yukarı ancak ML olasılığı düşük ({ml_prob:.2f})")
    if _ok(row.get("vol_ratio")) and row["vol_ratio"] < 0.6:
        conflicts.append("Hacim 20 günlük ortalamanın belirgin altında")

    if has_position:
        reasons = []
        if not trend_up:
            reasons.append(f"Trend dönüşü: EMA{params['ema_fast']} ({ef:.2f}) < EMA{params['ema_slow']} ({es:.2f})")
        exit_rsi = float(params.get("exit_rsi_max", 0) or 0)
        if exit_rsi > 0 and r > exit_rsi:
            reasons.append(f"RSI aşırı alım bölgesinde ({r:.1f} > {exit_rsi:.0f})")
        ml_exit = float(params.get("ml_exit_prob", 0) or 0)
        if use_ml and ml_exit > 0 and _ok(ml_prob) and ml_prob < ml_exit:
            reasons.append(f"ML olasılığı çıkış eşiğinin altında ({ml_prob:.2f} < {ml_exit:.2f})")
        if reasons:
            return Decision("SELL", reasons, conflicts)
        return Decision("HOLD", ["Pozisyon korunuyor: çıkış koşulu yok"], conflicts)

    reasons, fails = [], []
    (reasons if trend_up else fails).append(
        f"EMA{params['ema_fast']} {'>' if trend_up else '<='} EMA{params['ema_slow']} ({ef:.2f} / {es:.2f})")
    lo, hi = float(params["rsi_entry_min"]), float(params["rsi_entry_max"])
    rsi_ok = lo <= r <= hi
    (reasons if rsi_ok else fails).append(f"RSI {r:.1f} {'aralıkta' if rsi_ok else 'aralık dışında'} [{lo:.0f}-{hi:.0f}]")
    if use_ml:
        thr = float(params["ml_prob_threshold"])
        if not _ok(ml_prob):
            fails.append("Geçerli ML modeli/olasılığı yok - giriş engellendi")
        elif ml_prob > thr:
            reasons.append(f"ML olasılığı {ml_prob:.2f} > {thr:.2f}")
        else:
            fails.append(f"ML olasılığı {ml_prob:.2f} <= {thr:.2f}")
    if not fails:
        return Decision("BUY", reasons, conflicts)
    return Decision("HOLD", fails + reasons, conflicts)
