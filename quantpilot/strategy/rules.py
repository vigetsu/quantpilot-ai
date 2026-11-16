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


def rank_score(ml_prob: float | None, row: dict, params: dict) -> float:
    """Priority of a BUY candidate when cash/exposure is limited (higher = first). Shared by live + backtest."""
    if params.get("rank_by", "ml") == "momentum":
        m = row.get("mom_rank")
        return float(m) if _ok(m) else -1e9
    return float(ml_prob) if _ok(ml_prob) else 0.0


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
        if not trend_up and params.get("exit_on_ema_cross", True):
            reasons.append(f"Trend dönüşü: EMA{params['ema_fast']} ({ef:.2f}) < EMA{params['ema_slow']} ({es:.2f})")
        ch, close = row.get("chandelier"), row.get("close")
        if params.get("exit_chandelier", False) and _ok(ch) and _ok(close) and close < ch:
            reasons.append(f"İz süren stop (Chandelier): kapanış {close:.2f} < {ch:.2f} "
                           f"(son {int(params.get('chandelier_period', 22))} günün zirvesi − "
                           f"{float(params.get('chandelier_atr_mult', 3.0)):g}×ATR)")
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
    if params.get("use_regime_filter", False):
        reg = row.get("regime_ok")
        if reg is True or reg == 1:
            reasons.append(f"Piyasa rejimi olumlu: QQQ > SMA{int(params.get('regime_sma', 200))}")
        elif reg is None or (isinstance(reg, float) and math.isnan(reg)):
            fails.append("Piyasa rejimi bilinmiyor (QQQ verisi yok) - giriş engellendi")
        else:
            fails.append(f"Piyasa rejimi olumsuz: QQQ < SMA{int(params.get('regime_sma', 200))} - yeni alım yok")
    if params.get("momentum_filter", False):
        mom, mmin = row.get("mom_rank"), float(params.get("min_momentum", 0.0))
        if not _ok(mom):
            fails.append("Momentum hesaplanamadı (yetersiz geçmiş)")
        elif mom > mmin:
            reasons.append(f"6 aylık momentum {mom:+.1%} > {mmin:+.0%}")
        else:
            fails.append(f"6 aylık momentum {mom:+.1%} <= {mmin:+.0%}")
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
