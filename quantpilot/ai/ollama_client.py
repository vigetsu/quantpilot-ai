"""Local Ollama (Qwen3 8B) client with a deterministic Turkish TEMPLATE fallback.

- Advisory ONLY: receives already-computed numbers and returns explanatory Turkish text.
- When local Ollama (`localhost:11434`) is reachable and has `qwen3:8b`, it uses Ollama (real LLM).
- In cloud environments (e.g. Streamlit Community Cloud) or when Ollama is not running locally,
  it falls back to a deterministic text TEMPLATE filled with the same numbers. This is NOT an AI
  model, and it is labeled as such (``source_label() == BUILTIN_LABEL``) in the DB and dashboard.
- Output is stored strictly as text and can never create or alter an order decision.
"""
from __future__ import annotations

import json
import logging
import re
import time

import requests

log = logging.getLogger("quantpilot.ai")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
BUILTIN_LABEL = "şablon-özet (AI değil)"


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x  # excludes NaN

SYSTEM_PROMPT = (
    "Sen deneyimli bir kantitatif analistsin. Sana verilen yapılandırılmış sayısal sinyalleri Türkçe, kısa ve "
    "dürüst biçimde yorumla. Rakam uydurma; yalnızca verilen veriyi kullan. Yatırım tavsiyesi verme. "
    "Karar zaten deterministik kurallar ve risk motoru tarafından verildi; sen yalnızca açıklarsın."
)


def _builtin_analysis(user_prompt: str) -> str:
    """Synthesize a structured, truthful Turkish explanation strictly from the JSON numbers in the prompt."""
    try:
        idx = user_prompt.find("{")
        data = json.loads(user_prompt[idx:]) if idx != -1 else {}
    except Exception:
        data = {}

    if "hisse" in data:
        sym = data.get("hisse", "?")
        act = data.get("karar", "HOLD")
        prob = data.get("ml_olasiligi")
        ind = data.get("gostergeler") or {}
        reasons = data.get("kural_gerekceleri") or []
        conflicts = data.get("celisen_sinyaller") or []
        rnotes = data.get("risk_notlari") or []

        close = ind.get("close")
        ef, es = ind.get("ema_fast"), ind.get("ema_slow")
        rsi = ind.get("rsi")
        atr = ind.get("atr")
        hv = ind.get("hv20")
        mh = ind.get("macd_hist")

        if _num(ef) and _num(es):
            trend_txt = (f"hızlı EMA ({ef:.2f}) > yavaş EMA ({es:.2f}) ile yükseliş eğiliminde" if ef > es
                         else f"hızlı EMA ({ef:.2f}) <= yavaş EMA ({es:.2f}) ile baskı/düşüş eğiliminde")
        else:
            trend_txt = "trend göstergeleri hesaplanamadı"
        rsi_txt = f"RSI {rsi:.1f}" if _num(rsi) else "RSI hesaplanmadı"
        prob_txt = f"%{prob * 100:.1f}" if _num(prob) else "mevcut değil"
        atr_txt = (f"ATR ${atr:.2f} (günlük oynaklık ~%{(atr / close * 100):.1f})"
                   if (_num(atr) and _num(close) and close > 0) else "ATR hesaplanamadı")
        hv_txt = f", yıllıklandırılmış 20g tarihsel volatilite %{hv * 100:.1f}" if _num(hv) else ""
        close_txt = f"${close:.2f}" if _num(close) else "bilinmeyen fiyat"
        mh_txt = f"{mh:.4f}" if _num(mh) else "—"

        return (
            f"1) Piyasa yorumu: {sym} son kapanışta {close_txt} seviyesinde; {trend_txt}. "
            f"{rsi_txt}, MACD histogramı {mh_txt} ve ileri getiri ML olasılığı {prob_txt}.\n"
            f"2) Kararın gerekçesi ({act}): {'; '.join(reasons) if reasons else 'Standart kurallar değerlendirildi.'}\n"
            f"3) Riskler: {atr_txt}{hv_txt}. Stop-loss emirleri fiyat boşluklarında (gap) stop fiyatından dolmayabilir. "
            f"{('Risk motoru: ' + '; '.join(rnotes)) if rnotes else ''}\n"
            f"4) Çelişen sinyaller: {'; '.join(conflicts) if conflicts else 'Belirgin bir gösterge çelişkisi tespit edilmedi.'}"
        )

    if "ozsermaye" in data:
        eq = float(data.get("ozsermaye") or 0)
        cash = float(data.get("nakit") or 0)
        chg = float(data.get("gunluk_degisim") or data.get("bugun_kz") or 0)
        pos = data.get("pozisyonlar") or []
        sigs = data.get("sinyaller") or data.get("son_sinyaller") or []
        buys = [s.get("symbol") or s.get("hisse") for s in sigs if (s.get("action") or s.get("karar")) == "BUY"]
        sells = [s.get("symbol") or s.get("hisse") for s in sigs if (s.get("action") or s.get("karar")) == "SELL"]
        holds = [s.get("symbol") or s.get("hisse") for s in sigs if (s.get("action") or s.get("karar")) == "HOLD"]
        return (
            f"Portföy Özeti: Paper hesap özsermayesi ${eq:,.2f}, kullanılabilir nakit ${cash:,.2f} "
            f"(nakit oranı %{(cash / max(eq, 1) * 100):.1f}), günlük değişim ${chg:+,.2f}. "
            f"Açık pozisyon sayısı: {len(pos)}. "
            f"Son taramada AL (BUY) sinyali: {', '.join(buys) if buys else 'yok'}; "
            f"SAT (SELL) sinyali: {', '.join(sells) if sells else 'yok'}; "
            f"BEKLE (HOLD): {', '.join(holds) if holds else 'yok'}. "
            f"Risk Uyarıları: Hisse başı ≤%10 özsermaye, toplam ≤%50 maruziyet, işlem başı ≤%1 planlanan risk "
            f"ve ≤%3 günlük düşüş tavanları aktiftir; tüm emirler yalnızca PAPER hesabına iletilir."
        )

    return "Yapılandırılmış şablon özet üretildi."


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:8b", timeout: int = 90,
                 session: requests.Session | None = None, allow_builtin_fallback: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.http = session
        self.allow_builtin_fallback = allow_builtin_fallback
        self.last_source = model  # who produced the last chat() text: the model name or BUILTIN_LABEL

    def source_label(self) -> str:
        return self.last_source

    def _session(self) -> requests.Session:
        return self.http or requests.Session()

    def _builtin(self, user_prompt: str, t0: float) -> tuple[bool, str, float]:
        self.last_source = BUILTIN_LABEL
        try:
            return True, _builtin_analysis(user_prompt), time.monotonic() - t0
        except Exception as exc:  # noqa: BLE001
            log.warning("Şablon özet üretilemedi: %s", exc)
            return False, f"Şablon özet üretilemedi: {type(exc).__name__}", time.monotonic() - t0

    def health(self) -> dict:
        try:
            r = self._session().get(f"{self.base_url}/api/tags", timeout=2)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            want = self.model if ":" in self.model else self.model + ":latest"
            has = any(n == want or n == self.model for n in names)
            if has:
                return {"online": True, "model_available": True, "mode": "ollama", "models": names,
                        "message": f"Aktif (Ollama {self.model})"}
            if self.allow_builtin_fallback and self.http is None:
                return {"online": True, "model_available": True, "mode": "builtin", "models": names,
                        "message": f"Şablon özet modu (AI değil) • '{self.model}' yok: ollama pull {self.model}"}
            return {"online": True, "model_available": False, "mode": "ollama", "models": names,
                    "message": f"Ollama çalışıyor ancak '{self.model}' modeli yok. Çalıştırın: ollama pull {self.model}"}
        except Exception as exc:  # noqa: BLE001
            if self.allow_builtin_fallback and self.http is None:
                return {"online": True, "model_available": True, "mode": "builtin", "models": [],
                        "message": f"Şablon özet modu (AI değil) • Ollama'ya ulaşılamadı ({type(exc).__name__})"}
            return {"online": False, "model_available": False, "mode": "offline", "models": [],
                    "message": f"Ollama'ya ulaşılamadı ({self.base_url}): {type(exc).__name__}"}

    def chat(self, user_prompt: str) -> tuple[bool, str, float]:
        t0 = time.monotonic()
        payload = {"model": self.model, "stream": False, "think": False,
                   "options": {"temperature": 0.2, "num_predict": 600},
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt}]}
        try:
            r = self._session().post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            text = r.json().get("message", {}).get("content", "")
            text = _THINK_RE.sub("", text).strip()
            if text:
                self.last_source = self.model
                return True, text, time.monotonic() - t0
            if self.allow_builtin_fallback and self.http is None:
                return self._builtin(user_prompt, t0)
            self.last_source = self.model
            return False, "Model boş yanıt döndürdü.", time.monotonic() - t0
        except Exception as exc:  # noqa: BLE001
            if self.allow_builtin_fallback and self.http is None:
                return self._builtin(user_prompt, t0)
            self.last_source = self.model
            log.warning("Ollama isteği başarısız: %s", exc)
            return False, f"AI çevrimdışı: {type(exc).__name__}", time.monotonic() - t0


def signal_prompt(symbol: str, action: str, indicators: dict, ml_prob: float | None, reasons: list[str],
                  conflicts: list[str], risk_notes: list[str]) -> str:
    data = {"hisse": symbol, "karar": action, "ml_olasiligi": ml_prob,
            "gostergeler": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in indicators.items()},
            "kural_gerekceleri": reasons, "celisen_sinyaller": conflicts, "risk_notlari": risk_notes}
    return ("Aşağıdaki JSON verisini analiz et ve şu başlıklarla Türkçe yanıt ver (toplam en fazla 150 kelime):\n"
            "1) Piyasa yorumu\n2) Kararın gerekçesi\n3) Riskler\n4) Çelişen sinyaller\n\n"
            + json.dumps(data, ensure_ascii=False, default=str))


def portfolio_prompt(summary: dict) -> str:
    return ("Aşağıdaki paper trading portföy ve sinyal özetine göre 4-6 cümlelik Türkçe bir günlük özet yaz. "
            "Genel piyasa yorumunu, öne çıkan sinyalleri ve risk uyarılarını belirt:\n"
            + json.dumps(summary, ensure_ascii=False, default=str))
