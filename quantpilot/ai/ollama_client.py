"""Local Ollama (Qwen3 8B) client - ADVISORY ONLY.

The LLM receives already-computed numbers and returns explanatory text. Its output is
stored as text and is never parsed into, or used to influence, order decisions.
"""
from __future__ import annotations

import json
import logging
import re
import time

import requests

log = logging.getLogger("quantpilot.ai")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

SYSTEM_PROMPT = (
    "Sen deneyimli bir kantitatif analistsin. Sana verilen yapılandırılmış sayısal sinyalleri Türkçe, kısa ve "
    "dürüst biçimde yorumla. Rakam uydurma; yalnızca verilen veriyi kullan. Yatırım tavsiyesi verme. "
    "Karar zaten deterministik kurallar ve risk motoru tarafından verildi; sen yalnızca açıklarsın."
)


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:8b", timeout: int = 90,
                 session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.http = session or requests.Session()

    def health(self) -> dict:
        try:
            r = self.http.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            want = self.model if ":" in self.model else self.model + ":latest"
            has = any(n == want or n == self.model for n in names)
            if not has:
                return {"online": True, "model_available": False, "models": names,
                        "message": f"Ollama çalışıyor ancak '{self.model}' modeli yok. Çalıştırın: ollama pull {self.model}"}
            return {"online": True, "model_available": True, "models": names, "message": "Bağlı"}
        except Exception as exc:  # noqa: BLE001
            return {"online": False, "model_available": False, "models": [],
                    "message": f"Ollama'ya ulaşılamadı ({self.base_url}): {type(exc).__name__}"}

    def chat(self, user_prompt: str) -> tuple[bool, str, float]:
        t0 = time.monotonic()
        payload = {"model": self.model, "stream": False, "think": False,
                   "options": {"temperature": 0.2, "num_predict": 600},
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt}]}
        try:
            r = self.http.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            text = r.json().get("message", {}).get("content", "")
            text = _THINK_RE.sub("", text).strip()
            if not text:
                return False, "Model boş yanıt döndürdü.", time.monotonic() - t0
            return True, text, time.monotonic() - t0
        except Exception as exc:  # noqa: BLE001
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
