"""Claude görsel analizi ile fotoğrafın çekildiği ülke/şehri tahmin etme."""
from __future__ import annotations

import base64
import logging
import os

import anthropic
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

MODEL = os.getenv("TRAVEL_AI_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = """Sen bir seyahat fotoğrafı konum uzmanısın. Sana bir fotoğraf verilecek; \
fotoğrafın çekildiği ülkeyi ve şehri tespit et.

İpuçları: tanınmış yapılar ve simgeler, tabelalar ve yazı dili, plakalar, mimari tarz, \
bitki örtüsü, coğrafya, trafik yönü, kıyafetler, yemekler.

Kurallar:
- Ülke ve şehir adlarını Türkçe yaz (ör. "İtalya", "Roma", "Almanya", "Münih").
- Şehri kestiremiyorsan city alanını boş bırak (null); uydurma.
- Ülkeyi de kestiremiyorsan country alanına "Bilinmiyor" yaz ve confidence 0 olsun.
- confidence 0 ile 1 arasında, ne kadar emin olduğunu gösterir.
- description fotoğrafı 1 cümleyle Türkçe anlatsın.
- reasoning tahminine dayanak olan ipuçlarını kısaca açıklasın."""


class LocationGuess(BaseModel):
    country: str = Field(description="Ülke adı (Türkçe)")
    country_code: str | None = Field(description="ISO 3166-1 alpha-2 ülke kodu, ör. IT")
    city: str | None = Field(description="Şehir adı (Türkçe) veya bilinmiyorsa null")
    landmark: str | None = Field(description="Görünen önemli yer/yapı adı veya null")
    description: str = Field(description="Fotoğrafın tek cümlelik açıklaması")
    confidence: float = Field(description="0-1 arası güven")
    reasoning: str = Field(description="Kısa gerekçe")


class AIUnavailable(RuntimeError):
    pass


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def detect_location(jpeg_bytes: bytes, hint: str | None = None,
                    client: anthropic.Anthropic | None = None) -> LocationGuess:
    """Fotoğrafı Claude'a gönderip konum tahmini alır."""
    client = client or _client()
    text = "Bu fotoğraf hangi ülkede ve hangi şehirde çekilmiş?"
    if hint:
        text += f"\nEk bilgi: {hint}"

    try:
        response = client.beta.messages.parse(
            model=MODEL,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(jpeg_bytes).decode("ascii"),
                    }},
                    {"type": "text", "text": text},
                ],
            }],
            output_format=LocationGuess,
        )
    except anthropic.AuthenticationError as exc:
        raise AIUnavailable("Claude API anahtarı geçersiz veya tanımlı değil (ANTHROPIC_API_KEY).") from exc
    except anthropic.APIConnectionError as exc:
        raise AIUnavailable("Claude API'ye bağlanılamadı. İnternet bağlantınızı kontrol edin.") from exc
    except anthropic.RateLimitError as exc:
        raise AIUnavailable("Claude API istek sınırına ulaşıldı, biraz sonra tekrar deneyin.") from exc
    except anthropic.APIStatusError as exc:
        raise AIUnavailable(f"Claude API hatası ({exc.status_code}): {exc.message}") from exc

    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise AIUnavailable("Yapay zekâ bu fotoğraf için konum tahmini yapamadı.")

    guess = response.parsed_output
    guess.confidence = max(0.0, min(1.0, guess.confidence))
    if guess.country_code:
        guess.country_code = guess.country_code.upper()[:2]
    return guess


def flag(country_code: str | None) -> str:
    """'TR' → 🇹🇷"""
    if not country_code or len(country_code) != 2 or not country_code.isalpha():
        return "🌍"
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in country_code.upper())
