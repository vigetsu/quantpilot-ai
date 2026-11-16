"""Bir fotoğrafı işleyip konumunu bulan ve kaydeden ana akış."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import ai, geo
from .storage import PhotoStore


@dataclass
class ProcessResult:
    filename: str
    status: str            # 'eklendi', 'zaten_var', 'hata'
    message: str
    photo_id: int | None = None


def process_photo(store: PhotoStore, data: bytes, filename: str, *, use_gps: bool = True, ai_for_gps: bool = True,
                  detect: Callable[..., ai.LocationGuess] = ai.detect_location,
                  geocode: Callable[..., geo.GeoResult | None] = geo.reverse_geocode) -> ProcessResult:
    if store.exists(store.hash_bytes(data)):
        return ProcessResult(filename, "zaten_var", "Bu fotoğraf zaten albümde.")

    try:
        image = geo.open_image(data)
    except Exception:
        return ProcessResult(filename, "hata", "Dosya bir görsel olarak açılamadı.")

    meta = geo.read_metadata(image)
    jpeg = geo.to_jpeg(image)
    common = dict(original=data, jpeg=jpeg, filename=filename, latitude=meta.latitude,
                  longitude=meta.longitude, taken_at=meta.taken_at, camera=meta.camera)

    # 1) GPS varsa koordinattan kesin konum
    gps_result = None
    if use_gps and meta.latitude is not None:
        gps_result = geocode(meta.latitude, meta.longitude)

    # 2) Yapay zekâ: GPS yoksa konumu bulur; varsa sadece açıklama/simge ekler
    hint = None
    if gps_result:
        hint = f"GPS verisine göre fotoğraf {gps_result.city or ''} {gps_result.country} konumunda çekildi."
    guess = None
    if not gps_result or ai_for_gps:
        try:
            guess = detect(jpeg, hint=hint)
        except ai.AIUnavailable as exc:
            if not gps_result:
                return ProcessResult(filename, "hata", f"Konum bulunamadı: {exc}")

    if gps_result:
        pid = store.add(**common, country=gps_result.country, city=gps_result.city,
                        country_code=gps_result.country_code, source="gps", confidence=1.0,
                        landmark=guess.landmark if guess else None,
                        description=guess.description if guess else None,
                        reasoning="Fotoğrafın GPS koordinatlarından bulundu.")
        where = ", ".join(x for x in (gps_result.city, gps_result.country) if x)
        return ProcessResult(filename, "eklendi", f"📍 GPS: {where}", pid)

    pid = store.add(**common, country=guess.country, city=guess.city,
                    country_code=guess.country_code, landmark=guess.landmark,
                    description=guess.description, source="ai",
                    confidence=guess.confidence, reasoning=guess.reasoning)
    where = ", ".join(x for x in (guess.city, guess.country) if x)
    return ProcessResult(filename, "eklendi", f"🤖 AI: {where} (%{guess.confidence * 100:.0f} emin)", pid)
