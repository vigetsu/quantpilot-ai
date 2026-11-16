"""Fotoğraf EXIF verisinden GPS/tarih okuma ve koordinattan ülke/şehir bulma."""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import datetime

from PIL import ExifTags, Image, ImageOps

log = logging.getLogger(__name__)

try:  # iPhone HEIC fotoğrafları için (isteğe bağlı)
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass

GPS_IFD = 0x8825
EXIF_IFD = 0x8769


@dataclass
class PhotoMeta:
    latitude: float | None = None
    longitude: float | None = None
    taken_at: str | None = None  # ISO format
    camera: str | None = None


@dataclass
class GeoResult:
    country: str
    city: str | None
    country_code: str | None


def _to_degrees(value) -> float:
    d, m, s = (float(x) for x in value)
    return d + m / 60.0 + s / 3600.0


def read_metadata(image: Image.Image) -> PhotoMeta:
    meta = PhotoMeta()
    try:
        exif = image.getexif()
    except Exception:  # bozuk EXIF
        return meta
    if not exif:
        return meta

    try:
        gps = exif.get_ifd(GPS_IFD)
        if gps and 2 in gps and 4 in gps:
            lat = _to_degrees(gps[2])
            lon = _to_degrees(gps[4])
            if gps.get(1) in ("S", b"S"):
                lat = -lat
            if gps.get(3) in ("W", b"W"):
                lon = -lon
            if not (lat == 0 and lon == 0):
                meta.latitude, meta.longitude = round(lat, 6), round(lon, 6)
    except Exception as exc:
        log.debug("GPS okunamadı: %s", exc)

    try:
        sub = exif.get_ifd(EXIF_IFD)
        raw = sub.get(ExifTags.Base.DateTimeOriginal) or exif.get(ExifTags.Base.DateTime)
        if raw:
            meta.taken_at = datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S").isoformat()
    except Exception as exc:
        log.debug("Tarih okunamadı: %s", exc)

    make = exif.get(ExifTags.Base.Make)
    model = exif.get(ExifTags.Base.Model)
    if make or model:
        meta.camera = " ".join(str(x).strip() for x in (make, model) if x)
    return meta


def open_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def to_jpeg(image: Image.Image, max_side: int = 1568, quality: int = 85) -> bytes:
    """Yönü düzeltilmiş, küçültülmüş JPEG (AI'ye gönderim ve önizleme için)."""
    img = ImageOps.exif_transpose(image)
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def reverse_geocode(lat: float, lon: float, language: str = "tr") -> GeoResult | None:
    """OpenStreetMap Nominatim ile koordinat → ülke/şehir. İnternet yoksa None döner."""
    try:
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="gezi-albumu-app", timeout=10)
        loc = geolocator.reverse((lat, lon), language=language, zoom=10)
    except Exception as exc:
        log.warning("Ters coğrafi kodlama başarısız: %s", exc)
        return None
    if not loc:
        return None
    addr = loc.raw.get("address", {})
    country = addr.get("country")
    if not country:
        return None
    city = (
        addr.get("city")
        or addr.get("town")
        or addr.get("province")
        or addr.get("state")
        or addr.get("municipality")
        or addr.get("village")
        or addr.get("county")
    )
    code = (addr.get("country_code") or "").upper() or None
    return GeoResult(country=country, city=city, country_code=code)
