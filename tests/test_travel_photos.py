import io

from PIL import Image

from travel_photos import ai, geo
from travel_photos.pipeline import process_photo
from travel_photos.storage import PhotoStore


def _jpeg(color=(200, 100, 50), gps=None) -> bytes:
    img = Image.new("RGB", (64, 48), color)
    exif = Image.Exif()
    if gps:
        lat, lon = gps
        def dms(v):
            v = abs(v); d = int(v); m = int((v - d) * 60); s = (v - d - m / 60) * 3600
            return (d, m, round(s, 4))
        exif[geo.GPS_IFD] = {1: "N" if lat >= 0 else "S", 2: dms(lat),
                             3: "E" if lon >= 0 else "W", 4: dms(lon)}
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def _guess(**kw):
    base = dict(country="İtalya", country_code="it", city="Roma", landmark="Kolezyum",
                description="Kolezyum önünde bir fotoğraf.", confidence=0.9, reasoning="Kolezyum görünüyor.")
    base.update(kw)
    return ai.LocationGuess(**base)


def test_read_gps_metadata():
    meta = geo.read_metadata(geo.open_image(_jpeg(gps=(41.8902, -12.4922))))
    assert abs(meta.latitude - 41.8902) < 1e-3
    assert abs(meta.longitude + 12.4922) < 1e-3


def test_ai_path_when_no_gps(tmp_path):
    store = PhotoStore(tmp_path)
    res = process_photo(store, _jpeg(), "a.jpg", detect=lambda jpeg, hint=None: _guess(),
                        geocode=lambda *a: None)
    assert res.status == "eklendi"
    (p,) = store.all()
    assert (p.country, p.city, p.source) == ("İtalya", "Roma", "ai")
    assert store.image_path(p).exists()


def test_gps_path_overrides_ai(tmp_path):
    store = PhotoStore(tmp_path)
    res = process_photo(store, _jpeg(gps=(48.137, 11.575)), "b.jpg",
                        detect=lambda jpeg, hint=None: _guess(),
                        geocode=lambda lat, lon: geo.GeoResult("Almanya", "Münih", "DE"))
    assert res.status == "eklendi"
    (p,) = store.all()
    assert (p.country, p.city, p.source) == ("Almanya", "Münih", "gps")


def test_duplicate_and_ai_failure(tmp_path):
    store = PhotoStore(tmp_path)
    data = _jpeg()
    ok = lambda jpeg, hint=None: _guess()
    assert process_photo(store, data, "a.jpg", detect=ok).status == "eklendi"
    assert process_photo(store, data, "a.jpg", detect=ok).status == "zaten_var"

    def fail(jpeg, hint=None):
        raise ai.AIUnavailable("yok")
    assert process_photo(store, _jpeg((1, 2, 3)), "c.jpg", detect=fail).status == "hata"


def test_grouping_edit_delete(tmp_path):
    store = PhotoStore(tmp_path)
    process_photo(store, _jpeg((1, 1, 1)), "1.jpg", detect=lambda j, hint=None: _guess())
    process_photo(store, _jpeg((2, 2, 2)), "2.jpg", detect=lambda j, hint=None: _guess(city=None))
    g = store.grouped()
    assert set(g["İtalya"]) == {"Roma", "Diğer"}
    pid = g["İtalya"]["Diğer"][0].id
    store.update_location(pid, "İtalya", "Floransa")
    assert "Floransa" in store.grouped()["İtalya"]
    store.delete(pid)
    assert len(store.all()) == 1


def test_flag():
    assert ai.flag("TR") == "🇹🇷"
    assert ai.flag(None) == "🌍"
