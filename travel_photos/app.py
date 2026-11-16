"""Gezi Albümü — Streamlit arayüzü.

Çalıştırma:  streamlit run travel_photos/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from travel_photos import ai  # noqa: E402
from travel_photos.pipeline import process_photo  # noqa: E402
from travel_photos.storage import PhotoStore  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="Gezi Albümü", page_icon="🌍", layout="wide")


@st.cache_resource
def get_store() -> PhotoStore:
    return PhotoStore(os.getenv("TRAVEL_PHOTOS_DIR") or Path(__file__).resolve().parent / "data")


store = get_store()

st.title("🌍 Gezi Albümü")
st.caption("Fotoğraflarını yükle; yapay zekâ çekildikleri ülke ve şehri bulup albümü kendisi düzenlesin.")

# ---------------------------------------------------------------- kenar çubuğu
with st.sidebar:
    st.header("📤 Fotoğraf Yükle")
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        st.warning("ANTHROPIC_API_KEY tanımlı değil. GPS'siz fotoğraflar için yapay zekâ gerekli; "
                   "anahtarı `.env` dosyasına ekleyin.")
    files = st.file_uploader("Fotoğraflar", type=["jpg", "jpeg", "png", "webp", "heic", "heif"],
                             accept_multiple_files=True)
    use_gps = st.toggle("Varsa GPS konumunu kullan", value=True,
                        help="Telefon fotoğraflarında konum bilgisi varsa en kesin sonuç budur.")
    ai_for_gps = st.toggle("GPS'li fotoğraflara da AI açıklaması ekle", value=True,
                           help="Kapatırsanız GPS'li fotoğraflar için yapay zekâ çağrılmaz (daha ucuz).")
    if st.button("🔍 Analiz et ve ekle", type="primary", disabled=not files, width="stretch"):
        progress = st.progress(0.0)
        for i, f in enumerate(files, start=1):
            with st.spinner(f"{f.name} inceleniyor..."):
                res = process_photo(store, f.getvalue(), f.name, use_gps=use_gps, ai_for_gps=ai_for_gps)
            {"eklendi": st.success, "zaten_var": st.info}.get(res.status, st.error)(
                f"**{f.name}** — {res.message}")
            progress.progress(i / len(files))

photos = store.all()
if not photos:
    st.info("Albüm henüz boş. Soldaki panelden fotoğraf yükleyerek başla. ✈️")
    st.stop()

# ---------------------------------------------------------------- özet
grouped = store.grouped()
cities = {(p.country, p.city) for p in photos if p.city}
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fotoğraf", len(photos))
c2.metric("Ülke", len(grouped))
c3.metric("Şehir", len(cities))
c4.metric("AI ile bulunan", sum(p.source == "ai" for p in photos))

tab_album, tab_map, tab_list = st.tabs(["📚 Albüm", "🗺️ Harita", "📋 Liste"])

# ---------------------------------------------------------------- albüm
with tab_album:
    countries = sorted(grouped, key=lambda c: (-sum(len(v) for v in grouped[c].values()), c))
    selected = st.multiselect("Ülke filtrele", countries, placeholder="Tüm ülkeler")
    for country in selected or countries:
        city_map = grouped[country]
        code = next((p.country_code for ps in city_map.values() for p in ps if p.country_code), None)
        total = sum(len(v) for v in city_map.values())
        st.subheader(f"{ai.flag(code)} {country} · {total} fotoğraf")
        for city in sorted(city_map, key=lambda c: (c == "Diğer", c)):
            items = city_map[city]
            with st.expander(f"🏙️ {city} ({len(items)})", expanded=len(city_map) <= 3):
                cols = st.columns(4)
                for idx, p in enumerate(items):
                    with cols[idx % 4]:
                        st.image(str(store.image_path(p)), width="stretch")
                        badge = {"gps": "📍 GPS", "ai": "🤖 AI", "manuel": "✏️ Manuel"}[p.source]
                        conf = f" · %{p.confidence * 100:.0f}" if p.source == "ai" and p.confidence is not None else ""
                        st.caption(f"{badge}{conf}" + (f" · {p.landmark}" if p.landmark else ""))
                        if p.description:
                            st.caption(p.description)
                        with st.popover("Düzenle", width="stretch"):
                            if p.reasoning:
                                st.caption(f"Gerekçe: {p.reasoning}")
                            if p.taken_at:
                                st.caption(f"Çekim: {p.taken_at.replace('T', ' ')}")
                            new_country = st.text_input("Ülke", p.country, key=f"c{p.id}")
                            new_city = st.text_input("Şehir", p.city or "", key=f"s{p.id}")
                            b1, b2 = st.columns(2)
                            if b1.button("Kaydet", key=f"save{p.id}") and new_country.strip():
                                store.update_location(p.id, new_country, new_city)
                                st.rerun()
                            if b2.button("Sil", key=f"del{p.id}"):
                                store.delete(p.id)
                                st.rerun()

# ---------------------------------------------------------------- harita
with tab_map:
    pts = pd.DataFrame([{"lat": p.latitude, "lon": p.longitude} for p in photos if p.latitude is not None])
    if pts.empty:
        st.info("Haritada göstermek için GPS bilgisi olan fotoğraf yok.")
    else:
        st.map(pts, size=2000)
        st.caption(f"{len(pts)} fotoğrafın GPS konumu gösteriliyor.")

# ---------------------------------------------------------------- liste
with tab_list:
    df = pd.DataFrame([{
        "Dosya": p.filename, "Ülke": p.country, "Şehir": p.city, "Yer": p.landmark,
        "Kaynak": p.source, "Güven": p.confidence, "Çekim tarihi": p.taken_at,
    } for p in photos])
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("CSV indir", df.to_csv(index=False).encode("utf-8-sig"),
                       "gezi_albumu.csv", "text/csv")
