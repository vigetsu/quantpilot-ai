# 🌍 Gezi Albümü

Gezdiğin yerlerde çektiğin fotoğrafları yükle; uygulama her fotoğrafın **hangi ülke ve şehirde**
çekildiğini kendisi bulup albümü **Ülke → Şehir** olarak düzenlesin.

## Konum nasıl bulunuyor?

1. **GPS (en kesin):** Telefon fotoğraflarında konum bilgisi (EXIF GPS) varsa, koordinat
   OpenStreetMap ile ülke/şehre çevrilir.
2. **Yapay zekâ (Claude):** GPS yoksa fotoğraf Claude'a gönderilir; tabelalar, yapılar, mimari,
   doğa gibi ipuçlarından ülke ve şehri tahmin eder, güven oranı ve gerekçesini de yazar.
   GPS'li fotoğraflarda da isteğe bağlı olarak görünen simgeyi (ör. "Kolezyum") ve kısa açıklama ekler.

Yanlış tahmin olursa her fotoğrafın **Düzenle** menüsünden ülke/şehri elle düzeltebilirsin.

## Kurulum

```powershell
pip install -r requirements.txt
```

`.env` dosyasına Claude API anahtarını ekle (https://console.anthropic.com):

```ini
ANTHROPIC_API_KEY=sk-ant-...
```

## Çalıştırma

- Windows: `start_gezi_albumu.bat` dosyasına çift tıkla → http://localhost:8502
- veya: `streamlit run travel_photos/app.py`

## Özellikler

- Çoklu yükleme: JPG, PNG, WEBP ve iPhone **HEIC** fotoğrafları
- Ülke bayraklı albüm görünümü, ülke filtresi, şehir bazlı gruplar
- GPS'li fotoğrafların haritası
- Aynı fotoğraf ikinci kez eklenmez
- Liste görünümü ve CSV dışa aktarma
- Veriler yerelde: `travel_photos/data/` (fotoğraflar + `photos.db` SQLite)

## Notlar

- Her GPS'siz fotoğraf bir Claude API çağrısı yapar. Maliyeti düşürmek için kenar çubuğundaki
  "GPS'li fotoğraflara da AI açıklaması ekle" seçeneğini kapatabilirsin.
- Fotoğraflar AI'ye gönderilmeden önce en fazla 1568 px'e küçültülür.
