# QuantPilot AI — Yerel AI Destekli NASDAQ Paper Trading Sistemi (Windows 11)

**QuantPilot AI**, NASDAQ hisselerini kantitatif göstergeler, sızıntısız (walk-forward) makine öğrenmesi ve yerel **Ollama (Qwen3 8B)** açıklamalarıyla analiz eden; **yalnızca gerçek Alpaca Paper Trading (sanal para)** hesabı üzerinden emir gönderen ve performansını **SQLite + Streamlit** arayüzünde izlemenizi sağlayan eksiksiz bir masaüstü uygulamasıdır.

> **Güvenlik Garantisi (PAPER ONLY):** Sistem yalnızca `https://paper-api.alpaca.markets` uç noktasına ve `TradingClient(..., paper=True)` moduna bağlanır. Canlı (live) URL veya `AK...` ile başlayan canlı anahtar algılanırsa uygulama başlamayı reddeder.

---

## 1. Hızlı Kurulum (Windows 11)

1. Proje klasöründe **`setup_windows.bat`** dosyasına çift tıklayın (veya PowerShell/CMD içinde çalıştırın).
   - Python 3.12 (veya uyumlu Python 3.10+) sanal ortamını (`.venv`) oluşturur.
   - Tüm paketleri yükler, `.env.example` dosyasından `.env` oluşturur, `check_setup.py` ve `pytest` testlerini çalıştırır.
   - Python 3.12 kurulu değilse kurmak için:
     ```powershell
     winget install Python.Python.3.12
     ```

2. **Gösterge panelini başlatmak için:** `start_dashboard.bat` → tarayıcıda `http://localhost:8501` açılır.
3. **Arka plan işlemcisini (worker) başlatmak için:** `start_bot.bat`.

---

## 2. Alpaca Paper Trading Hesabı Açma ve API Anahtarı Alma (Manuel Adımlar)

QuantPilot AI otomatik hesap oluşturmaz. Aşağıdaki adımları izleyerek ücretsiz sanal hesabınızı açın:

1. <https://alpaca.markets> adresine gidin ve **Sign Up** ile ücretsiz hesap oluşturun.
2. Giriş yaptıktan sonra sol üst/orta menüden **Paper Trading** görünümünde olduğunuzdan emin olun (sarı/mavi **Paper** etiketi görünmelidir).
3.Varsayılan sanal bakiyeyi **$100,000** yapmak için Paper Trading panelinde **Account → Reset Account** seçeneğinden bakiyeyi `100,000 USD` olarak ayarlayabilirsiniz.
4. **API Keys** kutusunda **Generate New Key** düğmesine tıklayın:
   - **`Key ID`** (`PK...` ile başlar) ve **`Secret Key`** değerlerini kopyalayın (Secret Key yalnızca bir kez gösterilir).
5. Proje klasöründeki `.env` dosyasını Not Defteri ile açın ve yapıştırın:
   ```ini
   ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxx
   ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ALPACA_PAPER=true
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ```
6. Doğrulamak için:
   ```powershell
   .\.venv\Scripts\python.exe check_setup.py
   ```
   `[5/6] ✅ Alpaca PAPER hesabı bağlı!` çıktısını görmelisiniz.

---

## 3. Yerel Ollama ve Qwen3 8B Kurulumu (İsteğe Bağlı)

Bot, Ollama olmadan da deterministik kurallar ve risk motoruyla güvenle çalışır (panelde **"AI Çevrimdışı"** uyarısı gösterilir). Yerel AI açıklamalarını etkinleştirmek için:

1. <https://ollama.com/download/windows> adresinden **Ollama for Windows** kurulumunu indirip çalıştırın (veya `winget install Ollama.Ollama`).
2. Yeni bir PowerShell penceresi açıp modeli indirin:
   ```powershell
   ollama pull qwen3:8b
   ```
3. Ollama arka planda `http://localhost:11434` üzerinde çalışacaktır. Gösterge panelindeki **Ollama** rozeti otomatik olarak **Aktif** durumuna geçer.

> **Yetki Sınırı:** LLM yanıtları yalnızca bilgilendirme/açıklama amaçlıdır. Hiçbir LLM çıktısı risk motorunu aşamaz veya doğrudan emir oluşturamaz.

---

## 4. Otomatik Paper Emirlerini Etkinleştirme Sırası

Güvenlik gereği otomatik emirler ilk kurulumda **KAPALI** gelir. Açabilmek için **Güvenlik Kapısı** koşulları sağlanmalıdır:

1. `.env` içine Alpaca Paper anahtarlarınızı girin.
2. `start_dashboard.bat` ile paneli açın → **🧪 Backtest & ML Eğitimi** sekmesinden:
   - **🚀 Backtest Çalıştır** düğmesine basın (en az 1 backtest kaydı zorunludur).
   - **🎓 ML Modelini Eğit ve Kaydet** düğmesine basın (varsayılan strateji ML olasılığı `> 0.55` ister).
3. **🎛️ Bot Kontrolleri** sekmesinde tüm güvenlik kapısı maddelerinin `✅` olduğunu görün, ardından **Otomatik Paper Emirlerini Etkinleştir** anahtarını açın ve **▶️ Botu Başlat** düğmesine basın.
4. `start_bot.bat` penceresini çalıştırın.

---

## 5. Önemli Operasyonel Uyarılar

- **Worker veya Windows Kapatıldığında İşlem Durur:** Gösterge panelini (tarayıcıyı) kapatmak `start_bot.bat` işlemcisini durdurmaz; ancak `start_bot.bat` penceresini kapatırsanız, bilgisayar uyku moduna girerse veya Windows kapanırsa yeni sinyal değerlendirmesi ve yeni emir gönderimi **durur** (Alpaca'ya önceden iletilmiş GTC stop emirleri borsada aktif kalmaya devam eder).
- **Stop-Loss Garantisi Yoktur:** Koruyucu stop emirleri tetiklendiğinde piyasa emrine dönüşür. Seans açılış boşluklarında (gap) veya sert fiyat hareketlerinde emir, stop fiyatının **altında** gerçekleşebilir.
- **IEX Veri Sınırlaması:** Ücretsiz Alpaca veri planı anlık fiyatları yalnızca **IEX** borsasından sağlar (ABD hacminin ~%2–3'ü). Tarihsel günlük çubuklar (>15 dk gecikmeli) konsolide **SIP** akışından (gerekirse IEX yedekli) alınır.

---

## 6. İsteğe Bağlı: Windows Görev Zamanlayıcı (Task Scheduler) ile Otomatik Başlatma

Worker'ın hafta içi her gün ABD seansı öncesinde (veya Windows oturumu açıldığında) otomatik başlaması için PowerShell'de şu komutu (kendi kullanıcı hesabınızla) çalıştırabilirsiniz:

```powershell
$Action = New-ScheduledTaskAction -Execute "C:\Users\emire\.gemini\antigravity\scratch\quantpilot_ai\.venv\Scripts\python.exe" -Argument "-m quantpilot.worker" -WorkingDirectory "C:\Users\emire\.gemini\antigravity\scratch\quantpilot_ai"
$Trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "QuantPilotAI_PaperWorker" -Action $Action -Trigger $Trigger -Description "QuantPilot AI Paper Trading Worker"
```

Görevi kaldırmak için:
```powershell
Unregister-ScheduledTask -TaskName "QuantPilotAI_PaperWorker" -Confirm:$false
```
