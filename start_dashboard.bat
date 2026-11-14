@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [HATA] Sanal ortam .venv bulunamadi. Once setup_windows.bat calistirin.
    pause
    exit /b 1
)

echo ============================================================================
echo QuantPilot AI - Gosterge Paneli  http://localhost:8501
echo Tarayicida otomatik acilacak. Kapatmak icin bu pencerede Ctrl+C basin.
echo Not: Bu pencereyi kapatmak arka planda calisan start_bot.bat islemcisini durdurmaz.
echo ============================================================================
".venv\Scripts\python.exe" -m streamlit run dashboard\app.py --server.address localhost --server.port 8501 --browser.gatherUsageStats false
pause
