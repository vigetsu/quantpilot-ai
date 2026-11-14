@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [HATA] Sanal ortam .venv bulunamadi. Once setup_windows.bat calistirin.
    pause
    exit /b 1
)

echo ============================================================================
echo QuantPilot AI - Arka Plan Paper Trading Worker  YALNIZCA PAPER
echo UYARI: Bu pencereyi veya Windows'u kapattiginizda otomatik islem durur!
echo        Ayni anda yalnizca tek bir worker calisabilir.
echo ============================================================================
".venv\Scripts\python.exe" -m quantpilot.worker
pause
