@echo off
setlocal
cd /d "%~dp0"

echo ============================================================================
echo QuantPilot AI - Windows 11 Kurulum Betigi
echo ============================================================================

set PY_CMD=
py -3.12 -V >nul 2>&1
if not errorlevel 1 set PY_CMD=py -3.12

if not defined PY_CMD (
    py -3.10 -V >nul 2>&1
    if not errorlevel 1 set PY_CMD=py -3.10
)

if not defined PY_CMD (
    python -V >nul 2>&1
    if not errorlevel 1 set PY_CMD=python
)

if not defined PY_CMD (
    echo [HATA] Python bulunamadi! Lutfen Python 3.12 kurun:
    echo        winget install Python.Python.3.12
    pause
    exit /b 1
)

echo [BILGI] Kullanilan Python komutu: %PY_CMD%

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Sanal ortam olusturuluyor...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [HATA] Sanal ortam olusturulamadi.
        pause
        exit /b 1
    )
)

echo [2/4] Paketler yukleniyor / guncelleniyor...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 (
    echo [HATA] Paket kurulumu basarisiz oldu. Internet baglantinizi kontrol edin.
    pause
    exit /b 1
)

if not exist ".env" (
    echo [3/4] .env dosyasi .env.example sablonundan olusturuluyor...
    copy ".env.example" ".env" >nul
)

echo [4/4] Sistem dogrulamasi ve otomatik testler calistiriliyor...
".venv\Scripts\python.exe" check_setup.py
".venv\Scripts\pytest.exe" -q

echo.
echo ============================================================================
echo Kurulum tamamlandi!
echo 1 - .env dosyasini acip ALPACA_API_KEY ve ALPACA_SECRET_KEY Paper girin.
echo 2 - Gosterge panelini baslatmak icin: start_dashboard.bat
echo 3 - Otomatik islem worker'ini baslatmak icin: start_bot.bat
echo ============================================================================
pause
