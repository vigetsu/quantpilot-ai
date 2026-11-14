@echo off
setlocal
cd /d "%~dp0"

echo ============================================================================
echo QuantPilot AI - Tek Tikla GitHub Bulut Kurulumu (7/24 Bilgisayar Kapaliyken)
echo ============================================================================

where gh >nul 2>&1
if errorlevel 1 (
    echo [1/4] GitHub CLI gh kuruluyor...
    winget install --id GitHub.cli -e --accept-source-agreements --accept-package-agreements
    set "PATH=%PATH%;C:\Program Files\GitHub CLI"
)

where gh >nul 2>&1
if errorlevel 1 (
    echo [HATA] GitHub CLI bulunamadi. Lutfen yeni bir pencere acip tekrar deneyin.
    pause
    exit /b 1
)

gh auth status >nul 2>&1
if errorlevel 1 (
    echo [2/4] GitHub hesabiniza giris yapin tarayici acilacak...
    gh auth login --web -p https -h github.com
    if errorlevel 1 (
        echo [HATA] GitHub girisi tamamlanamadi.
        pause
        exit /b 1
    )
)

echo [3/4] Gizli GitHub deposu olusturuluyor ve kodlar yukleniyor...
git config user.name >nul 2>&1 || git config user.name "QuantPilot User"
git config user.email >nul 2>&1 || git config user.email "quantpilot@users.noreply.github.com"
git add -A
git commit -m "QuantPilot AI cloud setup" >nul 2>&1

gh repo view quantpilot-ai >nul 2>&1
if errorlevel 1 (
    gh repo create quantpilot-ai --private --source=. --remote=origin --push
) else (
    git push -u origin HEAD
)

echo [4/4] .env icindeki Alpaca Paper API anahtarlari GitHub Secrets'a aktariliyor...
".venv\Scripts\python.exe" -c "from quantpilot.config import load_settings; import subprocess; s=load_settings(require_credentials=True); subprocess.run(['gh','secret','set','ALPACA_API_KEY','--body',s.api_key.get()],check=True); subprocess.run(['gh','secret','set','ALPACA_SECRET_KEY','--body',s.secret_key.get()],check=True); print('API anahtarlari guvenle GitHub Secrets icine kaydedildi.')"

echo.
echo Ilk bulut calismasi tetikleniyor...
gh workflow run paper_trader.yml -f force_eval=true

echo ============================================================================
echo TAMAMLANDI! Artik bilgisayariniz kapaliyken bile hafta ici her 15 dakikada
echo bir GitHub sunucularinda otomatik calisacak.
echo ============================================================================
pause
