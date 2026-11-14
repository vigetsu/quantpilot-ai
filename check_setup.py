"""QuantPilot AI kurulum ve bağlantı doğrulama aracı (Türkçe yönlendirmeli)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

REQUIRED = ["alpaca", "pandas", "numpy", "sklearn", "streamlit", "plotly", "apscheduler", "dotenv", "pytest",
            "requests", "joblib", "pydantic"]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 72)
    print("QuantPilot AI — Sistem ve Bağımlılık Doğrulaması")
    print("=" * 72)
    v = sys.version_info
    print(f"[1/6] Python sürümü: {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) < (3, 10):
        print("  ❌ Python 3.10+ (önerilen: 3.12) gereklidir. Kurulum: winget install Python.Python.3.12")
        return 1
    if (v.major, v.minor) != (3, 12):
        print("  ℹ️  Hedef sürüm Python 3.12'dir (mevcut sürüm uyumlu modda çalışıyor). "
              "3.12 kurmak için: winget install Python.Python.3.12")

    missing = []
    for mod in REQUIRED:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[2/6] ❌ Eksik paketler: {', '.join(missing)} -> Çözüm: setup_windows.bat çalıştırın.")
        return 1
    print("[2/6] ✅ Tüm Python paketleri yüklü.")

    from quantpilot.config import ConfigError, load_settings
    from quantpilot.db import Database
    from quantpilot.risk.selftest import run_self_test

    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print("[3/6] ⚠️  .env dosyası yok. .env.example kopyalanıyor...")
        env_path.write_text((Path(__file__).parent / ".env.example").read_text(encoding="utf-8"), encoding="utf-8")

    try:
        settings = load_settings(require_credentials=False)
        print(f"[3/6] ✅ PAPER-ONLY koruması aktif ({settings.base_url}).")
    except ConfigError as exc:
        print(f"[3/6] ❌ Yapılandırma hatası: {exc}")
        return 1

    db = Database(settings.db_path)
    ok_st, msgs = run_self_test(db.get_params())
    print(f"[4/6] {'✅' if ok_st else '❌'} SQLite veritabanı ve risk motoru öz-testi ({len(msgs)} kontrol).")
    if not ok_st:
        return 1

    if not settings.has_credentials:
        print("[5/6] ⚠️  Alpaca Paper API anahtarları henüz girilmedi (.env -> ALPACA_API_KEY / ALPACA_SECRET_KEY).")
        print("      Gösterge paneli DEMO modunda açılabilir; gerçek paper işlem için anahtarlarınızı ekleyin.")
    else:
        try:
            from quantpilot.broker.alpaca_client import AlpacaBroker
            brk = AlpacaBroker(settings)
            acct = brk.get_account()
            clk = brk.get_clock()
            print(f"[5/6] ✅ Alpaca PAPER hesabı bağlı! Özsermaye=${acct['equity']:,.2f}, "
                  f"Nakit=${acct['cash']:,.2f}, Piyasa açık mı={clk['is_open']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[5/6] ❌ Alpaca bağlantısı başarısız: {exc}")
            print("      Çözüm: .env içindeki Paper API anahtarlarınızı (PK...) ve internet bağlantınızı kontrol edin.")

    from quantpilot.ai.ollama_client import OllamaClient
    h = OllamaClient(settings.ollama_url, settings.ollama_model, settings.ollama_timeout).health()
    if h["online"] and h["model_available"]:
        print(f"[6/6] ✅ Ollama ve '{settings.ollama_model}' modeli hazır.")
    else:
        print(f"[6/6] ⚠️  Ollama durumu: {h['message']}")
        print("      Not: Ollama isteğe bağlıdır. Kurmak için https://ollama.com adresinden indirip")
        print(f"      komut satırında `ollama pull {settings.ollama_model}` çalıştırın.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
