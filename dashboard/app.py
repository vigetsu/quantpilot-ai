"""QuantPilot AI — Streamlit Dashboard (Turkish UI, dark terminal theme).

Run with: start_dashboard.bat  (or: streamlit run dashboard/app.py)
The trading worker runs in a separate process (start_bot.bat) so closing this browser tab
does NOT stop trading. When no Alpaca keys are present in .env, the dashboard shows clearly
labeled DEMO data so you can explore the interface before connecting your paper account.
"""
from __future__ import annotations

import hmac
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.mock_data import demo_bars, demo_bundle  # noqa: E402
# NOTE: never importlib.reload() quantpilot modules here. Streamlit re-runs this script on every
# click; reloading embedded_worker reset its lock state and started an extra scheduler each time.
from quantpilot.ai.ollama_client import OllamaClient, portfolio_prompt, signal_prompt  # noqa: E402
from quantpilot.analytics.performance import (  # noqa: E402
    MIN_OBS_FOR_SHARPE,
    drawdown_series,
    equity_metrics,
    fifo_realized,
    trade_stats,
)
from quantpilot.backtest.runner import run_and_store  # noqa: E402
from quantpilot.broker.alpaca_client import AlpacaBroker  # noqa: E402
from quantpilot.broker.sync import OPEN_STATUSES, Synchronizer  # noqa: E402
from quantpilot.config import ConfigError, load_settings  # noqa: E402
from quantpilot.data.market_data import MarketData  # noqa: E402
from quantpilot.data.universe import validate_watchlist  # noqa: E402
from quantpilot.db import HARD_LIMITS, Database  # noqa: E402
from quantpilot.embedded_worker import ensure_embedded_worker  # noqa: E402
from quantpilot.engine import TradingEngine, safety_gate  # noqa: E402
from quantpilot.ml.model import ModelTrainer, load_active_predictor  # noqa: E402
from quantpilot.risk.selftest import run_self_test  # noqa: E402
from quantpilot.scheduler.calendar import get_session, market_status_text  # noqa: E402
from quantpilot.strategy.indicators import compute_indicators  # noqa: E402
import quantpilot as _qp_pkg  # noqa: E402

# Strategy keys added after the first cloud deploy. If the running process still has an older
# quantpilot.db module loaded (see stale-code check below), its DEFAULT_PARAMS lacks these keys;
# merging them here keeps the dashboard from crashing with KeyError until the app is rebooted.
_PARAM_FALLBACKS = {
    "use_ml": False, "use_regime_filter": False, "regime_sma": 200, "rank_by": "ml", "momentum_filter": False,
    "min_momentum": 0.0, "momentum_lookback": 126, "momentum_skip": 21, "exit_chandelier": False,
    "chandelier_period": 22, "chandelier_atr_mult": 3.0, "exit_on_ema_cross": True,
    "extended_hours_trading": True, "extended_limit_buffer_pct": 0.003, "extended_max_price_age_min": 15,
}


def load_params(database) -> dict:
    return {**_PARAM_FALLBACKS, **database.get_params()}


def _disk_fingerprint() -> str:
    """Same algorithm as quantpilot.code_fingerprint(), computed here because this script is always fresh."""
    import hashlib
    pkg = ROOT / "quantpilot"
    h = hashlib.sha256()
    for p in sorted(pkg.rglob("*.py")):
        h.update(str(p.relative_to(pkg)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


STALE_CODE = getattr(_qp_pkg, "LOADED_FINGERPRINT", None) != _disk_fingerprint()

st.set_page_config(page_title="QuantPilot AI — NASDAQ Paper Trader", page_icon="📈", layout="wide",
                   initial_sidebar_state="auto")  # auto: collapsed on phones, open on desktop
st.markdown(f"<style>{(Path(__file__).parent / 'theme.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)


# ------------------------------------------------------------------ formatting
# Chart series colors: first three slots of the categorical palette, dark-mode steps, validated
# against the chart surface #0f1626 (lightness band, chroma, CVD and contrast checks all pass).
SERIES = ["#3987e5", "#d95926", "#199e70"]
ACTION_LABEL = {"BUY": "🟢 AL", "SELL": "🔴 SAT", "HOLD": "⚪ BEKLE"}


def usd(x: float | None, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = "+" if signed and x > 0 else ("-" if x < 0 else "")
    return f"{s}${abs(x):,.2f}"


def pct(x: float | None, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = "+" if signed and x > 0 else ""
    return f"{s}{x * 100:.2f}%"


def cls(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)) or abs(x) < 1e-9:
        return "neu"
    return "pos" if x > 0 else "neg"


def ago(ts: str | None) -> str:
    """ISO timestamp -> short Turkish relative time ('3 dk önce'); '—' when missing/unparseable."""
    if not ts:
        return "—"
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        s = int((datetime.now(timezone.utc) - t).total_seconds())
    except Exception:
        return str(ts)
    if s < 0:
        return "az önce"
    if s < 60:
        return f"{s} sn önce"
    if s < 3600:
        return f"{s // 60} dk önce"
    if s < 86400:
        return f"{s // 3600} sa önce"
    return f"{s // 86400} gün önce"


def card(label: str, val: str, delta: str = "", c: str = "neu", tag: str = "") -> str:
    t = f"<span class='qp-tag'>{tag}</span>" if tag else ""
    d = f"<div class='qp-delta {c}'>{delta}</div>" if delta else ""
    return f"<div class='qp-card'><div class='qp-label'>{label}{t}</div><div class='qp-val {c}'>{val}</div>{d}</div>"


def dark_fig(height: int = 340) -> go.Figure:
    fig = go.Figure()
    # Title on its own line, legend on the line below it (they overlapped with t=28 / y=1.08).
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=16, r=16, t=64, b=24),
                      paper_bgcolor="#0f1626", plot_bgcolor="#0f1626",
                      font=dict(family="Inter, sans-serif", size=12, color="#cbd5e1"),
                      title=dict(y=0.97, yanchor="top", x=0.01, font=dict(size=14, color="#f1f5f9")),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0),
                      hoverlabel=dict(bgcolor="#0b111d", bordercolor="#334155", font=dict(color="#e5e7eb")))
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


# Streamlit re-runs the WHOLE script (all tabs) on every click. Cache slow Alpaca calls so a
# click does not trigger several broker/data API requests each time.
@st.cache_data(ttl=300, show_spinner=False)
def cached_portfolio_history(_broker, period: str = "6M") -> dict:
    return _broker.get_portfolio_history(period=period, timeframe="1D")


@st.cache_data(ttl=900, show_spinner=False)
def cached_daily_bars(_md, symbols: tuple[str, ...], days: int) -> dict:
    return _md.daily_bars(list(symbols), days=days)


@st.cache_data(ttl=60, show_spinner=False)
def cached_safety_gate(_db, _broker, params_json: str) -> tuple[bool, list[str]]:
    return safety_gate(_db, _broker, json.loads(params_json))


def worker_alive(db: Database) -> tuple[bool, str]:
    hb = db.get_state("worker_heartbeat")
    if not hb:
        return False, "Kapalı"
    try:
        dt = (datetime.now(timezone.utc) - datetime.fromisoformat(hb)).total_seconds()
        return (dt < 90), ("Aktif" if dt < 90 else f"Yanıt yok (son: {ago(hb)})")
    except Exception:
        return False, "Bilinmiyor"


def dispatch(db: Database, broker: AlpacaBroker | None, md: MarketData | None, ollama: OllamaClient,
             settings, cmd: str, payload: dict | None = None) -> str:
    """If the background worker is running, queue the command; otherwise execute immediately in-process.

    EMERGENCY STOP is always executed immediately (never only queued): it must not depend on a
    worker that may be stuck, asleep or missing its command loop.
    """
    if cmd == "emergency_stop" and broker:
        msg = TradingEngine(settings, db, broker, md, ollama).exec.emergency_stop(
            close_positions=bool((payload or {}).get("close_positions")))
        db.enqueue_command("refresh_gate")
        return msg
    alive, _ = worker_alive(db)
    if alive:
        db.enqueue_command(cmd, payload)
        return "Komut çalışan worker'a iletildi (1-3 sn içinde uygulanır)."
    if cmd == "start":
        db.set_state("bot_running", True)
        return "Bot durumu 'Başlatıldı' olarak kaydedildi. Otomatik döngü için start_bot.bat çalıştırın."
    if cmd == "pause":
        db.set_state("bot_running", False)
        return "Bot duraklatıldı."
    if cmd == "clear_emergency":
        db.set_state("emergency_stop", False)
        return "Acil durdurma kaldırıldı (otomatik emirler güvenlik için kapalı bırakıldı)."
    if not broker:
        return "Alpaca anahtarları yapılandırılmadı (.env)."
    eng = TradingEngine(settings, db, broker, md, ollama)
    if cmd == "emergency_stop":
        return eng.exec.emergency_stop(close_positions=bool((payload or {}).get("close_positions")))
    if cmd == "sync":
        Synchronizer(broker, db).sync_all()
        return "Alpaca paper hesabı senkronize edildi."
    if cmd == "validate_watchlist":
        validate_watchlist(broker, db)
        return "İzleme listesi NASDAQ/aktif/işlem yapılabilir kriterlerine göre doğrulandı."
    if cmd == "run_cycle":
        res = eng.run_cycle(force_eval=True)
        return f"Değerlendirme tamamlandı ({len(res.get('signals', []))} sinyal)."
    return "Komut işlendi."


# ------------------------------------------------------------------ bootstrap
try:
    settings = load_settings(require_credentials=False)
    config_err = None
except ConfigError as exc:
    st.error(f"Yapılandırma güvenlik hatası: {exc}")
    st.stop()

import os as _os
_dash_pw = _os.environ.get("DASHBOARD_PASSWORD", "")
if not _dash_pw:
    try:
        _dash_pw = str(st.secrets.get("DASHBOARD_PASSWORD", ""))
    except Exception:
        _dash_pw = ""
if _dash_pw and not st.session_state.get("pw_ok"):
    st.markdown("<div class='qp-topbar'><div class='qp-brand'>🔒 QuantPilot AI — Giriş Koruması</div></div>", unsafe_allow_html=True)
    _entered = st.text_input("Panel Şifresi", type="password")
    if st.button("Giriş Yap", type="primary"):
        if hmac.compare_digest(_entered.encode("utf-8"), _dash_pw.encode("utf-8")):
            st.session_state["pw_ok"] = True
            st.rerun()
        else:
            st.error("Hatalı şifre.")
    st.stop()

db = Database(settings.db_path)
ollama = OllamaClient(settings.ollama_url, settings.ollama_model, settings.ollama_timeout)
broker: AlpacaBroker | None = None
md: MarketData | None = None
alpaca_ok = False
alpaca_msg = "API anahtarları girilmedi (.env)"

if settings.has_credentials:
    try:
        broker = AlpacaBroker(settings)
        md = MarketData(settings)
        if "boot_synced" not in st.session_state:
            Synchronizer(broker, db).sync_all()
            if not db.watchlist(only_valid=True):
                validate_watchlist(broker, db)
            st.session_state["boot_synced"] = True
        ensure_embedded_worker(settings, db, broker, md, ollama)
        alpaca_ok = True
        alpaca_msg = "PAPER hesabına bağlı"
    except Exception as exc:  # noqa: BLE001
        alpaca_msg = f"Bağlantı hatası: {exc}"

demo_mode = not alpaca_ok
demo = demo_bundle() if demo_mode else None
session = get_session(broker) if broker else None
w_alive, w_text = worker_alive(db)
ol_health = ollama.health()

# ------------------------------------------------------------------ top bar
p_alp = f"<span class='qp-pill {'ok' if alpaca_ok else 'warn'}'>Alpaca: {'PAPER Bağlı' if alpaca_ok else 'DEMO Modu'}</span>"
_ai_mode = ol_health.get("mode")
if _ai_mode == "ollama" and ol_health["model_available"]:
    _ai_label, _ai_cls = f"Ollama ({settings.ollama_model}): Aktif", "ok"
elif _ai_mode == "builtin":
    _ai_label, _ai_cls = "Açıklamalar: Şablon Mod (AI değil)", "warn"
else:
    _ai_label, _ai_cls = f"Ollama ({settings.ollama_model}): Çevrimdışı", "warn"
p_ol = f"<span class='qp-pill {_ai_cls}'>{_ai_label}</span>"
_phase = session.phase if session else "closed"
_mkt_cls = {"regular": "ok", "pre": "warn", "post": "warn"}.get(_phase, "neu")
p_mkt = f"<span class='qp-pill {_mkt_cls}'>ABD Piyasası: {market_status_text(session) if session else 'DEMO'}</span>"
p_wrk = f"<span class='qp-pill {'ok' if w_alive else 'warn'}'>Worker: {w_text}</span>"
p_auto = f"<span class='qp-pill {'ok' if (db.flag('auto_trading') and not db.flag('emergency_stop')) else 'bad'}'>Oto. Paper Emir: {'AÇIK' if (db.flag('auto_trading') and not db.flag('emergency_stop')) else 'KAPALI'}</span>"

st.markdown(f"""
<div class="qp-topbar">
  <div>
    <div class="qp-brand">QuantPilot AI <span class="qp-pill paper">YALNIZCA PAPER TRADING</span></div>
    <div class="qp-sub">Yerel AI Destekli NASDAQ Kantitatif Algoritmik İşlem Terminali • Gerçek Para Riski Yoktur</div>
  </div>
  <div class="qp-pills">{p_alp}{p_ol}{p_mkt}{p_wrk}{p_auto}</div>
</div>
""", unsafe_allow_html=True)

if STALE_CODE:
    st.markdown("<div class='qp-banner stop'>🔄 <b>Uygulamanın yeniden başlatılması gerekiyor.</b> Yeni kod yüklendi, ancak "
                "arka planda çalışan bot ve bazı modüller hâlâ <b>eski sürümle</b> çalışıyor; yeni strateji ayarları "
                "uygulanmıyor ve bazı bölümler hata verebilir. share.streamlit.io → uygulamanın <b>⋮</b> menüsü → "
                "<b>Reboot app</b> ile yeniden başlatın.</div>", unsafe_allow_html=True)

if db.flag("emergency_stop"):
    st.markdown("<div class='qp-banner stop'>🛑 ACİL DURDURMA AKTİF — Tüm yeni emirler engellendi ve açık emirler iptal edildi. 'Bot Kontrolleri' sekmesinden kaldırabilirsiniz.</div>", unsafe_allow_html=True)

if demo_mode:
    st.markdown(f"""<div class='qp-banner demo'>
    ⚠️ <b>DEMO modu — rakamlar örnektir, gerçek değildir.</b> Alpaca Paper hesabına bağlanılamadı ({alpaca_msg}).
    Bağlamak için <code>ALPACA_API_KEY</code> ve <code>ALPACA_SECRET_KEY</code> değerlerini <code>.env</code> dosyasına
    veya Streamlit Secrets'a ekleyin.
    </div>""", unsafe_allow_html=True)

if alpaca_ok and not _dash_pw:
    st.markdown("<div class='qp-banner stop'>🔓 <b>Panel şifresiz:</b> Bu sayfanın adresini bilen herkes botu başlatıp durdurabilir, "
                "parametreleri değiştirebilir ve pozisyonları kapatabilir. Streamlit Cloud → App settings → Secrets bölümüne "
                "<code>DASHBOARD_PASSWORD = \"güçlü-bir-şifre\"</code> ekleyin.</div>", unsafe_allow_html=True)

_am_banner = db.active_model()
if _am_banner and load_params(db).get("use_ml", True):
    try:
        _auc = (json.loads(_am_banner["metrics"]).get("oos_overall") or {}).get("roc_auc")
    except Exception:
        _auc = None
    if _auc is not None and _auc < 0.52:
        st.markdown(f"<div class='qp-banner stop'>📉 <b>ML modeli tahmin gücü göstermiyor:</b> örneklem dışı ROC-AUC = {_auc:.3f} "
                    "(0,50 = yazı-tura). ML olasılık filtresi şu an alım kararlarına anlamlı bir bilgi katmıyor. "
                    "'Backtest & ML' sekmesinde ML'li ve ML'siz backtest sonuçlarını karşılaştırın.</div>", unsafe_allow_html=True)

with st.expander("ℹ️ Notlar ve sınırlamalar (açıklama modu, veri akışı, stop-loss)"):
    if _ai_mode == "builtin":
        st.markdown(f"**Açıklama modu — şablon (AI değil):** {ol_health['message']}. Sinyal açıklamaları bir yapay zekâ "
                    "tarafından değil, aynı sayılarla doldurulan sabit bir metin şablonuyla üretiliyor. "
                    "Strateji ve risk motoru bundan etkilenmez.")
    elif not (ol_health["online"] and ol_health["model_available"]):
        st.markdown(f"**Yerel AI (Ollama) çevrimdışı:** {ol_health['message']}. Strateji ve risk motoru AI olmadan çalışmaya devam eder.")
    else:
        st.markdown(f"**Açıklama modu:** Ollama `{settings.ollama_model}` aktif. Açıklamalar yalnızca danışma amaçlıdır; emir veremez.")
    st.markdown("**Veri akışı (IEX):** Ücretsiz Alpaca planı anlık veriyi yalnızca IEX borsasından verir (ABD hacminin ~%2–3'ü); "
                "anlık fiyatlar konsolide SIP akışından biraz farklı olabilir. Günlük çubuklar SIP'ten (>15 dk gecikmeli), "
                "olmazsa IEX'ten çekilir.")
    st.markdown("**Stop-loss:** Stop emri tetiklenince piyasa emrine dönüşür; fiyat boşluklarında (gap) stop fiyatından "
                "dolum **garanti edilmez**.")
    st.markdown("**Seans saatleri (Türkiye saati, kış):** seans öncesi 12:00–17:30 · normal seans 17:30–00:00 · "
                "seans sonrası 00:00–04:00 (yaz saatinde 1 saat erken). Seans dışında bot yalnızca limit emir verir, "
                "son işlem 15 dakikadan eskiyse emir vermez ve stop seviyesini kendisi kontrol eder. Strateji ve "
                "backtest günlük kapanış verisiyle kurulduğu için seans dışı işlemler backtest'e yansımaz.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚡ Hızlı Kontroller")
    c1, c2 = st.columns(2)
    if c1.button("🔄 Senkronize Et", width="stretch", disabled=demo_mode):
        st.toast(dispatch(db, broker, md, ollama, settings, "sync"))
        st.rerun()
    if c2.button("▶️ Şimdi Değerlendir", width="stretch", disabled=demo_mode):
        st.toast(dispatch(db, broker, md, ollama, settings, "run_cycle"))
        st.rerun()
    if st.button("🛑 ACİL DURDUR", type="primary", width="stretch", key="estop_sidebar"):
        st.toast(dispatch(db, broker, md, ollama, settings, "emergency_stop", {"close_positions": False}))
        st.rerun()
    st.markdown("### 🕒 Durum")
    st.markdown(
        f"<div class='qp-side-row'><span>Son senkronizasyon</span><b>{ago(db.get_state('last_sync'))}</b></div>"
        f"<div class='qp-side-row'><span>Son değerlendirme</span><b>{ago(db.get_state('last_evaluation'))}</b></div>"
        f"<div class='qp-side-row'><span>Başlangıç bakiyesi</span><b>${settings.initial_capital:,.0f}</b></div>",
        unsafe_allow_html=True)
    st.divider()
    st.caption("💡 Bilgisayarda çalışırken: `start_bot.bat` kapanırsa otomatik işlem durur. "
               "Bu sekmeyi kapatmak çalışan botu durdurmaz.")

# ------------------------------------------------------------------ data load
acct = demo["account"] if demo_mode else (db.latest_account() or {"equity": 0, "cash": 0, "buying_power": 0, "last_equity": 0})
positions = demo["positions"] if demo_mode else db.positions()
orders = demo["orders"] if demo_mode else db.orders(limit=500)
closed_trades, realized_by_order = fifo_realized(orders)
realized_pl = sum(t.pnl for t in closed_trades)
unrealized_pl = sum(float(p.get("unrealized_pl") or 0) for p in positions)
equity = float(acct.get("equity") or 0)
last_eq = float(acct.get("last_equity") or equity)
today_pl = equity - last_eq
today_pct = (today_pl / last_eq) if last_eq > 0 else 0.0
tot_ret = equity - settings.initial_capital
tot_pct = (tot_ret / settings.initial_capital) if settings.initial_capital > 0 else 0.0
tag = "DEMO" if demo_mode else "GERÇEK PAPER"

tabs = st.tabs([
    "📊 Portföy",
    "📈 Performans",
    "🧾 Emirler",
    "🤖 Sinyaller & Grafik",
    "🧪 Backtest & ML",
    "🎛️ Bot Kontrolü",
    "🖥️ Sistem",
])

# ======================================================= TAB 0: A + C
with tabs[0]:
    st.markdown("<div class='qp-section'>Portföy Genel Bakış</div>", unsafe_allow_html=True)
    cards = [
        card("Toplam Özsermaye (Equity)", usd(equity), f"Hedef başlangıç: {usd(settings.initial_capital)}", "neu", tag),
        card("Kullanılabilir Nakit", usd(acct.get("cash")), f"Maruziyet: {pct((equity - float(acct.get('cash') or 0)) / max(equity, 1))}", "neu", tag),
        card("Alım Gücü (Buying Power)", usd(acct.get("buying_power")), "Kaldıraç kullanılmaz (nakit sınırı)", "neu", tag),
        card("Toplam Getiri ($ / %)", usd(tot_ret, True), pct(tot_pct, True), cls(tot_ret), tag),
        card("Bugünkü K/Z (Günlük)", usd(today_pl, True), pct(today_pct, True), cls(today_pl), tag),
        card("Gerçekleşmiş K/Z (FIFO)", usd(realized_pl, True), f"{len(closed_trades)} kapanmış paper işlem", cls(realized_pl), "YALNIZCA DOLU EMİRLER"),
        card("Gerçekleşmemiş K/Z", usd(unrealized_pl, True), f"{len(positions)} açık pozisyon", cls(unrealized_pl), tag),
    ]
    st.markdown("<div class='qp-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)

    st.markdown("<div class='qp-section'>Açık Pozisyonlar</div>", unsafe_allow_html=True)
    if not positions:
        st.info("Açık pozisyon bulunmuyor.")
    else:
        rows = []
        for p in positions:
            mv = float(p.get("market_value") or 0)
            rows.append({
                "Hisse": p["symbol"],
                "Adet": p["qty"],
                "Ort. Giriş ($)": round(float(p["avg_entry_price"]), 2),
                "Güncel Fiyat ($)": round(float(p["current_price"]), 2),
                "Piyasa Değeri ($)": round(mv, 2),
                "Gerçekleşmemiş K/Z ($)": round(float(p["unrealized_pl"]), 2),
                "Gerçekleşmemiş K/Z (%)": round(float(p["unrealized_plpc"]) * 100, 2),
                "Portföy Payı (%)": round(mv / max(equity, 1) * 100, 2),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ======================================================= TAB 1: B
with tabs[1]:
    st.markdown("<div class='qp-section'>Performans Analitiği (Gerçek Paper Hesap — Backtest'ten Ayrı)</div>", unsafe_allow_html=True)
    eq_s = pd.Series(dtype=float)
    qqq_s = pd.Series(dtype=float)
    if demo_mode:
        eq_s, qqq_s = demo["equity_series"], demo["qqq_series"]
    else:
        if broker:
            try:
                ph = cached_portfolio_history(broker, "6M")
                pts = [(datetime.fromtimestamp(t, tz=timezone.utc).date(), e)
                       for t, e in zip(ph["timestamp"], ph["equity"]) if e and e > 0]
                if pts:
                    eq_s = pd.Series({ pd.Timestamp(d): float(v) for d, v in pts }).sort_index()
            except Exception:
                pass
        if len(eq_s) < 2:
            snaps = db.account_history()
            if snaps:
                df_s = pd.DataFrame(snaps)
                df_s["d"] = pd.to_datetime(df_s["ts"]).dt.tz_convert("America/New_York").dt.normalize()
                eq_s = df_s.groupby("d")["equity"].last()
        if md and len(eq_s) >= 2:
            try:
                qb = cached_daily_bars(md, ("QQQ",), max(40, (eq_s.index[-1] - eq_s.index[0]).days + 10)).get("QQQ")
                if qb is not None and not qb.empty:
                    q = qb["close"].reindex(eq_s.index).ffill().dropna()
                    if len(q):
                        qqq_s = eq_s.iloc[0] * q / q.iloc[0]
            except Exception:
                pass

    em = equity_metrics(eq_s)
    ts = trade_stats(closed_trades)
    pf_txt = "∞" if ts["profit_factor"] == float("inf") else (f"{ts['profit_factor']:.2f}" if ts["profit_factor"] is not None else "—")
    sh_txt = f"{em['sharpe']:.2f}" if em["sharpe"] is not None else f"Anlamlı değil (n={em['n_obs']}<{MIN_OBS_FOR_SHARPE})"
    m_cards = [
        card("Kümülatif Getiri", pct(em["total_return"], True), f"{em['n_obs']} günlük gözlem", cls(em["total_return"]), tag),
        card("Maksimum Düşüş (Max DD)", pct(em["max_drawdown"]), "Zirveden dibe", "neg" if em["max_drawdown"] else "neu", tag),
        card("Kazanma Oranı (Win Rate)", pct(ts["win_rate"]), f"{ts['n_trades']} kapanmış işlem", "neu", "FIFO"),
        card("Kâr Faktörü (Profit Factor)", pf_txt, f"Ort. işlem: {usd(ts['avg_trade'], True)}", cls(ts["total_pnl"]), "FIFO"),
        card("Sharpe Oranı (Yıllık)", sh_txt, f"Yıllık vol: {pct(em['volatility'])}", "neu", "≥30 gün şartı"),
    ]
    st.markdown("<div class='qp-grid'>" + "".join(m_cards) + "</div>", unsafe_allow_html=True)

    if len(eq_s) >= 2:
        fig = dark_fig(360)
        fig.add_trace(go.Scatter(x=eq_s.index, y=eq_s.values, name=f"Portföy ({tag})", line=dict(color="#38bdf8", width=2.4)))
        if len(qqq_s) >= 2:
            fig.add_trace(go.Scatter(x=qqq_s.index, y=qqq_s.values, name="QQQ Al-ve-Tut", line=dict(color="#a78bfa", width=1.8, dash="dot")))
        fig.update_layout(title="Portföy Özsermaye Eğrisi vs QQQ Al-ve-Tut (USD)", yaxis_title="USD")
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        dd = drawdown_series(eq_s) * 100
        f_dd = dark_fig(250)
        f_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="Drawdown %", line=dict(color="#f87171")))
        f_dd.update_layout(title="Düşüş (Drawdown %)")
        c1.plotly_chart(f_dd, width="stretch")

        rets = eq_s.pct_change().dropna() * 100
        f_r = dark_fig(250)
        colors = ["#34d399" if v >= 0 else "#f87171" for v in rets.values]
        f_r.add_trace(go.Bar(x=rets.index, y=rets.values, marker_color=colors, name="Günlük %"))
        f_r.update_layout(title="Günlük Getiriler (%)")
        c2.plotly_chart(f_r, width="stretch")
    else:
        st.info("Özsermaye eğrisi için henüz yeterli geçmiş gün birikmedi. Bot çalıştıkça günlük anlık görüntüler kaydedilir.")

# ======================================================= TAB 2: D
with tabs[2]:
    st.markdown("<div class='qp-section'>Emir ve İşlem Geçmişi (Dolu, Bekleyen ve Reddedilen Emirler Ayrı Gösterilir)</div>", unsafe_allow_html=True)
    st.caption("Gerçekleşmiş K/Z yalnızca **gerçekten dolmuş (filled / partially_filled)** paper emirlerinden FIFO ile hesaplanır; dolmamış sinyaller veya bekleyen emirler kâra dahil edilmez.")
    flt = st.radio("Filtre", ["Tümü", "Dolmuş (Filled)", "Açık / Bekleyen", "Reddedilen / İptal"], horizontal=True)
    rows = []
    for o in orders:
        st_o = o.get("status") or ""
        if flt == "Dolmuş (Filled)" and (o.get("filled_qty") or 0) <= 0:
            continue
        if flt == "Açık / Bekleyen" and st_o not in OPEN_STATUSES:
            continue
        if flt == "Reddedilen / İptal" and st_o not in ("rejected", "canceled", "expired"):
            continue
        r_pnl = realized_by_order.get(o["id"])
        rows.append({
            "Zaman": o.get("filled_at") or o.get("submitted_at"),
            "Hisse": o.get("symbol"),
            "Yön": (o.get("side") or "").upper(),
            "Tür": o.get("order_type"),
            "İstenen Adet": o.get("qty"),
            "Dolan Adet": o.get("filled_qty"),
            "İstenen/Ref Fiyat ($)": o.get("requested_price") or o.get("stop_price") or o.get("limit_price"),
            "Gerçek Dolum Fiyatı ($)": o.get("filled_avg_price"),
            "Durum": st_o,
            "Gerçekleşmiş K/Z ($)": round(r_pnl, 2) if r_pnl is not None else None,
            "Gerekçe": o.get("reason") or ("Harici / Manuel" if o.get("origin") != "quantpilot" else "—"),
            "Client Order ID": o.get("client_order_id"),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("Seçilen filtreye uygun emir yok.")

# ======================================================= TAB 3: E
with tabs[3]:
    st.markdown("<div class='qp-section'>Son Sinyaller</div>", unsafe_allow_html=True)
    sigs = db.latest_signals_per_symbol()
    sig_by_sym = {s["symbol"]: s for s in sigs}
    if sigs:
        n_buy = sum(s["action"] == "BUY" for s in sigs)
        n_sell = sum(s["action"] == "SELL" for s in sigs)
        n_hold = len(sigs) - n_buy - n_sell
        st.markdown("<div class='qp-grid'>"
                    + card("🟢 Al Sinyali", str(n_buy), "son taramada", "pos" if n_buy else "neu")
                    + card("🔴 Sat Sinyali", str(n_sell), "son taramada", "neg" if n_sell else "neu")
                    + card("⚪ Bekle", str(n_hold), "son taramada")
                    + card("Son Tarama", ago(max(s["ts"] for s in sigs)), f"çubuk: {max(s.get('bar_date') or '' for s in sigs)}")
                    + "</div>", unsafe_allow_html=True)
        srows = []
        for s in sigs:
            ind = json.loads(s.get("indicators") or "{}")
            srows.append({
                "Hisse": s["symbol"],
                "Karar": ACTION_LABEL.get(s["action"], s["action"]),
                "Fiyat ($)": round(float(s["price"]), 2) if s.get("price") else None,
                "RSI": round(ind["rsi"], 1) if ind.get("rsi") is not None else None,
                "Trend": ("↗ Yukarı" if ind["ema_fast"] > ind["ema_slow"] else "↘ Aşağı")
                         if ind.get("ema_fast") is not None and ind.get("ema_slow") is not None else "—",
                "ML Olasılığı": float(s["ml_prob"]) if s.get("ml_prob") is not None else None,
                "Risk": ("✅ Onay" if s["risk_approved"] else "⛔ Engellendi") if s["risk_approved"] is not None else "—",
            })
        st.dataframe(pd.DataFrame(srows), width="stretch", hide_index=True, column_config={
            "ML Olasılığı": st.column_config.ProgressColumn("ML Olasılığı", min_value=0.0, max_value=1.0, format="%.3f"),
        })
    else:
        st.info("Henüz kaydedilmiş sinyal yok. Sol menüden '▶️ Şimdi Değerlendir' ile ilk taramayı çalıştırabilirsiniz.")

    st.markdown("<div class='qp-section'>Hisse Grafiği ve Karar Detayı</div>", unsafe_allow_html=True)
    chart_syms = [w["symbol"] for w in db.watchlist(only_valid=not demo_mode)] or sorted(sig_by_sym)
    if chart_syms:
        g1, g2 = st.columns([1, 3])
        sel_sym = g1.selectbox("Hisse", chart_syms, key="chart_sym")
        span = g2.radio("Aralık", ["3 ay", "6 ay", "1 yıl"], index=1, horizontal=True, key="chart_span")
        n_bars = {"3 ay": 63, "6 ay": 126, "1 yıl": 252}[span]
        try:
            if demo_mode or not md:
                raw = demo_bars([sel_sym], n=420).get(sel_sym)
            else:
                raw = cached_daily_bars(md, (sel_sym,), 420).get(sel_sym)
        except Exception as exc:  # noqa: BLE001
            raw = None
            st.warning(f"Fiyat verisi alınamadı: {exc}")
        if raw is not None and len(raw) > 60:
            p_now = load_params(db)
            ind_df = compute_indicators(raw, p_now).tail(n_bars)
            ef, es = int(p_now["ema_fast"]), int(p_now["ema_slow"])
            f_px = dark_fig(360)
            f_px.add_trace(go.Scatter(x=ind_df.index, y=ind_df["close"], name="Kapanış", line=dict(color=SERIES[0], width=2)))
            f_px.add_trace(go.Scatter(x=ind_df.index, y=ind_df["ema_fast"], name=f"EMA {ef}", line=dict(color=SERIES[1], width=2)))
            f_px.add_trace(go.Scatter(x=ind_df.index, y=ind_df["ema_slow"], name=f"EMA {es}", line=dict(color=SERIES[2], width=2)))
            f_px.update_layout(title=f"{sel_sym} — günlük kapanış ve EMA'lar (USD){' • DEMO veri' if demo_mode else ''}",
                               hovermode="x unified", yaxis_tickprefix="$")
            st.plotly_chart(f_px, width="stretch")

            lo, hi = float(p_now["rsi_entry_min"]), float(p_now["rsi_entry_max"])
            f_rsi = dark_fig(200)
            f_rsi.add_hrect(y0=lo, y1=hi, fillcolor="rgba(148,163,184,0.12)", line_width=0,
                            annotation_text=f"giriş aralığı {lo:.0f}–{hi:.0f}", annotation_position="top left",
                            annotation_font_color="#94a3b8")
            for lvl in (30, 70):
                f_rsi.add_hline(y=lvl, line=dict(color="#475569", width=1, dash="dot"))
            f_rsi.add_trace(go.Scatter(x=ind_df.index, y=ind_df["rsi"], name="RSI", line=dict(color=SERIES[0], width=2),
                                       showlegend=False))
            f_rsi.update_layout(title="RSI (14)", yaxis_range=[0, 100], hovermode="x unified")
            st.plotly_chart(f_rsi, width="stretch")
        elif raw is not None:
            st.info("Grafik için yeterli geçmiş veri yok.")

        s_sel = sig_by_sym.get(sel_sym)
        if s_sel:
            reasons = json.loads(s_sel.get("reasons") or "[]")
            rnotes = json.loads(s_sel.get("risk_notes") or "[]")
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"**Karar: {ACTION_LABEL.get(s_sel['action'], s_sel['action'])}** · çubuk {s_sel.get('bar_date')} · "
                            f"model `{s_sel.get('model_version') or '—'}`")
                st.markdown("\n".join(f"- {r}" for r in reasons) or "_Gerekçe kaydı yok._")
            with d2:
                st.markdown("**Risk motoru notları**")
                st.markdown("\n".join(f"- {r}" for r in rnotes) or "_Risk kontrolü uygulanmadı (BEKLE kararı)._")
            if s_sel.get("ai_summary"):
                with st.expander("Açıklama metni"):
                    st.write(s_sel["ai_summary"])
        else:
            st.caption("Bu hisse için henüz sinyal kaydı yok.")

    st.markdown("<div class='qp-section'>Açıklamalar</div>", unsafe_allow_html=True)
    st.caption("🔒 Açıklamalar (Ollama veya şablon) yalnızca mevcut sayısal sinyalleri Türkçe anlatır; emir oluşturamaz, risk motorunu aşamaz.")

    col_ai1, col_ai2 = st.columns([1, 2])
    with col_ai1:
        st.markdown("**Yerel Ollama Bağlantısı**")
        st.write(f"URL: `{settings.ollama_url}` • Model: `{settings.ollama_model}`")
        st.write(f"Durum: **{ol_health['message']}**")
        if st.button("🧠 Portföy İçin Şimdi Özet Üret", disabled=not (ol_health["online"] and ol_health["model_available"])):
            with st.spinner("Qwen3 8B analiz ediyor..." if _ai_mode == "ollama" else "Şablon özet hazırlanıyor..."):
                summary = {"ozsermaye": equity, "nakit": acct.get("cash"), "bugun_kz": today_pl,
                           "pozisyonlar": positions, "son_sinyaller": [
                               {"hisse": s["symbol"], "karar": s["action"], "ml": s.get("ml_prob")} for s in sigs]}
                ok, txt, lat = ollama.chat(portfolio_prompt(summary))
                db.add_ai_analysis("portfolio", None, ollama.source_label(), ok, txt, lat)
                st.rerun()
    with col_ai2:
        st.markdown("**Son Açıklamalar (Danışma Amaçlı)**")
        # Show only successful outputs; failed rows stay in the DB for diagnostics (never deleted on view).
        ais = db.query("SELECT * FROM ai_analyses WHERE ok=1 ORDER BY id DESC LIMIT 8")
        if not ais and sigs and _ai_mode == "builtin":
            # Template mode is instant, so it is safe to fill in explanations during page render.
            # (A real Ollama call can take up to OLLAMA_TIMEOUT seconds per signal: never do that here.)
            for _s in sigs:
                _ind = json.loads(_s.get("indicators") or "{}")
                _rsn = json.loads(_s.get("reasons") or "[]")
                _rn = json.loads(_s.get("risk_notes") or "[]")
                _ok, _txt, _lat = ollama.chat(signal_prompt(_s["symbol"], _s["action"], _ind, _s.get("ml_prob"), _rsn, [], _rn))
                db.add_ai_analysis("signal", _s["symbol"], ollama.source_label(), _ok, _txt, _lat)
            _summary = {"ozsermaye": equity, "nakit": acct.get("cash"), "bugun_kz": today_pl,
                        "pozisyonlar": positions, "son_sinyaller": [
                            {"hisse": s["symbol"], "karar": s["action"], "ml": s.get("ml_prob")} for s in sigs]}
            _ok, _ptxt, _lat = ollama.chat(portfolio_prompt(_summary))
            db.add_ai_analysis("portfolio", None, ollama.source_label(), _ok, _ptxt, _lat)
            ais = db.query("SELECT * FROM ai_analyses WHERE ok=1 ORDER BY id DESC LIMIT 8")
        if not ais:
            st.caption("Henüz AI analizi kaydedilmedi.")
        for a in ais:
            scope_txt = "Portföy" if a["scope"] == "portfolio" else f"Sinyal · {a.get('symbol') or ''}"
            hdr = f"{scope_txt} — {ago(a['ts'])} · {a['model']}"
            with st.expander(hdr, expanded=(a == ais[0])):
                st.write(a["content"])

# ======================================================= TAB 4: Backtest & ML
with tabs[4]:
    st.markdown("<div class='qp-section'>🧪 Tarihsel Backtest ve Walk-Forward ML Model Eğitimi (SİMÜLASYON)</div>", unsafe_allow_html=True)
    st.caption("Backtest sonuçları **simülasyondur** ve gerçek Paper hesap metriklerinden tamamen ayrı tutulur. Emirler sinyal çubuğunun kapanışında değil, **ertesi günün açılışında** kayma (slippage) ve komisyon uygulanarak simüle edilir.")
    params = load_params(db)
    b1, b2, b3 = st.columns(3)
    days_bt = b1.slider("Tarihsel Veri Penceresi (gün)", 400, 1825, 1100, 50)
    use_ml_bt = b2.checkbox("Backtest'te Walk-Forward ML Olasılıklarını Kullan", value=bool(params["use_ml"]))
    kind_ml = b3.selectbox("ML Model Türü", ["hgb", "logreg"], format_func=lambda x: "HistGradientBoosting (Kalibre)" if x == "hgb" else "Lojistik Regresyon (Baz)")

    cb1, cb2 = st.columns(2)
    if cb1.button("🚀 Backtest Çalıştır", type="primary", width="stretch"):
        with st.spinner("Tarihsel veriler hazırlanıyor ve walk-forward backtest çalıştırılıyor..."):
            syms = [w["symbol"] for w in db.watchlist(only_valid=True)] or [w["symbol"] for w in db.watchlist()]
            if md:
                bars_bt = md.daily_bars(list(set(syms + ["QQQ"])), days=days_bt)
                feed_bt = f"alpaca:{md.last_feed_used}"
            else:
                bars_bt = demo_bars(list(set(syms + ["QQQ"])), n=min(days_bt, 650))
                feed_bt = "DEMO_SENTETİK"
            p_bt = dict(params, use_ml=use_ml_bt)
            run_id, _ = run_and_store(bars_bt, p_bt, db, settings.model_dir, settings.initial_capital, feed_bt)
            st.success(f"Backtest #{run_id} tamamlandı ({feed_bt}).")
            st.rerun()

    if cb2.button("🎓 ML Modelini Eğit ve Kaydet (Walk-Forward)", width="stretch"):
        with st.spinner("Walk-forward doğrulama ve kalibrasyon ile model eğitiliyor..."):
            syms = [w["symbol"] for w in db.watchlist(only_valid=True)] or [w["symbol"] for w in db.watchlist()]
            bars_tr = md.daily_bars(syms, days=days_bt) if md else demo_bars(syms, n=min(days_bt, 650))
            res_m = ModelTrainer(settings.model_dir, db).train_and_register(bars_tr, params, kind=kind_ml)
            st.success(f"Model {res_m['version']} eğitildi ve aktif edildi.")
            st.rerun()

    runs = db.backtests(limit=10)
    if runs:
        sel = st.selectbox("Kayıtlı Backtest Koşuları", runs,
                           format_func=lambda r: f"#{r['id']} • {r['ts']} ({json.loads(r['metrics']).get('data_feed', 'sim')})")
        m = json.loads(sel["metrics"])
        full = m.get("test") or m.get("full", {})
        qqq_m = m.get("benchmark_qqq_test") or m.get("benchmark_qqq") or {}
        st.caption(f"📌 **SİMÜLASYON** • {sel.get('notes') or ''} • Varsayımlar: `{json.dumps(m.get('assumptions', {}), ensure_ascii=False)}`")
        bc = [
            card("Backtest Getirisi", pct(full.get("total_return"), True), f"QQQ: {pct(qqq_m.get('total_return'), True)}", cls(full.get("total_return")), "SİMÜLASYON"),
            card("CAGR (Yıllık Bileşik)", pct(full.get("cagr"), True), f"QQQ CAGR: {pct(qqq_m.get('cagr'), True)}", cls(full.get("cagr")), "SİMÜLASYON"),
            card("Maksimum Düşüş", pct(full.get("max_drawdown")), f"QQQ DD: {pct(qqq_m.get('max_drawdown'))}", "neg", "SİMÜLASYON"),
            card("Sharpe Oranı", f"{full['sharpe']:.2f}" if full.get("sharpe") is not None else "Anlamlı değil", f"İşlem sayısı: {full.get('n_trades', 0)}", "neu", "SİMÜLASYON"),
            card("Kazanma Oranı / Kâr Faktörü", pct(full.get("win_rate")),
                 f"PF: {full['profit_factor']:.2f}" if isinstance(full.get("profit_factor"), (int, float)) and full["profit_factor"] != float("inf") else "PF: —", "neu", "SİMÜLASYON"),
        ]
        st.markdown("<div class='qp-grid'>" + "".join(bc) + "</div>", unsafe_allow_html=True)

        eq_pts = json.loads(sel["equity"])
        bm_pts = json.loads(sel["benchmark"])
        if eq_pts:
            f_bt = dark_fig(340)
            f_bt.add_trace(go.Scatter(x=[p[0] for p in eq_pts], y=[p[1] for p in eq_pts], name="Strateji (Simülasyon)", line=dict(color="#34d399", width=2.3)))
            if bm_pts:
                f_bt.add_trace(go.Scatter(x=[p[0] for p in bm_pts], y=[p[1] for p in bm_pts], name="QQQ Al-ve-Tut", line=dict(color="#a78bfa", width=1.8, dash="dot")))
            f_bt.update_layout(title="Backtest Özsermaye Eğrisi vs QQQ (SİMÜLASYON — Gerçek Paper İşlemi Değildir)", yaxis_title="USD")
            st.plotly_chart(f_bt, width="stretch")

        bt_trades = json.loads(sel["trades"])
        if bt_trades:
            with st.expander(f"Simüle Edilen İşlemler ({len(bt_trades)} adet)"):
                st.dataframe(pd.DataFrame(bt_trades), width="stretch", hide_index=True)

    st.markdown("<div class='qp-section'>Aktif Makine Öğrenmesi Modeli (Gerçek Walk-Forward Metrikleri)</div>", unsafe_allow_html=True)
    am = db.active_model()
    if am:
        mm = json.loads(am["metrics"])
        oos = mm.get("oos_overall", {})
        base_m = mm.get("baseline_logreg_oos") or {}
        st.write(f"**Sürüm:** `{am['version']}` • **Eğitim aralığı:** `{am['train_start']} → {am['train_end']}` • **Veri özeti (hash):** `{am['data_hash']}` • **Oluşturulma:** `{am['created_at']}`")
        mc = [
            card("OOS ROC-AUC", f"{oos['roc_auc']:.3f}" if oos.get("roc_auc") else "—", f"Baz (LogReg): {base_m.get('roc_auc', 0):.3f}" if base_m.get("roc_auc") else "", "neu", "WALK-FORWARD"),
            card("OOS Doğruluk (Accuracy)", pct(oos.get("accuracy")), f"Çoğunluk sınıfı bazı: {pct(oos.get('majority_baseline_acc'))}", "neu", "WALK-FORWARD"),
            card("Brier Skoru (Kalibrasyon)", f"{oos['brier']:.4f}" if oos.get("brier") else "—", "Düşük = daha iyi", "neu", "WALK-FORWARD"),
            card("Log Loss", f"{oos['log_loss']:.4f}" if oos.get("log_loss") else "—", f"Örneklem: {oos.get('n', 0)}", "neu", "WALK-FORWARD"),
        ]
        st.markdown("<div class='qp-grid'>" + "".join(mc) + "</div>", unsafe_allow_html=True)
        c_f, c_cal = st.columns(2)
        if mm.get("folds"):
            c_f.caption("Walk-Forward Katman (Fold) Sonuçları")
            c_f.dataframe(pd.DataFrame(mm["folds"]), width="stretch", hide_index=True)
        if mm.get("calibration"):
            c_cal.caption("Olasılık Kalibrasyon Tablosu (Tahmin vs Gerçekleşen Frekans)")
            c_cal.dataframe(pd.DataFrame(mm["calibration"]), width="stretch", hide_index=True)
    else:
        st.warning("Henüz eğitilmiş aktif ML modeli yok. Yukarıdaki '🎓 ML Modelini Eğit ve Kaydet' düğmesini kullanın.")

# ======================================================= TAB 5: F
with tabs[5]:
    st.markdown("<div class='qp-section'>Bot Kontrolleri, Güvenlik Kapısı, Risk Limitleri ve İzleme Listesi</div>", unsafe_allow_html=True)
    params = load_params(db)
    gate_ok, gate_msgs = (cached_safety_gate(db, broker, json.dumps(params, sort_keys=True, default=str)) if broker
                          else (False, ["❌ Alpaca Paper API bağlantısı yok (.env)"]))
    st.markdown("**Otomatik Paper Emir Ön Koşulları (Güvenlik Kapısı)**")
    for gm in gate_msgs:
        st.write(gm)

    k1, k2, k3, k4 = st.columns(4)
    if k1.button("▶️ Botu Başlat", width="stretch"):
        st.toast(dispatch(db, broker, md, ollama, settings, "start"))
        st.rerun()
    if k2.button("⏸️ Botu Duraklat", width="stretch"):
        st.toast(dispatch(db, broker, md, ollama, settings, "pause"))
        st.rerun()
    if k3.button("🛑 Acil Durdur (Emirleri İptal Et)", type="primary", width="stretch", key="estop_tab"):
        st.toast(dispatch(db, broker, md, ollama, settings, "emergency_stop", {"close_positions": False}))
        st.rerun()
    if db.flag("emergency_stop") and k4.button("🔓 Acil Durdurmayı Kaldır", width="stretch"):
        st.toast(dispatch(db, broker, md, ollama, settings, "clear_emergency"))
        st.rerun()

    with st.expander("⚠️ Tüm Açık Pozisyonları Piyasa Emriyle Kapat (Onay Gerektirir)"):
        confirm_liq = st.checkbox("Tüm açık paper pozisyonlarının piyasa emriyle kapatılmasını onaylıyorum.")
        if st.button("💥 Tüm Pozisyonları Kapat ve Durdur", disabled=not (confirm_liq and broker)):
            st.toast(dispatch(db, broker, md, ollama, settings, "emergency_stop", {"close_positions": True}))
            st.rerun()

    auto_now = db.flag("auto_trading")
    new_auto = st.toggle("Otomatik Paper Emirlerini Etkinleştir (YALNIZCA PAPER)", value=auto_now,
                         disabled=not gate_ok or db.flag("emergency_stop"))
    if new_auto != auto_now:
        db.set_state("auto_trading", new_auto)
        st.rerun()

    st.divider()
    st.markdown("**Strateji ve Risk Parametreleri** *(Üst tavanlar kilitlidir: hisse başı ≤%10, toplam maruziyet ≤%50, işlem başı risk ≤%1, günlük düşüş ≤%3)*")
    with st.form("params_form"):
        f1, f2, f3 = st.columns(3)
        ema_fast = f1.number_input("Hızlı EMA", 5, 100, int(params["ema_fast"]))
        ema_slow = f1.number_input("Yavaş EMA", 20, 250, int(params["ema_slow"]))
        rsi_min = f1.number_input("RSI Giriş Alt Sınır", 20.0, 70.0, float(params["rsi_entry_min"]), 1.0)
        rsi_max = f1.number_input("RSI Giriş Üst Sınır", 40.0, 85.0, float(params["rsi_entry_max"]), 1.0)
        use_ml = f2.checkbox("ML filtresi (önerilmez)", value=bool(params["use_ml"]), help="2017–2026 testlerinde modelin örneklem dışı ROC-AUC değeri 0,47–0,50 çıktı (tahmin gücü yok) ve ML filtresi getiriyi düşürdü.")
        ml_thr = f2.slider("Minimum ML Olasılığı", 0.50, 0.80, float(params["ml_prob_threshold"]), 0.01)
        atr_mult = f2.number_input("ATR Stop Çarpanı", 1.0, 5.0, float(params["atr_stop_mult"]), 0.25)
        place_stops = f2.checkbox("Dolum Sonrası Koruyucu Stop Emri Gönder", value=bool(params["place_stop_orders"]))
        max_pos = f3.slider("Hisse Başına Maks. Özsermaye (%)", 1, int(HARD_LIMITS["max_position_pct"] * 100), int(params["max_position_pct"] * 100)) / 100
        max_exp = f3.slider("Toplam Portföy Maruziyeti (%)", 5, int(HARD_LIMITS["max_exposure_pct"] * 100), int(params["max_exposure_pct"] * 100)) / 100
        max_rsk = f3.slider("İşlem Başına Maks. Planlanan Risk (%)", 0.2, HARD_LIMITS["max_risk_per_trade_pct"] * 100, float(params["max_risk_per_trade_pct"] * 100), 0.1) / 100
        max_dd = f3.slider("Günlük Düşüş Durdurma Limiti (%)", 0.5, HARD_LIMITS["max_daily_drawdown_pct"] * 100, float(params["max_daily_drawdown_pct"] * 100), 0.5) / 100
        cooldown = f3.number_input("İşlemler Arası Bekleme Süresi (dakika)", 0, 10080, int(params["cooldown_minutes"]), 60)

        st.markdown("**Ek kurallar (isteğe bağlı)** — 2017–2026 gerçek veri testlerinde karışık sonuç verdi: "
                    "geniş 44 hisselik listede ve düşüşlerde daha iyi, varsayılan 8 hisselik listede getiride daha zayıf. "
                    "Açmadan önce 'Backtest & ML' sekmesinde kendi listenizle karşılaştırın.")
        g1, g2, g3 = st.columns(3)
        use_regime = g1.checkbox("Piyasa rejimi filtresi", value=bool(params["use_regime_filter"]),
                                 help="QQQ 200 günlük ortalamasının altındayken yeni alım yapılmaz. Satışlar etkilenmez.")
        rank_mom = g1.checkbox("Momentuma göre sırala", value=params["rank_by"] == "momentum", help="Nakit/maruziyet sınırlıyken önce son 6 ayda (son ay hariç) en çok yükselen hisseler alınır.")
        mom_filter = g2.checkbox("Pozitif momentum şartı", value=bool(params["momentum_filter"]), help="Yalnızca 6 aylık (son ay hariç) getirisi pozitif hisseler alınır. Testlerde fayda göstermedi.")
        exit_ema = g2.checkbox("EMA kesişiminde sat", value=bool(params["exit_on_ema_cross"]))
        ext_on = g3.checkbox("Seans öncesi/sonrası işlem", value=bool(params["extended_hours_trading"]),
                             help="Seans öncesi (TSİ 12:00–17:30) ve sonrası (TSİ 00:00–04:00) da işlem yapar. "
                                  "Bu saatlerde yalnızca limit emir verilir (son fiyat ±%0,3), son işlem 15 dakikadan "
                                  "eskiyse emir verilmez; broker stop emirleri tetiklenmediği için stop seviyesi bot "
                                  "tarafından her döngüde kontrol edilir. Hacim düşük, fiyat farkları geniştir.")
        exit_ch = g3.checkbox("İz süren stop (Chandelier)", value=bool(params["exit_chandelier"]), help="Kapanış, son 22 günün en yüksek fiyatı − 3×ATR seviyesinin altına inerse ertesi gün satılır.")
        if st.form_submit_button("💾 Parametreleri Kaydet"):
            if not exit_ema and not exit_ch:
                st.error("En az bir satış kuralı (EMA kesişimi veya iz süren stop) açık olmalı.")
                st.stop()
            db.set_params({"ema_fast": ema_fast, "ema_slow": ema_slow, "rsi_entry_min": rsi_min, "rsi_entry_max": rsi_max,
                           "use_ml": use_ml, "ml_prob_threshold": ml_thr, "atr_stop_mult": atr_mult,
                           "place_stop_orders": place_stops, "max_position_pct": max_pos, "max_exposure_pct": max_exp,
                           "max_risk_per_trade_pct": max_rsk, "max_daily_drawdown_pct": max_dd, "cooldown_minutes": cooldown,
                           "use_regime_filter": use_regime, "rank_by": "momentum" if rank_mom else "ml",
                           "momentum_filter": mom_filter, "exit_on_ema_cross": exit_ema, "exit_chandelier": exit_ch,
                           "extended_hours_trading": ext_on})
            st.success("Parametreler kaydedildi.")
            st.rerun()

    st.divider()
    st.markdown("**İzleme Listesi (NASDAQ Doğrulamalı)**")
    w1, w2, w3 = st.columns([2, 1, 1])
    new_sym = w1.text_input("Sembol Ekle (örn. INTC, AVGO, QCOM)", placeholder="SEMBOL").upper().strip()
    if w2.button("➕ Ekle") and new_sym:
        db.add_symbol(new_sym)
        if broker:
            validate_watchlist(broker, db)
        st.rerun()
    if w3.button("🔍 Tümünü Doğrula", disabled=not broker):
        st.toast(dispatch(db, broker, md, ollama, settings, "validate_watchlist"))
        st.rerun()

    wl = db.watchlist()
    for row in wl:
        c_a, c_b, c_c, c_d = st.columns([1.2, 1.2, 5, 1])
        c_a.markdown(f"**{row['symbol']}**")
        c_b.markdown("✅ Geçerli" if row["is_valid"] else "⚠️ Doğrulanmadı")
        c_c.caption(row.get("note") or "Henüz Alpaca varlık üstverisi ile doğrulanmadı")
        if c_d.button("Sil", key=f"del_{row['symbol']}"):
            db.remove_symbol(row["symbol"])
            st.rerun()

# ======================================================= TAB 6: G
with tabs[6]:
    st.markdown("<div class='qp-section'>Sistem Durumu, Öz-Test ve Hata Kayıtları</div>", unsafe_allow_html=True)
    st_ok, st_msgs = run_self_test(load_params(db))
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Bağlantı ve Zamanlayıcı Durumu**")
        st.write({
            "Alpaca Bağlantısı": alpaca_msg,
            "Uç Nokta (PAPER)": settings.base_url,
            "Ollama Bağlantısı": ol_health["message"],
            "Piyasa Durumu": market_status_text(session) if session else "Bilinmiyor",
            "Son Veri Güncellemesi": db.get_state("last_data_update") or "—",
            "Son Strateji Değerlendirmesi": db.get_state("last_evaluation") or "—",
            "Son Başarılı Senkronizasyon": db.get_state("last_sync") or "—",
            "Aktif Veri Akışı": db.get_state("active_feed") or f"tarihsel={settings.historical_feed}, anlık={settings.data_feed}",
            "Arka Plan Worker": w_text,
        })
    with s2:
        st.markdown("**Risk Motoru Otomatik Öz-Testi**")
        for m in st_msgs:
            st.write(m)

    st.markdown("**Yapılandırılmış Uyarı ve Hata Kayıtları (SQLite + logs/worker.log)**")
    evs = db.recent_events(limit=100)
    if evs:
        st.dataframe(pd.DataFrame(evs), width="stretch", hide_index=True)
    else:
        st.caption("Kayıtlı uyarı veya hata bulunmuyor.")
