"""Environment configuration with a hard PAPER-ONLY guard.

Secrets are wrapped in :class:`Secret` so they never appear in repr/str/logs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
LIVE_TRADING_HOSTS = ("api.alpaca.markets",)  # anything without "paper-" is forbidden


class ConfigError(RuntimeError):
    """Raised when configuration is missing or unsafe."""


class Secret:
    """Holds a secret string; never reveals it via str()/repr()."""

    __slots__ = ("_value",)

    def __init__(self, value: str | None):
        self._value = (value or "").strip()

    def get(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def masked(self) -> str:
        if not self._value:
            return "<boş>"
        return self._value[:2] + "*" * 6 + self._value[-2:] if len(self._value) > 6 else "******"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "Secret(******)"

    __str__ = __repr__


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on", "evet")


@dataclass
class Settings:
    api_key: Secret
    secret_key: Secret
    paper: bool = True
    base_url: str = PAPER_TRADING_URL
    data_feed: str = "iex"              # live/latest feed (free plan: iex)
    historical_feed: str = "sip"        # historical bars (>15 min old) - falls back to iex
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "quantpilot.db")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_timeout: int = 90
    log_level: str = "INFO"
    initial_capital: float = 100_000.0

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key) and bool(self.secret_key)

    def secret_values(self) -> list[str]:
        return [s for s in (self.api_key.get(), self.secret_key.get()) if s]


def validate_paper_only(paper: bool, base_url: str, api_key: str) -> None:
    """Refuse any configuration that could reach a live brokerage endpoint."""
    if not paper:
        raise ConfigError(
            "ALPACA_PAPER=true olmalıdır. QuantPilot AI yalnızca PAPER (sanal) hesapta çalışır."
        )
    url = (base_url or "").lower().rstrip("/")
    if "paper-api.alpaca.markets" not in url:
        raise ConfigError(
            f"Geçersiz ALPACA_BASE_URL: yalnızca {PAPER_TRADING_URL} izinlidir. Canlı uç nokta reddedildi."
        )
    if api_key and api_key.upper().startswith("AK"):
        # Alpaca live keys start with 'AK', paper keys start with 'PK'.
        raise ConfigError(
            "Bu anahtar CANLI hesap anahtarına benziyor (AK...). Lütfen yalnızca Paper API anahtarı (PK...) kullanın."
        )


def load_settings(env_file: Path | None = None, require_credentials: bool = False) -> Settings:
    env_path = env_file or (PROJECT_ROOT / ".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)
    elif env_file is None and (PROJECT_ROOT / ".cloud_env").exists():
        load_dotenv(PROJECT_ROOT / ".cloud_env", override=False)

    def g(name: str, default: str = "") -> str:
        val = os.environ.get(name, "")
        if val:
            return val
        try:
            import streamlit as st
            if name in st.secrets:
                sval = str(st.secrets[name])
                os.environ[name] = sval
                return sval
        except Exception:
            pass
        return default

    settings = Settings(
        api_key=Secret(g("ALPACA_API_KEY")),
        secret_key=Secret(g("ALPACA_SECRET_KEY")),
        paper=_bool(g("ALPACA_PAPER"), True),
        base_url=g("ALPACA_BASE_URL", PAPER_TRADING_URL) or PAPER_TRADING_URL,
        data_feed=(g("ALPACA_DATA_FEED", "iex") or "iex").lower(),
        historical_feed=(g("ALPACA_HISTORICAL_FEED", "sip") or "sip").lower(),
        ollama_url=g("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=g("OLLAMA_MODEL", "qwen3:8b"),
        ollama_timeout=int(g("OLLAMA_TIMEOUT", "90") or 90),
        log_level=g("LOG_LEVEL", "INFO").upper(),
        initial_capital=float(g("INITIAL_CAPITAL", "100000") or 100000),
    )
    if g("QP_DB_PATH"):
        settings.db_path = Path(g("QP_DB_PATH"))
    if settings.data_feed not in ("iex", "sip"):
        raise ConfigError("ALPACA_DATA_FEED 'iex' veya 'sip' olmalıdır.")
    validate_paper_only(settings.paper, settings.base_url, settings.api_key.get())
    if require_credentials and not settings.has_credentials:
        raise ConfigError(
            "Alpaca Paper API anahtarları bulunamadı. .env dosyasına ALPACA_API_KEY ve ALPACA_SECRET_KEY ekleyin."
        )
    for d in (settings.db_path.parent, settings.log_dir, settings.model_dir):
        d.mkdir(parents=True, exist_ok=True)
    return settings
