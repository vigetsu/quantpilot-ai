import logging

import pytest

from quantpilot.config import ConfigError, Secret, load_settings, validate_paper_only
from quantpilot.logging_setup import RedactingFilter


def test_paper_flag_required():
    with pytest.raises(ConfigError):
        validate_paper_only(False, "https://paper-api.alpaca.markets", "PK123")


def test_live_url_rejected():
    with pytest.raises(ConfigError):
        validate_paper_only(True, "https://api.alpaca.markets", "PK123")


def test_live_key_rejected():
    with pytest.raises(ConfigError):
        validate_paper_only(True, "https://paper-api.alpaca.markets", "AKLIVEKEY123")


def test_paper_ok():
    validate_paper_only(True, "https://paper-api.alpaca.markets", "PKABC")


def test_secret_never_in_repr():
    s = Secret("supersecretvalue")
    assert "supersecret" not in repr(s) and "supersecret" not in str(s) and "supersecret" not in f"{s}"


def test_load_settings_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPACA_API_KEY", "PKXYZ")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "abc")
    monkeypatch.setenv("ALPACA_PAPER", "false")
    with pytest.raises(ConfigError):
        load_settings(env_file=tmp_path / "none.env")


def test_redaction():
    f = RedactingFilter(["mysecretvalue123"])
    rec = logging.LogRecord("x", logging.INFO, "", 0, "key=%s other PKABCDEFGHIJKLMNOP12", ("mysecretvalue123",), None)
    f.filter(rec)
    assert "mysecretvalue123" not in rec.getMessage()
    assert "PKABCDEFGHIJKLMNOP12" not in rec.getMessage()


def test_broker_requires_paper(settings, fake_client):
    from quantpilot.broker.alpaca_client import AlpacaBroker
    settings.paper = False
    with pytest.raises(ConfigError):
        AlpacaBroker(settings, trading_client=fake_client)
