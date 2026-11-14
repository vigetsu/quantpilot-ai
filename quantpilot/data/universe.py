"""Stock universe validation: must be an active, tradable, NASDAQ-listed US equity."""
from __future__ import annotations

import logging
import re

from quantpilot.db import Database

log = logging.getLogger("quantpilot.universe")
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")


def check_asset(asset) -> tuple[bool, str]:
    reasons = []
    if (asset.asset_class or "").lower() != "us_equity":
        reasons.append(f"varlık sınıfı {asset.asset_class}")
    if (asset.exchange or "").upper() != "NASDAQ":
        reasons.append(f"borsa {asset.exchange} (NASDAQ değil)")
    if (asset.status or "").lower() != "active":
        reasons.append(f"durum {asset.status}")
    if not asset.tradable:
        reasons.append("işlem yapılamaz")
    if reasons:
        return False, "Geçersiz: " + ", ".join(reasons)
    return True, f"Geçerli: NASDAQ, aktif, işlem yapılabilir ({asset.name or asset.symbol})"


def validate_watchlist(broker, db: Database) -> list[dict]:
    results = []
    for row in db.watchlist():
        sym = row["symbol"]
        if not SYMBOL_RE.match(sym):
            db.set_symbol_validation(sym, False, None, "Geçersiz sembol biçimi")
            continue
        try:
            asset = broker.get_asset(sym)
            ok, note = check_asset(asset)
            db.set_symbol_validation(sym, ok, asset.exchange, note)
        except Exception as exc:  # noqa: BLE001
            ok, note = False, f"Doğrulanamadı: {exc}"
            db.set_symbol_validation(sym, False, None, note)
        results.append({"symbol": sym, "valid": ok, "note": note})
    return results
