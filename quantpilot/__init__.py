"""QuantPilot AI - local, PAPER-ONLY NASDAQ quant trading system."""
from __future__ import annotations

import hashlib
from pathlib import Path

__version__ = "0.1.0"
_PKG = Path(__file__).resolve().parent


def code_fingerprint() -> str:
    """Hash of the quantpilot/*.py sources currently ON DISK."""
    h = hashlib.sha256()
    for p in sorted(_PKG.rglob("*.py")):
        h.update(str(p.relative_to(_PKG)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


# Fingerprint of the code that was actually IMPORTED into this process. Streamlit Cloud pulls new
# commits into the running process without re-importing modules (only the main script reruns), so
# a long-running process - including the embedded trading worker - can keep executing old code.
# The dashboard compares this with code_fingerprint() and asks for a reboot when they differ.
LOADED_FINGERPRINT = code_fingerprint()
