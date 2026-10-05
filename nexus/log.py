"""
log.py
Logging for NEXUS: warnings to stderr, everything at NEXUS_LOG_LEVEL to a
rotating file under the data directory.

Library modules only call `logging.getLogger(__name__)`; entry points (the
launcher, the UIs, the CLI) call `setup()` once.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from nexus import config

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_configured = False


def setup(console_level: int = logging.WARNING) -> None:
    """Idempotent: Streamlit re-runs the app script on every interaction."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("nexus")
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    root.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("[nexus] %(levelname)s %(message)s"))
    root.addHandler(console)

    try:
        config.APP_LOG.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            config.APP_LOG, maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8"
        )
    except OSError as exc:
        root.warning("file logging disabled (%s)", exc)
        return
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
