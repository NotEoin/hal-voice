"""Small shared helpers: logging setup and a state-event bus for the UI."""

from __future__ import annotations

import logging
import os
import queue
import ssl
import sys
from typing import Literal

from . import config

# System CA bundles, first match wins. A virtualenv Python can lack a default one.
CA_BUNDLES = (
    "/etc/ssl/certs/ca-certificates.crt",  # Debian, Ubuntu
    "/etc/pki/tls/certs/ca-bundle.crt",  # Fedora
    "/etc/ssl/cert.pem",  # Arch, Alpine
)

State = Literal["idle", "listening", "thinking", "speaking"]


def ca_bundle() -> str | None:
    """Sets `SSL_CERT_FILE` to a system CA bundle when Python's own verify paths do not exist.

    Returns the path it set, or None when nothing needed changing or no bundle was found.
    An `SSL_CERT_FILE` already in the environment is never replaced.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return None
    paths = ssl.get_default_verify_paths()
    if any(p and os.path.exists(p) for p in (paths.cafile, paths.capath)):
        return None
    for candidate in CA_BUNDLES:
        if os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return candidate
    return None


def get_logger(name: str = "halvoice") -> logging.Logger:
    """Returns the package logger, configured from the settings on first use."""
    cfg = config.load()
    log = logging.getLogger("halvoice")
    if log.handlers:  # already configured
        return logging.getLogger(name)
    level = getattr(logging, str(cfg.get("log.level", "INFO")).upper(), logging.INFO)
    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    target = cfg.get("log.file")
    handler: logging.Handler
    if target:
        handler = logging.FileHandler(config.resolve(target))
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    log.addHandler(handler)
    return logging.getLogger(name)


class StateBus:
    """Thread-safe broadcast of the current interaction state to the red-eye UI.

    The loop calls bus.set('listening'); the UI thread drains the queue to animate.
    Also carries a float 'level' (0..1) during speaking so the lens can pulse with audio.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[tuple[str, float]] = queue.Queue()
        self.state: State = "idle"
        self.level: float = 0.0
        self.on_change = None  # called with the new state whenever it changes

    def set(self, state: State, level: float = 0.0) -> None:
        """Publishes a new state and audio level."""
        changed = state != self.state
        self.state = state
        self.level = level
        self._q.put((state, level))
        if changed and self.on_change:
            self.on_change(state)

    def drain(self) -> tuple[State, float]:
        """Returns the newest queued state and level, or the current ones when none is queued."""
        last = (self.state, self.level)
        try:
            while True:
                last = self._q.get_nowait()  # type: ignore[assignment]  # only State values are queued
        except queue.Empty:
            pass
        return last  # type: ignore[return-value]  # same: the queue is typed with plain str
