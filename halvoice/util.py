"""Small shared helpers: logging setup and a state-event bus for the UI."""

from __future__ import annotations

import logging
import queue
import sys
from typing import Literal

from . import config

State = Literal["idle", "listening", "thinking", "speaking"]


def get_logger(name: str = "halvoice") -> logging.Logger:
    cfg = config.load()
    log = logging.getLogger("halvoice")
    if log.handlers:                      # already configured
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
        self.on_change = None   # called with the new state whenever it changes

    def set(self, state: State, level: float = 0.0) -> None:
        changed = state != self.state
        self.state = state
        self.level = level
        self._q.put((state, level))
        if changed and self.on_change:
            self.on_change(state)

    def drain(self) -> tuple[State, float]:
        last = (self.state, self.level)
        try:
            while True:
                last = self._q.get_nowait()  # type: ignore[assignment]
        except queue.Empty:
            pass
        return last  # type: ignore[return-value]
