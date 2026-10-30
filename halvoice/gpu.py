"""gpu-light claims for the voice loop: one for the resident set, one for each turn.

Follows gpu-light v2's "hal-voice" recipe. gpu-light is an optional import: without it, or
without a reachable daemon, every method is a no-op and the loop runs as before, because a
slower answer beats no answer and voice latency must not depend on the scheduler.
"""

from __future__ import annotations

import logging
import threading
import time
from types import ModuleType

log = logging.getLogger(__name__)

SERVICE_CLAIM = "hal-voice"
TURN_CLAIM = "hal-voice-turn"
GATEWAY_CLAIM = "hal-gateway"  # the turn claim reserves only what the gateway has yet to load
POLL_S = 0.05


def _import_gpu_light() -> ModuleType | None:
    """The `gpu_light` module, or None (logged once at INFO) when it is not installed."""
    try:
        import gpu_light
    except ImportError:
        log.info("gpu-light is not installed: the voice loop runs without GPU claims")
        return None
    return gpu_light


class VoiceClaims:
    """gpu-light claims for the resident set and for each turn; no-ops without gpu-light."""

    def __init__(self, resident_mib: int, brain_mib: int, linger_s: float) -> None:
        self.resident_mib = resident_mib
        self.brain_mib = brain_mib
        self.linger_s = linger_s
        self._gl = _import_gpu_light()
        self._service = None
        self._turn = None
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def _claim(self, name: str, **kwargs):
        """Starts a claim; a failure is logged and returns None."""
        if self._gl is None:
            return None
        try:
            return self._gl.claim(name, **kwargs).acquire()
        except Exception as e:  # noqa: BLE001 - a scheduler failure must never stop the voice loop
            log.warning("gpu-light claim %s failed (%s); continuing without it", name, e)
            return None

    def start_service(self) -> None:
        """Claims the resident set for the life of the process; call before any model loads."""
        self._service = self._claim(
            SERVICE_CLAIM, cls="service", vram_mib=self.resident_mib, reason="hal-voice models"
        )

    def turn_started(self) -> None:
        """Starts the turn claim without waiting; a claim already held is kept and renewed."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._turn is None:
                self._turn = self._claim(
                    TURN_CLAIM,
                    cls="interactive",
                    vram_mib=self.brain_mib,
                    exclusive=True,
                    uses=[GATEWAY_CLAIM],
                    reason="voice turn",
                )

    def wait_for_turn(self, timeout_s: float) -> bool:
        """True when the turn claim is granted within `timeout_s`; never raises."""
        turn = self._turn
        if turn is None:
            return False
        deadline = time.monotonic() + timeout_s
        while not getattr(turn, "granted", False):
            if time.monotonic() >= deadline:
                log.info("gpu-light did not grant the turn in %.1f s; going ahead", timeout_s)
                return False
            time.sleep(POLL_S)
        return True

    def turn_ended(self) -> None:
        """Releases the turn claim after `linger_s` unless a new turn renews it."""
        with self._lock:
            if self._turn is None:
                return
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.linger_s, self._release_turn)
            self._timer.daemon = True
            self._timer.start()

    def _release_turn(self) -> None:
        with self._lock:
            turn, self._turn, self._timer = self._turn, None, None
        if turn is not None:
            turn.release()

    def close(self) -> None:
        """Releases both claims at once."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
        self._release_turn()
        service, self._service = self._service, None
        if service is not None:
            service.release()
