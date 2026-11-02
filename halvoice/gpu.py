"""GPU sharing for the voice loop, through a small stdlib client of the gpu-light daemon.

`gpu.mode` picks what the loop asks for (see `halvoice/config.yaml`):
- `on_demand` (default): nothing is held while idle. A turn asks the daemon for room
  (`room.request`), the daemon parks background GPU work, and the room is given back
  `gpu.linger_s` after the last reply.
- `resident`: a service claim for the resident models for the life of the process, plus an
  interactive claim for each turn.
- `off`: gpu-light is never contacted.

Design decisions:
- hal-voice does not import `gpu_light`. The protocol is one JSON-RPC 2.0 object per line on a
  UNIX socket (gpu-light's docs/protocol-v2.md), small enough to speak with the stdlib.
- A room and a claim live as long as their connection, so a worker thread owns the
  connection for as long as the turn is held and closes it to give the room back.
- Voice must never stop because of the scheduler: every failure is logged once and the loop
  carries on, and every socket call has a timeout.
- Not done in resident mode: a daemon restart drops the claims and they are not made again;
  restart hal-voice to claim again.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import select
import socket
import threading
import time
from pathlib import Path
from typing import Any

from . import __version__
from .config import ConfigError

log = logging.getLogger(__name__)

MODES = ("on_demand", "resident", "off")
SERVICE_CLAIM = "hal-voice"
TURN_CLAIM = "hal-voice-turn"
GATEWAY_CLAIM = (
    "hal-gateway"  # the resident turn claim reserves only what the gateway has yet to load
)
PROTOCOL = 2
CONNECT_TIMEOUT_S = 2.0  # a local socket answers at once; a hung daemon must not hold the loop
CALL_TIMEOUT_S = 5.0  # hello, claim and release answer immediately
# room.request is answered when room is made; the daemon answers -32002 at its own
# `adopt.room_timeout_s` (600 s by default), so this only has to be longer than that
ROOM_WAIT_S = 660.0
HOLD_POLL_S = 0.5  # how often a held room checks that the daemon is still there
JOIN_TIMEOUT_S = 5.0
MAX_LINE_BYTES = 1024 * 1024  # gpu-light's own limit for one message
RECV_BYTES = 65536
NO_ROOM_CODE = -32002  # gpu-light: the room was not made within the daemon's timeout


class GpuLightError(Exception):
    """Any failure talking to the daemon: absent, refused, timed out, or a socket error."""


class RpcError(GpuLightError):
    """The daemon answered with a JSON-RPC error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{message} (code {code})")
        self.code = code


def gpu_mode(cfg: Any) -> str:
    """The validated `gpu.mode`; raises `ConfigError` naming the allowed values."""
    mode = str(cfg.get("gpu.mode", "on_demand"))
    if mode not in MODES:
        raise ConfigError(f"gpu.mode must be one of {', '.join(MODES)}, not {mode!r}")
    return mode


def socket_path() -> Path:
    """The daemon socket: `$GPU_LIGHT_SOCKET`, else `$XDG_RUNTIME_DIR/gpu-light/daemon.sock`."""
    override = os.environ.get("GPU_LIGHT_SOCKET")
    if override:
        return Path(override).expanduser()
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "gpu-light" / "daemon.sock"
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "gpu-light" / "daemon.sock"


class _Conn:
    """One daemon connection: `hello` on open, then blocking calls, each with a timeout."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._buf = b""
        self._next_id = 0
        self.notifications: collections.deque[dict[str, Any]] = collections.deque()
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(CONNECT_TIMEOUT_S)
        try:
            self.sock.connect(str(path))
            self.server = self.call(
                "hello",
                {"client": "hal-voice", "version": __version__, "protocol": PROTOCOL},
                CALL_TIMEOUT_S,
            )
        except (OSError, GpuLightError) as exc:
            self.sock.close()
            raise GpuLightError(f"no gpu-light daemon answered on {path}: {exc}") from None
        if not isinstance(self.server, dict) or self.server.get("protocol") != PROTOCOL:
            self.sock.close()
            raise GpuLightError(f"the daemon on {path} does not speak protocol {PROTOCOL}")

    def close(self) -> None:
        """Closes the connection; the daemon ends the room and claims it held."""
        self.sock.close()

    def _line(self, deadline: float) -> dict[str, Any]:
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GpuLightError("the daemon did not answer in time")
            self.sock.settimeout(remaining)
            try:
                data = self.sock.recv(RECV_BYTES)
            except TimeoutError:
                raise GpuLightError("the daemon did not answer in time") from None
            except OSError as exc:
                raise GpuLightError(f"socket error: {exc}") from None
            if not data:
                raise GpuLightError("the daemon closed the connection")
            self._buf += data
            if len(self._buf) > MAX_LINE_BYTES:
                raise GpuLightError("the daemon sent a message that is too long")
        raw, _, self._buf = self._buf.partition(b"\n")
        try:
            doc = json.loads(raw)
        except ValueError:
            raise GpuLightError("the daemon sent something that is not JSON") from None
        if not isinstance(doc, dict):
            raise GpuLightError("the daemon sent something that is not a JSON object")
        return doc

    def call(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        """Sends a request and returns its result; notifications that arrive are kept aside."""
        self._next_id += 1
        request = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}
        deadline = time.monotonic() + timeout
        try:
            self.sock.settimeout(timeout)
            self.sock.sendall(json.dumps(request).encode("ascii") + b"\n")
        except OSError as exc:
            raise GpuLightError(f"socket error: {exc}") from None
        while True:
            doc = self._line(deadline)
            if "id" not in doc:
                self.notifications.append(doc)
            elif doc["id"] == self._next_id:
                if "error" in doc:
                    err = doc["error"] if isinstance(doc["error"], dict) else {}
                    raise RpcError(int(err.get("code", 0)), str(err.get("message", "error")))
                return doc.get("result")

    def wait_notification(self, method: str, timeout: float) -> None:
        """Waits for a notification named `method`; raises `GpuLightError` on timeout."""
        deadline = time.monotonic() + timeout
        while True:
            while self.notifications:
                if self.notifications.popleft().get("method") == method:
                    return
            doc = self._line(deadline)
            if "id" not in doc:
                self.notifications.append(doc)

    def alive(self) -> bool:
        """False once the daemon has closed the connection; drains anything it sent."""
        try:
            while select.select([self.sock], [], [], 0)[0]:
                data = self.sock.recv(RECV_BYTES)
                if not data:
                    return False
        except OSError:
            return False
        return True


def probe(path: Path | None = None) -> dict[str, Any]:
    """The daemon's `hello` answer; raises `GpuLightError` when no daemon answers."""
    conn = _Conn(path or socket_path())
    conn.close()
    return conn.server


class _Turn:
    """One held room or claim: its worker thread, and what the loop waits on."""

    def __init__(self) -> None:
        self.ready = threading.Event()  # set when the room is made, or the attempt failed
        self.ok = False
        self.release = threading.Event()  # set to give the room back
        self.finished = threading.Event()  # set when the worker is done and the connection closed
        self.thread: threading.Thread | None = None


class VoiceClaims:
    """What the voice loop asks of gpu-light, by `gpu.mode`; never raises into the loop."""

    def __init__(
        self,
        mode: str,
        resident_mib: int,
        brain_mib: int,
        linger_s: float,
        path: Path | None = None,
    ) -> None:
        if mode not in MODES:
            raise ConfigError(f"gpu.mode must be one of {', '.join(MODES)}, not {mode!r}")
        self.mode = mode
        self.resident_mib = resident_mib
        self.brain_mib = brain_mib
        self.linger_s = linger_s
        self.path = path
        self._service: _Conn | None = None
        self._turn: _Turn | None = None
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._warned = False

    @classmethod
    def from_config(cls, cfg: Any) -> VoiceClaims:
        """Builds the claims from the `gpu:` settings."""
        return cls(
            gpu_mode(cfg),
            resident_mib=int(cfg.get("gpu.resident_mib", 3584)),
            brain_mib=int(cfg.get("gpu.brain_mib", 14000)),
            linger_s=float(cfg.get("gpu.linger_s", 30)),
        )

    def _failed(self, what: str, exc: Exception) -> None:
        """Logs the first failure at WARNING; later ones at DEBUG until a success."""
        level = logging.DEBUG if self._warned else logging.WARNING
        self._warned = True
        log.log(level, "gpu-light: %s failed (%s); continuing without it", what, exc)

    def start_service(self) -> None:
        """Resident mode: claims the resident models for the life of the process."""
        if self.mode != "resident":
            return
        try:
            conn = _Conn(self.path or socket_path())
            conn.call(
                "claim",
                {
                    "name": SERVICE_CLAIM,
                    "class": "service",
                    "vram_mib": self.resident_mib,
                    "reason": "hal-voice models",
                },
                CALL_TIMEOUT_S,
            )
        except GpuLightError as exc:
            self._failed("the service claim", exc)
            return
        self._service = conn

    def turn_started(self) -> None:
        """Starts making room for a turn without waiting; a room already held is kept."""
        if self.mode == "off":
            return
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            turn = self._turn
            if turn is not None and not turn.finished.is_set():
                return
            turn = self._turn = _Turn()
            turn.thread = threading.Thread(
                target=self._work, args=(turn,), name="gpu-light-turn", daemon=True
            )
            turn.thread.start()

    def _work(self, turn: _Turn) -> None:
        """Worker: makes the room, holds the connection until released, then gives it back."""
        conn = None
        try:
            conn = _Conn(self.path or socket_path())
            self._acquire(conn)
            turn.ok = True
            self._warned = False
            turn.ready.set()
            while not turn.release.wait(HOLD_POLL_S):
                if not conn.alive():
                    log.info("gpu-light closed the connection; the room is gone")
                    break
            if conn.alive():
                self._give_back(conn)
        except GpuLightError as exc:
            self._failed("making room for the turn", exc)
        finally:
            if conn is not None:
                conn.close()
            turn.ready.set()
            turn.finished.set()

    def _acquire(self, conn: _Conn) -> None:
        """Makes the room (on_demand) or the interactive claim (resident)."""
        if self.mode == "on_demand":
            answer = conn.call(
                "room.request",
                {"name": TURN_CLAIM, "gpu": 0, "vram_mib": self.brain_mib},
                ROOM_WAIT_S,
            )
            if isinstance(answer, dict) and not answer.get("ok", True):
                log.warning("gpu-light made room but it does not fit: %s", answer.get("notes"))
            return
        answer = conn.call(
            "claim",
            {
                "name": TURN_CLAIM,
                "class": "interactive",
                "vram_mib": self.brain_mib,
                "exclusive": True,
                "uses": [GATEWAY_CLAIM],
                "reason": "voice turn",
            },
            CALL_TIMEOUT_S,
        )
        claim = answer.get("claim", {}) if isinstance(answer, dict) else {}
        if not claim.get("granted"):
            conn.wait_notification("claim_granted", ROOM_WAIT_S)

    def _give_back(self, conn: _Conn) -> None:
        method = "room.release" if self.mode == "on_demand" else "claim.release"
        try:
            conn.call(method, {"name": TURN_CLAIM}, CALL_TIMEOUT_S)
        except GpuLightError as exc:
            log.debug("gpu-light: releasing the turn failed (%s); closing the connection", exc)

    def wait_for_turn(self, timeout_s: float) -> bool:
        """True when room is made within `timeout_s`; False on failure or timeout. Never raises."""
        turn = self._turn
        if turn is None:
            return False
        if not turn.ready.wait(timeout_s):
            log.info("gpu-light did not make room in %.1f s; going ahead", timeout_s)
            return False
        return turn.ok

    def turn_ended(self) -> None:
        """Gives the room back after `linger_s` unless a new turn starts first."""
        with self._lock:
            if self._turn is None:
                return
            if self._timer is not None:
                self._timer.cancel()
            timer = threading.Timer(self.linger_s, lambda: self._expire(timer))
            timer.daemon = True
            self._timer = timer
            timer.start()

    def _expire(self, timer: threading.Timer) -> None:
        """Linger is over; a timer that a new turn replaced or cancelled does nothing."""
        with self._lock:
            if self._timer is not timer:
                return
            self._timer = None
            turn = self._turn
        if turn is not None:
            turn.release.set()

    def close(self) -> None:
        """Gives back the room and the service claim at once."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            turn = self._turn
        if turn is not None:
            turn.release.set()
            if turn.thread is not None:
                turn.thread.join(JOIN_TIMEOUT_S)
        service, self._service = self._service, None
        if service is not None:
            service.close()
