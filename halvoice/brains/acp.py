"""The `hal` brain: an ACP client of `hal acp --channel voice --toolset assistant`.

Hal's boxed assistant agent answers; this module only speaks JSON-RPC over the child's
stdio. It imports nothing from Hal (REQ-VOICE-02).

Design decisions:

- One long-lived child and one ACP session for the loop's lifetime. A child that exits is
  restarted once; a second exit raises `BrainUnavailable`. Nothing else answers in its place.
- `session/new` gets an empty directory this package owns, never the vault: `hal acp`
  replaces the `cwd` with the voice channel scratch directory, so the value is not load-bearing.
- A reader thread owns the child's stdout, so a turn deadline and `cancel` work while the
  agent is silent.
- Approval: only a broker request (`_meta["hal/origin"] == "broker"`) can be approved, and only
  by a whole reply of "yes" or "allow" to the host-made `hal/summary` text. Everything else is
  answered `cancelled`.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .. import config
from ..util import get_logger
from . import BrainError, BrainUnavailable, TurnTimeout

log = get_logger(__name__)

PROTOCOL_VERSION = 1  # ACP v1
DEFAULT_COMMAND = ["hal", "acp", "--channel", "voice", "--toolset", "assistant"]
HANDSHAKE_TIMEOUT_S = 30.0  # first start boots the box, which can take a while
DRAIN_TIMEOUT_S = 5.0  # a cancelled turn answers well inside this
CLOSE_TIMEOUT_S = 5.0
APPROVAL_WORDS = frozenset({"yes", "allow"})  # REQ-VOICE-05: the whole reply, nothing longer
BROKER_ORIGIN = "broker"
MAX_RESTARTS = 1
_EOF = object()


def work_dir() -> Path:
    """The empty directory sent as `cwd`: `$XDG_STATE_HOME/hal-voice/work`, created on use."""
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    path = Path(base) / "hal-voice" / "work"
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_approval(reply: str) -> bool:
    """True when the whole spoken reply is "yes" or "allow"."""
    words = re.sub(r"[^a-z ]", "", reply.lower()).split()
    return len(words) == 1 and words[0] in APPROVAL_WORDS


def _allow_option(params: dict) -> str | None:
    """The `optionId` of the `allow_once` option of a permission request, if it has one."""
    for option in params.get("options") or []:
        if option.get("kind") == "allow_once":
            return option.get("optionId")
    return None


class AcpBrain:
    """A spoken conversation with Hal's assistant agent over one ACP session."""

    def __init__(
        self,
        cfg: config.Cfg | None = None,
        listen: Callable[[], str] | None = None,
    ) -> None:
        cfg = cfg if cfg is not None else config.load()
        command = cfg.get("brain.hal_command") or DEFAULT_COMMAND
        if not isinstance(command, list) or not all(isinstance(c, str) for c in command):
            raise BrainUnavailable("brain.hal_command must be a list of strings")
        self.command: list[str] = command
        self.turn_timeout = float(cfg.get("brain.turn_timeout_s", 300))
        self.persona = str(cfg.get("brain.persona") or "").strip()
        self.voice_approvals = bool(cfg.get("brain.voice_approvals", True))
        #: Returns the next spoken reply; set by the loop. None answers every request `cancelled`.
        self.listen = listen
        self._proc: subprocess.Popen | None = None
        self._inbox: queue.Queue = queue.Queue()
        self._session_id = ""
        self._next_id = 0
        self._restarts = 0
        self._persona_sent = False
        self._stale_id: int | None = None
        self._cancelled = threading.Event()
        self._start()

    # -- process ---------------------------------------------------------------
    def _start(self) -> None:
        """Spawns `hal acp`, initializes it and opens the session."""
        self._inbox = queue.Queue()
        try:
            proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as e:
            raise BrainUnavailable(
                f"brain.mode is hal but {self.command[0]!r} cannot start: {e}. Install Hal and "
                "put `hal` on PATH, or set brain.mode to openai (a local server) or claude "
                "(Claude Code on the host: not private, not sandboxed)."
            ) from e
        self._proc = proc
        threading.Thread(target=self._read, args=(proc, self._inbox), daemon=True).start()
        try:
            self._request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "clientCapabilities": {
                        "fs": {"readTextFile": False, "writeTextFile": False},
                        "terminal": False,
                    },
                },
                HANDSHAKE_TIMEOUT_S,
            )
            result = self._request(
                "session/new", {"cwd": str(work_dir()), "mcpServers": []}, HANDSHAKE_TIMEOUT_S
            )
        except BrainError as e:
            self._kill()
            raise BrainUnavailable(f"hal acp did not start a session: {e}") from e
        self._session_id = str(result.get("sessionId") or "")
        if not self._session_id:
            self._kill()
            raise BrainUnavailable("hal acp answered session/new without a sessionId")
        self._persona_sent = False
        self._stale_id = None
        log.info("brain: hal acp session %s", self._session_id)

    @staticmethod
    def _read(proc: subprocess.Popen, inbox: queue.Queue) -> None:
        """Moves the child's stdout lines into `inbox` as parsed objects, then an EOF mark."""
        assert proc.stdout is not None  # noqa: S101 - PIPE was requested
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log.warning("brain: skipped a non-JSON line from hal acp: %.80s", line)
                continue
            if isinstance(msg, dict):
                inbox.put(msg)
        inbox.put(_EOF)

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        with contextlib.suppress(OSError):
            proc.kill()
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            proc.wait(timeout=CLOSE_TIMEOUT_S)

    def _ensure(self) -> None:
        """Restarts a dead child once; a second death raises `BrainUnavailable`."""
        if self._alive():
            return
        if self._restarts >= MAX_RESTARTS:
            raise BrainUnavailable("hal acp exited twice: run `hal doctor`, then restart hal-voice")
        self._restarts += 1
        log.warning("brain: hal acp exited; restarting it once")
        self._kill()
        self._start()

    # -- JSON-RPC --------------------------------------------------------------
    def _send(self, msg: dict[str, Any]) -> None:
        if not self._alive():
            raise BrainError("hal acp is not running")
        assert self._proc is not None and self._proc.stdin is not None  # noqa: S101
        try:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise BrainError(f"cannot write to hal acp: {e}") from e

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _request(self, method: str, params: dict, timeout: float) -> dict:
        """Sends a request and waits for its result; other traffic is answered or dropped."""
        rid = self._new_id()
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            msg = self._get(deadline)
            if msg.get("id") == rid and "method" not in msg:
                if "error" in msg:
                    raise BrainError(f"{method} failed: {msg['error']}")
                return msg.get("result") or {}
            self._other(msg)

    def _get(self, deadline: float) -> dict:
        try:
            msg = self._inbox.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            raise TurnTimeout("hal acp did not answer in time") from None
        if msg is _EOF:
            raise BrainError("hal acp exited")
        return msg

    def _other(self, msg: dict) -> None:
        """Handles agent requests outside a turn (permission asks); notifications are dropped."""
        if msg.get("method") == "session/request_permission":
            self._answer_permission(msg, speak=False)
        elif "method" in msg and "id" in msg:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "error": {"code": -32601, "message": "method not supported"},
                }
            )

    def _answer_permission(self, msg: dict, speak: bool) -> str:
        """Cancels a permission request, or returns the summary to speak for a broker one.

        When a summary is returned the request stays open; `_finish_approval` answers it.
        """
        params = msg.get("params") or {}
        meta = params.get("_meta") or {}
        can_ask = (
            speak
            and self.voice_approvals
            and self.listen is not None
            and meta.get("hal/origin") == BROKER_ORIGIN
            and _allow_option(params) is not None
        )
        summary = str(meta.get("hal/summary") or "").strip() if can_ask else ""
        if not summary:
            self._send(
                {"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": {"outcome": "cancelled"}}}
            )
        return summary

    # -- the Brain interface ---------------------------------------------------
    def ask(self, text: str) -> Iterator[str]:
        """Streams the reply to one spoken turn as text deltas.

        Raises `TurnTimeout` past `brain.turn_timeout_s` (after sending `session/cancel`),
        and `BrainError` when the agent fails mid-turn.
        """
        self._ensure()
        self._drain_stale()
        self._cancelled.clear()
        blocks = []
        if self.persona and not self._persona_sent:
            blocks.append(
                {"type": "text", "text": f"[voice channel instructions]\n{self.persona}\n"}
            )
            self._persona_sent = True
        blocks.append({"type": "text", "text": text})
        rid = self._new_id()
        self._send(
            {
                "jsonrpc": "2.0",
                "id": rid,
                "method": "session/prompt",
                "params": {"sessionId": self._session_id, "prompt": blocks},
            }
        )
        self._stale_id = rid
        deadline = time.monotonic() + self.turn_timeout
        while True:
            try:
                msg = self._get(deadline)
            except TurnTimeout:
                self.cancel()
                raise TurnTimeout(
                    f"the reply took longer than brain.turn_timeout_s ({self.turn_timeout:g}s)"
                ) from None
            except BrainError:
                self._kill()
                raise
            if "method" not in msg and msg.get("id") == rid:
                self._stale_id = None
                if "error" in msg:
                    raise BrainError(f"the agent reported an error: {msg['error']}")
                return
            method = msg.get("method")
            if method == "session/update" and not self._cancelled.is_set():
                update = (msg.get("params") or {}).get("update") or {}
                content = update.get("content") or {}
                if update.get("sessionUpdate") == "agent_message_chunk":
                    chunk = content.get("text")
                    if isinstance(chunk, str) and chunk:
                        yield chunk
            elif method == "session/request_permission":
                summary = self._answer_permission(msg, speak=True)
                if summary:
                    yield f" {summary} "
                    self._finish_approval(msg)

    def _finish_approval(self, msg: dict) -> None:
        """Listens for the spoken reply and answers the pending broker request."""
        assert self.listen is not None  # noqa: S101 - _answer_permission checked it
        try:
            reply = self.listen()
        except Exception as e:  # noqa: BLE001 - a failed microphone must deny, not crash the turn
            log.warning("brain: approval listen failed: %s", e)
            reply = ""
        allow = _allow_option(msg.get("params") or {})
        if allow and is_approval(reply):
            outcome: dict = {"outcome": "selected", "optionId": allow}
        else:
            outcome = {"outcome": "cancelled"}
        log.info("brain: spoken approval answered %s", outcome["outcome"])
        self._send({"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": outcome}})

    def _drain_stale(self) -> None:
        """Waits for the answer to a cancelled turn so its chunks never reach the next one."""
        if self._stale_id is None:
            return
        deadline = time.monotonic() + DRAIN_TIMEOUT_S
        try:
            while True:
                msg = self._get(deadline)
                if "method" not in msg and msg.get("id") == self._stale_id:
                    break
                self._other(msg)
        except BrainError:
            log.warning("brain: no answer to the cancelled turn; restarting hal acp")
            self._kill()
            self._ensure()
        self._stale_id = None

    def cancel(self) -> None:
        """Sends `session/cancel` (barge-in); `ask` then ends without raising."""
        self._cancelled.set()
        if not self._alive():
            return
        with contextlib.suppress(BrainError):
            self._send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": {"sessionId": self._session_id},
                }
            )

    def close(self) -> None:
        """Ends the child process."""
        self._kill()
