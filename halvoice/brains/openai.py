"""The `openai` brain: streams replies from any OpenAI-compatible Chat Completions server.

This mode is for running hal-voice without Hal (llama-server, Ollama's `/v1`, vLLM, a
hosted API); with Hal installed, voice reaches models through Hal instead. It has no tools.

Design decisions:

- Stdlib `http.client`, not `requests`: the turn deadline must bound every read of the
  stream, so the socket timeout is reset to the time left before each line.
- `brain.base_url` has no default and an empty value is an error. Guessing a local port
  would send the user's speech to whatever happens to listen there.
- `cancel` shuts the socket down, which wakes a read blocked in another thread; the
  stream then ends quietly instead of raising.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import socket
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import urlsplit

from .. import config
from ..util import get_logger
from . import BrainError, BrainUnavailable, TurnTimeout

log = get_logger(__name__)

DEFAULT_CONNECT_TIMEOUT_S = 5.0
DEFAULT_TURN_TIMEOUT_S = 300.0
DEFAULT_HISTORY_TURNS = 8
ERROR_BODY_CHARS = 300  # enough of a server's error body to show the reason in a log line


def iter_sse_deltas(lines: Iterable[bytes | str]) -> Iterator[str]:
    """Yields the `choices[0].delta.content` text of each server-sent event line.

    Stops at `data: [DONE]`. Raises `BrainError` for an `error` event. Comment lines,
    blank keep-alives and events without content (role or reasoning deltas) are skipped.
    """
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            log.warning("openai brain: skipped a malformed stream event: %.80s", payload)
            continue
        if not isinstance(event, dict):
            continue
        if event.get("error"):
            raise BrainError(f"the server reported an error: {event['error']}")
        for choice in event.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if isinstance(content, str) and content:
                yield content


class OpenAIBrain:
    """A spoken conversation with an OpenAI-compatible server, `history_turns` exchanges long."""

    def __init__(self, cfg: config.Cfg | None = None) -> None:
        cfg = cfg if cfg is not None else config.load()
        self.base_url = str(cfg.get("brain.base_url") or "").rstrip("/")
        if not self.base_url:
            raise BrainUnavailable(
                "brain.mode is openai but brain.base_url is empty: set it to an "
                "OpenAI-compatible server (for example http://127.0.0.1:8080/v1)"
            )
        parts = urlsplit(self.base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise BrainUnavailable(f"brain.base_url must be an http(s) URL, not {self.base_url!r}")
        self._https = parts.scheme == "https"
        self._host = parts.hostname
        self._port = parts.port
        self._prefix = parts.path.rstrip("/")
        self.model = str(cfg.get("brain.model") or "")
        self.persona = str(cfg.get("brain.persona") or "").strip()
        self.connect_timeout = float(
            cfg.get("brain.connect_timeout_s") or DEFAULT_CONNECT_TIMEOUT_S
        )
        self.turn_timeout = float(cfg.get("brain.turn_timeout_s") or DEFAULT_TURN_TIMEOUT_S)
        turns = cfg.get("brain.history_turns")
        self.history: deque[tuple[str, str]] = deque(
            maxlen=int(turns) if turns not in (None, "") else DEFAULT_HISTORY_TURNS
        )
        self._token = self._read_token(str(cfg.get("brain.api_key_file") or ""))
        self._cancelled = threading.Event()
        self._sock: socket.socket | None = None
        log.info("brain: OpenAI-compatible server at %s", self.base_url)

    @staticmethod
    def _read_token(key_file: str) -> str:
        """The bearer token in `key_file`, or an empty string when no file is set."""
        if not key_file:
            return ""
        path = Path(key_file).expanduser()
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise BrainUnavailable(f"brain.api_key_file {path} cannot be read: {e}") from e
        if not token:
            raise BrainUnavailable(f"brain.api_key_file {path} is empty")
        return token

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def probe(self) -> None:
        """Raises `BrainError` unless `GET {base_url}/models` answers 200 within the connect
        timeout."""
        conn = self._connection()
        try:
            conn.request("GET", f"{self._prefix}/models", headers=self._auth())
            resp = conn.getresponse()
            resp.read(ERROR_BODY_CHARS)
        except (OSError, http.client.HTTPException) as e:
            raise BrainError(f"cannot reach {self.base_url}: {e}") from e
        finally:
            conn.close()
        if resp.status != 200:
            raise BrainError(f"{self.base_url}/models answered {resp.status}")

    def _messages(self, text: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.persona}] if self.persona else []
        for asked, answered in self.history:
            messages.append({"role": "user", "content": asked})
            messages.append({"role": "assistant", "content": answered})
        messages.append({"role": "user", "content": text})
        return messages

    def _connection(self) -> http.client.HTTPConnection:
        if self._https:
            return http.client.HTTPSConnection(self._host, self._port, timeout=self.connect_timeout)
        return http.client.HTTPConnection(self._host, self._port, timeout=self.connect_timeout)

    def _lines(
        self, resp: http.client.HTTPResponse, sock: socket.socket, deadline: float
    ) -> Iterator[bytes]:
        """Lines of the response body, each read bounded by the time left in the turn."""
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TurnTimeout(f"no complete reply within {self.turn_timeout:g} s")
            sock.settimeout(left)
            line = resp.readline()
            if not line:
                return
            yield line

    def ask(self, text: str) -> Iterator[str]:
        """Streams the reply to `text`; see `Brain.ask`."""
        self._cancelled.clear()
        deadline = time.monotonic() + self.turn_timeout
        body = {"messages": self._messages(text), "stream": True}
        if self.model:
            body["model"] = self.model
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        headers.update(self._auth())
        conn = self._connection()
        resp: http.client.HTTPResponse | None = None
        reply: list[str] = []
        try:
            try:
                conn.connect()
            except OSError as e:
                raise BrainError(f"cannot reach {self.base_url}: {e}") from e
            # Kept here: for a response that closes the connection, http.client hands the
            # socket to the response and clears `conn.sock`.
            sock = self._sock = conn.sock
            sock.settimeout(max(deadline - time.monotonic(), 0.001))
            conn.request("POST", f"{self._prefix}/chat/completions", json.dumps(body), headers)
            resp = conn.getresponse()
            if resp.status != 200:
                detail = resp.read(ERROR_BODY_CHARS).decode("utf-8", "replace")
                raise BrainError(f"{self.base_url} answered {resp.status}: {detail}")
            for delta in iter_sse_deltas(self._lines(resp, sock, deadline)):
                reply.append(delta)
                yield delta
        except TimeoutError as e:
            if not self._cancelled.is_set():
                raise TurnTimeout(f"no complete reply within {self.turn_timeout:g} s") from e
        except (OSError, http.client.HTTPException):
            if not self._cancelled.is_set():
                raise
        finally:
            self._sock = None
            if resp is not None:
                resp.close()
            conn.close()
            if reply:
                # A cut-off reply is still what was said, so the next turn can refer to it.
                self.history.append((text, "".join(reply)))

    def cancel(self) -> None:
        """Ends the turn in progress; a read blocked in another thread wakes up."""
        self._cancelled.set()
        sock = self._sock
        if sock is not None:
            with contextlib.suppress(OSError):  # already closed by the server or by `ask`
                sock.shutdown(socket.SHUT_RDWR)

    def close(self) -> None:
        """Ends any turn in progress; there is nothing else to release."""
        self.cancel()
