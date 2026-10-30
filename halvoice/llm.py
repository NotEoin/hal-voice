"""A stdlib OpenAI Chat Completions client for hal-voice's own text jobs (note summaries).

hal-voice imports nothing from Hal, so this reimplements the gateway rules: with Hal installed
the request goes to the gateway socket `$XDG_RUNTIME_DIR/hal/gateway.sock` (the gateway
authenticates by peer uid, so no token is read) under the `hal-chat` alias, or to `[gateway].url`
of `~/.config/hal/config.toml` with the token that `[gateway].token_file` names. Without Hal it
uses `brain.base_url`. Every request has a timeout.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import tomllib
import urllib.parse
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

GATEWAY_ALIAS = "hal-chat"
CHAT_PATH = "/chat/completions"
CONNECT_TIMEOUT_S = 5  # a local server answers in well under a second


class Unavailable(RuntimeError):
    """No model server answered; the message names what was tried."""


class _UnixConnection(http.client.HTTPConnection):
    """HTTP over a Unix socket path."""

    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


def gateway_socket() -> Path:
    """Where Hal's gateway listens."""
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(runtime) / "hal" / "gateway.sock"


def _hal_config() -> dict:
    path = Path(os.environ.get("HAL_CONFIG", "~/.config/hal/config.toml")).expanduser()
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _read_token(path: str) -> str:
    try:
        return Path(path).expanduser().read_text(encoding="utf-8").strip()
    except OSError as e:
        raise Unavailable(f"cannot read the token file {path}: {e}") from e


def _post(url: str, headers: dict, body: bytes, timeout: float, sock: str | None = None) -> dict:
    parts = urllib.parse.urlsplit(url)
    if sock:
        conn: http.client.HTTPConnection = _UnixConnection(sock, timeout)
    elif parts.scheme == "https":
        conn = http.client.HTTPSConnection(parts.netloc, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(parts.netloc, timeout=timeout)
    try:
        conn.request("POST", parts.path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
    except (OSError, http.client.HTTPException) as e:
        raise Unavailable(f"{url}: {e}") from e
    finally:
        conn.close()
    if resp.status != 200:
        raise Unavailable(f"{url}: HTTP {resp.status}: {data[:200].decode(errors='replace')}")
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise Unavailable(f"{url}: reply is not JSON") from e


def chat(system: str, user: str, *, model: str | None = None, timeout: float = 300) -> str:
    """Returns the model's reply to one prompt; raises `Unavailable` when no server answers."""
    headers = {"Content-Type": "application/json"}
    gateway = _hal_config().get("gateway", {})
    sock: str | None = None
    if gateway.get("url"):
        url = gateway["url"].rstrip("/") + CHAT_PATH
        if gateway.get("token_file"):
            headers["Authorization"] = "Bearer " + _read_token(gateway["token_file"])
        name = model or GATEWAY_ALIAS
    elif gateway_socket().exists():
        sock = str(gateway_socket())
        url = "http://gateway/v1" + CHAT_PATH
        name = model or GATEWAY_ALIAS
    else:
        cfg = config.load()
        base = str(cfg.get("brain.base_url", "") or "")
        if not base:
            raise Unavailable(
                "no model server: Hal's gateway is not running and brain.base_url is empty"
            )
        url = base.rstrip("/") + CHAT_PATH
        key_file = str(cfg.get("brain.api_key_file", "") or "")
        if key_file:
            headers["Authorization"] = "Bearer " + _read_token(key_file)
        name = model or str(cfg.get("brain.model", "") or "")
    payload = {
        "model": name,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    reply = _post(url, headers, json.dumps(payload).encode(), timeout, sock)
    try:
        return reply["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise Unavailable(f"{url}: unexpected reply shape") from e
