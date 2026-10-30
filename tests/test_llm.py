"""Tests for halvoice.llm: the stdlib chat client, against fake servers."""

import http.server
import json
import socketserver
import threading
from typing import ClassVar

import pytest

from halvoice import config, llm

REPLY = {"choices": [{"message": {"content": "A summary."}}]}


class Handler(http.server.BaseHTTPRequestHandler):
    seen: ClassVar[list] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append((self.path, self.headers.get("Authorization"), body["model"]))
        data = json.dumps(REPLY).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class UnixServer(socketserver.UnixStreamServer):
    def get_request(self):
        request, _ = super().get_request()
        return request, ("local", 0)


@pytest.fixture
def env(tmp_path, monkeypatch):
    Handler.seen = []
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("HAL_CONFIG", str(tmp_path / "hal.toml"))
    return tmp_path


def serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_gateway_socket_is_used_with_the_hal_chat_alias(env):
    sock = llm.gateway_socket()
    sock.parent.mkdir(parents=True)
    server = serve(UnixServer(str(sock), Handler))
    try:
        assert llm.chat("sys", "user") == "A summary."
    finally:
        server.shutdown()
    assert Handler.seen == [("/v1/chat/completions", None, "hal-chat")]


def test_base_url_is_used_when_hal_is_absent(env):
    server = serve(http.server.HTTPServer(("127.0.0.1", 0), Handler))
    key = env / "key"
    key.write_text("sekrit\n")
    cfg = config.load(force=True)
    cfg["brain"].update(
        base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key_file=str(key), model="m"
    )
    try:
        assert llm.chat("sys", "user") == "A summary."
    finally:
        server.shutdown()
    assert Handler.seen == [("/v1/chat/completions", "Bearer sekrit", "m")]


def test_no_server_is_reported_as_unavailable(env):
    with pytest.raises(llm.Unavailable, match=r"brain\.base_url"):
        llm.chat("sys", "user")
