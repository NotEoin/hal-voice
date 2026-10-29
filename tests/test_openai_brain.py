"""Tests for halvoice.brains.openai against a local fake Chat Completions server."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from halvoice import config
from halvoice.brains import BrainError, BrainUnavailable, TurnTimeout
from halvoice.brains.openai import OpenAIBrain, iter_sse_deltas

SERVER_JOIN_S = 5
POLL_S = 0.02  # how often the fake server checks for shutdown; keeps teardown fast

# Recorded from llama-server's /v1/chat/completions stream (shape, not a real reply).
RECORDED = [
    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}\n',
    b"\n",
    b'data: {"choices":[{"index":0,"delta":{"content":"Good "}}]}\n',
    b"\n",
    b": keep-alive\n",
    b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"hmm"}}]}\n',
    b'data: {"choices":[{"index":0,"delta":{"content":"evening, Dave."}}]}\n',
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n',
    b"data: [DONE]\n",
    b'data: {"choices":[{"index":0,"delta":{"content":"after done"}}]}\n',
]


class FakeServer:
    """A threaded HTTP server; `script(handler, body)` writes each response."""

    def __init__(self, script):
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                outer.requests.append({"path": self.path, "headers": self.headers, "body": body})
                script(self, body)

            def do_GET(self):
                outer.requests.append({"path": self.path, "headers": self.headers})
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data":[]}')

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": POLL_S}, daemon=True
        )
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(SERVER_JOIN_S)


def stream(lines, pause_s=0.0):
    """A script that answers with SSE `lines`, sleeping `pause_s` before the last one."""

    def script(handler, body):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        for i, line in enumerate(lines):
            if pause_s and i == len(lines) - 1:
                time.sleep(pause_s)
            try:
                handler.wfile.write(line)
                handler.wfile.flush()
            except OSError:
                return

    return script


@pytest.fixture
def serve():
    servers = []

    def start(script):
        server = FakeServer(script)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.stop()


def brain_cfg(**brain):
    cfg = config.load(force=True)
    cfg["brain"].update(brain)
    return cfg


def test_openai_brain_parses_sse_deltas():
    assert list(iter_sse_deltas(RECORDED)) == ["Good ", "evening, Dave."]


def test_error_event_raises_brain_error():
    with pytest.raises(BrainError, match="overloaded"):
        list(iter_sse_deltas([b'data: {"error":{"message":"overloaded"}}\n']))


def test_openai_brain_without_base_url_fails_loudly():
    with pytest.raises(BrainUnavailable, match=r"brain\.base_url is empty"):
        OpenAIBrain(brain_cfg(base_url=""))


def test_base_url_must_be_http():
    with pytest.raises(BrainUnavailable, match="http"):
        OpenAIBrain(brain_cfg(base_url="file:///etc/passwd"))


def test_reply_streams_with_persona_model_and_token(serve, tmp_path):
    server = serve(stream(RECORDED))
    key = tmp_path / "key"
    key.write_text("s3cret\n")
    brain = OpenAIBrain(
        brain_cfg(base_url=server.url, model="hal-chat", api_key_file=str(key), persona="Be HAL.")
    )
    assert "".join(brain.ask("Hello")) == "Good evening, Dave."
    sent = server.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer s3cret"
    assert sent["body"]["stream"] is True
    assert sent["body"]["model"] == "hal-chat"
    assert sent["body"]["messages"] == [
        {"role": "system", "content": "Be HAL."},
        {"role": "user", "content": "Hello"},
    ]


def test_empty_model_is_left_to_the_server(serve):
    server = serve(stream(RECORDED))
    brain = OpenAIBrain(brain_cfg(base_url=server.url, model=""))
    list(brain.ask("Hello"))
    assert "model" not in server.requests[0]["body"]


def test_history_keeps_the_last_history_turns_exchanges(serve):
    server = serve(stream(RECORDED))
    brain = OpenAIBrain(brain_cfg(base_url=server.url, history_turns=2, persona=""))
    for text in ("one", "two", "three", "four"):
        list(brain.ask(text))
    messages = server.requests[-1]["body"]["messages"]
    assert [m["content"] for m in messages if m["role"] == "user"] == ["two", "three", "four"]
    assert messages[1] == {"role": "assistant", "content": "Good evening, Dave."}


def test_openai_brain_times_out(serve):
    first = b'data: {"choices":[{"delta":{"content":"Let me "}}]}\n'
    server = serve(stream([first, b"data: [DONE]\n"], pause_s=3))
    brain = OpenAIBrain(brain_cfg(base_url=server.url, turn_timeout_s=0.5))
    got = []
    started = time.monotonic()
    with pytest.raises(TurnTimeout):
        for delta in brain.ask("Count to a million"):
            got.append(delta)
    assert got == ["Let me "]
    assert time.monotonic() - started < 2


def test_server_error_status_names_the_status(serve):
    def script(handler, body):
        handler.send_response(503)
        handler.end_headers()
        handler.wfile.write(b"loading model")

    server = serve(script)
    brain = OpenAIBrain(brain_cfg(base_url=server.url))
    with pytest.raises(BrainError, match="503: loading model"):
        list(brain.ask("Hello"))


def test_unreachable_server_is_a_brain_error():
    brain = OpenAIBrain(brain_cfg(base_url="http://127.0.0.1:9/v1", connect_timeout_s=1))
    with pytest.raises(BrainError, match="cannot reach"):
        list(brain.ask("Hello"))


def test_cancel_ends_a_blocked_stream_quietly(serve):
    first = b'data: {"choices":[{"delta":{"content":"Well, "}}]}\n'
    server = serve(stream([first, b"data: [DONE]\n"], pause_s=3))
    brain = OpenAIBrain(brain_cfg(base_url=server.url))
    replies = brain.ask("Tell me a story")
    assert next(replies) == "Well, "
    threading.Timer(0.2, brain.cancel).start()
    started = time.monotonic()
    assert list(replies) == []
    assert time.monotonic() - started < 2


def test_unreadable_api_key_file_names_the_file(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(BrainUnavailable, match="nope"):
        OpenAIBrain(brain_cfg(base_url="http://127.0.0.1:1/v1", api_key_file=str(missing)))


def test_probe_reads_models(serve):
    server = serve(stream(RECORDED))
    OpenAIBrain(brain_cfg(base_url=server.url)).probe()
    assert server.requests[0]["path"] == "/v1/models"
