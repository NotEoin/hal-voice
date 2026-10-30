"""Tests for voicetest.brain: tool calls are recorded, never run, and halcore is never imported."""

import http.server
import json
import sys
import threading
from typing import ClassVar

import pytest

from halvoice import config
from voicetest import brain

MANIFEST = """
[[tools]]
name = "calendar_add"
description = "Add an event."
[tools.params.title]
required = true
"""


class ModelHandler(http.server.BaseHTTPRequestHandler):
    bodies: ClassVar[list] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ModelHandler.bodies.append(body)
        if any(m["role"] == "tool" for m in body["messages"]):
            msg = {"role": "assistant", "content": "Done, Dave."}
        else:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "calendar_add", "arguments": '{"title": "x"}'},
                    }
                ],
            }
        data = json.dumps({"choices": [{"message": msg}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    ModelHandler.bodies = []
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("HAL_CONFIG", str(tmp_path / "hal.toml"))
    server = http.server.HTTPServer(("127.0.0.1", 0), ModelHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config.load(force=True)["brain"]["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
    yield tmp_path
    server.shutdown()


def test_voicetest_brain_records_calls_without_running_them(endpoint):
    manifest = endpoint / "hal-module.toml"
    manifest.write_text(MANIFEST)
    schemas = brain.tool_schemas([manifest])
    assert schemas[0]["function"]["parameters"]["required"] == ["title"]
    recorder = brain.RecordingBrain("You are HAL.", schemas)
    assert recorder.reply("Hal, add a meeting") == "Done, Dave."
    assert [t["name"] for t in recorder.last_turn["tools"]] == ["calendar_add"]
    tool_msgs = [m for m in ModelHandler.bodies[-1]["messages"] if m["role"] == "tool"]
    assert "not carried out" in tool_msgs[0]["content"]
    assert ModelHandler.bodies[0]["tools"] == schemas


def test_voicetest_brain_does_not_import_halcore(endpoint):
    assert "halcore" not in sys.modules


def test_missing_manifests_stop_with_the_fix_named(monkeypatch):
    monkeypatch.delenv(brain.MANIFEST_ENV, raising=False)
    with pytest.raises(SystemExit, match="--manifest"):
        brain.manifest_paths()
