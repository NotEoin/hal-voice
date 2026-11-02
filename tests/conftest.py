"""Shared fixtures: every test sees an empty user config directory, never the real one."""

import json
import socket
import threading
import time

import pytest

from halvoice import config

WAIT_S = 5.0


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Points `XDG_CONFIG_HOME` at a temp dir and drops cached and env-named settings."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.delenv("HALVOICE_CONFIG", raising=False)
    # no test may reach the real gpu-light daemon; a test that wants one sets its own socket
    monkeypatch.setenv("GPU_LIGHT_SOCKET", str(tmp_path / "no-gpu-light.sock"))
    monkeypatch.setattr(config, "_CACHE", None)
    return tmp_path / "xdg-config" / "hal-voice"


class FakeDaemon:
    """Speaks just enough of gpu-light's protocol 2: hello, room.*, claim, claim.release."""

    def __init__(self, path):
        self.path = path
        self.log = []  # (method, params), in arrival order
        self.open = 0  # connections currently open
        self.room_delay_s = 0.0
        self.room_error = None
        self.lock = threading.Lock()
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(path))
        self.server.listen(8)
        self.server.settimeout(0.1)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()

    def methods(self):
        with self.lock:
            return [m for m, _ in self.log]

    def count(self, method):
        return self.methods().count(method)

    def wait_for(self, predicate, timeout=WAIT_S):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def close(self):
        self.stop.set()
        self.thread.join(WAIT_S)
        self.server.close()

    def _accept(self):
        while not self.stop.is_set():
            try:
                conn, _ = self.server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        with self.lock:
            self.open += 1
        conn.settimeout(0.1)
        buf = b""
        try:
            while not self.stop.is_set():
                try:
                    data = conn.recv(4096)
                except TimeoutError:
                    continue
                if not data:
                    return
                buf += data
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    self._answer(conn, json.loads(line))
        except OSError:
            pass
        finally:
            with self.lock:
                self.open -= 1
            conn.close()

    def _answer(self, conn, req):
        with self.lock:
            self.log.append((req["method"], req.get("params", {})))
        method = req["method"]
        if method == "hello":
            result = {"server": "gpu-light", "protocol": 2, "hostname": "fake"}
        elif method == "room.request":
            time.sleep(self.room_delay_s)
            if self.room_error:
                self._send(conn, {"jsonrpc": "2.0", "id": req["id"], "error": self.room_error})
                return
            result = {"ok": True, "parked": ["batchjob"], "free_mib": 20000, "notes": []}
        elif method == "claim":
            result = {"claim": {"name": req["params"]["name"], "granted": True}}
        elif method == "room.release":
            result = True
        else:
            result = {"released": True}
        self._send(conn, {"jsonrpc": "2.0", "id": req["id"], "result": result})

    @staticmethod
    def _send(conn, doc):
        try:
            conn.sendall(json.dumps(doc).encode("ascii") + b"\n")
        except OSError:
            pass


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    path = tmp_path / "gl.sock"
    monkeypatch.setenv("GPU_LIGHT_SOCKET", str(path))
    fake = FakeDaemon(path)
    yield fake
    fake.close()
