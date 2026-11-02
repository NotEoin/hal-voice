"""VoiceClaims against a fake gpu-light daemon on a UNIX socket in a temp directory."""

import json
import socket
import threading
import time

import pytest

from halvoice import config, gpu
from halvoice.config import ConfigError

WAIT_S = 5.0


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


def make(mode="on_demand", linger_s=30.0):
    return gpu.VoiceClaims(mode, resident_mib=3584, brain_mib=14000, linger_s=linger_s)


def test_room_is_requested_when_a_request_is_heard_and_released_after_linger(daemon):
    claims = make(linger_s=0.2)
    claims.turn_started()
    assert claims.wait_for_turn(WAIT_S) is True
    request = next(p for m, p in daemon.log if m == "room.request")
    assert request == {"name": "hal-voice-turn", "gpu": 0, "vram_mib": 14000}
    assert daemon.open == 1  # the connection is the room's lifetime
    claims.turn_ended()
    assert daemon.wait_for(lambda: daemon.count("room.release") == 1)
    assert daemon.wait_for(lambda: daemon.open == 0)
    claims.close()


def test_hello_names_protocol_2(daemon):
    claims = make()
    claims.turn_started()
    claims.wait_for_turn(WAIT_S)
    hello = daemon.log[0]
    assert hello[0] == "hello"
    assert hello[1]["protocol"] == 2
    assert hello[1]["client"] == "hal-voice"
    claims.close()


def test_follow_up_turns_keep_one_room(daemon):
    claims = make(linger_s=0.4)
    claims.turn_started()
    assert claims.wait_for_turn(WAIT_S)
    claims.turn_ended()
    time.sleep(0.1)
    claims.turn_started()  # a follow-up inside the linger
    assert claims.wait_for_turn(WAIT_S)
    time.sleep(0.6)  # past the first linger: the timer must have been cancelled
    assert daemon.count("room.release") == 0
    assert daemon.count("room.request") == 1
    claims.turn_ended()
    assert daemon.wait_for(lambda: daemon.count("room.release") == 1)
    claims.close()
    assert daemon.count("room.request") == 1


def test_close_gives_the_room_back_at_once(daemon):
    claims = make()
    claims.turn_started()
    assert claims.wait_for_turn(WAIT_S)
    claims.close()
    assert daemon.count("room.release") == 1
    assert daemon.wait_for(lambda: daemon.open == 0)


def test_no_daemon_means_no_error_and_the_turn_proceeds(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("GPU_LIGHT_SOCKET", str(tmp_path / "absent.sock"))
    claims = make()
    started = time.monotonic()
    for _ in range(3):
        claims.turn_started()
        assert claims.wait_for_turn(WAIT_S) is False
        claims.turn_ended()
    assert time.monotonic() - started < WAIT_S
    claims.close()
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1


def test_a_refused_room_is_logged_once_and_the_turn_proceeds(daemon, caplog):
    daemon.room_error = {"code": -32002, "message": "timeout"}
    claims = make()
    for _ in range(2):
        claims.turn_started()
        assert claims.wait_for_turn(WAIT_S) is False
        claims.close()
        claims._turn = None
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_a_slow_room_answer_is_waited_for_only_up_to_turn_wait_s(daemon):
    daemon.room_delay_s = 1.5
    claims = make()
    claims.turn_started()
    started = time.monotonic()
    assert claims.wait_for_turn(0.3) is False  # the brain runs without the room
    assert time.monotonic() - started < 1.0
    assert claims.wait_for_turn(WAIT_S) is True  # the room still arrives
    claims.close()


def test_off_mode_never_connects(daemon):
    claims = make("off")
    claims.start_service()
    claims.turn_started()
    assert claims.wait_for_turn(0.2) is False
    claims.turn_ended()
    claims.close()
    assert daemon.log == []


def test_on_demand_holds_no_resident_claim(daemon):
    claims = make()
    claims.start_service()
    claims.close()
    assert daemon.log == []


def test_resident_mode_still_claims_the_models_and_each_turn(daemon):
    claims = make("resident", linger_s=0.1)
    claims.start_service()
    service = next(p for m, p in daemon.log if m == "claim")
    assert service["name"] == "hal-voice"
    assert service["class"] == "service"
    assert service["vram_mib"] == 3584
    claims.turn_started()
    assert claims.wait_for_turn(WAIT_S)
    turn = [p for m, p in daemon.log if m == "claim"][1]
    assert (turn["name"], turn["class"], turn["exclusive"], turn["uses"]) == (
        "hal-voice-turn",
        "interactive",
        True,
        ["hal-gateway"],
    )
    assert daemon.count("room.request") == 0
    claims.turn_ended()
    assert daemon.wait_for(lambda: daemon.count("claim.release") == 1)
    assert daemon.open == 1  # the service claim stays until close
    claims.close()
    assert daemon.wait_for(lambda: daemon.open == 0)


def test_resident_without_a_daemon_carries_on(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_LIGHT_SOCKET", str(tmp_path / "absent.sock"))
    claims = make("resident")
    claims.start_service()
    claims.turn_started()
    assert claims.wait_for_turn(WAIT_S) is False
    claims.close()


def test_socket_path_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("GPU_LIGHT_SOCKET")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert gpu.socket_path() == tmp_path / "gpu-light" / "daemon.sock"
    monkeypatch.setenv("GPU_LIGHT_SOCKET", "/x/y.sock")
    assert str(gpu.socket_path()) == "/x/y.sock"


def test_probe_returns_the_hello_answer(daemon):
    assert gpu.probe()["protocol"] == 2


def test_probe_without_a_daemon_raises_gpu_light_error():
    with pytest.raises(gpu.GpuLightError):
        gpu.probe()


def test_unknown_mode_is_a_config_error():
    with pytest.raises(ConfigError, match=r"gpu\.mode"):
        gpu.gpu_mode(config.Cfg({"gpu": {"mode": "sometimes"}}))
    with pytest.raises(ConfigError, match=r"gpu\.mode"):
        make("sometimes")


def test_the_packaged_default_mode_is_on_demand():
    assert gpu.gpu_mode(config.load()) == "on_demand"
