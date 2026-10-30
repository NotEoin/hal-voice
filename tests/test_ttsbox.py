"""Tests for the TTS container launcher: argv, preflight, socket forwarder, `tts-server` wiring."""

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from halvoice import cli, config, ttsbox

WAIT_S = 30


def test_docker_argv_is_the_locked_down_gpu_run():
    argv = ttsbox.docker_argv("/g", "/w", "/run/u/hal-voice", 1000, 1001, "hal-voice-tts:9")
    joined = " ".join(argv)
    for part in (
        "--runtime=nvidia --gpus all",
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--pids-limit 512",
        "-u 1000:1001",
        "-e GPU_LIGHT_CLAIM=hal-voice",
        "-v /g:/opt/gsv:ro",
        "-v /w:/weights:ro",
        "-v /run/u/hal-voice:/run/hal-voice",
    ):
        assert part in joined
    assert argv[-1] == "hal-voice-tts:9"


def test_preflight_without_docker_names_docker(monkeypatch):
    monkeypatch.setattr(ttsbox.shutil, "which", lambda name: None)
    with pytest.raises(ttsbox.TtsBoxError, match="docker is not installed"):
        ttsbox.preflight()


def fake_info(out, code=0):
    return lambda *a, **k: SimpleNamespace(returncode=code, stdout=out, stderr="boom")


def test_preflight_without_nvidia_runtime_names_it(monkeypatch):
    monkeypatch.setattr(ttsbox.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(ttsbox.subprocess, "run", fake_info('{"runc":{}}'))
    with pytest.raises(ttsbox.TtsBoxError, match="nvidia"):
        ttsbox.preflight()


def test_preflight_with_a_dead_daemon_says_so(monkeypatch):
    monkeypatch.setattr(ttsbox.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(ttsbox.subprocess, "run", fake_info("", code=1))
    with pytest.raises(ttsbox.TtsBoxError, match="unreachable"):
        ttsbox.preflight()


def echo_server(path: str):
    """A unix-socket server that upper-cases one message; returns its listening socket."""
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        conn.sendall(conn.recv(100).upper())
        conn.close()

    import threading

    threading.Thread(target=serve, daemon=True).start()
    return srv


def test_forwarder_carries_a_tcp_request_to_the_unix_socket(tmp_path):
    path = str(tmp_path / "tts.sock")
    srv = echo_server(path)
    fwd = ttsbox.Forwarder(path, 0)
    fwd.start()
    try:
        with socket.create_connection(("127.0.0.1", fwd.port), timeout=WAIT_S) as c:
            c.sendall(b"hello")
            assert c.recv(100) == b"HELLO"
    finally:
        fwd.stop()
        srv.close()


def test_forwarder_closes_the_client_when_the_socket_is_absent(tmp_path):
    fwd = ttsbox.Forwarder(str(tmp_path / "none.sock"), 0)
    fwd.start()
    try:
        with socket.create_connection(("127.0.0.1", fwd.port), timeout=WAIT_S) as c:
            assert c.recv(10) == b""
    finally:
        fwd.stop()


def test_tts_server_in_container_mode_exits_1_naming_the_missing_piece(
    monkeypatch, tmp_path, capsys
):
    (tmp_path / "gsv").mkdir()
    monkeypatch.setattr(
        config,
        "load",
        lambda force=False: config.Cfg(
            {"tts": {"server": {"gptsovits_dir": str(tmp_path / "gsv"), "isolation": "container"}}}
        ),
    )
    monkeypatch.setattr(ttsbox.shutil, "which", lambda name: None)
    assert cli.main(["tts-server"]) == 1
    assert "docker is not installed" in capsys.readouterr().err


def test_default_isolation_is_container():
    assert config.load(force=True).get("tts.server.isolation") == "container"


@pytest.mark.docker
def test_hardened_container_serves_a_socket_the_forwarder_reaches(tmp_path):
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")
    sock_dir = tmp_path / "run"
    sock_dir.mkdir(mode=0o700)
    code = (
        "import socket;s=socket.socket(socket.AF_UNIX);s.bind('/run/hal-voice/tts.sock');"
        "s.listen(1);c,_=s.accept();c.sendall(c.recv(100).upper())"
    )
    argv = [
        "docker", "run", *ttsbox.box_flags(str(sock_dir), os.getuid(), os.getgid()),
        "python:3.12-slim", "python", "-c", code,
    ]  # fmt: skip
    proc = subprocess.Popen(argv, stderr=subprocess.PIPE)
    fwd = ttsbox.Forwarder(str(sock_dir / "tts.sock"), 0)
    fwd.start()
    try:
        deadline = time.monotonic() + WAIT_S
        while not Path(sock_dir / "tts.sock").exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        with socket.create_connection(("127.0.0.1", fwd.port), timeout=WAIT_S) as c:
            c.sendall(b"hal")
            assert c.recv(100) == b"HAL"
    finally:
        fwd.stop()
        proc.terminate()
        proc.wait(timeout=WAIT_S)
