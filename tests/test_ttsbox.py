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


def test_cpu_docker_argv_has_no_gpu_and_no_claim():
    argv = ttsbox.docker_argv(
        "/g", "/w", "/run/u/hal-voice", 1000, 1001, "t:1", device="cpu", half=True, claim=None
    )
    joined = " ".join(argv)
    assert "--gpus" not in joined
    assert "--runtime" not in joined
    assert "GPU_LIGHT_CLAIM" not in joined
    assert "-e HALVOICE_TTS_DEVICE=cpu" in joined
    assert "-e HALVOICE_TTS_HALF=false" in joined  # the CPU has no fp16


def test_gpu_docker_argv_without_a_claim_is_unmarked():
    argv = ttsbox.docker_argv("/g", "/w", "/r", 1, 1, "t:1", claim=None)
    assert "--gpus all" in " ".join(argv)
    assert "GPU_LIGHT_CLAIM" not in " ".join(argv)


def test_cpu_preflight_does_not_need_the_nvidia_runtime(monkeypatch):
    monkeypatch.setattr(ttsbox.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(ttsbox.subprocess, "run", fake_info('{"runc":{}}'))
    ttsbox.preflight("cpu")


def tts_cfg(tmp_path, isolation, **tts):
    (tmp_path / "gsv").mkdir(exist_ok=True)
    tree = {
        "tts": {
            "api_url": "http://127.0.0.1:9999",
            "server": {
                "gptsovits_dir": str(tmp_path / "gsv"),
                "weights_dir": str(tmp_path / "w"),
                "isolation": isolation,
            },
            **tts,
        },
        "gpu": {"mode": "on_demand"},
    }
    return config.Cfg(tree)


def test_container_tts_server_runs_on_the_cpu_by_default(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(config, "load", lambda force=False: tts_cfg(tmp_path, "container"))
    monkeypatch.setattr(ttsbox, "serve", lambda *a, **k: seen.update(args=a, kwargs=k) or 0)
    assert cli.main(["tts-server"]) == 0
    assert seen["args"][2] == 9999
    assert seen["kwargs"] == {"device": "cpu", "half": False, "claim": None}


def test_container_tts_server_on_a_gpu_in_resident_mode_is_claimed(monkeypatch, tmp_path):
    seen = {}
    tree = tts_cfg(tmp_path, "container", device="cuda", half=True)
    tree["gpu"]["mode"] = "resident"
    monkeypatch.setattr(config, "load", lambda force=False: tree)
    monkeypatch.setattr(ttsbox, "serve", lambda *a, **k: seen.update(kwargs=k) or 0)
    assert cli.main(["tts-server"]) == 0
    assert seen["kwargs"] == {"device": "cuda", "half": True, "claim": "hal-voice"}


def test_host_tts_server_is_told_the_cpu_by_default(monkeypatch, tmp_path):
    from halvoice import ttspins

    monkeypatch.setattr(config, "load", lambda force=False: tts_cfg(tmp_path, "host"))
    monkeypatch.setattr(ttspins, "verify", lambda *a: None)
    started = {}
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: started.update(path=path))
    for key in ("HALVOICE_RESOLVED_TTS_DEVICE", "HALVOICE_RESOLVED_TTS_HALF"):
        monkeypatch.delenv(key, raising=False)
    cli.main(["tts-server"])
    assert str(started["path"]).endswith("gptsovits/run-api.sh")
    assert cli.os.environ["HALVOICE_RESOLVED_TTS_DEVICE"] == "cpu"
    assert cli.os.environ["HALVOICE_RESOLVED_TTS_HALF"] == "false"


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
