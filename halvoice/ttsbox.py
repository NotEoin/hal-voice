"""Runs the GPT-SoVITS server in a GPU container and bridges its socket to a TCP port.

`hal-voice tts-server` (tts.server.isolation: container, the default) calls `serve`.
Design decisions:
- The container has `--network none`; it talks through a unix socket in a bind-mounted
  run directory, and `Forwarder` publishes that on 127.0.0.1 so `hal-say` keeps working.
- The `nvidia` runtime is runc-based and reaches a bind-mounted unix socket; plain runsc
  would need `--host-uds=open`. `runsc-gpu` fails on the target host (08, "TTS isolation").
- Missing docker or runtime is an error that names the piece; there is no fallback to
  host mode, because host mode loads pickle checkpoints outside a box.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
from pathlib import Path

from . import __version__

log = logging.getLogger(__name__)

IMAGE_REPO = "hal-voice-tts"
SOCKET_NAME = "tts.sock"
CONTAINER_RUN_DIR = "/run/hal-voice"
CONTAINER_TMP = "/tmp"  # noqa: S108 - a path inside the container, backed by tmpfs
PIDS_LIMIT = 512  # api_v2 plus its worker threads and ffmpeg children stay well below this
DOCKER_INFO_TIMEOUT_S = 10
IMAGE_BUILD_TIMEOUT_S = 3600  # CUDA base layers and the pip install are several GB
SOCKET_WAIT_S = 600  # the first start loads every model
CONNECT_TIMEOUT_S = 3
STREAM_IDLE_S = 120  # a synthesis call can be silent while the model runs
STOP_TIMEOUT_S = 30
BUFFER = 65536
BUILD_FILES = ("tts.Dockerfile", "tts_launcher.py")


class TtsBoxError(RuntimeError):
    """The container cannot start; the message names what is missing."""


def image_tag() -> str:
    """The image reference for this hal-voice version."""
    return f"{IMAGE_REPO}:{__version__}"


def run_dir() -> Path:
    """The per-user directory that holds the socket; created private (0700)."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    if not base:
        raise TtsBoxError("XDG_RUNTIME_DIR is not set; the TTS socket needs a runtime directory")
    path = Path(base) / "hal-voice"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def preflight() -> None:
    """Raises `TtsBoxError` naming the missing piece: docker, its daemon or the nvidia runtime."""
    if shutil.which("docker") is None:
        raise TtsBoxError(
            "docker is not installed or not on PATH (needed for tts.server.isolation: container)"
        )
    try:
        done = subprocess.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            capture_output=True,
            text=True,
            timeout=DOCKER_INFO_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TtsBoxError(
            "docker did not answer `docker info` in time; is the daemon running?"
        ) from exc
    if done.returncode != 0:
        raise TtsBoxError(
            f"docker daemon unreachable: {done.stderr.strip() or 'docker info failed'}"
        )
    if '"nvidia"' not in done.stdout:
        raise TtsBoxError(
            "docker has no `nvidia` runtime; install the NVIDIA Container Toolkit "
            "(`nvidia-ctk runtime configure --runtime=docker`)"
        )


def box_flags(sock_dir: str, uid: int, gid: int) -> list[str]:
    """The isolation flags every TTS container gets: no network, read-only, no capabilities."""
    return [
        "--rm", "--network", "none", "--read-only", "--tmpfs", CONTAINER_TMP,
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", str(PIDS_LIMIT), "-u", f"{uid}:{gid}",
        "-v", f"{sock_dir}:{CONTAINER_RUN_DIR}",
    ]  # fmt: skip


def docker_argv(
    gsv_dir: str, weights_dir: str, sock_dir: str, uid: int, gid: int, tag: str
) -> list[str]:
    """The `docker run` argv for the locked-down GPU container."""
    return [
        "docker", "run", "--name", "hal-voice-tts", "--runtime=nvidia", "--gpus", "all",
        *box_flags(sock_dir, uid, gid),
        "-e", "GPU_LIGHT_CLAIM=hal-voice",
        "-v", f"{gsv_dir}:/opt/gsv:ro",
        "-v", f"{weights_dir}:/weights:ro",
        tag,
    ]  # fmt: skip


class Forwarder:
    """Bridges `127.0.0.1:port` to a unix socket, one thread per connection."""

    def __init__(self, sock_path: str, port: int, host: str = "127.0.0.1") -> None:
        self.sock_path = sock_path
        self.server = socket.create_server((host, port))
        self.server.settimeout(0.5)
        self.port = self.server.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept, daemon=True)

    def start(self) -> None:
        """Starts accepting connections in a background thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stops accepting and closes the listener."""
        self._stop.set()
        self._thread.join(timeout=STOP_TIMEOUT_S)
        self.server.close()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self.server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(target=self._bridge, args=(client,), daemon=True).start()

    def _bridge(self, client: socket.socket) -> None:
        upstream = socket.socket(socket.AF_UNIX)
        try:
            upstream.settimeout(CONNECT_TIMEOUT_S)
            upstream.connect(self.sock_path)
        except OSError as exc:
            log.warning("tts socket %s not reachable: %s", self.sock_path, exc)
            client.close()
            upstream.close()
            return
        for s in (client, upstream):
            s.settimeout(STREAM_IDLE_S)
        back = threading.Thread(target=_pump, args=(upstream, client), daemon=True)
        back.start()
        _pump(client, upstream)
        back.join(timeout=STREAM_IDLE_S)


def _pump(src: socket.socket, dst: socket.socket) -> None:
    """Copies bytes until `src` closes or idles, then half-closes `dst` and closes both."""
    try:
        while data := src.recv(BUFFER):
            dst.sendall(data)
        dst.shutdown(socket.SHUT_WR)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.close()
            except OSError:
                pass


def build_image(gsv_dir: str, docker_dir: Path) -> int:
    """Builds the TTS image; the context is a temp dir with the Dockerfile, launcher and requirements."""
    reqs = Path(gsv_dir) / "requirements.txt"
    if not reqs.is_file():
        raise TtsBoxError(f"{reqs} not found; run ./setup-voice.sh to clone GPT-SoVITS")
    with tempfile.TemporaryDirectory() as ctx:
        for name in BUILD_FILES:
            shutil.copy(docker_dir / name, ctx)
        shutil.copy(reqs, Path(ctx) / "requirements.txt")
        return subprocess.run(
            ["docker", "build", "-f", str(Path(ctx) / "tts.Dockerfile"), "-t", image_tag(), ctx],
            timeout=IMAGE_BUILD_TIMEOUT_S,
            check=False,
        ).returncode


def serve(gsv_dir: str, weights_dir: str, port: int) -> int:
    """Runs the container and the forwarder until the container exits or a signal arrives."""
    preflight()
    sock_dir = run_dir()
    sock = sock_dir / SOCKET_NAME
    argv = docker_argv(gsv_dir, weights_dir, str(sock_dir), os.getuid(), os.getgid(), image_tag())
    fwd = Forwarder(str(sock), port)
    fwd.start()
    # no timeout: the server runs until stopped; SIGTERM below ends it (see POPEN_ALLOWED)
    proc = subprocess.Popen(argv)

    def stop(*_: object) -> None:
        proc.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        return proc.wait()
    finally:
        fwd.stop()
