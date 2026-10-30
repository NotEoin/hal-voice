"""Checksum pins for the checkpoints host-mode TTS loads (`hal-voice tts-pins record|verify`).

Host mode runs GPT-SoVITS outside a box with torch's `weights_only=True` loading kept on, so
a swapped checkpoint is the remaining way to run code there. `record` is a human command: it
hashes every `*.ckpt` and `*.pth` under the pretrained models and the weights directory into
`models.lock` in the user's hal-voice config dir; `verify` runs before each host start and
fails on a missing pin, a changed hash or an unpinned file, naming the file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

LOCK_NAME = "models.lock"
PRETRAINED = "GPT_SoVITS/pretrained_models"
PATTERNS = ("*.ckpt", "*.pth")
CHUNK = 1 << 20


class PinError(RuntimeError):
    """A checkpoint does not match the lock; the message names the file."""


def lock_path() -> Path:
    """Where the lock file lives: the user's hal-voice config directory."""
    return config.user_config_dir() / LOCK_NAME


def sha256(path: Path) -> str:
    """The SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def collect(gsv_dir: Path, weights_dir: Path) -> dict[str, Path]:
    """Maps a stable key to each checkpoint: `pretrained/...` or `weights/...`."""
    found: dict[str, Path] = {}
    for prefix, base in (("pretrained", gsv_dir / PRETRAINED), ("weights", weights_dir)):
        for pattern in PATTERNS:
            for path in sorted(base.rglob(pattern)) if base.is_dir() else []:
                found[f"{prefix}/{path.relative_to(base).as_posix()}"] = path
    return found


def record(gsv_dir: Path, weights_dir: Path, lock: Path | None = None) -> dict[str, str]:
    """Hashes every checkpoint and writes the lock; raises `PinError` when there is none."""
    files = collect(gsv_dir, weights_dir)
    if not files:
        raise PinError(f"no *.ckpt or *.pth under {gsv_dir / PRETRAINED} or {weights_dir}")
    pins = {key: sha256(path) for key, path in files.items()}
    lock = lock or lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return pins


def verify(gsv_dir: Path, weights_dir: Path, lock: Path | None = None) -> None:
    """Raises `PinError` naming the first missing, changed or unpinned checkpoint."""
    lock = lock or lock_path()
    if not lock.is_file():
        raise PinError(
            f"{lock} not found; review the checkpoints, then run `hal-voice tts-pins record`"
        )
    pins = json.loads(lock.read_text(encoding="ascii"))
    files = collect(gsv_dir, weights_dir)
    for key in sorted(files):
        if key not in pins:
            raise PinError(f"{key} is not pinned in {lock}")
        if sha256(files[key]) != pins[key]:
            raise PinError(f"{key} does not match its pin in {lock}")
    for key in sorted(set(pins) - set(files)):
        raise PinError(f"{key} is pinned in {lock} but missing")
