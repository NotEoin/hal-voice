"""Container entry point: runs GPT-SoVITS api_v2 on a unix socket instead of a TCP port.

The container has no network, so the host reaches the server through a socket in a
bind-mounted directory. api_v2.py only offers `host`/`port`, so this launcher imports its
app and starts uvicorn with `uds`. Unverified against the pinned GPT-SoVITS commit
(api_v2 parses argv at import time); check it with `hal-voice tts-image build` and a run.
The device follows HALVOICE_TTS_DEVICE and HALVOICE_TTS_HALF, which `hal-voice tts-server`
sets, and is written into a copy of the config, because the mounted checkout is read-only.
Also unverified: that api_v2 reads the device from the `custom` block of that file.
"""

import os
import sys

import uvicorn
import yaml

SOCKET = os.environ.get("HALVOICE_TTS_SOCKET", "/run/hal-voice/tts.sock")
CONFIG = os.environ.get("HALVOICE_TTS_CONFIG", "GPT_SoVITS/configs/tts_infer.yaml")
DEVICE = os.environ.get("HALVOICE_TTS_DEVICE", "cpu")
HALF = os.environ.get("HALVOICE_TTS_HALF", "false").lower() == "true" and DEVICE != "cpu"
PLACED_CONFIG = "/tmp/tts_infer.yaml"  # noqa: S108 - inside the container, on its tmpfs


def place(source: str, target: str) -> str:
    """Writes a copy of the config with the device hal-voice chose; the mount is read-only."""
    with open(source, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("custom", {}).update({"device": DEVICE, "is_half": HALF})
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    return target


sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "GPT_SoVITS"))
sys.argv = ["api_v2.py", "-c", place(CONFIG, PLACED_CONFIG)]
import api_v2  # noqa: E402 - needs argv and sys.path set first

if os.path.exists(SOCKET):
    os.unlink(SOCKET)
uvicorn.run(api_v2.APP, uds=SOCKET, log_level="info")
