"""Container entry point: runs GPT-SoVITS api_v2 on a unix socket instead of a TCP port.

The container has no network, so the host reaches the server through a socket in a
bind-mounted directory. api_v2.py only offers `host`/`port`, so this launcher imports its
app and starts uvicorn with `uds`. Unverified against the pinned GPT-SoVITS commit
(api_v2 parses argv at import time); check it with `hal-voice tts-image build` and a run.
"""

import os
import sys

import uvicorn

SOCKET = os.environ.get("HALVOICE_TTS_SOCKET", "/run/hal-voice/tts.sock")
CONFIG = os.environ.get("HALVOICE_TTS_CONFIG", "GPT_SoVITS/configs/tts_infer.yaml")

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "GPT_SoVITS"))
sys.argv = ["api_v2.py", "-c", CONFIG]
import api_v2  # noqa: E402 - needs argv and sys.path set first

if os.path.exists(SOCKET):
    os.unlink(SOCKET)
uvicorn.run(api_v2.APP, uds=SOCKET, log_level="info")
