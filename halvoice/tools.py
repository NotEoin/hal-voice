"""hal-voice's hal-hub tools, run as a subprocess with the voice virtualenv.

    python -m halvoice.tools --stdin say    JSON arguments on stdin, JSON result on stdout
"""

from __future__ import annotations

import json
import sys

from . import config


def _server_up() -> bool:
    """True when the GPT-SoVITS server answers at all (it returns 404 on / when up)."""
    import requests

    url = str(config.load().get("tts.api_url", "http://127.0.0.1:9880")).rstrip("/")
    try:
        requests.get(url + "/", timeout=2)
        return True
    except requests.RequestException:
        return False


def say(text: str) -> dict:
    """Speaks a line in HAL's voice through the running voice server."""
    text = (text or "").strip()
    if not text:
        raise ValueError("nothing to say")
    if not _server_up():
        return {"spoken": False, "reason": "the voice server (hal-tts) is not running"}
    from . import tts

    tts.speak(text)
    return {"spoken": True}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ["--stdin"] or len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    fn = {"say": say}.get(argv[1])
    if fn is None:
        print(f"no function {argv[1]}", file=sys.stderr)
        return 2
    print(json.dumps(fn(**json.loads(sys.stdin.read() or "{}")), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
