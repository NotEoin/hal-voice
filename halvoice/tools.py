"""hal-voice's hal-hub tool functions; `halvoice.hubtool` runs them one call at a time."""

from __future__ import annotations

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
