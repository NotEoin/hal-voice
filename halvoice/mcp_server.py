"""hal-voice MCP server — gives the assistant a voice as MCP tools.

The voice loop speaks every reply. This server is the other way in: it lets HAL speak,
or take dictation, on demand from inside a normal interactive Claude Code session.
Register it in that project's `.mcp.json`:

  "hal-voice": {
    "command": "/abs/path/to/hal-voice/.venv/bin/python",
    "args": ["-m", "halvoice.mcp_server"],
    "env": { "HALVOICE_ROOT": "/abs/path/to/hal-voice" }
  }

Tools:
  speak(text)        — synthesize `text` in HAL's voice and play it on the machine.
  listen(seconds)    — capture one spoken utterance from the mic and return the transcript.
  set_voice_speed(x) — nudge HAL's speaking rate at runtime (0.8–1.1 sensible).
"""

from __future__ import annotations

import os

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    raise SystemExit(
        "The 'mcp' package isn't installed in this environment. Install hal-voice's deps:\n"
        "  ./setup-voice.sh   (or: pip install -e .)")

from . import config, tts
from .stt import listen_once
from .util import get_logger

log = get_logger(__name__)
mcp = FastMCP("hal-voice")


@mcp.tool()
def speak(text: str) -> str:
    """Speak the given text aloud in HAL 9000's voice on the local machine's speakers.
    Use this when the user asks you to *say* something, to read out a short summary, or to
    respond verbally. Keep spoken text concise and conversational — do not read code or long
    file contents aloud. Returns once playback has finished.
    """
    text = (text or "").strip()
    if not text:
        return "Nothing to speak."
    try:
        tts.speak(text)
    except Exception as e:  # noqa: BLE001
        return f"Could not speak (is the GPT-SoVITS server running?): {e}"
    return f"Spoke {len(text)} characters."


@mcp.tool()
def listen(seconds: int = 30) -> str:
    """Capture one spoken utterance from the microphone and return the transcribed text.
    Use this to take dictation or get a spoken answer from the user. Records until the user
    stops talking (or `seconds` elapse), then returns the transcript. `seconds` caps the
    recording length.
    """
    prev = os.environ.get("HALVOICE_AUDIO_MAX_UTTERANCE_S")
    os.environ["HALVOICE_AUDIO_MAX_UTTERANCE_S"] = str(int(seconds))
    config.load(force=True)
    try:
        text = listen_once()
    except Exception as e:  # noqa: BLE001
        return f"Could not listen (is a microphone available?): {e}"
    finally:
        if prev is None:
            os.environ.pop("HALVOICE_AUDIO_MAX_UTTERANCE_S", None)
        else:
            os.environ["HALVOICE_AUDIO_MAX_UTTERANCE_S"] = prev
        config.load(force=True)
    return text or "(heard nothing)"


@mcp.tool()
def set_voice_speed(speed: float) -> str:
    """Set HAL's speaking rate for subsequent speech. 1.0 is the tuned default; lower is
    slower and more deliberate (0.85–0.95 is most HAL-like), higher is faster. Range 0.7–1.2.
    """
    speed = max(0.7, min(1.2, float(speed)))
    os.environ["HALVOICE_TTS_SPEED"] = str(speed)
    config.load(force=True)
    return f"HAL speaking rate set to {speed:.2f}."


def main() -> None:
    """Runs the MCP server on stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
