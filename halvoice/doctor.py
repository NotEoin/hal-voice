"""`hal-voice doctor` -- check that everything voice-related is wired up and reachable.

Verifies: python deps, audio I/O, the STT model, the direct brain (core hal, Ollama,
hal-hub), the GPT-SoVITS server + HAL weights, the reference clip, and Claude Code on
PATH. Prints a tidy pass/fail report; exits non-zero if anything essential is missing.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from . import config

OK, WARN, BAD = "[x]", "!", "[ ]"


def _check(label, fn):
    try:
        detail = fn()
        print(f"  {OK} {label}{(' - ' + detail) if detail else ''}")
        return True
    except _Warn as w:
        print(f"  {WARN} {label} - {w}")
        return True
    except Exception as e:  # noqa: BLE001 - doctor reports every failure as a line, never crashes
        print(f"  {BAD} {label} - {e}")
        return False


class _Warn(Exception):
    pass


def _imports():
    for essential in ("numpy", "requests", "soundfile", "sounddevice", "yaml"):
        __import__(essential)  # the import itself is the check; a failure is reported by _check
    mods = []
    for m in ("faster_whisper", "webrtcvad", "pynput", "pedalboard", "pyloudnorm", "pygame", "mcp"):
        try:
            __import__(m)
        except Exception:  # noqa: BLE001 - any import failure, not only ImportError, means unusable
            mods.append(m)
    if mods:
        raise _Warn("modules missing: " + ", ".join(mods) + " - re-run ./setup-voice.sh")
    # Wake word is an opt-in extra (the `wake` extra), so its absence is normal
    # and should not read as a broken install.
    try:
        __import__("openwakeword")
        return "all deps import, including wake word"
    except Exception:  # noqa: BLE001 - any import failure means wake word is not usable
        return "all deps import (wake word not installed - optional, push-to-talk unaffected)"


def _audio():
    import sounddevice as sd

    ins = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
    outs = [d for d in sd.query_devices() if d["max_output_channels"] > 0]
    if not ins:
        raise Exception("no input (microphone) devices found")
    if not outs:
        raise Exception("no output (speaker) devices found")
    return f"{len(ins)} mic(s), {len(outs)} output(s)"


def _stt_model():
    cfg = config.load()
    name = cfg.get("stt.model", "small.en")
    from faster_whisper import WhisperModel  # noqa: F401 - the import is the check

    return f"faster-whisper available (model '{name}' downloads on first use)"


def _old_voice_backups() -> list[Path]:
    """`config.yaml.*.bak` files an older voice build left in the checkout, newest first."""
    root = config.root_dir()
    found = [*root.glob("config.yaml.*.bak"), *(root / "halvoice").glob("config.yaml.*.bak")]
    return sorted(found, reverse=True)


def _reference_clip():
    cfg = config.load()
    ref = config.resolve(cfg.get("tts.ref_audio_path"))
    if not ref:
        hint = ""
        backups = _old_voice_backups()
        if not config.built_file().exists() and backups:
            hint = f"; to keep an older voice: hal-voice config import-built {backups[0]}"
        raise Exception(config.NO_VOICE + hint)
    if not Path(ref).exists():
        raise Exception(f"reference clip not found: {ref} (see voice-data/README.md)")
    return ref


def _weights():
    cfg = config.load()
    if not cfg.get("tts.gpt_weights") and not cfg.get("tts.sovits_weights"):
        raise _Warn("no fine-tuned weights set; the server's own voice speaks (run make-hal-voice)")
    missing = []
    for key in ("tts.gpt_weights", "tts.sovits_weights"):
        p = config.resolve(cfg.get(key))
        if not p or not Path(p).exists():
            missing.append(f"{key}={p}")
    if missing:
        raise _Warn(
            "fine-tuned weights not found yet (train them - gptsovits/train-hal.md): "
            + "; ".join(missing)
        )
    return "HAL fine-tuned checkpoints present"


def _gptsovits_server():
    import requests

    cfg = config.load()
    url = str(cfg.get("tts.api_url", "http://127.0.0.1:9880")).rstrip("/")
    try:
        requests.get(url + "/", timeout=2)
    except Exception:  # noqa: BLE001 - any failure of the first probe falls through to the second
        # try a harmless control endpoint
        try:
            requests.get(url + "/set_gpt_weights", timeout=2)
        except Exception as e:
            raise _Warn(
                f"GPT-SoVITS api not reachable at {url} (start it: gptsovits/run-api.sh): {e}"
            ) from e
    return f"reachable at {url}"


def _direct() -> bool:
    return str(config.load().get("brain.mode", "direct")) == "direct"


def _brain():
    """The direct brain: core hal, its model in Ollama, and the tools hal-hub found."""
    if not _direct():
        raise _Warn("brain.mode is claude: every turn is a Claude Code run")
    import json
    import urllib.request

    from . import brain

    core = brain.find_core()
    b = brain.direct_brain()
    try:
        with urllib.request.urlopen(f"{b.url}/api/tags", timeout=3) as r:  # noqa: S310 - a configured local http URL
            names = {m["name"] for m in json.loads(r.read()).get("models", [])}
    except OSError as e:
        raise Exception(f"Ollama not reachable at {b.url}: {e}") from e
    if b.model not in names and f"{b.model}:latest" not in names:
        raise Exception(f"model {b.model} is not in Ollama (ollama pull {b.model})")
    tools = f"{len(b.hub.all_tools())} tools" if b.hub else "no tools (hal-hub not found)"
    return f"core hal at {core}; {b.model} via {b.url}; {tools}"


def _claude():
    if not shutil.which(str(config.load().get("claude.command", "claude"))):
        msg = "Claude Code ('claude') not on PATH - install it from https://claude.com/claude-code"
        if _direct():
            raise _Warn(msg + " (direct mode needs it only for long background jobs)")
        raise Exception(msg)
    return "claude on PATH"


def _connection():
    """What hal-voice will actually drive, and where. See halvoice/connect.py."""
    from . import connect

    conn = connect.resolve()
    if not Path(conn.project_dir).is_dir():
        raise Exception(f"project_dir does not exist: {conn.project_dir}")
    return f"{conn.describe()}  [from {' -> '.join(conn.sources)}]"


def _voice_profile():
    """The VRAM profile chosen by setup-voice.sh -- separate from the Claude connection."""
    import os

    profile = os.environ.get("HALVOICE_PROFILE")
    cfg = config.load()
    stt_dev = cfg.get("stt.device", "cpu")
    tts_dev = os.environ.get("HALVOICE_TTS_DEVICE", "cuda")
    detail = f"speech recognition on {stt_dev}, voice on {tts_dev}"
    if not profile:
        raise _Warn(f"no voice.conf yet (run ./setup-voice.sh) - defaults: {detail}")
    return f"{profile} - {detail}"


def main() -> None:
    print("hal-voice doctor\n----------------")
    results = [
        _check("python dependencies", _imports),
        _check("audio devices", _audio),
        _check("voice profile", _voice_profile),
        _check("speech-to-text", _stt_model),
        _check("brain", _brain),
        _check("Claude Code", _claude),
        _check("Claude Code connection", _connection),
        _check("GPT-SoVITS server", _gptsovits_server),
        _check("HAL reference clip", _reference_clip),
        _check("HAL fine-tuned weights", _weights),
    ]
    print()
    if all(results):
        print("All essential checks passed. Run:  ./hal-voice")
        sys.exit(0)
    print("Some essential checks failed (see [ ] above). Fix those, then re-run doctor.")
    sys.exit(1)


if __name__ == "__main__":
    main()
