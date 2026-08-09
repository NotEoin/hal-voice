"""`hal-voice doctor` — check that everything voice-related is wired up and reachable.

Verifies: python deps, audio I/O, the STT model, the GPT-SoVITS server + HAL weights,
the reference clip, and Claude Code on PATH. Prints a tidy pass/fail report; exits non-zero
if anything essential is missing.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from . import config

OK, WARN, BAD = "✓", "!", "✗"


def _check(label, fn):
    try:
        detail = fn()
        print(f"  {OK} {label}{(' — ' + detail) if detail else ''}")
        return True
    except _Warn as w:
        print(f"  {WARN} {label} — {w}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  {BAD} {label} — {e}")
        return False


class _Warn(Exception):
    pass


def _imports():
    import numpy, requests, soundfile, sounddevice, yaml  # noqa: F401
    mods = []
    for m in ("faster_whisper", "webrtcvad", "pynput",
              "pedalboard", "pyloudnorm", "pygame", "mcp"):
        try:
            __import__(m)
        except Exception:  # noqa: BLE001
            mods.append(m)
    if mods:
        raise _Warn("modules missing: " + ", ".join(mods)
                    + " — re-run ./setup-voice.sh")
    # Wake word is an opt-in extra (requirements-wake.txt), so its absence is normal
    # and should not read as a broken install.
    try:
        __import__("openwakeword")
        return "all deps import, including wake word"
    except Exception:  # noqa: BLE001
        return "all deps import (wake word not installed — optional, push-to-talk unaffected)"


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
    from faster_whisper import WhisperModel  # noqa: F401
    return f"faster-whisper available (model '{name}' downloads on first use)"


def _reference_clip():
    cfg = config.load()
    ref = config.resolve(cfg.get("tts.ref_audio_path"))
    if not ref or not Path(ref).exists():
        raise Exception(f"reference clip not found: {ref} (see voice-data/README.md)")
    return ref


def _weights():
    cfg = config.load()
    missing = []
    for key in ("tts.gpt_weights", "tts.sovits_weights"):
        p = config.resolve(cfg.get(key))
        if not p or not Path(p).exists():
            missing.append(f"{key}={p}")
    if missing:
        raise _Warn("fine-tuned weights not found yet (train them — gptsovits/train-hal.md): "
                    + "; ".join(missing))
    return "HAL fine-tuned checkpoints present"


def _gptsovits_server():
    import requests
    cfg = config.load()
    url = str(cfg.get("tts.api_url", "http://127.0.0.1:9880")).rstrip("/")
    try:
        requests.get(url + "/", timeout=2)
    except Exception:
        # try a harmless control endpoint
        try:
            requests.get(url + "/set_gpt_weights", timeout=2)
        except Exception as e:  # noqa: BLE001
            raise _Warn(f"GPT-SoVITS api not reachable at {url} "
                        f"(start it: gptsovits/run-api.sh): {e}")
    return f"reachable at {url}"


def _claude():
    if not shutil.which(str(config.load().get("claude.command", "claude"))):
        raise Exception("Claude Code ('claude') not on PATH — install it from "
                        "https://claude.com/claude-code")
    return "claude on PATH"


def _connection():
    """What hal-voice will actually drive, and where. See halvoice/connect.py."""
    from . import connect
    conn = connect.resolve()
    if not Path(conn.project_dir).is_dir():
        raise Exception(f"project_dir does not exist: {conn.project_dir}")
    return f"{conn.describe()}  [from {' -> '.join(conn.sources)}]"


def _voice_profile():
    """The VRAM profile chosen by setup-voice.sh — separate from the Claude connection."""
    import os
    profile = os.environ.get("HALVOICE_PROFILE")
    cfg = config.load()
    stt_dev = cfg.get("stt.device", "cpu")
    tts_dev = os.environ.get("HALVOICE_TTS_DEVICE", "cuda")
    detail = f"speech recognition on {stt_dev}, voice on {tts_dev}"
    if not profile:
        raise _Warn(f"no voice.conf yet (run ./setup-voice.sh) — defaults: {detail}")
    return f"{profile} — {detail}"


def main() -> None:
    print("hal-voice doctor\n----------------")
    results = [
        _check("python dependencies", _imports),
        _check("audio devices", _audio),
        _check("voice profile", _voice_profile),
        _check("speech-to-text", _stt_model),
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
    print("Some essential checks failed (see ✗ above). Fix those, then re-run doctor.")
    sys.exit(1)


if __name__ == "__main__":
    main()
