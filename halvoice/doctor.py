"""`hal-voice doctor` -- check that everything voice-related is wired up and reachable.

Verifies: python deps, audio I/O, the STT model, the brain (`hal` on PATH in hal
mode; the server in openai mode), the GPT-SoVITS server + HAL weights, the reference
clip, and Claude Code on PATH. Under WSL2 it also checks the WSLg audio server, the
display and systemd, because each is off or missing in a fresh distro. It also reports the GPU mode, whether the gpu-light daemon
answers, and the devices speech-to-text and the voice will use. Prints a tidy pass/fail
report; exits non-zero if anything essential is missing.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from . import config, gpu, hostplatform, placement

OK, WARN, BAD = "[x]", "!", "[ ]"

# WSLg serves PulseAudio on this socket; Windows owns the microphone and speakers
WSLG_PULSE = Path("/mnt/wslg/PulseServer")
SYSTEMD_RUN_DIR = Path("/run/systemd/system")  # exists only when systemd is PID 1


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


def _platform():
    """Linux or WSL2 passes; anything else fails with where to run hal-voice instead."""
    found = hostplatform.require_supported()
    release = hostplatform.osrelease_text() or "unknown kernel"
    return f"{'WSL2' if found == hostplatform.WSL2 else 'Linux'} ({release})"


def _wsl_audio_server():
    """WSL2 only: the WSLg PulseAudio socket must exist, or sounddevice sees no devices."""
    pulse = os.environ.get("PULSE_SERVER", "")
    if pulse.startswith("unix:"):
        socket_path = Path(pulse[len("unix:") :])
    else:
        socket_path = WSLG_PULSE
    if not socket_path.exists():
        raise Exception(
            f"no audio server at {socket_path}: update WSL with `wsl --update` in PowerShell, "
            f"then `wsl --shutdown` and reopen the distro (see {hostplatform.DOCS_HINT})"
        )
    return f"WSLg audio server at {socket_path}"


def _wsl_display():
    """WSL2 only: WSLg sets `DISPLAY` or `WAYLAND_DISPLAY`; the eye UI and pynput need one."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise _Warn(
            "no DISPLAY or WAYLAND_DISPLAY: the eye window and push-to-talk keys will not work; "
            "use hands-free mode, or run `wsl --update` to get WSLg"
        )
    return "display set (WSLg)"


def _wsl_systemd():
    """WSL2 only: the voice units need systemd, which a fresh distro leaves off."""
    if not SYSTEMD_RUN_DIR.is_dir():
        raise _Warn(
            "systemd is not running, so hal-voice.service and hal-tts.service cannot start; "
            "add `[boot]` and `systemd=true` to /etc/wsl.conf, then `wsl --shutdown`"
        )
    return "systemd is running"


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


def _mode() -> str:
    return str(config.load().get("brain.mode", "hal"))


def _openai_brain():
    """The openai brain: `brain.base_url` set, and its `/models` answering."""
    from .brains.openai import OpenAIBrain

    b = OpenAIBrain()  # raises BrainUnavailable naming the key when base_url is empty
    b.probe()
    return f"OpenAI-compatible server at {b.base_url}, model {b.model or '(server default)'}"


def _hal_brain():
    """The hal brain: `hal` on PATH (or the first word of `brain.hal_command`)."""
    command = config.load().get("brain.hal_command") or []
    if not command or not shutil.which(str(command[0])):
        raise Exception(
            f"{command[0] if command else 'brain.hal_command'} not found: install Hal and put "
            "`hal` on PATH, or set brain.mode to openai or claude"
        )
    return f"hal acp via {shutil.which(str(command[0]))}"


def _brain():
    """The configured brain: Hal's ACP agent, or the openai server."""
    if _mode() == "hal":
        return _hal_brain()
    if _mode() == "openai":
        return _openai_brain()
    raise _Warn(
        "brain.mode is claude: every turn is a Claude Code run (not private, not sandboxed)"
    )


def _claude():
    if not shutil.which(str(config.load().get("claude.command", "claude"))):
        msg = "Claude Code ('claude') not on PATH - install it from https://claude.com/claude-code"
        if _mode() != "claude":
            raise _Warn(msg + f" ({_mode()} mode does not use it)")
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
    placed = placement.resolve(config.load())
    detail = f"speech recognition on {placed.stt_device}, voice on {placed.tts_device}"
    if placed.mode == "on_demand" and profile:
        return f"{profile} profile ignored in on_demand mode - {detail}"
    if not profile:
        raise _Warn(f"no voice.conf yet (run ./setup-voice.sh) - defaults: {detail}")
    return f"{profile} - {detail}"


def _gpu_sharing():
    """The gpu.mode, whether the gpu-light daemon answers, and the devices that will be used."""
    placed = placement.resolve(config.load())
    devices = (
        f"speech recognition on {placed.stt_device}, voice on {placed.tts_device} "
        f"(from {placed.source})"
    )
    if placed.mode == "off":
        return f"gpu.mode off: gpu-light is not contacted; {devices}"
    try:
        daemon = gpu.probe()
    except gpu.GpuLightError as e:
        raise _Warn(
            f"gpu.mode {placed.mode}, but no gpu-light daemon answers ({e}); the loop runs "
            f"without GPU sharing; {devices}"
        ) from e
    return f"gpu.mode {placed.mode}, gpu-light {daemon.get('version', '?')} answers; {devices}"


def main() -> None:
    print("hal-voice doctor\n----------------")
    on_wsl = hostplatform.detect_platform() == hostplatform.WSL2
    results = [
        _check("platform", _platform),
        *(
            [
                _check("WSL2 audio server", _wsl_audio_server),
                _check("WSL2 display", _wsl_display),
                _check("WSL2 systemd", _wsl_systemd),
            ]
            if on_wsl
            else []
        ),
        _check("python dependencies", _imports),
        _check("audio devices", _audio),
        _check("voice profile", _voice_profile),
        _check("GPU sharing", _gpu_sharing),
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
