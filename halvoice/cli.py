"""Command-line entry point: `hal-voice [--project DIR] <command>`.

Each command hands over to the module that implements it. Modules are imported only when
their command runs, because most of them pull in audio libraries that `help` and `doctor`
must work without.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

USAGE = """\
hal-voice - a HAL-9000 voice: speak to it, it answers aloud.

  hal-voice              start the voice loop (push-to-talk or hands-free -> the brain -> spoken reply)
  hal-voice repl         alias for the default voice loop
  hal-voice mcp          run ONLY the MCP speak/listen server (for use inside another harness)
  hal-voice tts "text"   one-shot: synthesize and speak a line in HAL's voice
  hal-voice tts --out FILE.wav "text"
                         synthesize to a WAV file instead of speaking
  hal-voice transcribe-file PATH [--json]
                         print the transcript of an audio or video file
  hal-voice tts-server   run the GPT-SoVITS server the voice speaks through
  hal-voice tts-image build   build the TTS container image
  hal-voice tts-pins record|verify   pin and check the checkpoints host-mode TTS loads
  hal-voice listen       one-shot: capture one utterance and print the transcript
  hal-voice devices      list audio input/output devices
  hal-voice eye          run the red-eye UI on its own (demo the animation states)
  hal-voice doctor       check that every dependency and the GPT-SoVITS server are reachable
  hal-voice voicetest    replay a recording of voicetest/script.txt through hands-free listening
  hal-voice connection   show which Claude Code setup it will drive, and why
  hal-voice config import-built OLDFILE
                         copy the voice settings of an older checkout's config.yaml into built.yaml
  hal-voice help         show this help

Global option:  --project DIR   run Claude Code in DIR (claude mode; default: [hal].vault)
"""

PROJECT_ENV = "HALVOICE_CLAUDE_PROJECT_DIR"
CONFIG_ERROR_EXIT = 2  # a usage error: the user's settings, not hal-voice, need fixing


def _split_global_options(argv: list[str]) -> list[str]:
    """Removes `--project DIR` (accepted before or after the command) and exports it."""
    rest: list[str] = []
    it = iter(argv)
    for arg in it:
        if arg == "--project":
            os.environ[PROJECT_ENV] = next(it, "")
        elif arg.startswith("--project="):
            os.environ[PROJECT_ENV] = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    return rest


def _repl(args: list[str]) -> int:
    from . import loop

    loop.main()
    return 0


def _mcp(args: list[str]) -> int:
    from . import mcp_server

    mcp_server.main()
    return 0


def _tts(args: list[str]) -> int:
    from . import tts

    out = None
    if args[:1] == ["--out"]:
        out, args = (args[1] if len(args) > 1 else None), args[2:]
        if out is None:
            print("hal-voice tts: --out needs a WAV path", file=sys.stderr)
            return 2
    text = " ".join(args).strip()
    if not text:
        print('usage: hal-voice tts [--out FILE.wav] "line to speak"', file=sys.stderr)
        return 2
    if out is None:
        tts.speak(text)
        return 0
    import soundfile

    audio, rate = tts.synth(text)
    soundfile.write(out, audio, rate)
    print(out)
    return 0


def _transcribe_file(args: list[str]) -> int:
    """`transcribe-file PATH [--json]`: prints the transcript of an audio or video file."""
    import json

    from haltranscribe import tools

    as_json = "--json" in args
    paths = [a for a in args if a != "--json"]
    if len(paths) != 1:
        print("usage: hal-voice transcribe-file PATH [--json]", file=sys.stderr)
        return 2
    try:
        result = tools.transcribe(paths[0])
    except (ValueError, OSError) as e:
        print(f"hal-voice: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result) if as_json else result["text"])
    return 0


def _tts_server(args: list[str]) -> int:
    """Starts the GPT-SoVITS server, or exits 1 naming the directory it could not find."""
    from . import config

    cfg = config.load()
    server_dir = config.resolve(str(cfg.get("tts.server.gptsovits_dir", "")))
    if not server_dir or not Path(server_dir).is_dir():
        print(
            f"GPT-SoVITS not found at {server_dir or '(tts.server.gptsovits_dir is empty)'}.\n"
            "Run ./setup-voice.sh (it clones it), or set tts.server.gptsovits_dir.\n"
            "If it lives on a drive that mounts late, add RequiresMountsFor=<path> to hal-tts.service\n"
            "with `systemctl --user edit hal-tts.service`.",
            file=sys.stderr,
        )
        return 1
    isolation = str(cfg.get("tts.server.isolation", "container"))
    if isolation == "container":
        return _tts_container(cfg, server_dir)
    if isolation != "host":
        print(f"tts.server.isolation must be container or host, not {isolation!r}", file=sys.stderr)
        return 1
    from . import placement, ttspins

    weights = config.resolve(str(cfg.get("tts.server.weights_dir", "")))
    try:
        ttspins.verify(Path(server_dir), Path(weights))
    except ttspins.PinError as e:
        print(f"hal-voice: refusing to start the host TTS server: {e}", file=sys.stderr)
        return 1
    script = config.root_dir() / "gptsovits" / "run-api.sh"
    placed = placement.resolve(cfg)
    os.environ["HALVOICE_RESOLVED_TTS_DEVICE"] = placed.tts_device
    os.environ["HALVOICE_RESOLVED_TTS_HALF"] = "true" if placed.tts_half else "false"
    os.environ["GPTSOVITS_DIR"] = server_dir
    # host mode keeps torch.load(weights_only=True); the pins above are what makes that enough
    os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    os.execv(script, [str(script), *args])  # noqa: S606 - our own script; the server keeps this pid


def _tts_container(cfg, server_dir: str) -> int:
    """Runs the TTS server in its container; exits 1 naming a missing docker piece."""
    from urllib.parse import urlparse

    from . import config, placement, ttsbox

    port = urlparse(str(cfg.get("tts.api_url", "http://127.0.0.1:9880"))).port or 9880
    weights = config.resolve(str(cfg.get("tts.server.weights_dir", "")))
    placed = placement.resolve(cfg)
    # the container's VRAM belongs to the resident claim only in resident mode
    claim = "hal-voice" if placed.mode == "resident" else None
    try:
        return ttsbox.serve(
            server_dir, weights, port, device=placed.tts_device, half=placed.tts_half, claim=claim
        )
    except ttsbox.TtsBoxError as e:
        print(f"hal-voice: {e}", file=sys.stderr)
        return 1


def _tts_pins(args: list[str]) -> int:
    """`hal-voice tts-pins record|verify`: pins the checkpoints host-mode TTS may load."""
    from . import config, ttspins

    if args not in (["record"], ["verify"]):
        print("usage: hal-voice tts-pins record|verify", file=sys.stderr)
        return 2
    cfg = config.load()
    gsv = Path(config.resolve(str(cfg.get("tts.server.gptsovits_dir", ""))) or "")
    weights = Path(config.resolve(str(cfg.get("tts.server.weights_dir", ""))) or "")
    try:
        if args == ["record"]:
            pins = ttspins.record(gsv, weights)
            print(f"pinned {len(pins)} checkpoints in {ttspins.lock_path()}")
        else:
            ttspins.verify(gsv, weights)
            print("all checkpoints match their pins")
    except ttspins.PinError as e:
        print(f"hal-voice: {e}", file=sys.stderr)
        return 1
    return 0


def _tts_image(args: list[str]) -> int:
    """`hal-voice tts-image build`: builds the TTS container image."""
    from . import config, ttsbox

    if args != ["build"]:
        print("usage: hal-voice tts-image build", file=sys.stderr)
        return 2
    server_dir = config.resolve(str(config.load().get("tts.server.gptsovits_dir", "")))
    try:
        return ttsbox.build_image(server_dir, config.root_dir() / "docker")
    except ttsbox.TtsBoxError as e:
        print(f"hal-voice: {e}", file=sys.stderr)
        return 1


def _listen(args: list[str]) -> int:
    from . import stt

    print("Listening... speak now.", file=sys.stderr)
    print(stt.listen_once())
    return 0


def _devices(args: list[str]) -> int:
    from . import devices

    devices.main()
    return 0


def _eye(args: list[str]) -> int:
    from . import ui_eye

    ui_eye.demo()
    return 0


def _doctor(args: list[str]) -> int:
    from . import doctor

    doctor.main()
    return 0


def _connection(args: list[str]) -> int:
    from . import connect

    connect.explain()
    return 0


def _config(args: list[str]) -> int:
    """`config import-built OLDFILE`: moves an old in-repo voice into `built.yaml`."""
    from . import config

    if len(args) != 2 or args[0] != "import-built":
        print("usage: hal-voice config import-built OLDFILE", file=sys.stderr)
        return 2
    old = Path(args[1]).expanduser()
    if not old.is_file():
        print(f"hal-voice: no such file: {old}", file=sys.stderr)
        return 1
    print(f"voice settings written to {config.import_built(old)}")
    return 0


def _voicetest(args: list[str]) -> int:
    from voicetest.__main__ import main as voicetest_main

    return voicetest_main(args)


COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "repl": _repl,
    "mcp": _mcp,
    "tts": _tts,
    "tts-server": _tts_server,
    "tts-image": _tts_image,
    "tts-pins": _tts_pins,
    "transcribe-file": _transcribe_file,
    "listen": _listen,
    "devices": _devices,
    "eye": _eye,
    "doctor": _doctor,
    "connection": _connection,
    "config": _config,
    "voicetest": _voicetest,
}


def main(argv: list[str] | None = None) -> int:
    """Runs one hal-voice command and returns the process exit status."""
    args = _split_global_options(list(sys.argv[1:] if argv is None else argv))
    command = args[0] if args else "repl"
    if command in ("help", "-h", "--help"):
        print(USAGE, end="")
        return 0
    handler = COMMANDS.get(command)
    if handler is None:
        print(f"Unknown command: {command}\n", file=sys.stderr)
        print(USAGE, end="", file=sys.stderr)
        return 1
    from . import config, hostplatform, util  # not at the top: `help` must run without them

    if command != "doctor":  # doctor reports an unsupported platform as a line instead
        try:
            hostplatform.require_supported()
        except hostplatform.UnsupportedPlatform as e:
            print(f"hal-voice: {e}", file=sys.stderr)
            return 1
    util.ca_bundle()
    try:
        return handler(args[1:])
    except config.ConfigError as e:
        print(f"hal-voice: settings error: {e}", file=sys.stderr)
        return CONFIG_ERROR_EXIT


if __name__ == "__main__":
    sys.exit(main())
