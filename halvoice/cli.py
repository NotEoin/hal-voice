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

    server_dir = config.resolve(str(config.load().get("tts.server.gptsovits_dir", "")))
    if not server_dir or not Path(server_dir).is_dir():
        print(
            f"GPT-SoVITS not found at {server_dir or '(tts.server.gptsovits_dir is empty)'}.\n"
            "Run ./setup-voice.sh (it clones it), or set tts.server.gptsovits_dir.\n"
            "If it lives on a drive that mounts late, add RequiresMountsFor=<path> to hal-tts.service\n"
            "with `systemctl --user edit hal-tts.service`.",
            file=sys.stderr,
        )
        return 1
    script = config.root_dir() / "gptsovits" / "run-api.sh"
    os.environ["GPTSOVITS_DIR"] = server_dir
    os.execv(script, [str(script), *args])  # noqa: S606 - our own script; the server keeps this pid


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
    from . import config, util  # not at the top: `help` must run without the dependencies

    util.ca_bundle()
    try:
        return handler(args[1:])
    except config.ConfigError as e:
        print(f"hal-voice: settings error: {e}", file=sys.stderr)
        return CONFIG_ERROR_EXIT


if __name__ == "__main__":
    sys.exit(main())
