"""python -m voicebuild: builds HAL's voice from a copy of the film. ./make-hal-voice runs this.

./make-hal-voice                          everything in voice-data/film/ (drop the film there)
./make-hal-voice FILM [FILM ...]          these films, with subtitles found beside them
./make-hal-voice FILM --subs FILE.srt     with this subtitle file as well
./make-hal-voice --from select            redo the later stages of the last build
./make-hal-voice --only find              one stage, for a look at what it finds
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import build, media

# The same text as the README section of that name; tests/test_disclaimer.py keeps them equal.
DISCLAIMER = (Path(__file__).parent / "disclaimer.md").read_text(encoding="utf-8")

DROP = build.ROOT / "voice-data" / "film"
GPU_STAGES = {"find", "clean", "train", "select"}
# Stopping Hal's units waits for their own stop timeouts (15 s each).
HAL_STATUS_TIMEOUT_S = 10
HAL_SWITCH_TIMEOUT_S = 60
STAGE_TIMEOUT_S = 12 * 3600  # a queued stage waits for the GPU, then trains for under 6 h
DENOISE_PROBE_TIMEOUT_S = 120  # importing torch-based packages can be slow on a cold disk


def films_in(args: list[Path]) -> tuple[list[Path], list[Path]]:
    """Returns the films and subtitle files named, found in the folders named, or dropped in."""
    films, subs = [], []
    for p in args or [DROP]:
        p = p.expanduser()
        items = sorted(p.iterdir()) if p.is_dir() else [p]
        for item in items:
            suffix = item.suffix.lower()
            if suffix in media.SUBTITLE_SUFFIXES:
                subs.append(item)
            elif suffix in media.VIDEO_SUFFIXES and item.is_file():
                films.append(item)
    return films, subs


def denoise_python(given: str | None) -> str:
    """Returns a Python with demucs and DeepFilterNet: the one given, HALVOICE_DENOISE_PYTHON, or a
    conda environment called haldenoise or hal-voice-denoise."""
    candidates = [given, os.environ.get("HALVOICE_DENOISE_PYTHON")]
    base = Path(os.environ.get("CONDA_EXE", Path.home() / "miniconda3" / "bin" / "conda")).parents[
        1
    ]
    for root in {
        base,
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path.home() / "miniforge3",
    }:
        candidates += [
            str(root / "envs" / n / "bin" / "python") for n in ("haldenoise", "hal-voice-denoise")
        ]
    candidates.append(str(build.ROOT / ".venv-denoise" / "bin" / "python"))
    for c in candidates:
        if c and Path(c).exists():
            try:
                ok = subprocess.run(
                    [c, "-c", "import demucs, df, librosa, soundfile"],
                    capture_output=True,
                    check=False,
                    timeout=DENOISE_PROBE_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                continue
            if ok.returncode == 0:
                return c
    raise SystemExit(
        "No Python with demucs and DeepFilterNet found. Run ./setup-voice.sh --with-training, "
        "or set HALVOICE_DENOISE_PYTHON."
    )


def hal_is_on() -> bool:
    """True when `hal status` reports Hal running, which would hold the GPU."""
    if not shutil.which("hal"):
        return False
    try:
        out = subprocess.run(
            ["hal", "status"],
            capture_output=True,
            text=True,
            check=False,
            timeout=HAL_STATUS_TIMEOUT_S,
        ).stdout
    except subprocess.TimeoutExpired:
        print("`hal status` did not answer; leaving Hal as it is.", file=sys.stderr)
        return False
    return "Hal is on" in out and not out.startswith("Hal is on: 0 of")


def hal_switch(state: str) -> None:
    """Runs `hal on` or `hal off`; a timeout is reported, not raised, so the build carries on."""
    try:
        subprocess.run(["hal", state, "--quiet"], check=False, timeout=HAL_SWITCH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"`hal {state}` timed out after {HAL_SWITCH_TIMEOUT_S} s.", file=sys.stderr)


def stage_argv(stage: str, argv: list[str]) -> list[str]:
    """The command line of this build, narrowed to one stage."""
    kept: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
        elif arg in ("--from", "--only"):
            skip = True
        elif not arg.startswith(("--from=", "--only=")):
            kept.append(arg)
    return [sys.executable, "-m", "voicebuild", *kept, "--only", stage]


def run_under_gpu_light(stage: str) -> None:
    """Runs one GPU stage as a `gpu-light run` batch job; a failed stage stops the build."""
    argv = build.gpu_light_argv(stage_argv(stage, sys.argv[1:]))
    env = {**os.environ, build.IN_GPU_RUN_ENV: "1"}
    subprocess.run(argv, env=env, check=True, timeout=STAGE_TIMEOUT_S)


def report(b: build.Build) -> Path:
    """Writes report.md: where the lines came from, what was kept and what was dropped, and why."""
    out = [
        f"# HAL voice build, {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Log",
        "",
        "```text",
        *b.log,
        "```",
        "",
    ]
    for film in b.films:
        f = b.work / build.slug(film) / "lines.json"
        if not f.exists():
            continue
        lines = json.loads(f.read_text())
        out += [
            f"## {film.name}",
            "",
            f"Kept {len(lines['kept'])} lines:",
            "",
            "| Start | Line |",
            "| --- | --- |",
        ]
        out += [f"| {it['start']:.1f} | {it['text']} |" for it in lines["kept"]]
        out += [
            "",
            f"Dropped {len(lines['dropped'])} lines in HAL's voice:",
            "",
            "| Start | Reason | Line |",
            "| --- | --- | --- |",
        ]
        out += [f"| {d['start']:.1f} | {d['why']} | {d['text']} |" for d in lines["dropped"]]
        out.append("")
    spoken = b.work / "spoken.json"
    if spoken.exists():
        rows = [r for r in json.loads(spoken.read_text()).values() if r["spoken"] != r["subtitle"]]
        if rows:
            out += [
                "## Spoken differently from the subtitles",
                "",
                "The transcript trained on is what HAL says.",
                "",
                "| Spoken | Subtitle |",
                "| --- | --- |",
            ]
            out += [f"| {r['spoken']} | {r['subtitle']} |" for r in rows]
            out.append("")
    unusable = b.work / "unusable.json"
    if unusable.exists():
        why = json.loads(unusable.read_text())
        out += ["## Found but not trained on", "", "| Line | Reason |", "| --- | --- |"]
        for film in b.films:
            f = b.work / build.slug(film) / "lines.json"
            if f.exists():
                out += [
                    f"| {it['text']} | {why[it['id']]} |"
                    for it in json.loads(f.read_text())["kept"]
                    if it["id"] in why
                ]
        out.append("")
    path = b.work / "report.md"
    path.write_text("\n".join(out))
    return path


def main() -> int:
    """Runs the command line."""
    ap = argparse.ArgumentParser(
        prog="make-hal-voice",
        description=__doc__,
        epilog=DISCLAIMER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "films", nargs="*", type=Path, help="films or folders; voice-data/film/ when none given"
    )
    ap.add_argument(
        "--subs", nargs="*", type=Path, default=[], help="subtitle files to use as well"
    )
    ap.add_argument(
        "--work",
        type=Path,
        default=Path(os.environ.get("HALVOICE_BUILD_DIR", build.ROOT / "voice-data" / "build")),
    )
    ap.add_argument(
        "--version",
        default="v2",
        choices=["v2", "v2Pro", "v2ProPlus"],
        help="GPT-SoVITS model version",
    )
    ap.add_argument("--sovits-epochs", type=int, default=12)
    ap.add_argument("--gpt-epochs", type=int, default=12)
    ap.add_argument("--ocr", action="store_true", help="also OCR image subtitles inside the film")
    ap.add_argument("--from", dest="start", choices=build.STAGES, default="find")
    ap.add_argument("--only", choices=build.STAGES)
    ap.add_argument(
        "--no-install", action="store_true", help="build and select, but leave the voice as it is"
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PHRASE",
        help="leave out lines containing this phrase; remembered in the build folder (repeatable)",
    )
    ap.add_argument(
        "--verbatim",
        action="store_true",
        help="train on what each clip is heard to say rather than its subtitle text",
    )
    ap.add_argument(
        "--hold-out",
        action="store_true",
        help='leave "I\'m sorry, Dave" out of training and judge voices on it as an unheard line',
    )
    ap.add_argument("--denoise-python", help="Python with demucs and DeepFilterNet")
    args = ap.parse_args()

    films, subs = films_in(args.films)
    if not films:
        raise SystemExit(
            f"No film found. Put your copy (and any .srt) in {DROP}, or name the file."
        )
    stages = [args.only] if args.only else build.STAGES[build.STAGES.index(args.start) :]
    if args.no_install and "install" in stages:
        stages.remove("install")
    b = build.Build(
        films=films,
        subtitles=subs + args.subs,
        work=args.work.expanduser(),
        version=args.version,
        sovits_epochs=args.sovits_epochs,
        gpt_epochs=args.gpt_epochs,
        use_ocr=args.ocr,
        install="install" in stages,
        hold_out=args.hold_out,
        verbatim=args.verbatim,
        exclude=args.exclude,
    )
    b.work.mkdir(parents=True, exist_ok=True)
    if "clean" in stages:
        b.denoise_python = denoise_python(args.denoise_python)
    b.say(
        f"building from {', '.join(f.name for f in films)}; stages: {', '.join(stages)}; "
        f"work in {b.work}"
    )
    use_scheduler = (
        bool(GPU_STAGES & set(stages))
        and not os.environ.get(build.IN_GPU_RUN_ENV)
        and build.gpu_light_answers()
    )
    restore = not use_scheduler and GPU_STAGES & set(stages) and hal_is_on()
    if restore:
        b.say("turning Hal off to free the GPU; it comes back on at the end")
        hal_switch("off")
    try:
        for stage in stages:
            if use_scheduler and stage in GPU_STAGES:
                run_under_gpu_light(stage)
            else:
                getattr(build, stage)(b)
    finally:
        path = report(b)
        if restore:
            hal_switch("on")
        print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
