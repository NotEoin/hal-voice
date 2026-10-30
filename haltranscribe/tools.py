"""hal-transcribe's tools: audio to text, and text to a note in the vault.

Files of any format ffmpeg reads are converted to 16 kHz mono and transcribed
with faster-whisper (on the GPU when CUDA works). Recordings from the
microphone run in the background through pw-record until stopped. A
transcript can be summarised by the local model and saved as a meeting note.
"""

from __future__ import annotations

import contextlib
import ctypes
import glob
import json
import logging
import os
import signal
import site
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from . import llm
from .config import data_dir, settings, vault

log = logging.getLogger(__name__)

CLAIM_NAME = "hal-transcribe"
CLAIM_WAIT_S = 120  # gpu-light's recipe: wait this long for the card, then use the CPU
FFMPEG_TIMEOUT_S = 1800
CPU_MODEL = "small.en"
SUMMARY_CHARS = 20000  # what a local model's context holds beside the instructions
STOP_SETTLE_S = 1  # pw-record needs a moment to close the WAV after SIGINT

_model = None


@contextlib.contextmanager
def _gpu_claim():
    """Holds a gpu-light claim while Whisper runs; yields False when the CPU must be used."""
    try:
        import gpu_light
    except ImportError:
        yield True
        return
    try:
        with gpu_light.claim(CLAIM_NAME, wait=True, timeout=CLAIM_WAIT_S):
            yield True
    except gpu_light.ClaimTimeout:
        log.warning("the GPU stayed busy for %s s; transcribing on the CPU", CLAIM_WAIT_S)
        yield False


def _pid_start(pid: int) -> str | None:
    """The start time field of `/proc/<pid>/stat`, which tells a pid's reuse apart."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    return stat.rsplit(")", 1)[1].split()[19]


def _whisper(use_gpu: bool = True):
    global _model
    if not use_gpu:
        from faster_whisper import WhisperModel

        return WhisperModel(CPU_MODEL, device="cpu", compute_type="int8")
    if _model is None:
        from faster_whisper import WhisperModel

        for base in site.getsitepackages():
            for pat in (
                "nvidia/cublas/lib/libcublasLt.so.*",
                "nvidia/cublas/lib/libcublas.so.*",
                "nvidia/cudnn/lib/libcudnn*.so.*",
            ):
                for path in sorted(glob.glob(os.path.join(base, pat))):
                    try:
                        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
        try:
            _model = WhisperModel(settings()["model"], device="cuda", compute_type="float16")
        except (RuntimeError, ValueError):
            _model = WhisperModel(CPU_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe(path: str, timestamps: bool = False) -> dict:
    """Transcribes an audio or video file."""
    src = Path(path).expanduser()
    if not src.is_file():
        raise ValueError(f"{path} is not a file")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav),
            ],
            check=True,
            timeout=FFMPEG_TIMEOUT_S,
        )
        with _gpu_claim() as use_gpu:
            segments, info = _whisper(use_gpu).transcribe(str(wav), beam_size=5, vad_filter=True)
            lines = [
                (f"[{int(s.start // 60):02d}:{int(s.start % 60):02d}] " if timestamps else "")
                + s.text.strip()
                for s in segments
            ]
    text = ("\n" if timestamps else " ").join(lines).strip()
    return {
        "file": src.name,
        "language": info.language,
        "minutes": round(info.duration / 60, 1),
        "text": text,
    }


def _state() -> Path:
    return data_dir() / "recording.json"


def record_start(title: str = "Recording") -> dict:
    """Starts recording the microphone in the background."""
    if _state().exists():
        raise RuntimeError("a recording is already running; stop it first")
    out = data_dir() / f"{datetime.now():%Y%m%d-%H%M%S}.wav"
    proc = subprocess.Popen(
        ["pw-record", "--rate", "16000", "--channels", "1", str(out)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _state().write_text(
        json.dumps(
            {
                "pid": proc.pid,
                "pid_start": _pid_start(proc.pid),
                "file": str(out),
                "title": title,
                "started": time.time(),
            }
        )
    )
    return {"recording": title, "file": str(out)}


def record_stop(save: bool = True) -> dict:
    """Stops the recording, transcribes it and (by default) saves a summarised note in the vault."""
    if not _state().exists():
        raise RuntimeError("nothing is being recorded")
    st = json.loads(_state().read_text())
    # Signal only the recorder this file describes: a reused pid belongs to another program
    if _pid_start(st["pid"]) == st.get("pid_start"):
        with contextlib.suppress(ProcessLookupError):
            os.kill(st["pid"], signal.SIGINT)
    else:
        log.warning("the recorder (pid %s) is gone; transcribing what was saved", st["pid"])
    time.sleep(STOP_SETTLE_S)
    _state().unlink()
    result = transcribe(st["file"])
    if save:
        result["note"] = save_note(result["text"], st["title"])
    return result


def save_note(text: str, title: str = "Transcript") -> str:
    """Saves a transcript to the vault as a note with a summary, decisions and actions."""
    summary = ""
    try:
        summary = llm.chat(
            "Write a meeting-style summary of this transcript in Markdown: a two-sentence "
            "Summary, then 'Decisions' and 'Actions' as bullet lists (actions as - [ ] items). "
            "Use only the transcript.",
            text[:SUMMARY_CHARS],
        ).strip()
    except llm.Unavailable as e:
        log.warning("saving the note without a summary: %s", e)
    folder = vault() / settings()["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.now():%Y-%m-%d %H%M} {title}".replace("/", "-")
    note = folder / f"{name}.md"
    note.write_text(
        f"---\ntype: transcript\ndate: {datetime.now():%Y-%m-%d}\ntags: [transcript]\nhal: needs-index\n---\n"
        f"# {title}\n\n{summary}\n\n## Transcript\n\n{text}\n"
    )
    return str(note)


def recording() -> dict:
    if not _state().exists():
        return {"recording": False}
    st = json.loads(_state().read_text())
    return {
        "recording": True,
        "title": st["title"],
        "minutes": round((time.time() - st["started"]) / 60, 1),
    }
