"""Finds the film's dialogue audio and subtitles, and extracts the audio with ffmpeg.

Prefers the centre channel of a surround mix, where film dialogue is mixed, so the score and
effects in the other channels are left behind before any cleaning. Stdlib only.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

VIDEO_SUFFIXES = {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm", ".ts", ".m2ts", ".mpg", ".mpeg", ".wmv",
                  ".flac", ".wav", ".mka", ".m4a", ".mp3", ".aac", ".ac3", ".dts", ".opus", ".ogg"}
SUBTITLE_SUFFIXES = {".srt", ".ass", ".ssa", ".vtt", ".webvtt", ".sub", ".idx", ".sup", ".txt"}
TEXT_CODECS = {"subrip", "ass", "ssa", "webvtt", "mov_text", "text", "srt"}
BITMAP_CODECS = {"dvd_subtitle", "hdmv_pgs_subtitle", "dvb_subtitle", "xsub"}
ENGLISH = {"eng", "en", "english", "und", ""}


@dataclass
class Stream:
    index: int
    kind: str                # audio | subtitle
    codec: str
    language: str
    title: str
    channels: int = 0
    layout: str = ""
    forced: bool = False
    hearing_impaired: bool = False
    default: bool = False


def probe(path: Path) -> list[Stream]:
    """Returns the audio and subtitle streams of a media file."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    streams = []
    for s in json.loads(out).get("streams", []):
        if s.get("codec_type") not in ("audio", "subtitle"):
            continue
        tags, disp = s.get("tags", {}), s.get("disposition", {})
        title = tags.get("title", "")
        streams.append(Stream(
            index=s["index"], kind=s["codec_type"], codec=s.get("codec_name", ""),
            language=tags.get("language", "").lower(), title=title,
            channels=int(s.get("channels") or 0), layout=s.get("channel_layout", ""),
            forced=bool(disp.get("forced")) or "forced" in title.lower(),
            hearing_impaired=bool(disp.get("hearing_impaired")) or bool(re.search(r"\bSDH\b|hearing", title, re.I)),
            default=bool(disp.get("default"))))
    return streams


def _is_commentary(s: Stream) -> bool:
    return bool(re.search(r"comment|director|descriptive|audio description", s.title, re.I))


def pick_audio(streams: list[Stream]) -> Stream:
    """Returns the main English dialogue track: no commentary, most channels, default first."""
    audio = [s for s in streams if s.kind == "audio" and not _is_commentary(s)]
    if not audio:
        audio = [s for s in streams if s.kind == "audio"]
    if not audio:
        raise ValueError("the file has no audio")
    return max(audio, key=lambda s: (s.language in ENGLISH, s.channels, s.default, -s.index))


def pick_subtitles(streams: list[Stream]) -> Stream | None:
    """Returns the best embedded English subtitle track, or None.

    Full (not forced) English subtitles win; SDH ones name off-screen speakers, which helps find
    HAL, and text tracks beat bitmap ones, which need OCR.
    """
    subs = [s for s in streams if s.kind == "subtitle" and not _is_commentary(s) and not s.forced
            and (s.codec in TEXT_CODECS or s.codec in BITMAP_CODECS)]
    english = [s for s in subs if s.language in ENGLISH]
    if not english:
        return None
    return max(english, key=lambda s: (s.language != "und", s.hearing_impaired, s.codec in TEXT_CODECS, -s.index))


def external_subtitles(video: Path, folder: Path | None = None) -> list[Path]:
    """Returns subtitle files beside the film (or in `folder`), the likeliest English one first."""
    folder = folder or video.parent
    stem = video.stem.lower()
    found = [p for p in folder.iterdir() if p.suffix.lower() in SUBTITLE_SUFFIXES and p.is_file()]
    # A .sub with an .idx beside it is VobSub, opened through its .idx.
    found = [p for p in found if not (p.suffix.lower() == ".sub" and p.with_suffix(".idx").exists())]

    def rank(p: Path) -> tuple:
        name = p.name.lower()
        return (name.startswith(stem[:12]), bool(re.search(r"(^|[._ -])(en|eng|english)([._ -]|$)", name)),
                "sdh" in name or "hi" in name.split("."), "forced" not in name, p.suffix.lower() in (".srt", ".ass", ".vtt"))
    return sorted(found, key=rank, reverse=True)


def dialogue_filter(s: Stream) -> str:
    """Returns the ffmpeg pan filter that keeps the dialogue: the centre channel when there is one."""
    layout = s.layout.lower()
    if s.channels >= 3 and ("5.1" in layout or "7.1" in layout or "6.1" in layout or "fc" in layout or s.channels >= 6):
        return "pan=mono|c0=FC"
    if s.channels == 2:
        return "pan=mono|c0=0.5*c0+0.5*c1"
    return "pan=mono|c0=c0"


def extract_audio(video: Path, s: Stream, out: Path, rate: int) -> Path:
    """Writes the dialogue as mono 16-bit WAV at `rate` Hz."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", f"0:{s.index}",
                    "-af", dialogue_filter(s), "-ar", str(rate), "-c:a", "pcm_s16le", str(out)], check=True)
    return out


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip() or 0)
