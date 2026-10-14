"""Reads subtitles in any common form into timed, speaker-tagged lines.

Text subtitles (SRT, WebVTT, ASS/SSA, embedded or beside the film) are parsed directly; image
subtitles (DVD/VobSub, Blu-ray PGS) are read with tesseract, after working out which colours
of the subtitle bitmaps are the letters. SDH speaker labels ("HAL: ...") and dash-led lines
split one subtitle into the lines of each speaker. The parsers are stdlib only; OCR needs
PyAV, numpy, Pillow and the tesseract command.
"""

from __future__ import annotations

import html
import io
import itertools
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import media


@dataclass
class Cue:
    start: float
    end: float
    text: str
    speaker: str | None = None  # from an SDH label, upper case
    cue: int = 0  # index of the subtitle the line came from
    tags: list[str] = field(default_factory=list)


# ---- reading files ------------------------------------------------------------------------


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and raw[:2] not in (b"\xff\xfe", b"\xfe\xff"):
                continue
            return text
        except UnicodeDecodeError:
            continue
    return raw.decode("cp1252", errors="replace")


_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{1,2})[.,:](\d{1,3})")


def _seconds(stamp: str) -> float:
    m = _TIME.search(stamp)
    if not m:
        raise ValueError(f"not a timestamp: {stamp!r}")
    h, mi, s, frac = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + int(frac.ljust(3, "0")[:3]) / 1000


def parse_srt(text: str) -> list[tuple[float, float, str]]:
    """Parses SRT and WebVTT: blocks with an `a --> b` line and text below it."""
    out = []
    for block in re.split(r"\r?\n\s*\r?\n", text.replace("\ufeff", "")):
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            if "-->" in line:
                a, b = line.split("-->", 1)
                try:
                    start, end = _seconds(a), _seconds(b.strip().split(" ")[0])
                except ValueError:
                    break
                body = "\n".join(lines[i + 1 :])
                if body.strip():
                    out.append((start, end, body))
                break
    return out


def parse_ass(text: str) -> list[tuple[float, float, str]]:
    """Parses the [Events] Dialogue lines of ASS/SSA; the Name field becomes a speaker label."""
    out, fields = [], None
    in_events = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            in_events = s.lower() == "[events]"
            continue
        if not in_events:
            continue
        if s.lower().startswith("format:"):
            fields = [f.strip().lower() for f in s.split(":", 1)[1].split(",")]
        elif s.lower().startswith("dialogue:") and fields:
            parts = s.split(":", 1)[1].split(",", len(fields) - 1)
            row = dict(zip(fields, (p.strip() for p in parts)))
            body = (
                re.sub(r"\{[^}]*\}", "", row.get("text", ""))
                .replace("\\N", "\n")
                .replace("\\n", "\n")
                .replace("\\h", " ")
            )
            name = row.get("name", "")
            if name and not re.match(r"^(default|sign|song|op|ed)$", name, re.I):
                body = f"{name.upper()}: {body}"
            if body.strip():
                out.append((_seconds(row["start"]), _seconds(row["end"]), body))
    return sorted(out)


def read_file(path: Path, ocr_lang: str = "eng") -> list[tuple[float, float, str]]:
    """Reads a subtitle file of any supported kind into (start, end, raw text)."""
    suffix = path.suffix.lower()
    if suffix in (".idx", ".sup", ".sub"):
        return read_bitmap(path, None, ocr_lang)
    text = read_text(path)
    if suffix in (".ass", ".ssa") or "[Events]" in text[:5000]:
        return parse_ass(text)
    return parse_srt(text)


def read_embedded(
    video: Path, s: media.Stream, ocr_lang: str = "eng"
) -> list[tuple[float, float, str]]:
    """Reads an embedded subtitle stream: text ones through ffmpeg's SRT writer, bitmaps by OCR."""
    if s.codec in media.BITMAP_CODECS:
        return read_bitmap(video, s.index, ocr_lang)
    srt = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-map", f"0:{s.index}", "-f", "srt", "-"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8", errors="replace")
    return parse_srt(srt)


# ---- image subtitles --------------------------------------------------------------------


def _bitmaps(path: Path, stream_index: int | None):
    """Yields (start, end, index array) for every subtitle bitmap in a file or stream."""
    import av
    import numpy as np

    with av.open(str(path)) as container:
        stream = (
            next(s for s in container.streams if s.index == stream_index)
            if stream_index is not None
            else container.streams.subtitles[0]
        )
        tb = float(stream.time_base)
        for pkt in container.demux(stream):
            if pkt.pts is None:
                continue
            start = pkt.pts * tb
            end = start + (pkt.duration * tb if pkt.duration else 3.0)
            for rect in stream.codec_context.decode(pkt):
                if not getattr(rect, "width", 0) or not rect.planes:
                    continue
                data = np.frombuffer(bytes(rect.planes[0]), dtype=np.uint8)
                stride = data.size // rect.height
                yield start, end, data.reshape(rect.height, stride)[:, : rect.width]


def _ocr(index, ink: tuple[int, ...] | None, lang: str) -> tuple[str, float]:
    """Returns the text tesseract reads from a subtitle bitmap and its mean word confidence."""
    import numpy as np
    from PIL import Image

    background = np.bincount(index.ravel()).argmax()
    mask = (index != background) if ink is None else np.isin(index, ink)
    img = np.pad(np.where(mask, 0, 255).astype(np.uint8), 12, constant_values=255)
    pic = Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.LANCZOS)
    buf = io.BytesIO()
    pic.save(buf, "PNG")
    tsv = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", "6", "-l", lang, "tsv"],
        input=buf.getvalue(),
        capture_output=True,
        check=False,
    ).stdout.decode("utf-8", "replace")
    lines: dict[tuple, list[str]] = {}
    confs = []
    for row in tsv.splitlines()[1:]:
        f = row.split("\t")
        if len(f) == 12 and f[11].strip() and float(f[10]) >= 0:
            lines.setdefault((f[2], f[3], f[4]), []).append(f[11])
            confs.append(float(f[10]))
    return "\n".join(" ".join(w) for w in lines.values()), (
        sum(confs) / len(confs) if confs else 0.0
    )


def read_bitmap(
    path: Path, stream_index: int | None, lang: str = "eng"
) -> list[tuple[float, float, str]]:
    """Reads image subtitles by OCR.

    The bitmaps carry palette indices but not the palette, so which indices are letters is
    found once, on a sample, by trying each choice and keeping the one tesseract reads best.
    """
    import numpy as np

    subs = list(_bitmaps(path, stream_index))
    if not subs:
        return []
    sample = subs[len(subs) // 4 : len(subs) // 4 + 16] or subs[:16]
    counts = np.bincount(np.concatenate([s[2].ravel() for s in sample]), minlength=256)
    background = int(counts.argmax())
    common = [int(i) for i in np.argsort(counts)[::-1] if i != background and counts[i] > 0][:3]
    choices: list[tuple[int, ...] | None] = [None] + [
        c for k in (1, 2) for c in itertools.combinations(common, k)
    ]
    best = max(choices, key=lambda ink: sum(_ocr(s[2], ink, lang)[1] for s in sample))
    return [(start, end, _ocr(index, best, lang)[0]) for start, end, index in subs]


# ---- cleaning and splitting -------------------------------------------------------------

_LABEL = re.compile(r"^\s*-?\s*([A-Z][A-Z0-9 .'\-]{1,24}?)(?:\s*\(([^)]*)\))?\s*:\s*")
_BRACKET_LABEL = re.compile(r"^\s*-?\s*\[([A-Z][A-Z0-9 .'\-]{0,24})\]\s*")
_SOUND = re.compile(r"\[[^\]]*\]|\([A-Z][A-Z ,.'-]+\)|[\u266a\u266b#*]+")
# Words that make a bracketed tag a sound, not a speaker: "[GRUNTING]", "[APE MEN SCREAM]".
_SOUND_WORDS = {
    "ALARM",
    "NOISE",
    "MUSIC",
    "CHATTER",
    "DIALOGUE",
    "LANGUAGE",
    "STATIC",
    "BEEP",
    "BEEPS",
    "SCREAM",
    "SCREAMS",
    "GROWLS",
    "ROARS",
    "SHRIEKS",
    "GIBBERS",
    "WHOOSHES",
    "FADES",
    "LAUGHS",
    "SIGHS",
    "GASPS",
    "GROANS",
    "CHUCKLES",
    "INAUDIBLE",
    "INDISTINCT",
    "APPLAUSE",
    "SILENCE",
    "THUD",
    "CRASH",
}


def is_speaker_tag(tag: str) -> bool:
    """True when a bracketed SDH tag names a speaker ("[HAL]", "[HAL SINGS]"), not a sound."""
    words = tag.split()
    if not words or len(words) > 3 or words[0] in _SOUND_WORDS or words[0].endswith("ING"):
        return False
    return not any(
        w in _SOUND_WORDS or w.endswith("ING") for w in words[1:] if w not in ("SINGS", "VOICE")
    )


def fix_ocr(text: str) -> str:
    """Repairs what tesseract reliably gets wrong in subtitle fonts."""
    t = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    t = re.sub(r"(?<![\w'])[|/](?=['\s]|$)", "I", t)  # | or / for a lone I
    t = re.sub(r"(?<![\w'])[|/]'", "I'", t)
    t = re.sub(r"(?<![\w'])ll\b", "I'll", t)
    t = re.sub(r"\b([Ii])t{2,}\b", r"\1t", t)
    t = re.sub(r"\bts\b", "is", t)
    t = re.sub(r"'\?", "?", t)
    return t


def clean(text: str) -> str:
    t = html.unescape(re.sub(r"<[^>]+>", "", text))
    t = _SOUND.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def split_speakers(raw: str) -> list[tuple[str | None, list[str], str]]:
    """Splits one subtitle into (speaker label, tags, text) for each speaker in it.

    A new speaker starts at an SDH label ("HAL:", "BOWMAN (ON RADIO):", "[HAL]", alone on its
    line or before the words) or a line that opens with a dash. Unlabelled parts keep speaker
    None. Italic text, the usual mark of a voice off screen or through a speaker, is tagged.
    """
    parts: list[tuple[str | None, list[str], list[str]]] = []
    italic = False
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line_italic = italic or bool(re.search(r"<i>|\{\\i1\}", line, re.I))
        italic = (italic or "<i>" in line.lower()) and "</i>" not in line.lower()
        bare = re.sub(r"<[^>]+>", "", line)
        m, b = _LABEL.match(bare), _BRACKET_LABEL.match(bare)
        if b and is_speaker_tag(b.group(1)):
            name = b.group(1).strip()
            tags = [w.lower() for w in name.split()[1:]] + (["italic"] if line_italic else [])
            parts.append((name.split()[0].upper(), tags, [bare[b.end() :]]))
        elif m and not re.fullmatch(r"\d+", m.group(1).strip()):
            name = m.group(1).strip()
            tags = [w.lower() for w in name.split()[1:]] + (
                [m.group(2).lower()] if m.group(2) else []
            )
            parts.append(
                (
                    name.split()[0].upper(),
                    tags + (["italic"] if line_italic else []),
                    [bare[m.end() :]],
                )
            )
        elif bare.startswith("-") or not parts:
            parts.append((None, ["italic"] if line_italic else [], [bare.lstrip("- ")]))
        else:
            parts[-1][2].append(bare)
    return [
        (spk, tags, clean(" ".join(lines))) for spk, tags, lines in parts if clean(" ".join(lines))
    ]


def cues(raw: list[tuple[float, float, str]], ocr: bool = False) -> list[Cue]:
    """Turns raw subtitles into speaker lines; a two-speaker subtitle's time is split by length."""
    out = []
    for n, (start, end, text) in enumerate(raw):
        if ocr:
            text = fix_ocr(text)
        parts = split_speakers(text)
        total = sum(len(p[2]) for p in parts) or 1
        t = start
        for spk, tags, body in parts:
            share = (end - start) * len(body) / total
            out.append(Cue(start=t, end=t + share, text=body, speaker=spk, cue=n, tags=tags))
            t += share
    return out


def sources(
    video: Path, extra: list[Path] | None = None, ocr_lang: str = "eng", with_embedded: bool = True
) -> list[tuple[str, list[Cue]]]:
    """Returns every subtitle source for the film as (name, lines): files given, files beside it
    and, unless text files exist and with_embedded is False, the best embedded track."""
    found: list[tuple[str, list[Cue]]] = []
    files = list(extra or []) + [
        p
        for p in (media.external_subtitles(video) if video.parent.exists() else [])
        if p not in (extra or [])
    ]
    for f in files:
        try:
            raw = read_file(f, ocr_lang)
        except (ValueError, OSError, StopIteration):
            continue
        if raw:
            found.append((f.name, cues(raw, ocr=f.suffix.lower() in (".idx", ".sup", ".sub"))))
    if with_embedded or not found:
        stream = media.pick_subtitles(media.probe(video))
        if stream is not None:
            raw = read_embedded(video, stream, ocr_lang)
            found.append(
                (
                    f"embedded stream {stream.index} ({stream.codec})",
                    cues(raw, ocr=stream.codec in media.BITMAP_CODECS),
                )
            )
    return found
