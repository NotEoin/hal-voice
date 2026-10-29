"""Cuts a recording of the voice test script into takes and lines, with a manifest.

Speech is found by level against the recording's own noise floor, each stretch of speech is
transcribed on its own, and the stretches are matched to the script's lines in order by
dynamic programming. Nothing relies on the markers being heard right ("Marker five" often
comes back as something else), only on the lines coming in the script's order. Each take's
silence after "Marker one" gives its noise level and spectrum.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import script

HOP_S = 0.01
WIN_S = 0.03
START_DB = 6.0  # speech is this far above the noise floor
KEEP_DB = 4.0  # and carries on while above this
MERGE_S = 0.3  # gaps shorter than this are inside a phrase
MIN_REGION_S = 0.12
MAX_GROUP = 8  # most stretches of speech one line can take
PRE_S, POST_S = 1.0, 1.5  # audio kept around each line's speech, gaps allowing
NAMES = {"hal", "hall", "how", "al", "howl", "hell", "hale", "haul", "hull", "owl", "hel"}


@dataclass
class Region:
    start: float
    end: float
    text: str = ""

    @property
    def tokens(self) -> list[str]:
        return [("hal" if w in NAMES else w) for w in script.words(self.text)]


def frame_db(y: np.ndarray, sr: int) -> np.ndarray:
    hop, win = int(HOP_S * sr), int(WIN_S * sr)
    n = max(1, (len(y) - win) // hop + 1)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return 10 * np.log10(np.mean(y[idx] ** 2, axis=1) + 1e-12)


def band(y: np.ndarray, sr: int) -> np.ndarray:
    """The speech band, 150 Hz to 4 kHz, without a PC's low hum and rumble."""
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(4, [150, 4000], "bandpass", fs=sr, output="sos"), y)


def level_regions(y: np.ndarray, sr: int) -> list[list[float]]:
    """Stretches above the speech band's noise floor, with hysteresis."""
    db = frame_db(band(y, sr), sr)
    floor = float(np.percentile(db, 10))
    out, on, start, below = [], False, 0, 0
    for i, d in enumerate(db):
        if not on and d > floor + START_DB + 1:
            on, start, below = True, i, 0
        elif on:
            below = below + 1 if d < floor + KEEP_DB else 0
            if below * HOP_S >= MERGE_S:
                out.append([start * HOP_S, (i - below + 1) * HOP_S + WIN_S])
                on = False
    if on:
        out.append([start * HOP_S, len(db) * HOP_S])
    return out


def regions(y: np.ndarray, sr: int) -> tuple[list[Region], float]:
    """Stretches of speech and the recording's noise floor.

    Silero's voice detector finds most speech, and the speech band's level finds the quiet
    words it misses: a quiet voice can be only a few dB above a PC's fans.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(
        threshold=0.3,
        min_silence_duration_ms=int(MERGE_S * 1000),
        min_speech_duration_ms=int(MIN_REGION_S * 1000),
        speech_pad_ms=60,
    )
    stamps = get_speech_timestamps(y, opts, sampling_rate=sr)
    spans = [[t["start"] / sr, t["end"] / sr] for t in stamps]
    spans = sorted(spans + level_regions(y, sr))
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    floor = float(np.percentile(frame_db(y, sr), 10))
    return [Region(a, b) for a, b in merged if b - a >= MIN_REGION_S], floor


def transcribe_regions(y: np.ndarray, sr: int, regs: list[Region], model) -> None:
    for r in regs:
        seg = y[int(max(0, r.start - 0.2) * sr) : int((r.end + 0.2) * sr)]
        segs, _ = model.transcribe(
            seg,
            language="en",
            beam_size=5,
            hotwords="Hal",
            vad_filter=False,
            condition_on_previous_text=False,
        )
        r.text = " ".join(
            s.text.strip() for s in segs if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0
        ).strip()


def score(got: list[str], want: list[str]) -> float:
    """Matched words, less a little for every word on either side left unmatched."""
    m = sum(
        b.size
        for b in difflib.SequenceMatcher(None, got, want, autojunk=False).get_matching_blocks()
    )
    return m - 0.3 * (len(got) - m) - 0.3 * (len(want) - m)


def align(regs: list[Region], lines: list[script.Line]) -> dict[str, tuple[int, int]]:
    """Assigns consecutive stretches of speech to each line, in order, best match overall."""
    n, m = len(regs), len(lines)
    want = [[("hal" if w in NAMES else w) for w in script.words(ln.text)] for ln in lines]
    toks = [r.tokens for r in regs]
    neg = -1e9
    best = np.full((n + 1, m + 1), neg)
    back: dict[tuple[int, int], tuple] = {}
    best[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            here = best[i, j]
            if here <= neg:
                continue
            if i < n:  # a stretch that is no line: a cough, a word to someone else
                v = here - 0.2 - 0.3 * len(toks[i])
                if v > best[i + 1, j]:
                    best[i + 1, j], back[(i + 1, j)] = v, (i, j, None)
            if j < m:  # a line not heard at all
                v = here - 0.3 * len(want[j]) - 0.5
                if v > best[i, j + 1]:
                    best[i, j + 1], back[(i, j + 1)] = v, (i, j, None)
            if i < n and j < m:
                got: list[str] = []
                for k in range(i, min(n, i + MAX_GROUP)):
                    got = got + toks[k]
                    if lines[j].kind == "chat":
                        span = regs[k].end - regs[i].start
                        if not 3.0 <= span <= 25.0:
                            continue
                        v = here + 0.2 * len(got)
                    else:
                        v = here + score(got, want[j])
                    if v > best[k + 1, j + 1]:
                        best[k + 1, j + 1], back[(k + 1, j + 1)] = v, (i, j, (i, k + 1))
    out, at = {}, (n, m)
    while at != (0, 0):
        i, j, span = back[at]
        if span is not None:
            out[lines[j].id] = span
        at = (i, j)
    return out


def take_starts(regs: list[Region]) -> list[tuple[int, str]]:
    """Finds each "Marker one, <take>" and the take it names."""
    found = []
    for i, r in enumerate(regs):
        text = " ".join(x.text for x in regs[i : i + 3]).lower()
        if not r.text or not re.match(r"^\W*mark\w*\W+(one|1)\b", text):
            continue
        tid = max(
            script.TAKES,
            key=lambda t: difflib.SequenceMatcher(
                None, script.TAKES[t][0].lower(), text[:40]
            ).ratio(),
        )
        found.append((i, tid))
    return found


def band_levels(y: np.ndarray, sr: int) -> dict[str, float]:
    """Third-octave band levels in dB, 63 Hz to 6.3 kHz, from Welch's method."""
    from scipy.signal import welch

    f, p = welch(y, sr, nperseg=4096)
    out = {}
    for k in range(-14, 6):
        c = 1000 * 2 ** (k / 3)
        sel = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        if sel.any():
            out[f"{c:.0f}"] = round(float(10 * np.log10(np.sum(p[sel]) * (f[1] - f[0]) + 1e-15)), 1)
    return out


def active_level(y: np.ndarray, sr: int, floor: float) -> float:
    """Mean level of the frames that are speech, in dBFS."""
    db = frame_db(y, sr)
    act = db[db > floor + START_DB]
    return round(float(10 * np.log10(np.mean(10 ** (act / 10)))), 1) if act.size else -120.0


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "chat"


def split(recording: Path, out: Path, model, take: str | None = None, say=print) -> list[dict]:
    """Writes the takes in a recording as clips under out/<take>/; returns the takes."""
    import soundfile as sf

    y, sr = sf.read(recording, dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    regs, floor = regions(y, sr)
    say(f"{recording.name}: {len(regs)} stretches of speech, noise floor {floor:.1f} dBFS")
    transcribe_regions(y, sr, regs, model)
    starts = take_starts(regs) or [(0, take or "desk-1")]
    if take:
        starts = [(starts[0][0], take)]
    lines = script.load()
    takes = []
    for t, (first, tid) in enumerate(starts):
        last = starts[t + 1][0] if t + 1 < len(starts) else len(regs)
        label, distance, noise = script.TAKES[tid]
        sub = regs[first:last]
        lines[0].text = f"Marker one. {label}"
        spans = align(sub, lines)
        tdir = out / tid
        tdir.mkdir(parents=True, exist_ok=True)
        # The silence is the longest gap between "Marker one" and the next line.
        m1 = spans.get("m1", (0, 1))
        gaps = [
            (sub[k + 1].start - sub[k].end, sub[k].end, sub[k + 1].start)
            for k in range(m1[1] - 1, min(len(sub) - 1, m1[1] + 2))
        ]
        g, a, b = max(gaps) if gaps else (0.0, 0.0, 0.0)
        quiet = y[int((a + 0.5) * sr) : int((b - 0.5) * sr)]
        sf.write(tdir / "silence.wav", quiet, sr)
        noise_db = round(float(10 * np.log10(np.mean(quiet**2) + 1e-12)), 1) if quiet.size else None
        clips = []
        for ln in lines:
            if ln.kind == "marker":
                continue
            row = {**ln.as_dict(), "take": tid, "distance": distance, "noise": noise}
            if ln.id not in spans:
                clips.append({**row, "missing": True})
                continue
            i, j = spans[ln.id]
            s0, s1 = sub[i].start, sub[j - 1].end
            prev_end = sub[i - 1].end if i > 0 else 0.0
            next_start = sub[j].start if j < len(sub) else len(y) / sr
            c0 = max(s0 - PRE_S, (prev_end + s0) / 2)
            c1 = min(s1 + POST_S, (s1 + next_start) / 2)
            clip = y[int(c0 * sr) : int(c1 * sr)]
            name = f"{int(ln.id):03d}-{slug(ln.text)}.wav"
            sf.write(tdir / name, clip, sr)
            level = active_level(y[int(s0 * sr) : int(s1 * sr)], sr, floor)
            clips.append(
                {
                    **row,
                    "file": f"{tid}/{name}",
                    "at": round(s0, 2),
                    "speech_start": round(s0 - c0, 3),
                    "speech_end": round(s1 - c0, 3),
                    "heard": " ".join(r.text for r in sub[i:j]),
                    "level_dbfs": level,
                    "snr_db": round(level - noise_db, 1) if noise_db is not None else None,
                }
            )
        missing = [c["id"] for c in clips if c.get("missing")]
        say(
            f"  {tid}: {len(clips) - len(missing)} lines cut, silence {g:.1f} s at "
            f"{noise_db} dBFS" + (f"; not found: {', '.join(missing)}" if missing else "")
        )
        takes.append(
            {
                "take": tid,
                "label": label,
                "distance": distance,
                "noise": noise,
                "recording": str(recording),
                "silence": f"{tid}/silence.wav",
                "silence_s": round(len(quiet) / sr, 2),
                "noise_dbfs": noise_db,
                "noise_bands_db": band_levels(quiet, sr) if quiet.size else {},
                "clips": clips,
            }
        )
    return takes


def write_manifest(out: Path, takes: list[dict]) -> Path:
    """Merges the takes into out/manifest.json, replacing any take split again."""
    path = out / "manifest.json"
    old = json.loads(path.read_text())["takes"] if path.exists() else []
    keep = {t["take"]: t for t in old}
    keep.update({t["take"]: t for t in takes})
    order = list(script.TAKES)
    path.write_text(
        json.dumps({"takes": sorted(keep.values(), key=lambda t: order.index(t["take"]))}, indent=1)
    )
    return path
