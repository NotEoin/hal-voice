"""Keeps only the HAL lines worth training on, and says why each other one was dropped.

A kept line is HAL alone (no other voice within a beat of it), heard the way the subtitle
says it, at HAL's usual pitch and pace (which rules out the slowed shutdown and the singing),
long enough to learn from, and, after cleaning, well above the background.
"""

from __future__ import annotations

import difflib

import numpy as np

from .sync import Line, tokens

MIN_S, MAX_S = 0.8, 12.0
MAX_DISAGREEMENT = 0.25  # word error rate between subtitle and transcript
MIN_WORD_PROB = 0.55
OVERLAP_S = 0.0  # seconds of clearance between HAL's words and anyone else's
PITCH_ST = 4.5  # semitones from HAL's median; the slowed shutdown is far lower
MIN_RATE = 1.2  # words per second
MIN_SNR_DB = 12.0
JOIN_GAP_S = 1.5  # HAL's lines this close, with nobody between, make one clip


def word_error(ref: list[str], hyp: list[str]) -> float:
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    same = sum(n for _, _, n in sm.get_matching_blocks())
    return 1 - same / max(len(ref), 1) if ref else 1.0


def split_long(ln: Line) -> list[Line]:
    """Splits a line longer than MAX_S at its longest pauses between words."""
    if ln.end - ln.start <= MAX_S or len(ln.words) < 4:
        return [ln]
    gaps = sorted(
        range(1, len(ln.words)),
        key=lambda k: ln.words[k]["start"] - ln.words[k - 1]["end"],
        reverse=True,
    )
    for k in gaps:
        a, b = ln.words[:k], ln.words[k:]
        if a[-1]["end"] - a[0]["start"] >= MIN_S and b[-1]["end"] - b[0]["start"] >= MIN_S:
            parts = []
            for ws in (a, b):
                text = " ".join(w["word"] for w in ws)
                parts += split_long(
                    Line(
                        start=ws[0]["start"],
                        end=ws[-1]["end"],
                        text=text,
                        heard=text,
                        speaker=ln.speaker,
                        tags=ln.tags,
                        cue=ln.cue,
                        matched=ln.matched,
                        words=ws,
                    )
                )
            return parts
    return [ln]


def join_runs(lines: list[Line], verdicts: list[dict]) -> list[tuple[Line, dict]]:
    """Returns HAL's lines with each run of them (a monologue broken only by short pauses, with
    nobody else speaking between) joined into one line, pauses kept, up to MAX_S long."""
    out: list[tuple[Line, dict]] = []
    prev = -2
    for i, (ln, v) in enumerate(zip(lines, verdicts, strict=False)):
        if not v["hal"]:
            continue
        if out and prev == i - 1 and "sings" not in ln.tags:
            last, lv = out[-1]
            if (
                ln.start - last.end <= JOIN_GAP_S
                and ln.end - last.start <= MAX_S
                and "sings" not in last.tags
            ):
                joined = Line(
                    start=last.start,
                    end=ln.end,
                    text=f"{last.text} {ln.text}",
                    heard=f"{last.heard} {ln.heard}",
                    speaker="HAL",
                    tags=sorted(set(last.tags) | set(ln.tags)),
                    cue=last.cue,
                    matched=min(last.matched, ln.matched),
                    words=last.words + ln.words,
                )
                sims = [s for s in (lv["sim"], v["sim"]) if s is not None]
                out[-1] = (joined, {**lv, "sim": max(sims) if sims else None})
                prev = i
                continue
        out.append((ln, v))
        prev = i
    return out


def pitch_semitones(seg: np.ndarray, rate: int) -> float | None:
    import librosa

    f0 = librosa.yin(seg, fmin=55, fmax=250, sr=rate, frame_length=1024)
    rms = librosa.feature.rms(y=seg, frame_length=1024, hop_length=256)[0][: len(f0)]
    f0 = f0[(rms > np.percentile(rms, 50)) & (f0 > 60) & (f0 < 240)]
    return float(np.median(12 * np.log2(f0 / 100.0))) if f0.size >= 10 else None


def screen(
    lines: list[Line], verdicts: list[dict], audio: np.ndarray, rate: int, have_subtitles: bool
) -> tuple[list[tuple[Line, dict]], list[dict]]:
    """Returns the HAL lines that pass every check, and a record of every HAL line dropped."""
    # Other voices by their recognised words, so a reply a beat later doesn't count as overlap.
    others = [
        (w["start"], w["end"])
        for ln, v in zip(lines, verdicts, strict=False)
        if not v["hal"]
        for w in ln.words
    ]
    candidates, dropped = [], []
    for ln, v in join_runs(lines, verdicts):
        for part in split_long(ln):
            candidates.append((part, v))
    pitches = {}
    for n, (ln, _) in enumerate(candidates):
        seg = audio[int(ln.start * rate) : int(ln.end * rate)]
        pitches[n] = pitch_semitones(seg, rate) if seg.size > rate // 2 else None
    centre = float(np.median([p for p in pitches.values() if p is not None])) if pitches else 0.0
    kept = []
    for n, (ln, v) in enumerate(candidates):
        dur = ln.end - ln.start
        heard = tokens(ln.heard)
        probs = [w["prob"] for w in ln.words]
        rate_wps = len(heard) / dur if dur else 0
        why = None
        if "sings" in ln.tags or "singing" in ln.tags:
            why = "singing"
        elif dur < MIN_S:
            why = f"too short ({dur:.1f}s)"
        elif dur > MAX_S:
            why = f"too long ({dur:.1f}s) with no pause to split at"
        elif have_subtitles and word_error(tokens(ln.text), heard) > MAX_DISAGREEMENT:
            why = "transcript disagrees with the subtitle"
        elif probs and float(np.mean(probs)) < MIN_WORD_PROB:
            why = "speech unclear to the recogniser"
        elif any(
            s < ln.end + OVERLAP_S and e > ln.start - OVERLAP_S
            for s, e in others
            if not any(w["start"] == s and w["end"] == e for w in ln.words)
        ):
            why = "another voice overlaps it"
        elif pitches[n] is None:
            why = "no clear pitch"
        elif abs(pitches[n] - centre) > PITCH_ST:
            why = f"pitch {pitches[n] - centre:+.1f} st from HAL's usual (effect or shutdown)"
        elif rate_wps < MIN_RATE:
            why = f"very slow ({rate_wps:.1f} words/s)"
        record = {
            "start": round(ln.start, 2),
            "end": round(ln.end, 2),
            "text": ln.text,
            "sim": v["sim"],
        }
        if why:
            dropped.append({**record, "why": why})
        else:
            kept.append((ln, {**record, "pitch_st": round(pitches[n], 2)}))
    return kept, dropped
