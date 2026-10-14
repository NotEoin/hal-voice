"""Lines up subtitles with the speech, and cuts each line at the words actually spoken.

Subtitles can be shifted, drift (made for another frame rate) or jump at cuts. Their text is
aligned to the transcript word by word across the whole film; the timing differences of the
matched lines then give a robust offset-plus-drift fit, with a running median on top for local
jumps. Each line's times come from its own recognised words wherever it has enough of them.
Stdlib only.
"""

from __future__ import annotations

import difflib
import re
import statistics
from dataclasses import asdict, dataclass, field

from .subtitles import Cue

_WORD = re.compile(r"[a-z0-9']+")
CONTEXT_S = 0.5  # how far outside a line's fitted times its words may lie
MIN_MATCHED = 2
EDGE_GAP_S = 0.4  # largest pause before an unmatched edge word still joins the line
MAX_EDGE_WORD_S = 0.8  # longest a line's first or last word may plausibly last


@dataclass
class Line:
    start: float
    end: float
    text: str  # subtitle text with transcript corrections
    heard: str  # what the recogniser heard over the same span
    speaker: str | None
    tags: list[str]
    cue: int
    matched: float  # share of the subtitle's words found in the transcript
    words: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("-", " "))


def theil_sen(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Returns (intercept, slope) of the median-of-slopes line through the points."""
    n = len(xs)
    step = max(1, n // 300)  # bounds the pair count on long films
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(0, n, step)
        for j in range(i + 1, n, step)
        if xs[j] - xs[i] > 30
    ]
    slope = statistics.median(slopes) if slopes else 0.0
    intercept = statistics.median(y - slope * x for x, y in zip(xs, ys))
    return intercept, slope


def rolling_median(values: list[float], width: int) -> list[float]:
    half = width // 2
    return [statistics.median(values[max(0, i - half) : i + half + 1]) for i in range(len(values))]


def tidy(text: str) -> str:
    """Drops the ellipses that join a sentence across subtitles, and capitalises the start."""
    t = re.sub(r"^[.\s]+|[\s]*\.{2,}\s*$", "", text).strip()
    t = re.sub(r"\s*\.{3}\s*", ", ", t)
    t = re.sub(r"[,;:]+$", "", t)
    if t and t[-1] not in ".?!":
        t += "."
    return t[:1].upper() + t[1:] if t else t


def correct(text: str, heard: list[str]) -> str:
    """Replaces single subtitle words the recogniser heard differently (OCR slips)."""
    pieces = re.findall(r"\S+", text)
    norm = [(_WORD.findall(p.lower()) or [""])[0] for p in pieces]
    norm = [n if n else re.sub(r"\W", "", p) or "?" for n, p in zip(norm, pieces)]
    sm = difflib.SequenceMatcher(None, norm, heard, autojunk=False)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "replace" and a1 - a0 == b1 - b0:
            for k in range(a1 - a0):
                old, new = pieces[a0 + k], heard[b0 + k]
                short = len(norm[a0 + k]) <= 2 and len(new) <= 3  # "!", "lf", "1" for I, It
                if short or (
                    norm[a0 + k] and difflib.SequenceMatcher(None, norm[a0 + k], new).ratio() >= 0.5
                ):
                    core = re.search(r"[A-Za-z0-9'!|]+", old)
                    if core:
                        upper = (
                            core.group(0)[:1].isupper()
                            or a0 + k == 0
                            or new == "i"
                            or new.startswith("i'")
                        )
                        rep = new[:1].upper() + new[1:] if upper else new
                        pieces[a0 + k] = old[: core.start()] + rep + old[core.end() :]
    return " ".join(pieces)


def align(cues: list[Cue], words: list[dict]) -> tuple[list[Line], dict]:
    """Returns the subtitle lines with spoken times, and a report of the timing fit."""
    asr_tokens, asr_index = [], []
    for i, w in enumerate(words):
        for t in tokens(w["word"]):
            asr_tokens.append(t)
            asr_index.append(i)
    sub_tokens, sub_cue, sub_pos = [], [], []
    for k, c in enumerate(cues):
        for n, t in enumerate(tokens(c.text)):
            sub_tokens.append(t)
            sub_cue.append(k)
            sub_pos.append(n)
    matches: dict[int, list[int]] = {}
    matched_pos: dict[int, list[int]] = {}
    sm = difflib.SequenceMatcher(None, sub_tokens, asr_tokens, autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for j in range(n):
            matches.setdefault(sub_cue[a + j], []).append(asr_index[b + j])
            matched_pos.setdefault(sub_cue[a + j], []).append(sub_pos[a + j])

    # Timing fit from lines found confidently in the transcript.
    xs, ys = [], []
    for k, c in enumerate(cues):
        got = matches.get(k, [])
        if len(got) >= MIN_MATCHED and len(got) >= 0.5 * len(tokens(c.text)):
            xs.append(c.start)
            ys.append(words[min(got)]["start"] - c.start)
    if len(xs) >= 5:
        a, b = theil_sen(xs, ys)
        local = rolling_median([y - (a + b * x) for x, y in zip(xs, ys)], 9)
    else:
        a, b, local = 0.0, 0.0, []

    def shift(t: float) -> float:
        if not local:
            return a + b * t
        i = min(range(len(xs)), key=lambda n: abs(xs[n] - t))
        return a + b * t + local[i]

    lines = []
    for k, c in enumerate(cues):
        want = tokens(c.text)
        got = sorted(set(matches.get(k, [])))
        lo_t, hi_t = c.start + shift(c.start) - CONTEXT_S, c.end + shift(c.end) + CONTEXT_S
        if got:
            pos = matched_pos.get(k, [])
            idx = list(range(got[0], got[-1] + 1))
            # Widen only where the subtitle's own first or last words went unmatched (OCR slips);
            # a fully matched edge is the edge, and the next word belongs to someone else.
            missing_left, missing_right = min(pos), len(want) - 1 - max(pos)
            i = got[0] - 1
            while (
                missing_left > 0
                and i >= 0
                and words[i]["start"] >= lo_t
                and _free(i, k, matches)
                and words[i + 1]["start"] - words[i]["end"] < EDGE_GAP_S
            ):
                idx.insert(0, i)
                i, missing_left = i - 1, missing_left - 1
            i = got[-1] + 1
            while (
                missing_right > 0
                and i < len(words)
                and words[i]["end"] <= hi_t
                and _free(i, k, matches)
                and words[i]["start"] - words[i - 1]["end"] < EDGE_GAP_S
            ):
                idx.append(i)
                i, missing_right = i + 1, missing_right - 1
        else:
            idx = [i for i, w in enumerate(words) if lo_t <= w["start"] and w["end"] <= hi_t]
        span = [dict(words[i]) for i in sorted(set(idx))]
        if span:
            # The recogniser stretches a word at a speaker change back to the last voice it heard.
            span[0]["start"] = max(span[0]["start"], span[0]["end"] - MAX_EDGE_WORD_S)
            span[-1]["end"] = min(span[-1]["end"], span[-1]["start"] + MAX_EDGE_WORD_S)
        heard = [t for w in span for t in tokens(w["word"])]
        if span:
            start, end = span[0]["start"], span[-1]["end"]
        else:
            start, end = c.start + shift(c.start), c.end + shift(c.end)
        lines.append(
            Line(
                start=start,
                end=end,
                text=tidy(correct(c.text, heard) if heard else c.text),
                heard=" ".join(w["word"] for w in span),
                speaker=c.speaker,
                tags=c.tags,
                cue=c.cue,
                matched=round(len(got) / max(len(want), 1), 3),
                words=span,
            )
        )
    report = {
        "subtitle_lines": len(cues),
        "fitted_on": len(xs),
        "offset_s": round(a, 3),
        "drift_ppm": round(b * 1e6, 1),
        "local_jumps_s": round(max((abs(v) for v in local), default=0.0), 3),
    }
    return lines, report


def _free(i: int, k: int, matches: dict[int, list[int]]) -> bool:
    """True when word i is not claimed by another subtitle line."""
    return not any(i in v for kk, v in matches.items() if kk != k and abs(kk - k) <= 2)


def from_transcript(words: list[dict], gap_s: float = 0.6) -> list[Line]:
    """Without subtitles, cuts the transcript into lines at pauses and sentence ends."""
    lines, cur = [], []
    for w in words:
        if cur and (
            w["start"] - cur[-1]["end"] > gap_s or cur[-1]["word"].endswith((".", "?", "!"))
        ):
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    out = []
    for n, span in enumerate(lines):
        text = " ".join(w["word"] for w in span)
        out.append(
            Line(
                start=span[0]["start"],
                end=span[-1]["end"],
                text=text,
                heard=text,
                speaker=None,
                tags=[],
                cue=n,
                matched=1.0,
                words=span,
            )
        )
    return out


def coverage(lines: list[Line]) -> float:
    """Returns the share of a source's subtitle words found in the speech, a measure of its fit."""
    total = sum(len(tokens(ln.text)) for ln in lines)
    found = sum(ln.matched * len(tokens(ln.text)) for ln in lines)
    return found / total if total else 0.0


def choose(aligned: list[tuple[str, list[Line], dict]]) -> tuple[str, list[Line], dict, list[dict]]:
    """Picks the source that fits the speech best and gives it every other source's speaker labels.

    All sources are aligned to the same transcript, so two lines over the same recognised words
    are the same line, whatever times their files gave them.
    """
    scored = sorted(aligned, key=lambda a: coverage(a[1]), reverse=True)
    name, best, report = scored[0]
    summary = [
        {
            "source": n,
            "coverage": round(coverage(ls), 3),
            "labelled": sum(bool(ln.speaker) for ln in ls),
            "offset_s": r["offset_s"],
            "drift_ppm": r["drift_ppm"],
        }
        for n, ls, r in scored
    ]
    transferred = 0
    for _, other, _ in scored[1:]:
        labelled = [ln for ln in other if ln.speaker and ln.words]
        for ln in best:
            if ln.speaker or not ln.words:
                continue
            mine = {w["start"] for w in ln.words}
            for o in labelled:
                shared = mine & {w["start"] for w in o.words}
                if len(shared) >= max(1, 0.5 * len(mine)):
                    ln.speaker, ln.tags = o.speaker, sorted(set(ln.tags) | set(o.tags))
                    transferred += 1
                    break
    report = {**report, "source": name, "labels_transferred": transferred}
    return name, best, report, summary
