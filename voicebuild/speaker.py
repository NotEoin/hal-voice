"""Finds the lines HAL speaks, by voice.

Every line is embedded with GPT-SoVITS's speaker-verification model (ERes2NetV2). HAL's voice
is seeded from lines the subtitles label "HAL:", a few phrases only HAL says, and any clips of
HAL already in voice-data/hal-voice; every other labelled character becomes a counter-example.
The threshold sits between the two distributions, so it adapts to each copy of the film.
Needs numpy, torch, soundfile and a GPT-SoVITS checkout.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

from .sync import Line, tokens

# Phrases nobody else in the film says; matched against whole lines, so a near miss fails safe.
ANCHORS = [
    "i'm sorry dave i'm afraid i can't do that",
    "good afternoon gentlemen",
    "this mission is too important",
    "affirmative dave i read you",
    "just what do you think you're doing dave",
    "i'm afraid that's something i cannot allow to happen",
    "thank you for a very enjoyable game",
]
MIN_SECONDS = 0.8
MARGIN = 0.08  # how much closer to HAL than to anyone else a line must be
MIN_FILM_SEEDS = 8  # below this, clips from voice-data/hal-voice join the seeds
FLOOR = 0.65


class Embedder:
    """Speaker embeddings from GPT-SoVITS's ERes2NetV2, on 16 kHz mono audio."""

    def __init__(self, gsv: Path) -> None:
        import torch

        cwd = os.getcwd()
        os.chdir(gsv)  # sv.py loads its checkpoint by a relative path
        for p in (str(gsv), str(gsv / "GPT_SoVITS")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from sv import SV

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sv = SV(self.device, False)
        os.chdir(cwd)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        wav = self.torch.tensor(audio, dtype=self.torch.float32)[None].to(self.device)
        with self.torch.no_grad():
            e = self.sv.compute_embedding3(wav)[0].float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)


def _is_anchor(text: str) -> bool:
    said = " ".join(tokens(text))
    return any(a == said or (len(a) > 20 and a in said) for a in ANCHORS)


def identify(
    lines: list[Line],
    audio: np.ndarray,
    rate: int,
    embed: Embedder,
    seed_clips: list[Path] | None = None,
) -> tuple[list[dict], dict]:
    """Returns each line's similarity to HAL and to others, whether it is HAL, and a report."""
    import soundfile as sf

    vecs: dict[int, np.ndarray] = {}
    for i, ln in enumerate(lines):
        if ln.end - ln.start >= MIN_SECONDS:
            seg = audio[int(ln.start * rate) : int(ln.end * rate)]
            if seg.size >= MIN_SECONDS * rate:
                vecs[i] = embed(seg)
    seeds = [
        i
        for i in vecs
        if (lines[i].speaker == "HAL" and "sings" not in lines[i].tags) or _is_anchor(lines[i].text)
    ]
    seed_vecs = [vecs[i] for i in seeds]
    # Clips from elsewhere differ in recording and processing, so they only fill in for a film
    # whose subtitles give too few certain lines.
    if len(seed_vecs) < MIN_FILM_SEEDS:
        for clip in seed_clips or []:
            try:
                y, sr = sf.read(str(clip), dtype="float32", always_2d=True)
            except (RuntimeError, OSError):
                continue
            y = y.mean(axis=1)
            if sr != rate:
                import librosa

                y = librosa.resample(y, orig_sr=sr, target_sr=rate)
            if y.size >= MIN_SECONDS * rate:
                seed_vecs.append(embed(y))
    if len(seed_vecs) < 3:
        raise RuntimeError(
            "found fewer than 3 lines that are certainly HAL; check the subtitles are the film's"
        )
    # A label can sit on a line another sound dominates; drop seeds far from the rest.
    hal = _unit(np.mean(seed_vecs, axis=0))
    sims = [float(v @ hal) for v in seed_vecs]
    seed_vecs = [v for v, s in zip(seed_vecs, sims) if s >= float(np.median(sims)) - 0.15]
    others: dict[str, list[np.ndarray]] = {}
    for i in vecs:
        spk = lines[i].speaker
        if spk and spk != "HAL" and not _is_anchor(lines[i].text):
            others.setdefault(spk, []).append(vecs[i])
    other_centres = {k: _unit(np.mean(v, axis=0)) for k, v in others.items() if len(v) >= 2}

    def nearest_other(v: np.ndarray) -> float:
        return max((float(v @ c) for c in other_centres.values()), default=-1.0)

    hal = _unit(np.mean(seed_vecs, axis=0))
    for _ in range(3):
        seed_sims = [float(v @ hal) for v in seed_vecs]
        other_sims = [
            float(vecs[i] @ hal) for i in vecs if lines[i].speaker and lines[i].speaker != "HAL"
        ]
        low_seed = float(np.percentile(seed_sims, 10))
        high_other = float(np.percentile(other_sims, 95)) if other_sims else FLOOR
        threshold = max(high_other + 0.02, FLOOR)
        confident = [
            v
            for i, v in vecs.items()
            if float(v @ hal) >= threshold + 0.05
            and float(v @ hal) - nearest_other(v) >= MARGIN
            and lines[i].speaker in (None, "HAL")
        ]
        hal = _unit(np.mean(seed_vecs + confident, axis=0))

    results = []
    for i, ln in enumerate(lines):
        if i not in vecs:
            results.append(
                {"line": i, "hal": False, "sim": None, "other": None, "why": "too short to tell"}
            )
            continue
        s = float(vecs[i] @ hal)
        nearest = nearest_other(vecs[i])
        labelled_other = ln.speaker is not None and ln.speaker != "HAL"
        is_hal = s >= threshold and s - nearest >= MARGIN and not labelled_other
        why = (
            "labelled as " + ln.speaker.title()
            if labelled_other
            else "seed"
            if i in seeds
            else "voice matches HAL"
            if is_hal
            else "voice is not HAL's"
        )
        results.append(
            {"line": i, "hal": is_hal, "sim": round(s, 3), "other": round(nearest, 3), "why": why}
        )
    report = {
        "lines_embedded": len(vecs),
        "seed_lines": len(seeds),
        "seeds_used": len(seed_vecs),
        "threshold": round(threshold, 3),
        "seed_10th_pct": round(low_seed, 3),
        "others_95th_pct": round(high_other, 3),
        "other_speakers": sorted(other_centres),
        "hal_lines": sum(r["hal"] for r in results),
    }
    return results, report


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def seed_clips(folder: Path) -> list[Path]:
    """Returns existing HAL clips to seed from, if the user has any."""
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if re.search(r"\.(wav|mp3|flac|ogg|m4a)$", p.name, re.I)
    )
