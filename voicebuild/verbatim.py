"""Writes down exactly what HAL says in each kept line, for the training transcripts.

Subtitles condense speech ("after four months of training" for "after four months of separate
training"), and a voice model learns the words it is told go with each sound, so every line is
transcribed again on its own audio and reconciled with its subtitle: words the subtitle left out
or condensed come from the speech, and a word swapped one for one ("greetings" for "credence",
a name's spelling) comes from the subtitle, as the recogniser mishears more than subtitles
invent. The subtitle is not given to the recogniser as a prompt: with it, Whisper takes a line
broken by a pause as finished at the pause.
"""

from __future__ import annotations

import difflib
import re

import numpy as np

from .sync import tokens

PAD_S = 0.15  # read just past the line so a clipped last word is still heard


class Verbatim:
    """faster-whisper, decoding carefully one line at a time."""

    def __init__(self, model: str = "medium.en") -> None:
        import torch
        from faster_whisper import WhisperModel

        gpu = torch.cuda.is_available()
        self.asr = WhisperModel(
            model, device="cuda" if gpu else "cpu", compute_type="float16" if gpu else "int8"
        )

    def __call__(self, audio: np.ndarray, rate: int, start: float, end: float, hint: str) -> str:
        seg = audio[int(max(0.0, start - PAD_S) * rate) : int((end + PAD_S) * rate)]
        segments, _ = self.asr.transcribe(
            seg.astype(np.float32),
            language="en",
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        heard = re.sub(r"\s+", " ", " ".join(s.text.strip() for s in segments)).strip()
        return reconcile(heard, hint)


def _norm(piece: str) -> str:
    return " ".join(tokens(piece)) or piece


def reconcile(heard: str, subtitle: str) -> str:
    """Returns the heard words, with one-for-one swaps taken back from the subtitle."""
    said = re.findall(r"\S+", heard)
    wrote = re.findall(r"\S+", subtitle)
    sm = difflib.SequenceMatcher(
        None, [_norm(p) for p in said], [_norm(p) for p in wrote], autojunk=False
    )
    out = []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "replace" and a1 - a0 == b1 - b0:
            for k in range(a1 - a0):
                core = re.search(r"[A-Za-z0-9'-]+", wrote[b0 + k])
                old = re.search(r"[A-Za-z0-9'-]+", said[a0 + k])
                if core and old:
                    out.append(
                        said[a0 + k][: old.start()] + core.group(0) + said[a0 + k][old.end() :]
                    )
                else:
                    out.append(said[a0 + k])
        else:
            out += said[a0:a1]
    return " ".join(out)
