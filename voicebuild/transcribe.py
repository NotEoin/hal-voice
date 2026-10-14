"""Transcribes the film's dialogue with word timings (faster-whisper), cached as JSON."""
from __future__ import annotations

import json
from pathlib import Path


def transcribe(wav: Path, out: Path, model: str = "medium.en", device: str = "auto") -> list[dict]:
    """Returns every recognised word as {start, end, word, prob}; reuses `out` when it exists."""
    if out.exists():
        return json.loads(out.read_text())
    import torch
    from faster_whisper import WhisperModel
    gpu = torch.cuda.is_available() if device == "auto" else device == "cuda"
    asr = WhisperModel(model, device="cuda" if gpu else "cpu", compute_type="float16" if gpu else "int8")
    # Each segment is decoded on its own, so one misheard line can't send the rest of the film
    # into a repetition loop.
    segments, _ = asr.transcribe(str(wav), language="en", word_timestamps=True, vad_filter=True,
                                 condition_on_previous_text=False, beam_size=5)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip(),
                          "prob": round(w.probability, 3)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(words))
    return words
