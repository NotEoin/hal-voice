"""Text-to-speech: synthesize a line in HAL's voice via GPT-SoVITS, apply the HAL FX
chain, and play it. Talks to a running GPT-SoVITS api_v2 server over HTTP (started by
gptsovits/run-api.sh). Keeping GPT-SoVITS in its own process/venv keeps its heavy torch
stack cleanly separate from hal-voice.

Public API:
  speak(text)                  synthesize + FX + play (blocking).
  synth(text) -> (audio, sr)   synthesize + FX, no playback (used by the MCP server).
  Player                        small playback helper with stop() for barge-in.

The HAL pacing trick: long replies are split into sentences and streamed — we start
speaking sentence 1 while sentence 2 synthesizes, which hides CPU TTS latency.
"""

from __future__ import annotations

import io
import re
import sys
import threading
from typing import Iterator

import numpy as np

from . import config, hal_fx
from .util import get_logger

log = get_logger(__name__)

_SENTENCE_RE = re.compile(r"(.+?(?:[.!?]+|\n|$))", re.S)


def split_sentences(text: str) -> list[str]:
    """Split into speakable chunks. Keeps HAL's measured, one-thought-at-a-time delivery."""
    text = text.strip()
    if not text:
        return []
    chunks = [m.group(1).strip() for m in _SENTENCE_RE.finditer(text)]
    return [c for c in chunks if c]


class GPTSoVITSClient:
    """Thin HTTP client for GPT-SoVITS api_v2 /tts (and weight hot-swap endpoints)."""

    def __init__(self) -> None:
        cfg = config.load()
        t = cfg.tts
        self.url = str(t.get("api_url", "http://127.0.0.1:9880")).rstrip("/")
        self.text_lang = str(t.get("text_lang", "en"))
        self.prompt_lang = str(t.get("prompt_lang", "en"))
        self.ref_audio = config.resolve(t.get("ref_audio_path"))
        # Extra references blended in for timbre; the main one still sets the cadence.
        self.aux_refs = [config.resolve(p) for p in (t.get("aux_ref_audio_paths") or [])]
        self.prompt_text = str(t.get("prompt_text", ""))
        self.split_method = str(t.get("text_split_method", "cut2"))
        self.batch_size = int(t.get("batch_size", 1))
        self.top_k = int(t.get("top_k", 10))
        self.top_p = float(t.get("top_p", 0.7))
        self.temperature = float(t.get("temperature", 0.7))
        self.speed = float(t.get("speed", 0.92))
        self._weights_set = False

    def ensure_weights(self) -> None:
        """Tell the server to load HAL's fine-tuned checkpoints (once per process)."""
        if self._weights_set:
            return
        import requests

        cfg = config.load()
        gpt = config.resolve(cfg.get("tts.gpt_weights"))
        sov = config.resolve(cfg.get("tts.sovits_weights"))
        try:
            if gpt:
                requests.get(f"{self.url}/set_gpt_weights", params={"weights_path": gpt}, timeout=60)
            if sov:
                requests.get(f"{self.url}/set_sovits_weights", params={"weights_path": sov}, timeout=60)
            self._weights_set = True
            log.info("GPT-SoVITS HAL weights loaded")
        except Exception as e:  # noqa: BLE001
            # Not fatal: the server may already have them baked in via -g/-s flags.
            log.warning("could not hot-swap weights (%s) — assuming server already has HAL loaded", e)
            self._weights_set = True

    def synth_raw(self, text: str) -> tuple[np.ndarray, int]:
        """POST one text chunk, return (mono float32 audio, sample_rate) — no FX."""
        import requests
        import soundfile as sf

        self.ensure_weights()
        payload = {
            "text": text,
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio,
            "aux_ref_audio_paths": self.aux_refs,
            "prompt_text": self.prompt_text,
            "prompt_lang": self.prompt_lang,
            "text_split_method": self.split_method,
            "batch_size": self.batch_size,
            "media_type": "wav",
            "streaming_mode": False,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "speed_factor": self.speed,
        }
        r = requests.post(f"{self.url}/tts", json=payload, timeout=300)
        if r.status_code != 200:
            raise RuntimeError(f"GPT-SoVITS /tts failed ({r.status_code}): {r.text[:200]}")
        audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), int(sr)


EDGE_GATE_DB = 38.0   # audio this far below the line's peak, before speech or after it, is cut
EDGE_FADE_S = 0.012


def gate_edges(audio: np.ndarray, sr: int) -> np.ndarray:
    """Trims the faint noise bed before the first word and after the last, with short fades.

    The bed is quiet under speech but audible on its own, so it is removed only where
    nothing else is playing: at the start and end of each synthesised line.
    """
    if audio.size < sr // 10:
        return audio
    frame = sr // 100
    n = audio.size // frame
    rms = np.sqrt(np.mean(audio[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    loud = np.flatnonzero(db > db.max() - EDGE_GATE_DB)
    if loud.size == 0:
        return audio
    start = max(0, loud[0] - 2) * frame
    end = min(audio.size, (loud[-1] + 3) * frame)
    out = audio[start:end].copy()
    fade = min(int(EDGE_FADE_S * sr), out.size // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    return out


def synth(text: str) -> tuple[np.ndarray, int]:
    """Synthesize the whole text (joined) + HAL FX. Returns (audio, sr)."""
    client = GPTSoVITSClient()
    parts = []
    sr = 32000
    for chunk in split_sentences(text) or [text]:
        audio, sr = client.synth_raw(chunk)
        parts.append(gate_edges(audio, sr))
    full = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return hal_fx.apply(full, sr), sr


_client: "GPTSoVITSClient | None" = None


def _shared_client() -> "GPTSoVITSClient":
    """One client per process, so HAL's weights are sent to the server once, not per sentence."""
    global _client
    if _client is None:
        _client = GPTSoVITSClient()
    return _client


def synth_stream(text: str) -> Iterator[tuple[np.ndarray, int]]:
    """Yield (audio, sr) per sentence, FX applied — lets the loop start playback early."""
    client = _shared_client()
    for chunk in split_sentences(text):
        audio, sr = client.synth_raw(chunk)
        yield hal_fx.apply(gate_edges(audio, sr), sr), sr


class Player:
    """Plays audio and can be interrupted (barge-in). Tracks an output level for the UI."""

    def __init__(self) -> None:
        cfg = config.load()
        self.device = cfg.get("audio.output_device")
        self._stop = threading.Event()
        self.level = 0.0

    def stop(self) -> None:
        self._stop.set()

    def reset(self) -> None:
        self._stop.clear()

    def play(self, audio: np.ndarray, sr: int, on_level=None) -> bool:
        """Blocking playback. Returns False if interrupted via stop(). Calls on_level(0..1)."""
        import sounddevice as sd

        self.reset()
        block = max(256, sr // 50)  # ~20 ms blocks for responsive barge-in + level metering
        idx = 0
        audio = np.asarray(audio, dtype=np.float32).flatten()
        with sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                            blocksize=block, device=self.device) as stream:
            while idx < len(audio) and not self._stop.is_set():
                seg = audio[idx:idx + block]
                stream.write(seg)
                self.level = float(np.sqrt(np.mean(seg ** 2)) * 3.0) if seg.size else 0.0
                if on_level:
                    on_level(min(1.0, self.level))
                idx += block
        self.level = 0.0
        if on_level:
            on_level(0.0)
        return not self._stop.is_set()


def speak(text: str, player: "Player | None" = None, on_level=None) -> None:
    """Synthesize + speak the full text (sentence-streamed). Blocking."""
    player = player or Player()
    for audio, sr in synth_stream(text):
        if not player.play(audio, sr, on_level=on_level):
            break  # interrupted


def main() -> None:
    # `python -m halvoice.tts --say "..."`
    args = sys.argv[1:]
    if "--say" in args:
        text = " ".join(args[args.index("--say") + 1:]).strip()
    else:
        text = " ".join(args).strip()
    if not text:
        print("usage: python -m halvoice.tts --say \"line to speak\"", file=sys.stderr)
        raise SystemExit(2)
    speak(text)


if __name__ == "__main__":
    main()
