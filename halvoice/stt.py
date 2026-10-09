"""Speech-to-text: record one utterance from the mic and transcribe it on the CPU.

Pipeline: capture 16 kHz mono frames -> webrtcvad trims to the spoken segment ->
faster-whisper transcribes. Device and precision come from the profile chosen by
setup-voice.sh — CPU on the `12gb` and `cloud` profiles, so the GPU stays free.
`record_utterance()` blocks until speech stops (trailing silence) or the
max-utterance cap is reached.
"""

from __future__ import annotations

import collections
import sys
import time

import numpy as np

from . import config
from .util import get_logger

log = get_logger(__name__)

_FRAME_MS = 30  # webrtcvad accepts 10/20/30 ms frames


class Recorder:
    """Captures a single utterance, returning float32 mono audio at the STT sample rate."""

    def __init__(self) -> None:
        cfg = config.load()
        self.sr = int(cfg.get("audio.sample_rate", 16000))
        self.device = cfg.get("audio.input_device")
        self.vad_level = int(cfg.get("audio.vad_aggressiveness", 2))
        self.silence_ms = int(cfg.get("audio.silence_ms", 800))
        self.max_s = float(cfg.get("audio.max_utterance_s", 30))

    def record_utterance(self, on_first_voice=None, hold_check=None) -> np.ndarray:
        """Capture one utterance as float32 mono.

        Two modes:
          * hold-to-talk (hold_check given): record ONLY while hold_check() is True
            (the push-to-talk key is held). Stops the instant the key is released —
            no VAD-silence guessing, so HAL responds immediately on release.
          * VAD (hold_check None): wait for speech, stop after `silence_ms` of trailing
            silence (used by wake-word activation and the `listen` one-shot).
        `max_utterance_s` always caps the recording as a safety stop.
        """
        import sounddevice as sd

        frame_len = int(self.sr * _FRAME_MS / 1000)
        t0 = time.time()
        log.debug("recording (sr=%d, device=%s, hold=%s)", self.sr, self.device, hold_check is not None)

        # ---- Hold-to-talk: bounded purely by the key being held ------------------
        if hold_check is not None:
            frames: list[np.ndarray] = []
            notified = False
            with sd.InputStream(samplerate=self.sr, channels=1, dtype="int16",
                                blocksize=frame_len, device=self.device) as stream:
                while hold_check():
                    if time.time() - t0 > self.max_s:
                        log.debug("max utterance length reached")
                        break
                    block, _ = stream.read(frame_len)
                    frames.append(block[:, 0])
                    if not notified and on_first_voice:
                        on_first_voice()
                        notified = True
            if not frames:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(frames).astype(np.float32) / 32768.0

        # ---- VAD-bounded capture (wake word / one-shot) --------------------------
        import webrtcvad

        vad = webrtcvad.Vad(self.vad_level)
        ring = collections.deque(maxlen=int(self.silence_ms / _FRAME_MS))
        voiced: list[np.ndarray] = []
        started = False

        with sd.InputStream(samplerate=self.sr, channels=1, dtype="int16",
                            blocksize=frame_len, device=self.device) as stream:
            while True:
                if time.time() - t0 > self.max_s:
                    log.debug("max utterance length reached")
                    break
                block, _ = stream.read(frame_len)
                frame = block[:, 0]
                is_speech = vad.is_speech(frame.tobytes(), self.sr)
                if not started:
                    if is_speech:
                        started = True
                        if on_first_voice:
                            on_first_voice()
                        voiced.append(frame)
                else:
                    voiced.append(frame)
                    ring.append(is_speech)
                    # Stop once the trailing window is fully silent.
                    if len(ring) == ring.maxlen and not any(ring):
                        break

        if not voiced:
            return np.zeros(0, dtype=np.float32)
        pcm16 = np.concatenate(voiced).astype(np.float32) / 32768.0
        return pcm16


def _preload_cuda() -> None:
    """Loads the cuBLAS and cuDNN wheels' libraries so CTranslate2 can find them.

    The dynamic loader reads LD_LIBRARY_PATH only at process start, so the
    nvidia-* pip packages' libraries are loaded explicitly, before CTranslate2.
    """
    import ctypes
    import glob
    import os
    import site

    for base in site.getsitepackages():
        for pattern in ("nvidia/cublas/lib/libcublasLt.so.*", "nvidia/cublas/lib/libcublas.so.*",
                        "nvidia/cudnn/lib/libcudnn*.so.*"):
            for path in sorted(glob.glob(os.path.join(base, pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


class Transcriber:
    """Lazy faster-whisper wrapper. Loads the model once, reuses it.

    `stt.device: auto` uses the GPU when CUDA works and falls back to the CPU.
    """

    _model = None
    _lock = __import__("threading").Lock()

    def __init__(self) -> None:
        cfg = config.load()
        self.model_name = str(cfg.get("stt.model", "small.en"))
        self.device = str(cfg.get("stt.device", "cpu"))
        self.compute_type = str(cfg.get("stt.compute_type", "int8"))
        self.cpu_model = str(cfg.get("stt.cpu_model", "small.en"))
        self.beam_size = int(cfg.get("stt.beam_size", 1))
        self.language = cfg.get("stt.language", "en")
        self.hotwords = cfg.get("stt.hotwords", "Hal")

    def _load(self):
        with Transcriber._lock:
            if Transcriber._model is None:
                from faster_whisper import WhisperModel

                if self.device in ("auto", "cuda"):
                    _preload_cuda()
                    try:
                        ct = self.compute_type if self.compute_type != "int8" else "float16"
                        log.info("loading STT model %s (cuda/%s)", self.model_name, ct)
                        Transcriber._model = WhisperModel(self.model_name, device="cuda", compute_type=ct)
                        return Transcriber._model
                    except (RuntimeError, ValueError) as e:
                        log.warning("STT on the GPU failed (%s); using the CPU", e)
                log.info("loading STT model %s (cpu/int8)", self.cpu_model)
                Transcriber._model = WhisperModel(self.cpu_model, device="cpu", compute_type="int8")
        return Transcriber._model

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> str:
        if audio.size == 0:
            return ""
        model = self._load()
        segments, _info = model.transcribe(
            audio, beam_size=self.beam_size, language=self.language, initial_prompt=prompt,
            hotwords=self.hotwords or None, vad_filter=False, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()


def listen_once(on_first_voice=None) -> str:
    """Record one utterance and return its transcript (used by the loop and `hal-voice listen`)."""
    audio = Recorder().record_utterance(on_first_voice=on_first_voice)
    return Transcriber().transcribe(audio)


def main() -> None:
    once = "--once" in sys.argv
    print("Listening… speak now." if once else "Listening (Ctrl-C to stop)…", file=sys.stderr)
    while True:
        text = listen_once()
        print(text)
        if once:
            return


if __name__ == "__main__":
    main()
