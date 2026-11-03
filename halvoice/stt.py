"""Speech-to-text: record one utterance from the mic and transcribe it on the CPU.

Pipeline: capture 16 kHz mono frames -> webrtcvad trims to the spoken segment ->
faster-whisper transcribes. Device and precision come from `halvoice.placement`: the CPU
unless you ask for a GPU, so the card stays free for background work.
`record_utterance()` blocks until speech stops (trailing silence) or the
max-utterance cap is reached.
"""

from __future__ import annotations

import collections
import re
import sys
import time

import numpy as np

from . import config, placement
from .util import get_logger

log = get_logger(__name__)

_FRAME_MS = 30  # webrtcvad accepts 10/20/30 ms frames
NO_SPEECH_MAX = 0.5  # segments Whisper thinks are more likely silence than speech are dropped
MIN_LOGPROB = -1.0  # and so are segments it was very unsure of
_WORDS = re.compile(r"[a-z0-9']+")


def _normal(text: str) -> str:
    return " ".join(_WORDS.findall(text.lower()))


def prompt_sentences(prompt: str | None) -> set[str]:
    """The normalised sentences of `prompt`; Whisper repeats them back when it hears only noise."""
    return {n for part in re.split(r"[.!?]+", prompt or "") if (n := _normal(part))}


class Recorder:
    """Captures a single utterance, returning float32 mono audio at the STT sample rate."""

    def __init__(self, max_utterance_s: float | None = None) -> None:
        """`max_utterance_s` replaces the configured recording cap for this recorder only."""
        cfg = config.load()
        self.sr = int(cfg.get("audio.sample_rate", 16000))
        self.device = cfg.get("audio.input_device")
        self.vad_level = int(cfg.get("audio.vad_aggressiveness", 2))
        self.silence_ms = int(cfg.get("audio.silence_ms", 800))
        self.max_s = float(
            max_utterance_s if max_utterance_s is not None else cfg.get("audio.max_utterance_s", 30)
        )

    def record_utterance(self, on_first_voice=None, hold_check=None) -> np.ndarray:
        """Capture one utterance as float32 mono.

        Two modes:
          * hold-to-talk (hold_check given): record ONLY while hold_check() is True
            (the push-to-talk key is held). Stops the instant the key is released --
            no VAD-silence guessing, so HAL responds immediately on release.
          * VAD (hold_check None): wait for speech, stop after `silence_ms` of trailing
            silence (used by wake-word activation and the `listen` one-shot).
        `max_utterance_s` always caps the recording as a safety stop.
        """
        import sounddevice as sd

        frame_len = int(self.sr * _FRAME_MS / 1000)
        t0 = time.time()
        log.debug(
            "recording (sr=%d, device=%s, hold=%s)", self.sr, self.device, hold_check is not None
        )

        # ---- Hold-to-talk: bounded purely by the key being held ------------------
        if hold_check is not None:
            frames: list[np.ndarray] = []
            notified = False
            with sd.InputStream(
                samplerate=self.sr,
                channels=1,
                dtype="int16",
                blocksize=frame_len,
                device=self.device,
            ) as stream:
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

        with sd.InputStream(
            samplerate=self.sr, channels=1, dtype="int16", blocksize=frame_len, device=self.device
        ) as stream:
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
        for pattern in (
            "nvidia/cublas/lib/libcublasLt.so.*",
            "nvidia/cublas/lib/libcublas.so.*",
            "nvidia/cudnn/lib/libcudnn*.so.*",
        ):
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
        placed = placement.resolve(cfg)
        self.device = placed.stt_device
        self.compute_type = placed.stt_compute
        self.cpu_model = str(cfg.get("stt.cpu_model", "small.en"))
        self.beam_size = int(cfg.get("stt.beam_size", 1))
        self.language = cfg.get("stt.language", "en")
        self.hotwords = cfg.get("stt.hotwords", "Hal")
        self.prompt = cfg.get("stt.initial_prompt") or None
        self.highpass_hz = float(cfg.get("stt.highpass_hz", 0) or 0)
        self.level_dbfs = cfg.get("stt.normalise_dbfs")

    def _load(self):
        with Transcriber._lock:
            if Transcriber._model is None:
                from faster_whisper import WhisperModel

                if self.device in ("auto", "cuda"):
                    _preload_cuda()
                    try:
                        ct = self.compute_type if self.compute_type != "int8" else "float16"
                        log.info("loading STT model %s (cuda/%s)", self.model_name, ct)
                        Transcriber._model = WhisperModel(
                            self.model_name, device="cuda", compute_type=ct
                        )
                        return Transcriber._model
                    except (RuntimeError, ValueError) as e:
                        log.warning("STT on the GPU failed (%s); using the CPU", e)
                log.info("loading STT model %s (cpu/int8)", self.cpu_model)
                Transcriber._model = WhisperModel(self.cpu_model, device="cpu", compute_type="int8")
        return Transcriber._model

    def prepare(self, audio: np.ndarray) -> np.ndarray:
        """Takes out what is below a voice, and brings a quiet one up to a set level."""
        if self.highpass_hz:
            from scipy.signal import butter, sosfilt

            sos = butter(4, self.highpass_hz, "highpass", fs=16000, output="sos")
            audio = sosfilt(sos, audio).astype(np.float32)
        if self.level_dbfs is not None:
            loud = np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-9
            gain = 10 ** (float(self.level_dbfs) / 20) / loud
            audio = np.clip(audio * min(gain, 30.0), -1.0, 1.0).astype(np.float32)
        return audio

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> str:
        """Returns the text of `audio`, or an empty string when it holds no speech."""
        if audio.size == 0:
            return ""
        model = self._load()
        audio = self.prepare(audio)
        segments, _info = model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=self.language,
            initial_prompt=prompt or self.prompt,
            hotwords=self.hotwords or None,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        # Noise and silence come back as confident-sounding words; Whisper's own scores
        # say which segments were really speech. On noise it also echoes the prompt it was
        # given, so a segment made only of prompt sentences is dropped too.
        echoes = prompt_sentences(prompt or self.prompt)
        kept = [
            s.text.strip()
            for s in segments
            if s.no_speech_prob < NO_SPEECH_MAX
            and s.avg_logprob > MIN_LOGPROB
            and not _only_echo(s.text, echoes)
        ]
        return " ".join(kept).strip()


def _only_echo(text: str, echoes: set[str]) -> bool:
    """True when every sentence of `text` is a prompt sentence (an empty text is not an echo)."""
    parts = prompt_sentences(text)
    return bool(parts) and parts <= echoes


def listen_once(on_first_voice=None, max_utterance_s: float | None = None) -> str:
    """Record one utterance and return its transcript (used by the loop and `hal-voice listen`).

    `max_utterance_s` caps this recording without touching the configuration.
    """
    audio = Recorder(max_utterance_s).record_utterance(on_first_voice=on_first_voice)
    return Transcriber().transcribe(audio)


def main() -> None:
    """Transcribes the microphone to the terminal, once with `--once` or until interrupted."""
    once = "--once" in sys.argv
    print("Listening... speak now." if once else "Listening (Ctrl-C to stop)...", file=sys.stderr)
    while True:
        text = listen_once()
        print(text)
        if once:
            return


if __name__ == "__main__":
    main()
