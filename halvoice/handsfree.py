"""Hands-free listening: one microphone stream, cut into utterances, each transcribed once.

The stream never closes, so nothing said between "Hal" and the request is
lost, and each utterance keeps PRE_ROLL of audio from before the voice
detector fired, so the soft "H" of the name survives. An utterance becomes a
request when it starts or ends with HAL's name, when it follows a bare "Hal"
within the command window, or when it comes during the follow-up window after
HAL has spoken. While HAL speaks, only his name counts, which interrupts him.
"""

from __future__ import annotations

import collections
import queue
import re
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import config
from .util import get_logger

log = get_logger(__name__)

FRAME_MS = 30           # webrtcvad accepts 10, 20 or 30 ms frames
PRE_ROLL_MS = 360       # audio kept from before the detector fired
MIN_SPEECH_MS = 240     # shorter bursts are clicks and coughs

# Whisper's usual spellings of a spoken "Hal", alone or after a greeting.
NAME = r"(?:hal|hall|hale|haul|howl|hull|hel|hell|al|owl|hel|hol)"
GREETING = r"(?:(?:hey|hi|hello|ok|okay|oi|right|so)[\s,]+)?"
LEADING = re.compile(rf"^\W*{GREETING}{NAME}\b[\s,.!?:;-]*", re.I)
TRAILING = re.compile(rf"[\s,]+{NAME}\W*$", re.I)


@dataclass
class Heard:
    text: str           # the request, with the name removed
    named: bool         # HAL's name was in it
    raw: str            # the full transcript


class Ears:
    """Background listener that yields requests from a queue."""

    def __init__(self, transcribe) -> None:
        cfg = config.load()
        self.sr = int(cfg.get("audio.sample_rate", 16000))
        self.device = cfg.get("audio.input_device")
        self.vad_level = int(cfg.get("audio.vad_aggressiveness", 2))
        self.end_ms = int(cfg.get("activation.keyword.end_silence_ms", 600))
        self.max_s = float(cfg.get("audio.max_utterance_s", 30))
        self.command_window = float(cfg.get("activation.keyword.command_window_s", 6))
        self.followup_s = float(cfg.get("activation.followup_s", 8))
        self.transcribe = transcribe
        self.requests: "queue.Queue[Heard]" = queue.Queue()
        self.interrupt = threading.Event()
        self.speaking = threading.Event()
        self._armed_until = 0.0         # a request without the name is accepted until then
        self._stop = threading.Event()
        self._audio: "queue.Queue[np.ndarray]" = queue.Queue()

    # -- windows ----------------------------------------------------------------
    def follow_up(self) -> None:
        """Accepts a request without the name for a short while (called when HAL stops talking)."""
        self._armed_until = time.time() + self.followup_s

    def disarm(self) -> None:
        self._armed_until = 0.0

    # -- threads ----------------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._capture, name="ears-capture", daemon=True).start()
        threading.Thread(target=self._understand, name="ears-stt", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _capture(self) -> None:
        """Splits the microphone stream into utterances and queues them for transcription."""
        import sounddevice as sd
        import webrtcvad

        vad = webrtcvad.Vad(self.vad_level)
        frame = int(self.sr * FRAME_MS / 1000)
        pre = collections.deque(maxlen=PRE_ROLL_MS // FRAME_MS)
        end_frames = self.end_ms // FRAME_MS
        max_frames = int(self.max_s * 1000 / FRAME_MS)
        voiced: list[np.ndarray] = []
        speech_frames = silent = 0
        with sd.InputStream(samplerate=self.sr, channels=1, dtype="int16", blocksize=frame,
                            device=self.device) as stream:
            log.info("listening for HAL's name (continuous, %d ms pre-roll)", PRE_ROLL_MS)
            while not self._stop.is_set():
                block, _ = stream.read(frame)
                chunk = block[:, 0].copy()
                is_speech = vad.is_speech(chunk.tobytes(), self.sr)
                if not voiced:
                    pre.append(chunk)
                    if is_speech:
                        voiced = list(pre)
                        speech_frames, silent = 1, 0
                    continue
                voiced.append(chunk)
                if is_speech:
                    speech_frames += 1
                    silent = 0
                else:
                    silent += 1
                if silent >= end_frames or len(voiced) >= max_frames:
                    if speech_frames * FRAME_MS >= MIN_SPEECH_MS:
                        self._audio.put(np.concatenate(voiced).astype(np.float32) / 32768.0)
                    voiced, speech_frames, silent = [], 0, 0
                    pre.clear()

    def _understand(self) -> None:
        """Transcribes utterances and decides which are requests."""
        pending_name_at = 0.0
        while not self._stop.is_set():
            try:
                audio = self._audio.get(timeout=0.5)
            except queue.Empty:
                continue
            text = self.transcribe(audio, prompt="Hal,").strip()
            if not text or re.fullmatch(r"\W*(you|thank you|thanks for watching)?\W*", text, re.I):
                continue
            lead = LEADING.match(text)
            trail = TRAILING.search(text)
            named = bool(lead or trail)
            request = text[lead.end():] if lead else text[:trail.start()] if trail else text
            request = request.strip(" ,.!?")
            now = time.time()
            if self.speaking.is_set():
                if named:
                    log.info("heard HAL's name while speaking: interrupting")
                    self.interrupt.set()
                    if request:
                        self.requests.put(Heard(request, True, text))
                    else:
                        pending_name_at = now
                continue
            if named and not request:
                log.info("heard HAL's name: listening for the request")
                pending_name_at = now
                self.requests.put(Heard("", True, text))
                continue
            armed = now < self._armed_until or now - pending_name_at < self.command_window
            if named or armed:
                pending_name_at = 0.0
                self.disarm()
                log.info("you: %s", request)
                self.requests.put(Heard(request, named, text))
            else:
                log.debug("ignored (no name): %r", text)
