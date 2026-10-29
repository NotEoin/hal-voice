"""Activation: decide when a turn begins.

Three triggers, combined per config.activation:
  * keyword      - hands-free: short speech bursts are transcribed on the CPU and a
                   turn starts when one begins with HAL's name. Words after the name
                   become the request, so "Hal, what's new?" is a single turn.
  * wake_word    - openWakeWord listens for a trained wake-word model (ONNX, CPU).
  * push_to_talk - hold a key (default: space) to start a turn.

`wait_for_activation()` blocks until one fires, then returns the trigger name.
All run on background listeners so they race naturally.
"""

from __future__ import annotations

import re
import threading
import time

import numpy as np

from . import config
from .util import get_logger

log = get_logger(__name__)


class _PushToTalk:
    """Sets an event while the configured key is held down."""

    def __init__(self, key_name: str) -> None:
        self.key_name = key_name
        self.pressed = threading.Event()
        self._listener = None

    def start(self) -> None:
        from pynput import keyboard

        target = self._resolve_key(keyboard, self.key_name)

        def on_press(key):
            if self._matches(key, target):
                self.pressed.set()

        def on_release(key):
            if self._matches(key, target):
                self.pressed.clear()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        log.info("push-to-talk armed (hold '%s')", self.key_name)

    @staticmethod
    def _resolve_key(keyboard, name: str):
        special = getattr(keyboard.Key, name, None)
        if special is not None:
            return special
        return keyboard.KeyCode.from_char(name[:1])

    @staticmethod
    def _matches(key, target) -> bool:
        try:
            return key == target or getattr(key, "char", None) == getattr(target, "char", None)
        except Exception:  # noqa: BLE001 - pynput key objects differ by platform; an odd key never matches
            return False

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()


class _WakeWord:
    """openWakeWord continuous listener; sets an event when the phrase is detected."""

    def __init__(self) -> None:
        cfg = config.load()
        self.sr = int(cfg.get("audio.sample_rate", 16000))
        self.device = cfg.get("audio.input_device")
        self.threshold = float(cfg.get("activation.wake.threshold", 0.6))
        self.model_name = str(cfg.get("activation.wake.model", "hey_jarvis"))
        self.custom = config.resolve(cfg.get("activation.wake.custom_model_path"))
        self.detected = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import sounddevice as sd
        from openwakeword.model import Model

        kwargs = {"wakeword_models": [self.custom]} if self.custom else {}
        try:
            oww = Model(**kwargs)
        except Exception as e:  # noqa: BLE001 - any model failure disables wake word; push-to-talk stays
            log.error("wake word model failed to load (%s) - disabling wake word.", e)
            return
        # openWakeWord expects 16 kHz int16 frames of 1280 samples (80 ms).
        chunk = 1280
        with sd.InputStream(
            samplerate=self.sr, channels=1, dtype="int16", blocksize=chunk, device=self.device
        ) as stream:
            log.info(
                "wake word listening ('%s', threshold %.2f)...",
                self.custom or self.model_name,
                self.threshold,
            )
            while not self._stop.is_set():
                block, _ = stream.read(chunk)
                preds = oww.predict(block[:, 0].astype(np.int16))
                if any(score >= self.threshold for score in preds.values()):
                    log.debug("wake word detected: %s", preds)
                    self.detected.set()
                    # Debounce: wait for the event to be cleared by the consumer.
                    while self.detected.is_set() and not self._stop.is_set():
                        time.sleep(0.05)

    def stop(self) -> None:
        self._stop.set()


DEFAULT_KEYWORD = r"^\W*(?:(?:hey|hi|hello|ok|okay)\W+)?(?:hal|hall|howl)\b[\s,.!?]*"
_FRAME_MS = 30  # webrtcvad accepts 10, 20 or 30 ms frames
_END_SILENCE_MS = 450  # a pause this long ends a burst


class _Keyword:
    """Hands-free trigger that matches HAL's name in transcribed speech."""

    def __init__(self) -> None:
        cfg = config.load()
        self.sr = int(cfg.get("audio.sample_rate", 16000))
        self.device = cfg.get("audio.input_device")
        self.vad_level = int(cfg.get("audio.vad_aggressiveness", 2))
        self.model_name = str(cfg.get("activation.keyword.model", "tiny.en"))
        self.max_seconds = float(cfg.get("activation.keyword.max_seconds", 8))
        self.pattern = re.compile(str(cfg.get("activation.keyword.pattern", DEFAULT_KEYWORD)), re.I)
        self.detected = threading.Event()
        self.paused = threading.Event()
        self.command = ""
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import sounddevice as sd
        import webrtcvad
        from faster_whisper import WhisperModel

        model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        vad = webrtcvad.Vad(self.vad_level)
        frame = int(self.sr * _FRAME_MS / 1000)
        end_frames = _END_SILENCE_MS // _FRAME_MS
        max_frames = int(self.max_seconds * 1000 / _FRAME_MS)
        voiced: list[np.ndarray] = []
        silent = 0
        with sd.InputStream(
            samplerate=self.sr, channels=1, dtype="int16", blocksize=frame, device=self.device
        ) as stream:
            log.info("listening for HAL's name (%s)", self.model_name)
            while not self._stop.is_set():
                block, _ = stream.read(frame)
                if self.paused.is_set() or self.detected.is_set():
                    voiced, silent = [], 0
                    continue
                if vad.is_speech(block.tobytes(), self.sr):
                    voiced.append(block[:, 0].copy())
                    silent = 0
                elif voiced:
                    voiced.append(block[:, 0].copy())
                    silent += 1
                if voiced and (silent >= end_frames or len(voiced) >= max_frames):
                    audio = np.concatenate(voiced).astype(np.float32) / 32768.0
                    voiced, silent = [], 0
                    if len(audio) >= self.sr * 0.3:
                        self._check(model, audio)

    def _check(self, model, audio: np.ndarray) -> None:
        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=1,
            initial_prompt="Hey Hal.",
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text for seg in segments).strip()
        match = self.pattern.match(text)
        if match:
            self.command = text[match.end() :].strip(" ,.!?")
            log.info("heard HAL's name: %r", text)
            self.detected.set()

    def take_command(self) -> str:
        """Returns and clears any request spoken in the same breath as the name."""
        command, self.command = self.command, ""
        return command

    def stop(self) -> None:
        self._stop.set()


class Activator:
    """Combines the enabled triggers and exposes a single blocking wait()."""

    def __init__(self, keyword: bool = True) -> None:
        """`keyword=False` leaves name detection to handsfree.Ears, which the voice loop uses."""
        cfg = config.load()
        self.mode = str(cfg.get("activation.mode", "both"))
        self.ptt = None
        self.wake = None
        self.keyword = None
        if self.mode in ("push_to_talk", "both"):
            self.ptt = _PushToTalk(str(cfg.get("activation.push_to_talk_key", "space")))
        if self.mode in ("wake_word", "both") and cfg.get("activation.wake.enabled", True):
            self.wake = _WakeWord()
        if keyword and cfg.get("activation.keyword.enabled", False):
            self.keyword = _Keyword()

    def start(self) -> None:
        if self.ptt:
            self.ptt.start()
        if self.wake:
            self.wake.start()
        if self.keyword:
            self.keyword.start()

    def heard(self) -> bool:
        """True while a hands-free trigger has fired and not been consumed."""
        return bool(
            (self.wake and self.wake.detected.is_set())
            or (self.keyword and self.keyword.detected.is_set())
        )

    def pause_keyword(self, paused: bool) -> None:
        """Stops the keyword listener transcribing while a turn is recorded."""
        if self.keyword:
            (self.keyword.paused.set if paused else self.keyword.paused.clear)()

    def wait_for_activation(self) -> str:
        """Block until a trigger fires; return 'push_to_talk', 'wake_word' or 'keyword'."""
        while True:
            if self.keyword and self.keyword.detected.is_set():
                self.keyword.detected.clear()
                return "keyword"
            if self.ptt and self.ptt.pressed.is_set():
                return "push_to_talk"
            if self.wake and self.wake.detected.is_set():
                self.wake.detected.clear()
                return "wake_word"
            time.sleep(0.02)

    def stop(self) -> None:
        if self.ptt:
            self.ptt.stop()
        if self.wake:
            self.wake.stop()
        if self.keyword:
            self.keyword.stop()
