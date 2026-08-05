"""Activation: decide when a turn begins.

Two triggers, either/both per config.activation.mode:
  * wake_word   — openWakeWord listens continuously for "Hello, HAL" (ONNX, CPU, cheap).
  * push_to_talk — hold a key (default: space) to start a turn.

`wait_for_activation()` blocks until either fires, then returns the trigger name.
Both run on background listeners so they race naturally.
"""

from __future__ import annotations

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
        except Exception:  # noqa: BLE001
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
        except Exception as e:  # noqa: BLE001
            log.error("wake word model failed to load (%s) — disabling wake word.", e)
            return
        # openWakeWord expects 16 kHz int16 frames of 1280 samples (80 ms).
        chunk = 1280
        with sd.InputStream(samplerate=self.sr, channels=1, dtype="int16",
                            blocksize=chunk, device=self.device) as stream:
            log.info("wake word listening ('%s', threshold %.2f)…",
                     self.custom or self.model_name, self.threshold)
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


class Activator:
    """Combines the enabled triggers and exposes a single blocking wait()."""

    def __init__(self) -> None:
        cfg = config.load()
        self.mode = str(cfg.get("activation.mode", "both"))
        self.ptt = None
        self.wake = None
        if self.mode in ("push_to_talk", "both"):
            self.ptt = _PushToTalk(str(cfg.get("activation.push_to_talk_key", "space")))
        if self.mode in ("wake_word", "both") and cfg.get("activation.wake.enabled", True):
            self.wake = _WakeWord()

    def start(self) -> None:
        if self.ptt:
            self.ptt.start()
        if self.wake:
            self.wake.start()

    def wait_for_activation(self) -> str:
        """Block until a trigger fires; return 'push_to_talk' or 'wake_word'."""
        while True:
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
