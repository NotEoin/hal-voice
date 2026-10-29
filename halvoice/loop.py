"""The voice loop: the spine of hal-voice.

    activate (HAL's name, the wake word, or push-to-talk)
        -> transcribe (STT)
            -> the brain streams a reply (halvoice/brains: direct, openai or claude)
                -> each sentence is synthesised while the previous one plays (TTS)
                    -> barge-in: HAL's name while he is speaking stops him
                        -> follow-up: for a few seconds after, talk without the name

The red-eye UI (if enabled) runs in its own thread and reflects the state via StateBus.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time

from . import config, tts
from .brains import BrainUnavailable, TurnTimeout, make_brain
from .stt import Recorder, Transcriber
from .util import StateBus, get_logger
from .wake import Activator

log = get_logger(__name__)

_END = object()


def _accumulate_sentences(delta_iter):
    """Turn a stream of text deltas into complete sentences for measured TTS delivery."""
    buf = ""
    enders = ".!?\n"
    for delta in delta_iter:
        buf += delta
        while True:
            cut = -1
            for i, ch in enumerate(buf):
                if ch in enders and i >= 2:
                    cut = i + 1
                    break
            if cut == -1:
                break
            sentence, buf = buf[:cut].strip(), buf[cut:]
            if sentence:
                yield sentence
    if buf.strip():
        yield buf.strip()


TOO_LONG = "That took too long."
UNAVAILABLE = "My brain is not available."


def _with_timeout_notice(deltas):
    """Passes the reply through; a turn timeout becomes a spoken notice instead of silence."""
    try:
        yield from deltas
    except TurnTimeout as e:
        log.warning("brain: %s", e)
        yield f" {TOO_LONG}"


class VoiceLoop:
    def __init__(self) -> None:
        self.cfg = config.load()
        self.bus = StateBus()
        self.transcriber = Transcriber()
        self.brain = make_brain()
        self.player = tts.Player()
        self._ui_thread = None
        self.ears = None
        self.activator = None
        if self.cfg.get("activation.keyword.enabled", False):
            from .handsfree import Ears

            self.ears = Ears(self.transcriber.transcribe)
        if self.cfg.get("activation.mode", "both") in ("push_to_talk", "both") or not self.ears:
            self.activator = Activator(keyword=False)
            self.recorder = Recorder()
        if hasattr(self.brain, "on_deep_done"):
            self.brain.on_deep_done = self._deep_done
        if self.cfg.get("ui.rgb_follow", False) and getattr(self.brain, "hub", None):
            hub = self.brain.hub
            # The lights show HAL's state through hal-rgb; a slow or missing light never holds up speech.
            self.bus.on_change = lambda state: threading.Thread(
                target=hub.call, args=("rgb_state", {"state": state}), daemon=True
            ).start()
        self._speak_lock = threading.Lock()

    # -- UI --------------------------------------------------------------------
    def _start_ui(self) -> None:
        if not self.cfg.get("ui.enabled", True):
            return
        try:
            from .ui_eye import RedEye
        except Exception as e:  # noqa: BLE001 - the UI is optional; any import failure means headless
            log.warning("red-eye UI unavailable (%s) - running headless", e)
            return

        def run():
            try:
                RedEye(self.bus).run()
            except Exception as e:  # noqa: BLE001 - a UI crash must not stop the voice loop
                log.warning("red-eye UI crashed (%s) - continuing headless", e)

        self._ui_thread = threading.Thread(target=run, daemon=True)
        self._ui_thread.start()

    # -- speaking --------------------------------------------------------------
    def _interrupted(self) -> bool:
        if self.ears and self.ears.interrupt.is_set():
            return True
        return bool(self.activator and self.activator.ptt and self.activator.ptt.pressed.is_set())

    def _watch_for_interrupt(self, stop_flag: threading.Event) -> None:
        while not stop_flag.is_set():
            if self._interrupted():
                self.player.stop()
                return
            time.sleep(0.03)

    def speak_stream(self, sentences) -> bool:
        """Speaks sentences as they arrive, synthesising the next while the current one plays.

        Returns False if HAL was interrupted.
        """
        audio_q: queue.Queue = queue.Queue(maxsize=3)
        cancel = threading.Event()

        def synthesise():
            try:
                for sentence in sentences:
                    if cancel.is_set():
                        break
                    log.info("HAL: %s", sentence)
                    if self.ears:
                        self.ears.spoke(sentence)
                    for seg in tts.synth_stream(sentence):
                        audio_q.put(seg)
            except Exception as e:  # noqa: BLE001 - a failed reply is logged, the loop carries on
                log.error("reply failed: %s", e)
            finally:
                audio_q.put(_END)

        threading.Thread(target=synthesise, name="synth", daemon=True).start()
        stop_flag = threading.Event()
        threading.Thread(target=self._watch_for_interrupt, args=(stop_flag,), daemon=True).start()
        if self.ears:
            self.ears.interrupt.clear()
            self.ears.speaking.set()
        finished = True
        try:
            with self._speak_lock:
                while True:
                    item = audio_q.get()
                    if item is _END:
                        break
                    audio, sr = item
                    self.bus.set("speaking")
                    if not self.player.play(
                        audio, sr, on_level=lambda lv: self.bus.set("speaking", lv)
                    ):
                        log.info("(barge-in: HAL stops)")
                        finished = False
                        cancel.set()
                        break
        finally:
            stop_flag.set()
            if self.ears:
                self.ears.stopped_speaking()
                self.ears.speaking.clear()
        return finished

    def _deep_done(self, text: str) -> None:
        """Speaks the outcome of background work when it finishes."""
        summary = " ".join(text.split()[:60])
        log.info("background task finished")
        self.speak_stream(iter([f"Dave, the task you gave me is finished. {summary}"]))
        self.bus.set("idle")

    # -- one full turn ---------------------------------------------------------
    def _handle_turn(self, text: str) -> None:
        if not text:
            self.bus.set("idle")
            return
        log.info("you: %s", text)
        self.bus.set("thinking")
        started = time.time()
        first = {"t": None}

        def timed(deltas):
            for d in deltas:
                if first["t"] is None:
                    first["t"] = time.time() - started
                yield d

        deltas = _with_timeout_notice(timed(self.brain.ask(text)))
        finished = self.speak_stream(_accumulate_sentences(deltas))
        if not finished:
            # The direct brain has no cancel; its reply simply stops being spoken.
            cancel = getattr(self.brain, "cancel", None)
            if cancel is not None:
                cancel()
        if first["t"] is not None:
            log.info("first words after %.2fs", first["t"])
        self.bus.set("idle")
        if finished and self.ears:
            self.ears.follow_up()

    def _record(self, trigger: str) -> str:
        hold_check = None
        if trigger == "push_to_talk" and self.activator.ptt is not None:
            hold_check = self.activator.ptt.pressed.is_set
            log.info("listening... (hold to talk; release to send)")
        audio = self.recorder.record_utterance(hold_check=hold_check)
        return self.transcriber.transcribe(audio)

    # -- main ------------------------------------------------------------------
    def run(self) -> None:
        self._start_ui()
        self.transcriber._load()
        if hasattr(self.brain, "warm"):
            threading.Thread(target=self.brain.warm, daemon=True).start()
        if self.ears:
            self.ears.start()
        if self.activator:
            self.activator.start()
        self.bus.set("idle")
        log.info("HAL is online.")
        try:
            while True:
                if self.ears:
                    try:
                        heard = self.ears.requests.get(timeout=0.05)
                    except queue.Empty:
                        heard = None
                    if heard is not None:
                        if not heard.text:
                            self.bus.set("listening")
                            continue
                        self._handle_turn(heard.text)
                        continue
                if self.activator and self.activator.ptt and self.activator.ptt.pressed.is_set():
                    self.bus.set("listening")
                    self._handle_turn(self._record("push_to_talk"))
                    continue
                if self.activator and self.activator.wake and self.activator.wake.detected.is_set():
                    self.activator.wake.detected.clear()
                    self.bus.set("listening")
                    self._handle_turn(self._record("wake_word"))
                    continue
                if not self.ears:
                    time.sleep(0.02)
        except KeyboardInterrupt:
            log.info("shutting down...")
        finally:
            if self.ears:
                self.ears.stop()
            if self.activator:
                self.activator.stop()


def _say_unavailable() -> None:
    """Says once that there is no brain, when the voice works; never raises."""
    try:
        tts.speak(UNAVAILABLE)
    except Exception as e:  # noqa: BLE001 - the brain error is the one to report; TTS is best effort
        log.debug("could not say the brain is unavailable: %s", e)


def main() -> None:
    """Runs the voice loop until interrupted; exits 1 when the brain cannot start."""
    try:
        voice = VoiceLoop()
    except BrainUnavailable as e:
        log.error("brain unavailable: %s", e)
        _say_unavailable()
        logging.shutdown()
        sys.exit(1)
    voice.run()
    # CTranslate2's CUDA state crashes the interpreter's own teardown (SIGSEGV), which systemd
    # records as a failure with a core dump; everything worth closing is closed by now.
    logging.shutdown()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
