"""The voice loop — the spine of hal-voice.

    activate (wake word / push-to-talk)
        -> listen + transcribe (STT)
            -> send to Claude Code, stream the reply
                -> speak each sentence in HAL's voice as it's ready (TTS)
                    -> barge-in: a new activation while HAL is speaking interrupts him

The red-eye UI (if enabled) runs in its own thread and reflects the state via StateBus.
"""

from __future__ import annotations

import threading
import time

from . import config, tts
from .claude_client import ClaudeSession
from .stt import Recorder, Transcriber
from .util import StateBus, get_logger
from .wake import Activator

log = get_logger(__name__)


def _accumulate_sentences(delta_iter):
    """Turn a stream of text deltas into complete sentences for measured TTS delivery."""
    buf = ""
    enders = ".!?\n"
    for delta in delta_iter:
        buf += delta
        # Flush whenever we have at least one sentence-ending punctuation with text after it.
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


class VoiceLoop:
    def __init__(self) -> None:
        self.cfg = config.load()
        self.bus = StateBus()
        self.activator = Activator()
        self.recorder = Recorder()
        self.transcriber = Transcriber()
        self.session = ClaudeSession()
        self.player = tts.Player()
        self._ui_thread = None

    # -- UI --------------------------------------------------------------------
    def _start_ui(self) -> None:
        if not self.cfg.get("ui.enabled", True):
            return
        try:
            from .ui_eye import RedEye
        except Exception as e:  # noqa: BLE001
            log.warning("red-eye UI unavailable (%s) — running headless", e)
            return

        def run():
            try:
                RedEye(self.bus).run()
            except Exception as e:  # noqa: BLE001
                log.warning("red-eye UI crashed (%s) — continuing headless", e)

        self._ui_thread = threading.Thread(target=run, daemon=True)
        self._ui_thread.start()

    # -- barge-in monitor ------------------------------------------------------
    def _watch_for_interrupt(self, stop_flag: threading.Event) -> None:
        """While HAL speaks, a fresh activation stops playback (barge-in)."""
        while not stop_flag.is_set():
            if self.activator.ptt and self.activator.ptt.pressed.is_set():
                self.player.stop()
                return
            if self.activator.heard():
                self.player.stop()
                return
            time.sleep(0.03)

    # -- one full turn ---------------------------------------------------------
    def _handle_turn(self, trigger: str = "push_to_talk") -> None:
        self.bus.set("listening")
        text = self.activator.keyword.take_command() if trigger == "keyword" else ""
        if not text:
            text = self._record(trigger)
        if not text:
            log.info("(heard nothing)")
            self.bus.set("idle")
            return
        log.info("you: %s", text)

        self.bus.set("thinking")
        sentences = _accumulate_sentences(self.session.ask(text))

        # Speak sentences as they complete; allow barge-in throughout.
        stop_flag = threading.Event()
        watcher = threading.Thread(target=self._watch_for_interrupt, args=(stop_flag,), daemon=True)
        watcher.start()
        spoke_any = False
        try:
            for sentence in sentences:
                log.info("HAL: %s", sentence)
                self.bus.set("speaking")
                for audio_seg, sr in tts.synth_stream(sentence):
                    ok = self.player.play(audio_seg, sr,
                                          on_level=lambda lv: self.bus.set("speaking", lv))
                    spoke_any = True
                    if not ok:  # interrupted
                        log.info("(barge-in — HAL stops)")
                        return
        finally:
            stop_flag.set()
        if not spoke_any:
            log.info("(no reply)")
        self.bus.set("idle")

    def _record(self, trigger: str) -> str:
        # Hold-to-talk records only while the key stays held; release ends the turn.
        hold_check = None
        if trigger == "push_to_talk" and self.activator.ptt is not None:
            hold_check = self.activator.ptt.pressed.is_set
            log.info("listening… (hold to talk; release to send)")
        else:
            log.info("listening…")
        self.activator.pause_keyword(True)
        try:
            audio = self.recorder.record_utterance(hold_check=hold_check)
            return self.transcriber.transcribe(audio)
        finally:
            self.activator.pause_keyword(False)

    # -- main ------------------------------------------------------------------
    def run(self) -> None:
        self._start_ui()
        self.activator.start()
        self.bus.set("idle")
        mode = self.cfg.get("activation.mode", "both")
        log.info("HAL is online. Activation: %s. Say the wake word or hold the talk key.", mode)
        try:
            while True:
                trigger = self.activator.wait_for_activation()
                log.debug("activated via %s", trigger)
                self._handle_turn(trigger)
        except KeyboardInterrupt:
            log.info("shutting down…")
        finally:
            self.activator.stop()


def main() -> None:
    VoiceLoop().run()


if __name__ == "__main__":
    main()
