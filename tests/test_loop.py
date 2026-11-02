"""Tests for halvoice.loop: what the loop does around the brain."""

import pytest

from halvoice import loop
from halvoice.brains import BrainUnavailable, TurnTimeout
from halvoice.gpu import VoiceClaims


def test_turn_timeout_is_spoken_instead_of_silence():
    def reply():
        yield "Let me check."
        raise TurnTimeout("no complete reply within 300 s")

    assert list(loop._accumulate_sentences(loop._with_timeout_notice(reply()))) == [
        "Let me check.",
        loop.TOO_LONG,
    ]


def test_unavailable_brain_exits_1_after_saying_so(monkeypatch):
    def no_brain():
        raise BrainUnavailable("brain.mode is openai but brain.base_url is empty")

    said = []
    monkeypatch.setattr(loop, "VoiceLoop", no_brain)
    monkeypatch.setattr(loop.tts, "speak", said.append)
    with pytest.raises(SystemExit) as stop:
        loop.main()
    assert stop.value.code == 1
    assert said == [loop.UNAVAILABLE]


def test_unavailable_brain_still_exits_1_when_the_voice_is_down(monkeypatch):
    def no_brain():
        raise BrainUnavailable("x")

    def voice_down(text):
        raise ConnectionError("TTS server down")

    monkeypatch.setattr(loop, "VoiceLoop", no_brain)
    monkeypatch.setattr(loop.tts, "speak", voice_down)
    with pytest.raises(SystemExit) as stop:
        loop.main()
    assert stop.value.code == 1


class FakeBrain:
    def __init__(self):
        self.cancelled = False

    def ask(self, text):
        yield "I am putting myself to the fullest possible use."

    def cancel(self):
        self.cancelled = True


def bare_loop(brain, finished):
    voice = loop.VoiceLoop.__new__(loop.VoiceLoop)
    voice.brain = brain
    voice.ears = None
    voice.cfg = {}
    voice.claims = VoiceClaims("off", 0, 0, 0)
    voice.bus = loop.StateBus()
    voice.speak_stream = lambda sentences: (list(sentences), finished)[1]
    return voice


def test_barge_in_cancels_the_brain():
    brain = FakeBrain()
    bare_loop(brain, finished=False)._handle_turn("Hal, tell me a story")
    assert brain.cancelled


def test_finished_reply_does_not_cancel():
    brain = FakeBrain()
    bare_loop(brain, finished=True)._handle_turn("Hal, hello")
    assert not brain.cancelled
