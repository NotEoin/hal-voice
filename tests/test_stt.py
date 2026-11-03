"""Tests for `halvoice.stt.Transcriber` with a fake Whisper model."""

from types import SimpleNamespace

import numpy as np
import pytest

from halvoice import stt

PROMPT = "Hey Hal. Hal, one moment."


def seg(text, no_speech=0.1, logprob=-0.2):
    return SimpleNamespace(text=text, no_speech_prob=no_speech, avg_logprob=logprob)


@pytest.fixture
def transcriber(monkeypatch):
    """A `Transcriber` whose Whisper model returns `fake.segments`."""
    fake = SimpleNamespace(segments=[])
    model = SimpleNamespace(transcribe=lambda audio, **kw: (iter(fake.segments), None))
    monkeypatch.setattr(stt.Transcriber, "_model", model)
    t = object.__new__(stt.Transcriber)
    t.beam_size, t.language, t.hotwords, t.prompt = 1, "en", "Hal", PROMPT
    t.highpass_hz, t.level_dbfs = 0.0, None
    t.fake = fake
    return t


@pytest.mark.parametrize(
    "echo", ["Hey Hal.", "Hal, one moment.", "Hey Hal. Hal, one moment.", "hal one moment"]
)
def test_an_echo_of_the_initial_prompt_is_not_heard_as_speech(transcriber, echo):
    """Regression: noise came back as the prompt "Hal, read me the headlines." and Hal acted on it."""
    transcriber.fake.segments = [seg(echo)]
    assert transcriber.transcribe(np.ones(160, np.float32)) == ""


def test_real_speech_is_kept_and_an_appended_prompt_echo_dropped(transcriber):
    transcriber.fake.segments = [seg("Hal, read me the headlines."), seg("Hal, one moment.")]
    assert transcriber.transcribe(np.ones(160, np.float32)) == "Hal, read me the headlines."


def test_the_shipped_prompt_holds_no_request():
    """Every sentence of the default prompt must be safe to drop: an echo of it is discarded."""
    from halvoice import config

    prompt = config.load().get("stt.initial_prompt")
    assert stt.prompt_sentences(prompt) == {"hey hal", "hal one moment"}
