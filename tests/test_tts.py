"""Tests for halvoice.tts: the weight swap that gives the server HAL's voice."""

import pytest
import requests

from halvoice import config, tts

BUILT = """\
tts:
  gpt_weights: gptsovits/weights/hal-e12.ckpt
  sovits_weights: gptsovits/weights/hal_e12_s288.pth
  ref_audio_path: voice-data/reference/hal_film_reference.wav
  prompt_text: I'm sorry, Dave.
  speed: 0.92
"""


@pytest.fixture
def built_voice(isolated_config):
    """A voice installed by the build, as `built.yaml`."""
    isolated_config.mkdir(parents=True, exist_ok=True)
    (isolated_config / "built.yaml").write_text(BUILT)


class FakeReply:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def make_client(monkeypatch, statuses):
    """A client whose GETs answer from `statuses`, one per call; returns (client, calls)."""
    calls = []
    answers = iter(statuses)

    def fake_get(url, params=None, timeout=None):
        calls.append(url.rsplit("/", 1)[1])
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return FakeReply(answer)

    monkeypatch.setattr(requests, "get", fake_get)
    client = tts.GPTSoVITSClient()
    return client, calls


@pytest.mark.usefixtures("built_voice")
def test_failed_weight_swap_is_retried_next_call(monkeypatch):
    """A swap that failed once must not be recorded as done (it used to be)."""
    client, calls = make_client(
        monkeypatch, [requests.ConnectionError("server starting"), 200, 200]
    )
    client.ensure_weights()
    assert client._weights_set is False
    client.ensure_weights()
    assert client._weights_set is True
    assert calls == ["set_gpt_weights", "set_gpt_weights", "set_sovits_weights"]


@pytest.mark.usefixtures("built_voice")
def test_non_200_swap_reply_is_not_recorded_as_done(monkeypatch):
    client, _calls = make_client(monkeypatch, [200, 400])
    client.ensure_weights()
    assert client._weights_set is False


@pytest.mark.usefixtures("built_voice")
def test_successful_weight_swap_is_sent_once(monkeypatch):
    client, calls = make_client(monkeypatch, [200, 200])
    client.ensure_weights()
    client.ensure_weights()
    assert calls == ["set_gpt_weights", "set_sovits_weights"]


def test_no_voice_built_fails_with_the_fix(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("must not reach the server"))
    with pytest.raises(RuntimeError, match="run make-hal-voice"):
        tts.GPTSoVITSClient().synth_raw("hello")


def test_empty_sampling_settings_leave_server_defaults(isolated_config, monkeypatch):
    isolated_config.mkdir(parents=True, exist_ok=True)
    (isolated_config / "config.yaml").write_text("tts:\n  ref_audio_path: /voices/ref.wav\n")
    sent = {}

    class Stop(Exception):
        pass

    def fake_post(url, json=None, timeout=None):
        sent.update(json)
        raise Stop

    monkeypatch.setattr(requests, "post", fake_post)
    client = tts.GPTSoVITSClient()
    with pytest.raises(Stop):
        client.synth_raw("hello")
    assert sent["ref_audio_path"] == "/voices/ref.wav"
    for key in ("top_k", "top_p", "temperature", "speed_factor"):
        assert key not in sent
    assert config.load().get("tts.speed") == ""


def test_whole_text_synth_sends_the_weights_once_per_process(monkeypatch):
    # regression: each `synth` call built a new client and swapped both checkpoints again
    import numpy as np

    made = []

    class Client:
        def __init__(self) -> None:
            made.append(self)

        def synth_raw(self, text):
            return np.zeros(320, dtype=np.float32), 32000

    monkeypatch.setattr(tts, "GPTSoVITSClient", Client)
    monkeypatch.setattr(tts, "_client", None)
    monkeypatch.setattr(tts.hal_fx, "apply", lambda audio, sr: audio)
    tts.synth("One.")
    tts.synth("Two.")
    assert len(made) == 1
