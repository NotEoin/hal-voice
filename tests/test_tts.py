"""Tests for halvoice.tts: the weight swap that gives the server HAL's voice."""

import requests

from halvoice import tts


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


def test_non_200_swap_reply_is_not_recorded_as_done(monkeypatch):
    client, _calls = make_client(monkeypatch, [200, 400])
    client.ensure_weights()
    assert client._weights_set is False


def test_successful_weight_swap_is_sent_once(monkeypatch):
    client, calls = make_client(monkeypatch, [200, 200])
    client.ensure_weights()
    client.ensure_weights()
    assert calls == ["set_gpt_weights", "set_sovits_weights"]
