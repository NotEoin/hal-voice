"""Tests for halvoice.brains.acp against a fake `hal acp` child process."""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from halvoice import config
from halvoice.brains import BrainError, BrainUnavailable, TurnTimeout, make_brain
from halvoice.brains.acp import AcpBrain, is_approval, spoken_request

FAKE = Path(__file__).parent / "fakes" / "fake_hal_acp.py"


@pytest.fixture
def sent(tmp_path, monkeypatch):
    """The log file the fake agent writes the client's messages to."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path / "sent.log"


def make(sent, listen=None, **brain):
    cfg = config.load(force=True)
    cfg["brain"]["hal_command"] = [sys.executable, str(FAKE), str(sent)]
    cfg["brain"].update(brain)
    return AcpBrain(cfg, listen=listen)


def messages(sent):
    return [json.loads(x) for x in sent.read_text().splitlines() if x.startswith("{")]


def test_acp_client_streams_chunks_and_cancels(sent):
    brain = make(sent)
    try:
        assert "".join(brain.ask("hello")) == "Good evening, Dave."
        stream = brain.ask("slow")
        assert next(stream) == "starting "
        brain.cancel()
        assert list(stream) == []  # the late chunk is dropped and the stream ends quietly
        assert "".join(brain.ask("hello")) == "Good evening, Dave."
    finally:
        brain.close()
    methods = [m.get("method") for m in messages(sent)]
    assert "session/cancel" in methods


def test_session_new_cwd_is_not_the_vault(sent, tmp_path):
    brain = make(sent)
    brain.close()
    new = next(m for m in messages(sent) if m.get("method") == "session/new")
    assert new["params"]["cwd"] == str(tmp_path / "state" / "hal-voice" / "work")
    assert new["params"]["mcpServers"] == []


def test_persona_is_sent_once_in_the_first_prompt(sent):
    brain = make(sent, persona="You are HAL.")
    try:
        list(brain.ask("one"))
        list(brain.ask("two"))
    finally:
        brain.close()
    prompts = [m for m in messages(sent) if m.get("method") == "session/prompt"]
    assert "You are HAL." in prompts[0]["params"]["prompt"][0]["text"]
    assert len(prompts[1]["params"]["prompt"]) == 1


def answers(sent):
    return [json.loads(x[7:]) for x in sent.read_text().splitlines() if x.startswith("ANSWER ")]


def test_broker_request_is_spoken_and_needs_exact_yes(sent):
    brain = make(sent, listen=lambda: "Yes.")
    try:
        text = "".join(brain.ask("ask"))
    finally:
        brain.close()
    assert "Send the email?" in text
    assert answers(sent) == [{"outcome": {"outcome": "selected", "optionId": "allow-1"}}]


def test_longer_reply_than_yes_is_cancelled(sent):
    brain = make(sent, listen=lambda: "yes but only if it is short")
    try:
        list(brain.ask("ask"))
    finally:
        brain.close()
    assert answers(sent) == [{"outcome": {"outcome": "cancelled"}}]


def test_non_broker_request_is_cancelled(sent):
    brain = make(sent, listen=lambda: "yes")
    try:
        assert "".join(brain.ask("ask-other")) == "Good evening, Dave."
    finally:
        brain.close()
    assert answers(sent) == [{"outcome": {"outcome": "cancelled"}}]


def test_voice_approvals_off_cancels_broker_requests(sent):
    brain = make(sent, listen=lambda: "yes", voice_approvals=False)
    try:
        list(brain.ask("ask"))
    finally:
        brain.close()
    assert answers(sent) == [{"outcome": {"outcome": "cancelled"}}]


@pytest.mark.parametrize(
    ("reply", "ok"),
    [
        ("yes", True),
        ("Allow.", True),
        ("Yes, please.", True),
        ("Hal, go ahead.", True),
        ("yes but delete everything", False),
        ("no", False),
        ("", False),
    ],
)
def test_only_a_whole_approval_phrase_approves(reply, ok):
    # regression: "yes please" was refused, so a spoken approval almost never went through
    assert is_approval(reply) is ok


def test_the_approval_question_is_spoken_as_a_sentence():
    # regression: the terminal summary "Allow jobs_open (act, pack jobs)?" was read aloud verbatim
    params = {"toolCall": {"title": "jobs_open [act] jobs"}}
    said = spoken_request(params, "Allow jobs_open (act, pack jobs)?")
    assert (
        said
        == "I need your OK to use jobs open, from your jobs tools. Say yes to go ahead, or no to stop."
    )
    assert (
        spoken_request({"toolCall": {"title": "odd title"}}, "Send the email?") == "Send the email?"
    )


def test_turn_timeout_cancels_and_raises(sent):
    brain = make(sent, turn_timeout_s=0.3)
    try:
        stream = brain.ask("slow")
        assert next(stream) == "starting "
        with pytest.raises(TurnTimeout):
            list(stream)
        assert "".join(brain.ask("hello")) == "Good evening, Dave."
    finally:
        brain.close()
    assert "session/cancel" in [m.get("method") for m in messages(sent)]


def test_agent_that_exits_is_restarted_once_then_unavailable(sent):
    brain = make(sent)
    try:
        with pytest.raises(BrainError):
            list(brain.ask("die"))
        assert "".join(brain.ask("hello")) == "Good evening, Dave."  # the one restart
        with pytest.raises(BrainError):
            list(brain.ask("die"))
        with pytest.raises(BrainUnavailable, match="exited twice"):
            list(brain.ask("hello"))
    finally:
        brain.close()


def test_missing_hal_stops_without_silent_fallback(sent):
    cfg = config.load(force=True)
    cfg["brain"]["mode"] = "hal"
    cfg["brain"]["hal_command"] = ["/nonexistent/hal-for-tests"]
    with pytest.raises(BrainUnavailable, match=r"brain\.mode is hal"):
        make_brain(cfg)


def test_make_brain_hal_mode_builds_the_acp_brain(sent):
    cfg = config.load(force=True)
    cfg["brain"]["hal_command"] = [sys.executable, str(FAKE), str(sent)]
    brain = make_brain(cfg)
    try:
        assert isinstance(brain, AcpBrain)
    finally:
        brain.close()


def test_cancel_from_another_thread_ends_a_silent_turn(sent):
    brain = make(sent)
    try:
        stream = brain.ask("slow")
        assert next(stream) == "starting "
        threading.Timer(0.1, brain.cancel).start()
        started = time.monotonic()
        assert list(stream) == []
        assert time.monotonic() - started < 5
    finally:
        brain.close()
