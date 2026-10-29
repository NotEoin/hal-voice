"""Tests for halvoice.brains.claude with a fake `claude` script in place of Claude Code."""

import logging
import sys
import threading
import time
from pathlib import Path

import pytest

from halvoice import connect
from halvoice.brains import BrainUnavailable, TurnTimeout
from halvoice.brains.claude import ClaudeSession

GONE_WAIT_S = 5  # how long a killed child may take to disappear from the process table

# Streams one delta, starts a child that would outlive it, records the child's pid, hangs.
FAKE_CLAUDE = """\
import json, subprocess, sys, time
print(json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}), flush=True)
event = {"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": "Just a moment."}}}
print(json.dumps(event), flush=True)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
open(sys.argv[0] + ".child", "w").write(str(child.pid))
time.sleep(60)
"""


@pytest.fixture
def fake_claude(tmp_path, isolated_config, monkeypatch):
    """Writes a fake `claude`, points `claude.command` at it, returns its path."""
    script = tmp_path / "claude"
    script.write_text(f"#!{sys.executable}\n{FAKE_CLAUDE}")
    script.chmod(0o755)
    isolated_config.mkdir(parents=True, exist_ok=True)
    (isolated_config / "config.yaml").write_text(
        f"claude:\n  command: {script}\nbrain:\n  turn_timeout_s: 1\n"
    )
    monkeypatch.setattr(connect, "resolve", lambda: connect.Connection(project_dir=str(tmp_path)))
    return script


def gone(pid: int) -> bool:
    """True once `pid` has exited (a zombie waiting for its reaper counts as exited)."""
    deadline = time.monotonic() + GONE_WAIT_S
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if state == "Z":
            return True
        time.sleep(0.05)
    return False


def test_claude_session_is_killed_after_turn_timeout(fake_claude):
    session = ClaudeSession()
    got = []
    started = time.monotonic()
    with pytest.raises(TurnTimeout):
        for delta in session.ask("Hal, count to a million"):
            got.append(delta)
    assert got == ["Just a moment."]
    assert time.monotonic() - started < 5
    child = int(Path(f"{fake_claude}.child").read_text())
    assert gone(child), "the claude process group, child included, must be killed"


def test_cancel_kills_the_turn_and_ends_quietly(fake_claude, isolated_config):
    (isolated_config / "config.yaml").write_text(
        f"claude:\n  command: {fake_claude}\nbrain:\n  turn_timeout_s: 30\n"
    )
    session = ClaudeSession()
    replies = session.ask("Hal, tell me a story")
    assert next(replies) == "Just a moment."
    threading.Timer(0.3, session.cancel).start()
    started = time.monotonic()
    assert list(replies) == []
    assert time.monotonic() - started < 5


def test_claude_permission_mode_defaults_to_default(fake_claude):
    assert "--permission-mode" in (cmd := ClaudeSession()._base_cmd())
    assert cmd[cmd.index("--permission-mode") + 1] == "default"


def test_claude_mode_warns_that_it_runs_outside_the_sandbox(fake_claude, caplog):
    with caplog.at_level(logging.WARNING, logger="halvoice.brains.claude"):
        ClaudeSession()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "outside Hal's sandbox" in warnings[0].getMessage()


def test_missing_claude_command_is_brain_unavailable(isolated_config, tmp_path):
    isolated_config.mkdir(parents=True, exist_ok=True)
    (isolated_config / "config.yaml").write_text(
        f"claude:\n  command: {tmp_path / 'no-such-claude'}\n"
    )
    with pytest.raises(BrainUnavailable, match=r"claude\.command"):
        ClaudeSession()
