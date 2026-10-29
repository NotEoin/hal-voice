"""Drive Claude Code headless and yield the assistant's text as it streams.

We use `claude -p <prompt> --output-format stream-json --include-partial-messages`,
which emits newline-delimited JSON events. We keep the conversation going across turns
with `--resume <session_id>` (the session id is reported in the init event), so HAL has
memory of the whole spoken conversation.

Why headless instead of the interactive TUI: it gives us the assistant's text
*deterministically* and *incrementally*, so HAL can start speaking sentence 1 while the
model is still generating sentence 2.

In claude mode Claude Code is the brain (direct mode uses core Hal's; see ../brain.py). Which
Claude Code, in which directory, and against which model is decided by halvoice/connect.py;
by default that is "whatever Claude Code you already have, in the current directory".

This module yields *text deltas*; the loop reassembles them into sentences for TTS.

Bounds: each turn's `claude -p` runs in its own process group, killed whole when the turn
passes `brain.turn_timeout_s` or on barge-in (`cancel`), so a hung tool call cannot keep
HAL silent or leave children behind. This mode runs Claude Code on the host, outside Hal's
sandbox, so `claude.permission_mode` defaults to `default`: headless Claude Code cannot ask,
so anything that needs a permission is refused rather than auto-accepted.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import threading
from collections.abc import Iterator

from .. import config, connect
from ..util import get_logger
from . import BrainUnavailable, TurnTimeout

log = get_logger(__name__)

DEFAULT_TURN_TIMEOUT_S = 300.0
EXIT_WAIT_S = 10  # Claude Code writes its session file after the result event, then exits


class ClaudeSession:
    """One persistent voice conversation with Claude Code (local model)."""

    def __init__(self, cwd: str | None = None) -> None:
        cfg = config.load()
        self.c = cfg.claude
        command = str(self.c.get("command") or "claude")
        self.command = shutil.which(command)
        if self.command is None:
            raise BrainUnavailable(
                f"brain.mode is claude but claude.command {command!r} is not an executable "
                "path or on PATH: install Claude Code or set claude.command to its full path"
            )
        self.turn_timeout = float(cfg.get("brain.turn_timeout_s") or DEFAULT_TURN_TIMEOUT_S)
        # Which Claude Code, and where -- see halvoice/connect.py for the full order.
        self.conn = connect.resolve()
        self.cwd = cwd or self.conn.project_dir
        self.session_id: str | None = None
        self._env = self._build_env()
        self._proc: subprocess.Popen[str] | None = None
        log.warning(
            "brain.mode claude runs Claude Code on the host, outside Hal's sandbox, in %s "
            "(permission mode %s)",
            self.cwd,
            self.c.get("permission_mode") or "Claude Code's own",
        )
        log.info("Claude Code: %s", self.conn.describe())

    def _build_env(self) -> dict:
        env = dict(os.environ)
        env.update(self.conn.env)
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.c.get("max_output_tokens", 8192))
        if self.conn.is_cloud:
            # Cloud: leave Claude Code's own configuration completely alone. Suppressing
            # its traffic here would also disable things a hosted setup relies on.
            return env
        # Local model: nothing should phone home.
        env["HAL_MODEL"] = self.conn.model
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["DISABLE_TELEMETRY"] = "1"
        env["DISABLE_ERROR_REPORTING"] = "1"
        env["DISABLE_AUTOUPDATER"] = "1"
        return env

    def _base_cmd(self) -> list[str]:
        cmd = [
            str(self.command),
            "-p",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
        ]  # required for stream-json on -p
        pm = self.c.get("permission_mode")
        if pm:
            cmd += ["--permission-mode", str(pm)]
        allowed = self.c.get("allowed_tools") or []
        if allowed:
            cmd += ["--allowedTools", ",".join(map(str, allowed))]
        for d in self.c.get("add_dirs") or []:
            cmd += ["--add-dir", str(d)]
        sysprompt = self.c.get("voice_system_prompt")
        if sysprompt:
            cmd += ["--append-system-prompt", str(sysprompt).strip()]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        return cmd

    def ask(self, prompt: str) -> Iterator[str]:
        """Send one user turn; yield assistant *text deltas* as they arrive."""
        cmd = [*self._base_cmd(), prompt]
        log.debug("claude cmd: %s", " ".join(cmd[:-1]) + " <prompt>")
        proc = subprocess.Popen(
            cmd,
            cwd=self.cwd,
            env=self._env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,  # its own process group, so a kill takes its tools too
        )
        self._proc = proc
        timed_out = threading.Event()

        def expire() -> None:
            timed_out.set()
            _kill_group(proc)

        timer = threading.Timer(self.turn_timeout, expire)
        timer.daemon = True
        timer.start()

        # Drain stderr in the background so it can't deadlock the pipe.
        def _drain_err():
            for line in proc.stderr:  # type: ignore[union-attr]  # a pipe, because Popen got stderr=PIPE
                log.debug("claude stderr: %s", line.rstrip())

        threading.Thread(target=_drain_err, daemon=True).start()

        emitted = 0  # chars of the final assistant message already yielded
        final_text = ""
        try:
            for line in proc.stdout:  # type: ignore[union-attr]  # a pipe, because Popen got stdout=PIPE
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = evt.get("type")

                if etype == "system" and evt.get("subtype") == "init":
                    self.session_id = evt.get("session_id") or self.session_id

                # Incremental deltas (present with --include-partial-messages).
                elif etype == "stream_event":
                    delta = (evt.get("event", {}) or {}).get("delta", {}) or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        final_text += delta["text"]
                        new = final_text[emitted:]
                        if new:
                            emitted = len(final_text)
                            yield new

                # Final assembled assistant message -- yield any tail we didn't stream.
                elif etype == "assistant":
                    msg = evt.get("message", {}) or {}
                    text = "".join(
                        b.get("text", "") for b in msg.get("content", []) if b.get("type") == "text"
                    )
                    if text and len(text) > emitted:
                        yield text[emitted:]
                        emitted = len(text)
                        final_text = text

                elif etype == "result":
                    if evt.get("session_id"):
                        self.session_id = evt["session_id"]
                    break
        finally:
            timer.cancel()
            with contextlib.suppress(OSError, ValueError):  # the pipe may already be closed
                proc.stdout.close()  # type: ignore[union-attr]  # a pipe, because Popen got stdout=PIPE
            try:
                proc.wait(timeout=EXIT_WAIT_S)
            except subprocess.TimeoutExpired:
                log.warning("claude did not exit %d s after its reply; killing it", EXIT_WAIT_S)
                _kill_group(proc)
                proc.wait(timeout=EXIT_WAIT_S)
            self._proc = None
        if timed_out.is_set():
            raise TurnTimeout(f"claude -p was killed after {self.turn_timeout:g} s")

    def cancel(self) -> None:
        """Ends the turn in progress (barge-in) by killing its process group."""
        proc = self._proc
        if proc is not None:
            _kill_group(proc)

    def close(self) -> None:
        """Ends any turn in progress; nothing else is held between turns."""
        self.cancel()


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILLs the process group `proc` leads; a group that has already gone is fine."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)


if __name__ == "__main__":  # quick manual test: python -m halvoice.brains.claude "hello"
    import sys

    s = ClaudeSession()
    q = " ".join(sys.argv[1:]) or "Say hello in one short sentence."
    for piece in s.ask(q):
        print(piece, end="", flush=True)
    print(f"\n[session={s.session_id}]")
