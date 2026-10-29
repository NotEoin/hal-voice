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
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
from collections.abc import Iterator

from .. import config, connect
from ..util import get_logger

log = get_logger(__name__)


class ClaudeSession:
    """One persistent voice conversation with Claude Code (local model)."""

    def __init__(self, cwd: str | None = None) -> None:
        cfg = config.load()
        self.c = cfg.claude
        # Which Claude Code, and where -- see halvoice/connect.py for the full order.
        self.conn = connect.resolve()
        self.cwd = cwd or self.conn.project_dir
        self.session_id: str | None = None
        self._env = self._build_env()
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
            str(self.c.get("command", "claude")),
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
        )

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
            with contextlib.suppress(OSError, ValueError):  # the pipe may already be closed
                proc.stdout.close()  # type: ignore[union-attr]  # a pipe, because Popen got stdout=PIPE
            proc.wait(timeout=10)


if __name__ == "__main__":  # quick manual test: python -m halvoice.brains.claude "hello"
    import sys

    s = ClaudeSession()
    q = " ".join(sys.argv[1:]) or "Say hello in one short sentence."
    for piece in s.ask(q):
        print(piece, end="", flush=True)
    print(f"\n[session={s.session_id}]")
