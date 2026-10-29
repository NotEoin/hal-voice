"""The brains that answer a spoken turn, one module per `brain.mode`.

A brain takes the transcript of one turn and streams the reply text back; the loop turns
the stream into sentences for the voice. Modes:

- `direct`: core Hal's brain (`halcore.brain.Brain`, see `halvoice/brain.py`);
- `openai`: any OpenAI-compatible server at `brain.base_url`, for use without Hal
  (`halvoice/brains/openai.py`);
- `claude`: Claude Code headless on the host (`halvoice/brains/claude.py`).

`make_brain` raises `BrainUnavailable` when the configured brain cannot start, with a
message that names the mode and the fix. The `direct` brain predates this interface: it
has no `cancel` or `close`, so the loop calls those only when they exist.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from .. import config
from ..util import get_logger

log = get_logger(__name__)

MODES = ("direct", "openai", "claude")


class BrainError(RuntimeError):
    """A turn failed: the brain answered with an error or could not be reached."""


class BrainUnavailable(BrainError):
    """The configured brain cannot start; the message names the mode and the fix."""


class TurnTimeout(BrainError):
    """The reply did not finish within `brain.turn_timeout_s`."""


class Brain(Protocol):
    """What the voice loop needs from a brain."""

    def ask(self, text: str) -> Iterator[str]:
        """Streams the reply to one spoken turn as text deltas.

        Raises `TurnTimeout` when the turn runs past `brain.turn_timeout_s`, and
        `BrainError` when the brain fails mid-turn.
        """
        ...

    def cancel(self) -> None:
        """Stops the turn in progress (barge-in); `ask` then ends without raising."""
        ...

    def close(self) -> None:
        """Releases the brain's connections and processes."""
        ...


def make_brain(cfg: config.Cfg | None = None) -> Brain:
    """Returns the brain `brain.mode` names; raises `BrainUnavailable` when it cannot start."""
    cfg = cfg if cfg is not None else config.load()
    mode = str(cfg.get("brain.mode", "direct"))
    if mode == "openai":
        from .openai import OpenAIBrain

        return OpenAIBrain(cfg)
    if mode == "claude":
        from .claude import ClaudeSession

        return ClaudeSession()
    if mode == "direct":
        from ..brain import CoreNotFound, DirectBrain

        try:
            return DirectBrain()
        except CoreNotFound as e:
            log.error("%s. Falling back to claude mode.", e)
        from .claude import ClaudeSession

        return ClaudeSession()
    raise BrainUnavailable(f"brain.mode '{mode}' is not one of: {', '.join(MODES)}")
