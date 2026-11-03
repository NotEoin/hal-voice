"""The brains that answer a spoken turn, one module per `brain.mode`.

A brain takes the transcript of one turn and streams the reply text back; the loop turns
the stream into sentences for the voice. Modes:

- `hal`: Hal's boxed assistant agent over ACP (`halvoice/brains/acp.py`);
- `openai`: any OpenAI-compatible server at `brain.base_url`, for use without Hal
  (`halvoice/brains/openai.py`);
- `claude`: Claude Code headless on the host (`halvoice/brains/claude.py`).

`make_brain` raises `BrainUnavailable` when the configured brain cannot start, with a
message that names the mode and the fix. The loop calls `cancel` and `close` only when a
brain has them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from .. import config
from ..util import get_logger

log = get_logger(__name__)

MODES = ("hal", "openai", "claude")


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


DEFAULT_USER = "Dave"  # the film's name for the person HAL talks to


def persona(cfg: config.Cfg) -> str:
    """`brain.persona` with each `{user}` replaced by `brain.user_name` (Dave when unset)."""
    name = str(cfg.get("brain.user_name") or "").strip() or DEFAULT_USER
    return str(cfg.get("brain.persona") or "").strip().replace("{user}", name)


def make_brain(cfg: config.Cfg | None = None) -> Brain:
    """Returns the brain `brain.mode` names; raises `BrainUnavailable` when it cannot start."""
    cfg = cfg if cfg is not None else config.load()
    mode = str(cfg.get("brain.mode", "hal"))
    if mode == "hal":
        from .acp import AcpBrain

        return AcpBrain(cfg)
    if mode == "openai":
        from .openai import OpenAIBrain

        return OpenAIBrain(cfg)
    if mode == "claude":
        from .claude import ClaudeSession

        return ClaudeSession()
    hint = " (the direct brain was removed: use hal)" if mode == "direct" else ""
    raise BrainUnavailable(f"brain.mode '{mode}' is not one of: {', '.join(MODES)}{hint}")
