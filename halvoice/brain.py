"""The direct brain: core Hal's agent loop, speaking as HAL.

The local model, the hal-hub tools, background deep work and the turn log all
live in core hal (halcore.brain.Brain). This module finds core hal and starts a
voice-channel Brain with config.yaml's brain.* values as overrides; empty
values leave the shared ~/.config/hal/config.toml [brain] settings in charge,
and HALVOICE_BRAIN_* variables override config.yaml as usual.

Core hal is found at HAL_CORE, else in the ecosystem layout
(<ecosystem>/core/hal beside extensions/hal-voice), else at [hal].core in the
shared config.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

from . import config
from .util import get_logger

log = get_logger(__name__)

# config.yaml keys passed to halcore's Brain when they are set.
OVERRIDES = ("ollama_url", "keep_alive", "history_turns", "max_tool_rounds", "tools_per_turn", "num_ctx",
             "temperature", "user_name")


class CoreNotFound(RuntimeError):
    pass


def _shared_config() -> dict:
    path = Path(os.environ.get("HAL_CONFIG", "~/.config/hal/config.toml")).expanduser()
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def find_core() -> Path:
    """Returns the core hal checkout; raises CoreNotFound when there is none."""
    candidates = [os.environ.get("HAL_CORE"), str(config.root_dir().resolve().parents[1] / "core" / "hal"),
                  _shared_config().get("hal", {}).get("core")]
    for c in candidates:
        if c and (Path(c).expanduser() / "halcore" / "brain.py").is_file():
            return Path(c).expanduser().resolve()
    raise CoreNotFound("core hal not found: install it with the Hal ecosystem installer, point HAL_CORE or "
                       "[hal].core in ~/.config/hal/config.toml at it, or set brain.mode: claude in config.yaml")


def direct_brain():
    """Returns a halcore Brain for the voice channel, set up from config.yaml."""
    core = str(find_core())
    if core not in sys.path:
        sys.path.insert(0, core)
    from halcore.brain import Brain

    cfg = config.load()
    hub_path = config.resolve(cfg.get("brain.hub_path"))
    if hub_path:
        os.environ.setdefault("HAL_HUB", hub_path)
    overrides = {k: cfg.get(f"brain.{k}") for k in OVERRIDES if cfg.get(f"brain.{k}") not in (None, "")}
    persona = str(cfg.get("brain.persona") or cfg.get("claude.voice_system_prompt") or "").strip() or None
    brain = Brain(channel="voice", persona=persona, model=cfg.get("brain.model") or None, **overrides)
    log.info("brain: %s via %s, %s (core hal at %s)", brain.model, brain.url,
             "with tools" if brain.hub else "no tools", core)
    return brain


# The name loop.py constructs.
DirectBrain = direct_brain
