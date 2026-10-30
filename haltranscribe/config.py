"""Settings for hal-transcribe: the `transcribe:` section of hal-voice's config.

Read through `halvoice.config`, so the packaged defaults, `built.yaml`, the user's
`config.yaml` and `HALVOICE_TRANSCRIBE_*` variables apply. A `[transcribe]` table in
`~/.config/hal/config.toml` is still read, with a deprecation warning, for one minor version
and loses to the hal-voice files. The vault is `transcribe.vault`, else `[hal].vault`.
"""

import logging
import os
import tomllib
from pathlib import Path

from halvoice import config as voice_config

log = logging.getLogger(__name__)

SECTION = "transcribe"
DEFAULT_VAULT = "~/Documents/vaults/Hal"


def config_path() -> Path:
    """Returns the shared config file; HAL_CONFIG overrides it, read on every call."""
    return Path(os.environ.get("HAL_CONFIG", "~/.config/hal/config.toml")).expanduser()


def _file() -> dict:
    try:
        return tomllib.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def settings() -> dict:
    """Returns the transcribe settings; a legacy `[transcribe]` table fills unset keys."""
    current = dict(voice_config.load().get(SECTION, {}))
    legacy = _file().get(SECTION)
    if legacy:
        log.warning(
            "[transcribe] in %s is deprecated: move it to hal-voice's config", config_path()
        )
        defaults = dict(voice_config.defaults().get(SECTION, {}))
        for key, value in legacy.items():
            if current.get(key) == defaults.get(key):
                current[key] = value
    return current


def shared() -> dict:
    """Returns the `[hal]` table of the shared config."""
    return dict(_file().get("hal", {}))


def vault() -> Path:
    """Returns the vault notes go to."""
    chosen = settings().get("vault") or shared().get("vault") or DEFAULT_VAULT
    return Path(chosen).expanduser()


def data_dir() -> Path:
    """Returns this module's data directory, creating it."""
    base = Path(os.environ.get("HAL_DATA", "~/.local/share/hal")).expanduser()
    path = base / SECTION
    path.mkdir(parents=True, exist_ok=True)
    return path
