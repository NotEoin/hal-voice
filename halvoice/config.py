"""Configuration loader for hal-voice.

Settings come from four layers, later wins:

1. packaged defaults, `halvoice/config.yaml` (every key exists here, with its comment);
2. `$XDG_CONFIG_HOME/hal-voice/built.yaml`, written by the voice build (`make-hal-voice`);
3. `$XDG_CONFIG_HOME/hal-voice/config.yaml`, the user's own overrides, or the file
   `HALVOICE_CONFIG` names;
4. `HALVOICE_<SECTION>_<KEY>` environment variables (uppercased, nesting joined by `_`).

The two user files may only set keys the defaults define: an unknown key raises
`ConfigError` naming the file and the key, because a misspelt key that is silently ignored
leaves the user debugging the wrong thing. Tables listed in `OPEN_TABLES` take any key.
Unknown environment variables are ignored, since the environment carries unrelated
`HALVOICE_*` names (`HALVOICE_PROFILE` from `voice.conf`, for one).

Access is dict-with-dots: `cfg.tts.speed`, `cfg["tts"]["speed"]`, or `cfg.get("tts.speed")`.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """A dict that also supports attribute access and dotted .get()."""

    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Cfg(val) if isinstance(val, dict) else val

    def get(self, dotted: str, default: Any = None) -> Any:  # type: ignore[override]  # takes a dotted path where dict.get takes a key, on purpose
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Cfg(node) if isinstance(node, dict) else node


def _coerce(text: str) -> Any:
    """Turn an env-var string into the most plausible scalar."""
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _apply_env_overrides(cfg: dict) -> None:
    """HALVOICE_TTS_SPEED=0.9 -> cfg['tts']['speed'] = 0.9 (only if the path exists)."""
    prefix = "HALVOICE_"
    skip = {"HALVOICE_ROOT", "HALVOICE_CONFIG"}
    for key, raw in os.environ.items():
        if not key.startswith(prefix) or key in skip:
            continue
        parts = key[len(prefix) :].lower().split("_")
        # Walk as far down the existing tree as the parts allow; the remainder, if any,
        # is treated as a single leaf key (covers leaf names that contain underscores).
        node: Any = cfg
        for i, part in enumerate(parts):
            if isinstance(node, dict) and part in node and isinstance(node[part], dict):
                node = node[part]
            else:
                leaf = "_".join(parts[i:])
                if isinstance(node, dict) and leaf in node:
                    node[leaf] = _coerce(raw)
                break


DEFAULTS_FILE = Path(__file__).resolve().parent / "config.yaml"
# Tables whose keys are the user's own names, not settings: environment variables for the
# claude subprocess.
OPEN_TABLES = frozenset({"claude.model_env"})


class ConfigError(ValueError):
    """A settings file is unreadable, malformed, or sets a key hal-voice does not know."""


def user_config_dir() -> Path:
    """`$XDG_CONFIG_HOME/hal-voice`, defaulting to `~/.config/hal-voice`."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "hal-voice"


def built_file() -> Path:
    """The file the voice build writes its settings to."""
    return user_config_dir() / "built.yaml"


def user_file() -> Path:
    """The user's override file: the one `HALVOICE_CONFIG` names, else `config.yaml`."""
    env = os.environ.get("HALVOICE_CONFIG")
    return Path(env).expanduser() if env else user_config_dir() / "config.yaml"


def _read_yaml(path: Path) -> dict:
    """Returns the mapping in `path`; an empty file is an empty mapping."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"{path}: cannot read: {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping of settings, got {type(data).__name__}")
    return data


def _merge(base: dict, over: dict, source: Path, prefix: str = "") -> None:
    """Overlays `over` onto `base` in place; every key must already exist in `base`."""
    for key, value in over.items():
        dotted = f"{prefix}{key}"
        if key not in base:
            raise ConfigError(f"{source}: unknown key '{dotted}'")
        current = base[key]
        if isinstance(current, dict) and dotted not in OPEN_TABLES:
            if not isinstance(value, dict):
                raise ConfigError(f"{source}: '{dotted}' must be a mapping of settings")
            _merge(current, value, source, f"{dotted}.")
        else:
            base[key] = value


NO_VOICE = "no voice built or configured: run make-hal-voice or set tts.ref_audio_path"

# `tts` keys the voice build owns: written only to `built.yaml`, empty in the packaged defaults.
BUILD_KEYS = (
    "gpt_weights",
    "sovits_weights",
    "ref_audio_path",
    "prompt_text",
    "aux_ref_audio_paths",
    "speed",
    "top_k",
    "top_p",
    "temperature",
)


def write_built(tts_values: dict[str, Any], origin: str) -> Path:
    """Writes the voice build's `tts` settings to `built.yaml` atomically and returns its path.

    The previous file is kept as `built.yaml.prev`; on any failure the old file stays in
    place. `origin` names the build (or import) in the header comment.
    """
    unknown = sorted(set(tts_values) - set(BUILD_KEYS))
    if unknown:
        raise ConfigError(f"not build-owned tts keys: {', '.join(unknown)}")
    path = built_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    header = (
        f"# Written by hal-voice ({origin}) at {stamp}.\n"
        "# Rerun make-hal-voice to replace it; set keys in config.yaml beside it to override.\n"
    )
    body = yaml.safe_dump({"tts": dict(tts_values)}, sort_keys=False, allow_unicode=False)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(header + body)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".prev"))
    os.replace(tmp, path)
    return path


def import_built(old_config: Path) -> Path:
    """Copies the build-owned `tts` keys of an old checkout `config.yaml` into `built.yaml`.

    Relative paths are copied as they are: they resolve against the checkout, as before.
    """
    tts = _read_yaml(old_config).get("tts") or {}
    values = {k: tts[k] for k in BUILD_KEYS if k in tts}
    if not values:
        raise ConfigError(f"{old_config}: no build-owned tts keys ({', '.join(BUILD_KEYS)})")
    return write_built(values, f"imported from {old_config}")


def layers() -> list[Path]:
    """The settings files that exist, in the order they are applied."""
    files = [DEFAULTS_FILE, built_file(), user_file()]
    return [f for f in files if f.is_file()]


def root_dir() -> Path:
    """Absolute path to the hal-voice checkout, which relative config paths resolve against."""
    env = os.environ.get("HALVOICE_ROOT")
    return Path(env) if env else Path(__file__).resolve().parent.parent


def resolve(path_like: str | None) -> str | None:
    """Resolve a (possibly relative) config path against the hal-voice root."""
    if not path_like:
        return path_like
    p = Path(path_like).expanduser()
    return str(p if p.is_absolute() else root_dir() / p)


_CACHE: Cfg | None = None


def load(force: bool = False) -> Cfg:
    """Returns the merged settings, cached after the first call unless `force` is set.

    Raises `ConfigError` for a missing `HALVOICE_CONFIG` file, malformed YAML, or an unknown
    key in a user file.
    """
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE
    if not DEFAULTS_FILE.is_file():
        raise ConfigError(f"{DEFAULTS_FILE}: packaged defaults missing; reinstall hal-voice")
    data = _read_yaml(DEFAULTS_FILE)
    explicit = os.environ.get("HALVOICE_CONFIG")
    if explicit and not user_file().is_file():
        raise ConfigError(f"{user_file()}: named by HALVOICE_CONFIG but does not exist")
    for override in (built_file(), user_file()):
        if override.is_file():
            _merge(data, _read_yaml(override), override)
    _apply_env_overrides(data)
    _CACHE = Cfg(data)
    return _CACHE


if __name__ == "__main__":  # `python -m halvoice.config` prints the resolved config
    import json
    import sys

    print("# layers: " + ", ".join(str(f) for f in layers()), file=sys.stderr)
    print(json.dumps(load(), indent=2, default=str))
