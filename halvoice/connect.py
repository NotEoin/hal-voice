"""Work out which Claude Code setup this voice loop should drive.

hal-voice never runs a model of its own. It records you, hands the transcript to
Claude Code, and speaks whatever comes back. So the only question this module answers
is: *which* Claude Code, running *where*?

Out of the box the answer is "plain Claude Code, in the current directory" -- clone the
repo, run it, and it talks to whatever Claude Code you already have logged in. Everything
below is about pointing it somewhere else without editing code.

Two independent things get resolved:

  project_dir   The working directory Claude Code is launched in. This is the whole
                integration mechanism: Claude Code automatically picks up that
                directory's CLAUDE.md, .mcp.json and .claude/ settings. Point it at a
                Hal vault and you get Hal's memory and MCP servers; point it at any
                other project and you get that project's.

  environment   Which model answers. Empty = Claude Code's own login (cloud). Set a
                base_url + model and it is redirected at a local Ollama instead.

Resolution order, lowest priority first:

  1. hal-voice settings     the `claude:` block (halvoice/config.py layers)
  2. hal.conf               if claude.hal_root points at (or auto-finds) a Hal checkout
  3. env_file               any KEY=VALUE file named by claude.env_file
  4. process environment    ANTHROPIC_* already exported in your shell
  5. HALVOICE_CLAUDE_*      env overrides, applied by config.py before we run

Later sources overwrite earlier ones key by key, so importing hal.conf and then
overriding one value in env_file behaves the way you'd expect.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .util import get_logger

log = get_logger(__name__)

# Names a hal.conf uses -> what they mean here.
_HAL_CONF_KEYS = ("HAL_PROFILE", "HAL_MODEL", "HAL_FAST_MODEL", "HAL_OLLAMA_URL")


@dataclass
class Connection:
    """The resolved answer: where Claude Code runs, and with what environment."""

    project_dir: str
    env: dict[str, str] = field(default_factory=dict)
    model: str = ""
    small_fast_model: str = ""
    base_url: str = ""
    sources: list[str] = field(default_factory=list)
    inherited: dict[str, str] = field(default_factory=dict)

    @property
    def is_cloud(self) -> bool:
        """True when we are not redirecting Claude Code ourselves.

        Note this says nothing about where Claude Code then goes: if you have your own
        ANTHROPIC_BASE_URL exported, it still applies -- we simply aren't the one setting it.
        """
        return not self.base_url and not self.model

    def describe(self) -> str:
        """Returns a one-line description of the brain connection for logs and `doctor`."""
        where = f"in {self.project_dir}"
        if not self.is_cloud:
            return f"{self.model or 'default model'} via {self.base_url or 'default endpoint'}, {where}"
        if self.inherited.get("ANTHROPIC_BASE_URL"):
            return f"Claude Code's own settings ({self.inherited['ANTHROPIC_BASE_URL']}), {where}"
        return f"Claude Code with its own login, {where}"


def _parse_kv_file(path: Path) -> dict[str, str]:
    """Read a shell-style KEY=VALUE file. Ignores comments, blank lines and `export`.

    Deliberately not `source`d through a shell: these files come from other tools and
    should never be able to execute anything.
    """
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("could not read %s (%s)", path, e)
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key.replace("_", "").isalnum():
            continue
        # shlex handles the quoting styles these files use ("x", 'x', bare).
        try:
            parts = shlex.split(value, comments=True)
        except ValueError:
            parts = [value.strip().strip("\"'")]
        out[key] = parts[0] if parts else ""
    return out


def _find_hal_root(configured: str | None) -> Path | None:
    """Locate a Hal checkout. 'auto' looks for a sibling of this repo."""
    if not configured:
        return None
    if str(configured).lower() != "auto":
        p = Path(configured).expanduser()
        # A relative hal_root is relative to the hal-voice repo, not to wherever you
        # happened to be standing when you launched it.
        if not p.is_absolute():
            p = config.root_dir() / p
        p = p.resolve()
        return p if p.exists() else None
    here = config.root_dir()
    for candidate in (here.parent / "Hal", here.parent / "hal"):
        if (candidate / "hal.conf").exists():
            log.info("auto-detected Hal checkout at %s", candidate)
            return candidate
    return None


def _from_hal_conf(hal_root: Path) -> tuple[dict[str, str], str | None]:
    """Translate a Hal profile into our settings. Returns (values, vault_dir)."""
    conf = hal_root / "hal.conf"
    if not conf.exists():
        log.warning("hal_root %s has no hal.conf - run Hal's ./setup.sh there first", hal_root)
        return {}, None
    raw = _parse_kv_file(conf)
    values: dict[str, str] = {}
    # An empty HAL_MODEL means Hal itself is on the cloud profile, so we must NOT
    # redirect Claude Code either -- inheriting a base_url with no model would break it.
    if raw.get("HAL_MODEL"):
        values["model"] = raw["HAL_MODEL"]
        values["small_fast_model"] = raw.get("HAL_FAST_MODEL", "")
        values["base_url"] = raw.get("HAL_OLLAMA_URL", "http://localhost:11434")
    vault = hal_root / "Vault"
    return values, str(vault) if vault.is_dir() else None


def _shared_vault() -> str | None:
    """Returns [hal].vault from ~/.config/hal/config.toml when that folder exists."""
    try:
        import tomllib  # Python 3.11+; without it there is no shared vault to find
    except ImportError:
        return None
    path = Path(os.environ.get("HAL_CONFIG", "~/.config/hal/config.toml")).expanduser()
    try:
        vault = tomllib.loads(path.read_text()).get("hal", {}).get("vault")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return vault if vault and Path(vault).expanduser().is_dir() else None


def resolve() -> Connection:
    """Figure out where and how to run Claude Code. Safe to call repeatedly."""
    cfg = config.load()
    c = cfg.claude if "claude" in cfg else config.Cfg({})

    # --- 1. hal-voice settings -----------------------------------------------
    settings = {
        "model": str(c.get("model", "") or ""),
        "small_fast_model": str(c.get("small_fast_model", "") or ""),
        "base_url": str(c.get("base_url", "") or ""),
    }
    project_dir = c.get("project_dir") or None
    sources = ["hal-voice settings"]

    # --- 2. hal.conf, if we were pointed at a Hal checkout --------------------
    hal_root = _find_hal_root(c.get("hal_root"))
    if hal_root:
        values, vault = _from_hal_conf(hal_root)
        if (hal_root / "hal.conf").exists():
            # Say so even when it contributed nothing -- "Hal is on the cloud profile too"
            # is a useful thing to see, and silence here looks like the import failed.
            note = "" if values else " - cloud profile, nothing to redirect"
            sources.append(f"hal.conf ({hal_root}){note}")
        settings.update(values)
        if vault and not project_dir:
            project_dir = vault

    # --- 3. env_file --------------------------------------------------------
    env_file = config.resolve(c.get("env_file"))
    extra_env: dict[str, str] = {}
    if env_file and Path(env_file).exists():
        raw = _parse_kv_file(Path(env_file))
        if raw:
            sources.append(f"env_file ({env_file})")
        # Recognised keys map onto our settings; anything else is passed through
        # verbatim, so an env_file can carry proxy tokens, gateway headers, whatever.
        if raw.get("ANTHROPIC_MODEL"):
            settings["model"] = raw["ANTHROPIC_MODEL"]
        if raw.get("ANTHROPIC_SMALL_FAST_MODEL"):
            settings["small_fast_model"] = raw["ANTHROPIC_SMALL_FAST_MODEL"]
        if raw.get("ANTHROPIC_BASE_URL"):
            settings["base_url"] = raw["ANTHROPIC_BASE_URL"]
        for k, v in raw.items():
            if k not in ("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "ANTHROPIC_BASE_URL"):
                extra_env[k] = v

    # --- 4. the surrounding shell -------------------------------------------
    # Anything you have already exported (your own launcher's env, a proxy, a gateway)
    # is inherited by the subprocess for free, because we pass os.environ through. So we
    # deliberately do NOT copy it into our own overrides: re-deriving settings from it
    # would make us re-inject placeholder credentials over real ones. We only note that
    # it is there, so `hal-voice connection` can tell you what will actually apply.
    inherited = {
        k: os.environ[k]
        for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL")
        if os.environ.get(k)
    }
    if inherited and not any(settings.values()):
        sources.append("inherited from your environment")

    # --- project dir --------------------------------------------------------
    # HAL_VAULT is still honoured so older setups keep working unchanged; then the vault
    # named in the shared Hal config, so a service needs no --project.
    if not project_dir and not os.environ.get("HAL_VAULT") and _shared_vault():
        sources.append("[hal].vault (~/.config/hal/config.toml)")
    project_dir = project_dir or os.environ.get("HAL_VAULT") or _shared_vault() or os.getcwd()
    # Always absolute: this becomes a subprocess cwd, so a relative path would depend on
    # where hal-voice was launched from.
    project_dir = str(Path(project_dir).expanduser().resolve())

    # --- build the subprocess environment ------------------------------------
    env: dict[str, str] = {}
    if settings["base_url"]:
        env["ANTHROPIC_BASE_URL"] = settings["base_url"]
        # A local Ollama accepts any token; a real gateway should supply its own via
        # env_file or model_env, which are merged after this.
        env.setdefault("ANTHROPIC_AUTH_TOKEN", "ollama")
        env.setdefault("ANTHROPIC_API_KEY", "ollama")
    if settings["model"]:
        env["ANTHROPIC_MODEL"] = settings["model"]
    if settings["small_fast_model"]:
        env["ANTHROPIC_SMALL_FAST_MODEL"] = settings["small_fast_model"]
    env.update(extra_env)
    # model_env is the raw escape hatch and wins over everything we derived.
    env.update({str(k): str(v) for k, v in (c.get("model_env") or {}).items()})

    return Connection(
        project_dir=project_dir, env=env, sources=sources, inherited=inherited, **settings
    )


def explain() -> None:
    """Prints which Claude Code setup hal-voice resolved, and where each setting came from."""
    conn = resolve()
    print("hal-voice -> Claude Code connection")
    print("----------------------------------")
    print(f"  target       : {conn.describe()}")
    print(f"  project dir  : {conn.project_dir}")
    print(
        f"  mode         : {'cloud (your own Claude Code login)' if conn.is_cloud else 'local model'}"
    )
    print(f"  resolved from: {' -> '.join(conn.sources)}")

    def _mask(key: str, value: str) -> str:
        return "***" if any(s in key.upper() for s in ("TOKEN", "KEY", "SECRET")) else value

    if conn.env:
        print("  we set       :")
        for k, v in sorted(conn.env.items()):
            print(f"      {k}={_mask(k, v)}")
    else:
        print("  we set       : (nothing - Claude Code's own configuration applies)")
    # Only what we do NOT override -- anything we set above wins over the ambient value.
    still_applies = {k: v for k, v in conn.inherited.items() if k not in conn.env}
    if still_applies:
        print("  inherited    :")
        for k, v in sorted(still_applies.items()):
            print(f"      {k}={_mask(k, v)}")


if __name__ == "__main__":
    explain()
