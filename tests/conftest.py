"""Shared fixtures: every test sees an empty user config directory, never the real one."""

import pytest

from halvoice import config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Points `XDG_CONFIG_HOME` at a temp dir and drops cached and env-named settings."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.delenv("HALVOICE_CONFIG", raising=False)
    monkeypatch.setattr(config, "_CACHE", None)
    return tmp_path / "xdg-config" / "hal-voice"
