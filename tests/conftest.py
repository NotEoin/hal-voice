"""Shared fixtures: every test sees an empty user config directory, never the real one."""

import pytest

from halvoice import config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Points `XDG_CONFIG_HOME` at a temp dir and drops cached and env-named settings."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.delenv("HALVOICE_CONFIG", raising=False)
    # no test may reach the real gpu-light daemon; a test that wants one sets its own socket
    monkeypatch.setenv("GPU_LIGHT_SOCKET", str(tmp_path / "no-gpu-light.sock"))
    monkeypatch.setattr(config, "_CACHE", None)
    return tmp_path / "xdg-config" / "hal-voice"
