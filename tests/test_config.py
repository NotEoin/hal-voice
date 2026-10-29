"""Tests for halvoice.config: where the settings come from."""

from pathlib import Path

from halvoice import config


def test_packaged_defaults_ship_inside_the_package(monkeypatch):
    monkeypatch.delenv("HALVOICE_CONFIG", raising=False)
    path = config._config_path()
    assert path == Path(config.__file__).resolve().parent / "config.yaml"
    cfg = config.load(force=True)
    assert cfg.get("tts.api_url") == "http://127.0.0.1:9880"
