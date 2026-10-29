"""Tests for halvoice.config: layers, strict keys and environment overrides."""

import pytest

from halvoice import config


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_packaged_defaults_ship_inside_the_package():
    assert config.DEFAULTS_FILE.parent.name == "halvoice"
    assert config.load(force=True).get("tts.api_url") == "http://127.0.0.1:9880"


def test_config_layers_in_order(isolated_config, monkeypatch):
    write(isolated_config / "built.yaml", "tts:\n  speed: 0.8\n  top_k: 20\n  top_p: 0.5\n")
    write(isolated_config / "config.yaml", "tts:\n  top_k: 30\n  top_p: 0.6\n")
    monkeypatch.setenv("HALVOICE_TTS_TOP_P", "0.7")
    cfg = config.load(force=True)
    assert cfg.get("tts.speed") == 0.8  # built.yaml over the defaults
    assert cfg.get("tts.top_k") == 30  # the user file over built.yaml
    assert cfg.get("tts.top_p") == 0.7  # the environment over everything
    assert cfg.get("tts.text_lang") == "en"  # untouched defaults stay
    assert config.layers() == [
        config.DEFAULTS_FILE,
        isolated_config / "built.yaml",
        isolated_config / "config.yaml",
    ]


@pytest.mark.parametrize("name", ["built.yaml", "config.yaml"])
def test_unknown_override_key_fails(isolated_config, name):
    path = write(isolated_config / name, "tts:\n  sped: 0.8\n")
    with pytest.raises(config.ConfigError) as err:
        config.load(force=True)
    assert str(path) in str(err.value)
    assert "tts.sped" in str(err.value)


def test_unknown_top_level_section_fails(isolated_config):
    write(isolated_config / "config.yaml", "voice:\n  speed: 1\n")
    with pytest.raises(config.ConfigError, match="'voice'"):
        config.load(force=True)


def test_scalar_in_place_of_a_table_fails(isolated_config):
    write(isolated_config / "config.yaml", "tts: fast\n")
    with pytest.raises(config.ConfigError, match="'tts' must be a mapping"):
        config.load(force=True)


def test_open_table_takes_any_key(isolated_config):
    write(isolated_config / "config.yaml", "claude:\n  model_env:\n    MY_PROXY_TOKEN: x\n")
    assert config.load(force=True).get("claude.model_env") == {"MY_PROXY_TOKEN": "x"}


def test_halvoice_config_names_the_user_file(tmp_path, monkeypatch):
    path = write(tmp_path / "elsewhere.yaml", "log:\n  level: DEBUG\n")
    monkeypatch.setenv("HALVOICE_CONFIG", str(path))
    assert config.load(force=True).get("log.level") == "DEBUG"


def test_missing_file_named_by_halvoice_config_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("HALVOICE_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(config.ConfigError, match="does not exist"):
        config.load(force=True)


def test_malformed_yaml_names_the_file(isolated_config):
    path = write(isolated_config / "config.yaml", "tts: [unclosed\n")
    with pytest.raises(config.ConfigError, match="not valid YAML") as err:
        config.load(force=True)
    assert str(path) in str(err.value)


def test_empty_user_file_changes_nothing(isolated_config):
    write(isolated_config / "config.yaml", "# nothing yet\n")
    assert config.load(force=True).get("tts.text_lang") == "en"


def test_unknown_environment_variable_is_ignored(monkeypatch):
    monkeypatch.setenv("HALVOICE_PROFILE", "24gb")
    monkeypatch.setenv("HALVOICE_TTS_NOT_A_KEY", "1")
    cfg = config.load(force=True)
    assert "not_a_key" not in cfg["tts"]


def test_build_owned_defaults_are_empty():
    cfg = config.load(force=True)
    for key in config.BUILD_KEYS:
        assert cfg.get(f"tts.{key}") in ("", []), key


def test_write_built_keeps_one_previous_file(isolated_config):
    first = config.write_built({"speed": 0.9}, "voice build one")
    config.write_built({"speed": 0.8}, "voice build two")
    assert first == isolated_config / "built.yaml"
    text = first.read_text()
    assert text.startswith("# Written by hal-voice (voice build two) at ")
    assert "Z.\n" in text.splitlines(keepends=True)[0]
    assert "speed: 0.8" in text
    assert "speed: 0.9" in (isolated_config / "built.yaml.prev").read_text()
    assert sorted(p.name for p in isolated_config.iterdir()) == ["built.yaml", "built.yaml.prev"]
    assert config.load(force=True).get("tts.speed") == 0.8


def test_write_built_refuses_keys_the_build_does_not_own(isolated_config):
    with pytest.raises(config.ConfigError, match="api_url"):
        config.write_built({"api_url": "http://x"}, "test")
    assert not (isolated_config / "built.yaml").exists()


def test_import_built_copies_only_build_owned_keys(tmp_path, isolated_config):
    old = write(
        tmp_path / "config.yaml.20261009-014500.bak",
        "tts:\n  api_url: http://elsewhere\n  gpt_weights: gptsovits/weights/hal-e12.ckpt\n"
        "  speed: 0.92\naudio:\n  input_device: 3\n",
    )
    config.import_built(old)
    cfg = config.load(force=True)
    assert cfg.get("tts.gpt_weights") == "gptsovits/weights/hal-e12.ckpt"
    assert cfg.get("tts.speed") == 0.92
    assert cfg.get("tts.api_url") == "http://127.0.0.1:9880"
    assert cfg.get("audio.input_device") is None
    assert f"imported from {old}" in (isolated_config / "built.yaml").read_text()


def test_import_built_without_voice_keys_fails(tmp_path):
    old = write(tmp_path / "config.yaml", "audio:\n  input_device: 3\n")
    with pytest.raises(config.ConfigError, match="no build-owned tts keys"):
        config.import_built(old)
