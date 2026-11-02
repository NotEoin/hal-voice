"""Tests for halvoice.placement: which device STT and the TTS server get, by gpu.mode."""

import pytest

from halvoice import config, placement
from halvoice.config import ConfigError
from halvoice.stt import Transcriber

PROFILE_24GB = {
    "HALVOICE_PROFILE_STT_DEVICE": "auto",
    "HALVOICE_PROFILE_STT_COMPUTE_TYPE": "float16",
    "HALVOICE_PROFILE_TTS_DEVICE": "cuda",
    "HALVOICE_PROFILE_TTS_HALF": "true",
}


def cfg(mode, **gpu_devices):
    tree = {"gpu": {"mode": mode}, "stt": {"device": "cpu", "compute_type": "int8"}}
    tree["tts"] = {"device": "cpu", "half": False}
    for dotted, value in gpu_devices.items():
        section, key = dotted.split("__")
        tree[section][key] = value
    return config.Cfg(tree)


def test_on_demand_picks_the_cpu_by_default():
    placed = placement.resolve(cfg("on_demand"), {})
    assert (placed.stt_device, placed.tts_device) == ("cpu", "cpu")
    assert (placed.stt_compute, placed.tts_half) == ("int8", False)
    assert not placed.stt_on_gpu
    assert not placed.tts_on_gpu


def test_on_demand_ignores_a_gpu_voice_conf_profile():
    placed = placement.resolve(cfg("on_demand"), PROFILE_24GB)
    assert (placed.stt_device, placed.tts_device) == ("cpu", "cpu")
    assert placed.source == "config"


def test_on_demand_follows_a_gpu_device_set_in_the_settings():
    placed = placement.resolve(
        cfg("on_demand", stt__device="cuda", tts__device="cuda", tts__half=True), {}
    )
    assert (placed.stt_device, placed.tts_device, placed.tts_half) == ("cuda", "cuda", True)
    assert placed.tts_on_gpu
    assert placed.stt_on_gpu


def test_on_demand_follows_an_environment_variable_asking_for_the_gpu(monkeypatch):
    monkeypatch.setenv("HALVOICE_TTS_DEVICE", "cuda")
    monkeypatch.setenv("HALVOICE_TTS_HALF", "true")
    placed = placement.resolve(config.load(force=True), {})
    assert (placed.tts_device, placed.tts_half) == ("cuda", True)
    assert placed.stt_device == "cpu"


def test_resident_lets_the_voice_conf_profile_win():
    placed = placement.resolve(cfg("resident"), PROFILE_24GB)
    assert (placed.stt_device, placed.stt_compute) == ("auto", "float16")
    assert (placed.tts_device, placed.tts_half) == ("cuda", True)
    assert placed.source == "voice.conf profile"


def test_off_lets_the_voice_conf_profile_win():
    placed = placement.resolve(cfg("off"), PROFILE_24GB)
    assert placed.tts_device == "cuda"


def test_resident_without_a_profile_uses_the_settings():
    placed = placement.resolve(cfg("resident"), {})
    assert placed.tts_device == "cpu"
    assert placed.source == "config"


def test_an_empty_profile_value_is_not_a_device():
    placed = placement.resolve(cfg("resident"), {"HALVOICE_PROFILE_TTS_DEVICE": ""})
    assert placed.tts_device == "cpu"


def test_a_cpu_device_forces_int8_and_no_half_precision():
    placed = placement.resolve(cfg("on_demand", stt__compute_type="float16", tts__half=True), {})
    assert placed.stt_compute == "int8"
    assert placed.tts_half is False


def test_packaged_defaults_put_stt_and_tts_on_the_cpu():
    placed = placement.resolve(config.load(force=True), {})
    assert (placed.mode, placed.stt_device, placed.tts_device) == ("on_demand", "cpu", "cpu")


def test_unknown_mode_is_a_config_error():
    with pytest.raises(ConfigError):
        placement.resolve(cfg("sometimes"), {})


def test_transcriber_stays_on_the_cpu_in_on_demand_despite_the_profile(monkeypatch):
    for key, value in PROFILE_24GB.items():
        monkeypatch.setenv(key, value)
    assert Transcriber().device == "cpu"


def test_transcriber_follows_the_profile_in_resident_mode(monkeypatch):
    for key, value in PROFILE_24GB.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HALVOICE_GPU_MODE", "resident")
    placement_seen = Transcriber()
    assert (placement_seen.device, placement_seen.compute_type) == ("auto", "float16")
