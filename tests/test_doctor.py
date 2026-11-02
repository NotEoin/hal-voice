"""Tests for the GPU lines of `hal-voice doctor`, against a fake daemon in a temp directory."""

import pytest

from halvoice import doctor, gpu


def test_doctor_reports_the_mode_the_daemon_and_the_cpu_devices(daemon):
    line = doctor._gpu_sharing()
    assert "gpu.mode on_demand" in line
    assert "answers" in line
    assert "speech recognition on cpu, voice on cpu" in line


def test_doctor_warns_when_no_daemon_answers_but_does_not_fail():
    with pytest.raises(doctor._Warn, match="no gpu-light daemon answers"):
        doctor._gpu_sharing()


def test_doctor_in_off_mode_does_not_contact_the_daemon(daemon, monkeypatch):
    monkeypatch.setenv("HALVOICE_GPU_MODE", "off")
    assert "gpu-light is not contacted" in doctor._gpu_sharing()
    assert daemon.log == []


def test_doctor_names_a_profile_it_ignores_in_on_demand_mode(monkeypatch):
    monkeypatch.setenv("HALVOICE_PROFILE", "24gb")
    monkeypatch.setenv("HALVOICE_PROFILE_TTS_DEVICE", "cuda")
    assert "ignored in on_demand mode" in doctor._voice_profile()
    assert "voice on cpu" in doctor._voice_profile()


def test_doctor_shows_the_profile_devices_in_resident_mode(monkeypatch):
    monkeypatch.setenv("HALVOICE_GPU_MODE", "resident")
    monkeypatch.setenv("HALVOICE_PROFILE", "24gb")
    monkeypatch.setenv("HALVOICE_PROFILE_TTS_DEVICE", "cuda")
    assert "voice on cuda" in doctor._voice_profile()
    assert gpu.MODES  # the mode names stay in one place
