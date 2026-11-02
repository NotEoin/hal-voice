"""Tests for platform detection and the WSL2 lines of `hal-voice doctor`, with a fake kernel."""

import pytest

from halvoice import cli, doctor, hostplatform


def _release(tmp_path, text):
    path = tmp_path / "osrelease"
    path.write_text(text + "\n", encoding="ascii")
    return path


def test_wsl2_kernel_release_is_detected_as_wsl2(tmp_path):
    path = _release(tmp_path, "5.15.153.1-microsoft-standard-WSL2")
    assert hostplatform.detect_platform(path) == hostplatform.WSL2


def test_plain_linux_kernel_release_is_detected_as_linux(tmp_path):
    assert hostplatform.detect_platform(_release(tmp_path, "6.8.0-generic")) == "linux"


def test_osrelease_env_replaces_the_proc_file(tmp_path, monkeypatch):
    monkeypatch.setenv(hostplatform.OSRELEASE_ENV, str(_release(tmp_path, "5.15-microsoft")))
    assert hostplatform.detect_platform() == hostplatform.WSL2


def test_other_systems_are_unsupported_with_a_message_naming_wsl2():
    with pytest.raises(hostplatform.UnsupportedPlatform, match="WSL2"):
        hostplatform.require_supported(system="win32")


def test_a_missing_osrelease_file_is_unsupported(tmp_path):
    assert hostplatform.detect_platform(tmp_path / "none") == hostplatform.UNSUPPORTED


def test_cli_refuses_to_start_on_an_unsupported_system(monkeypatch, capsys):
    monkeypatch.setattr(hostplatform.sys, "platform", "darwin")
    assert cli.main(["listen"]) == 1
    assert "not supported" in capsys.readouterr().err


def test_doctor_fails_the_audio_server_line_when_wslg_has_no_socket(tmp_path, monkeypatch):
    monkeypatch.delenv("PULSE_SERVER", raising=False)
    monkeypatch.setattr(doctor, "WSLG_PULSE", tmp_path / "PulseServer")
    with pytest.raises(Exception, match="wsl --update"):
        doctor._wsl_audio_server()


def test_doctor_accepts_the_wslg_audio_socket(tmp_path, monkeypatch):
    monkeypatch.delenv("PULSE_SERVER", raising=False)
    socket_path = tmp_path / "PulseServer"
    socket_path.touch()
    monkeypatch.setattr(doctor, "WSLG_PULSE", socket_path)
    assert str(socket_path) in doctor._wsl_audio_server()


def test_doctor_follows_pulse_server_when_it_names_a_unix_socket(tmp_path, monkeypatch):
    socket_path = tmp_path / "other"
    socket_path.touch()
    monkeypatch.setenv("PULSE_SERVER", f"unix:{socket_path}")
    assert str(socket_path) in doctor._wsl_audio_server()


def test_doctor_warns_when_wsl_has_no_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(doctor._Warn, match="hands-free"):
        doctor._wsl_display()


def test_doctor_warns_when_systemd_is_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "SYSTEMD_RUN_DIR", tmp_path / "absent")
    with pytest.raises(doctor._Warn, match="systemd=true"):
        doctor._wsl_systemd()


def test_doctor_platform_line_names_wsl2(tmp_path, monkeypatch):
    monkeypatch.setenv(
        hostplatform.OSRELEASE_ENV, str(_release(tmp_path, "5.15-microsoft-standard-WSL2"))
    )
    assert doctor._platform().startswith("WSL2")
