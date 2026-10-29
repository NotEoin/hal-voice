"""Tests for halvoice.cli: command dispatch and the packaged console script."""

import importlib.util
import subprocess
import sys

import pytest

from halvoice import cli, config, util

INSTALL_TIMEOUT_S = 600  # pip builds the wheel and creates a venv; allow a slow disk
RUN_TIMEOUT_S = 60


@pytest.fixture(autouse=True)
def no_environment_changes(monkeypatch):
    """Keeps `main` from setting SSL_CERT_FILE in the test process."""
    monkeypatch.setattr(util, "ca_bundle", lambda: None)


def test_help_lists_every_command(capsys):
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    for command in cli.COMMANDS:
        assert f"hal-voice {command}" in out or command == "repl"


def test_unknown_command_exits_nonzero_and_names_it(capsys):
    assert cli.main(["frobnicate"]) == 1
    assert "Unknown command: frobnicate" in capsys.readouterr().err


def test_project_option_is_exported_before_or_after_the_command(monkeypatch):
    monkeypatch.delenv(cli.PROJECT_ENV, raising=False)
    seen = {}
    monkeypatch.setitem(cli.COMMANDS, "doctor", lambda args: seen.update(args=args) or 0)
    assert cli.main(["doctor", "--project", "/work/x", "extra"]) == 0
    assert seen["args"] == ["extra"]
    assert cli.os.environ[cli.PROJECT_ENV] == "/work/x"
    cli.os.environ.pop(cli.PROJECT_ENV)


def test_no_arguments_runs_the_voice_loop(monkeypatch):
    called = []
    monkeypatch.setitem(cli.COMMANDS, "repl", lambda args: called.append(args) or 0)
    assert cli.main([]) == 0
    assert called == [[]]


def test_tts_without_text_is_a_usage_error(capsys):
    assert cli.main(["tts"]) == 2
    assert "usage" in capsys.readouterr().err


@pytest.mark.integration
def test_wheel_installs_and_hal_voice_help_runs(tmp_path):
    """A wheel built from this checkout installs and its console script prints help."""
    # find_spec, not importorskip: importing setuptools inside pytest trips its distutils shim
    for needed in ("pip", "setuptools"):
        if importlib.util.find_spec(needed) is None:
            pytest.skip(f"{needed} is not installed in this environment")
    root = cli.__file__.rsplit("/halvoice/", 1)[0]
    wheels = tmp_path / "wheels"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(wheels),
            root,
        ],
        check=True,
        capture_output=True,
        timeout=INSTALL_TIMEOUT_S,
    )
    wheel = next(wheels.glob("hal_voice-*.whl"))
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=INSTALL_TIMEOUT_S)
    subprocess.run(
        [str(venv / "bin" / "python"), "-m", "pip", "install", "--no-deps", str(wheel)],
        check=True,
        capture_output=True,
        timeout=INSTALL_TIMEOUT_S,
    )
    run = subprocess.run(
        [str(venv / "bin" / "hal-voice"), "help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
    )
    assert run.returncode == 0
    assert "hal-voice mcp" in run.stdout


def fake_config(monkeypatch, server_dir):
    monkeypatch.setattr(
        config,
        "load",
        lambda force=False: config.Cfg({"tts": {"server": {"gptsovits_dir": server_dir}}}),
    )


def test_tts_server_without_the_checkout_exits_1_and_names_it(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "GPT-SoVITS"
    fake_config(monkeypatch, str(missing))
    monkeypatch.setattr(cli.os, "execv", lambda *a: pytest.fail("must not start the server"))
    assert cli.main(["tts-server"]) == 1
    err = capsys.readouterr().err
    assert str(missing) in err
    assert "RequiresMountsFor" in err


def test_tts_server_with_the_checkout_starts_run_api(monkeypatch, tmp_path):
    present = tmp_path / "GPT-SoVITS"
    present.mkdir()
    fake_config(monkeypatch, str(present))
    monkeypatch.delenv("GPTSOVITS_DIR", raising=False)
    started = {}
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: started.update(path=path, argv=argv))
    cli.main(["tts-server"])
    assert str(started["path"]).endswith("gptsovits/run-api.sh")
    assert cli.os.environ["GPTSOVITS_DIR"] == str(present)
    cli.os.environ.pop("GPTSOVITS_DIR")
