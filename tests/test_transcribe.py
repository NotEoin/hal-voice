"""Tests for hal-transcribe that need no model."""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pytest

from haltranscribe import config as tconfig
from haltranscribe import tools


class TranscribeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["HAL_DATA"] = self.tmp.name
        from haltranscribe import tools

        self.t = tools

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_and_no_recording(self):
        with self.assertRaises(ValueError):
            self.t.transcribe("/no/such.mp3")
        with self.assertRaises(RuntimeError):
            self.t.record_stop()
        self.assertEqual(self.t.recording(), {"recording": False})


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HAL_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("HAL_CONFIG", str(tmp_path / "hal.toml"))
    return tmp_path


def test_transcribe_manifest_keeps_tool_names():
    import tomllib

    manifest = tomllib.loads(
        (Path(__file__).parents[1] / "haltranscribe" / "hal-module.toml").read_text()
    )
    assert [t["name"] for t in manifest["tools"]] == [
        "transcribe_file",
        "transcribe_record_start",
        "transcribe_record_stop",
        "transcribe_status",
        "transcribe_save_note",
    ]


def test_transcribe_settings_come_from_the_voice_config(data_home):
    assert tconfig.settings() == {"model": "medium.en", "folder": "Transcripts", "vault": ""}


def test_legacy_transcribe_table_is_read_with_a_warning(data_home, caplog):
    (data_home / "hal.toml").write_text('[transcribe]\nfolder = "Meetings"\n')
    assert tconfig.settings()["folder"] == "Meetings"
    assert "deprecated" in caplog.text


def test_vault_falls_back_to_the_hal_table(data_home):
    (data_home / "hal.toml").write_text(f'[hal]\nvault = "{data_home}/v"\n')
    assert tconfig.vault() == data_home / "v"


def test_record_stop_does_not_signal_a_reused_pid(data_home, monkeypatch):
    """Regression: a recorder pid reused by another program used to be sent SIGINT."""
    killed = []
    monkeypatch.setattr(tools.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(tools, "STOP_SETTLE_S", 0)
    monkeypatch.setattr(tools, "transcribe", lambda f: {"text": "hi"})
    tools._state().write_text(
        json.dumps({"pid": os.getpid(), "pid_start": "1", "file": "x", "title": "t", "started": 0})
    )
    assert tools.record_stop(save=False) == {"text": "hi"}
    assert killed == []


def test_record_stop_signals_the_matching_recorder(data_home, monkeypatch):
    killed = []
    monkeypatch.setattr(tools.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(tools, "STOP_SETTLE_S", 0)
    monkeypatch.setattr(tools, "transcribe", lambda f: {"text": "hi"})
    me = os.getpid()
    tools._state().write_text(
        json.dumps(
            {"pid": me, "pid_start": tools._pid_start(me), "file": "x", "title": "t", "started": 0}
        )
    )
    tools.record_stop(save=False)
    assert killed == [me]


def test_busy_gpu_falls_back_to_the_cpu_model(monkeypatch):
    class ClaimTimeout(Exception):
        pass

    class Claim:
        def __init__(self, name, **kw):
            assert (name, kw["wait"], kw["timeout"]) == ("hal-transcribe", True, 120)

        def __enter__(self):
            raise ClaimTimeout

        def __exit__(self, *exc):
            return False

    fake = types.ModuleType("gpu_light")
    fake.claim, fake.ClaimTimeout = Claim, ClaimTimeout
    monkeypatch.setitem(sys.modules, "gpu_light", fake)
    with tools._gpu_claim() as use_gpu:
        assert use_gpu is False


def test_transcribing_runs_without_gpu_light(monkeypatch):
    monkeypatch.setitem(sys.modules, "gpu_light", None)
    with tools._gpu_claim() as use_gpu:
        assert use_gpu is True
