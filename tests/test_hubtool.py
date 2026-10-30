"""Tests for the exec manifests and the one-shot `halvoice.hubtool` worker."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from halvoice import hubtool

ROOT = Path(__file__).parents[1]
MANIFESTS = [ROOT / "halvoice" / "hal-module.toml", ROOT / "haltranscribe" / "hal-module.toml"]
CALL_TIMEOUT_S = 60


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.parent.name)
def test_manifest_is_schema_2_exec_through_hubtool(path):
    m = tomllib.loads(path.read_text())
    assert m["schema"] == 2
    assert m["module"]["pack"] == "voice"
    assert m["module"]["exec"][1:] == ["-m", "halvoice.hubtool"]
    assert m["module"]["protocol"] == "oneshot"


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.parent.name)
def test_every_manifest_tool_has_policy_fields_and_a_worker_function(path):
    for tool in tomllib.loads(path.read_text())["tools"]:
        assert tool["function"] in hubtool.FUNCTIONS
        assert tool["tier"] in {"read", "write", "act", "send", "system"}
        for flag in ("network", "untrusted_output", "reviewable", "reversible"):
            assert isinstance(tool[flag], bool), (tool["name"], flag)


def test_run_calls_the_function_with_json_arguments(monkeypatch):
    monkeypatch.setattr("halvoice.tools.say", lambda text: {"spoken": True, "text": text})
    assert json.loads(hubtool.run("say", '{"text": "hi"}')) == {"spoken": True, "text": "hi"}


def test_run_rejects_an_unknown_function():
    with pytest.raises(KeyError, match="no function"):
        hubtool.run("nope", "{}")


def test_run_rejects_non_object_arguments():
    with pytest.raises(ValueError, match="JSON object"):
        hubtool.run("say", "[1]")


def test_subprocess_round_trip_fails_loudly_on_empty_text():
    done = subprocess.run(
        [sys.executable, "-m", "halvoice.hubtool", "say"],
        input='{"text": " "}',
        capture_output=True,
        text=True,
        timeout=CALL_TIMEOUT_S,
        cwd=ROOT,
        check=False,
    )
    assert done.returncode == hubtool.USAGE_EXIT
    assert "nothing to say" in done.stderr
    assert done.stdout == ""
