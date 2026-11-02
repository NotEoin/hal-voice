"""Tests that the shipped systemd units hold no GPU claim of their own."""

from pathlib import Path

import pytest

UNITS = Path(__file__).resolve().parents[1] / "systemd"


@pytest.mark.parametrize("name", ["hal-voice.service", "hal-tts.service"])
def test_unit_is_not_marked_as_a_gpu_unit(name):
    lines = (UNITS / name).read_text(encoding="utf-8").splitlines()
    live = [line for line in lines if not line.lstrip().startswith("#")]
    assert not any("X-HalGPU" in line for line in live)
    assert not any("GPU_LIGHT_CLAIM" in line for line in live)
