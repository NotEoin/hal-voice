"""Tests for halvoice.brains: choosing the brain `brain.mode` names."""

import pytest

from halvoice import config
from halvoice.brains import BrainUnavailable, make_brain
from halvoice.brains.openai import OpenAIBrain


def cfg_with(**brain):
    cfg = config.load(force=True)
    cfg["brain"].update(brain)
    return cfg


def test_openai_mode_builds_the_openai_brain():
    brain = make_brain(cfg_with(mode="openai", base_url="http://127.0.0.1:8080/v1"))
    assert isinstance(brain, OpenAIBrain)


def test_openai_mode_without_base_url_stops_with_the_key_named():
    with pytest.raises(BrainUnavailable, match=r"brain\.base_url"):
        make_brain(cfg_with(mode="openai", base_url=""))


def test_unknown_mode_names_the_mode_and_the_choices():
    with pytest.raises(BrainUnavailable, match="'telepathy' is not one of: hal, openai"):
        make_brain(cfg_with(mode="telepathy"))


def test_direct_mode_is_removed_and_names_the_fix():
    """Regression: `direct` once ran core Hal in-process, and a missing core fell back to Claude."""
    with pytest.raises(BrainUnavailable, match="direct brain was removed: use hal"):
        make_brain(cfg_with(mode="direct"))


def test_default_brain_mode_is_hal():
    assert config.load(force=True).get("brain.mode") == "hal"
