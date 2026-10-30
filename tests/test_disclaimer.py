"""Tests for the film pipeline disclaimer: README, docs/howto/build-the-voice.md and `make-hal-voice --help`."""

import re
from pathlib import Path

import pytest

from voicebuild import __main__ as voicebuild_main

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = (ROOT / "voicebuild" / "disclaimer.md").read_text(encoding="utf-8")


def test_readme_and_build_help_carry_disclaimer(monkeypatch, capsys):
    assert DISCLAIMER in (ROOT / "README.md").read_text(encoding="utf-8")
    monkeypatch.setattr(voicebuild_main.sys, "argv", ["make-hal-voice", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        voicebuild_main.main()
    assert exit_info.value.code == 0
    help_text = re.sub(r"\s+", " ", capsys.readouterr().out)
    assert re.sub(r"\s+", " ", DISCLAIMER.strip()) in help_text


def test_build_guide_links_to_the_disclaimer_at_the_top():
    top = (
        (ROOT / "docs" / "howto" / "build-the-voice.md")
        .read_text(encoding="utf-8")
        .split("\n## ", 1)[0]
    )
    assert "README.md#about-the-film-voice-pipeline" in top


def test_disclaimer_says_no_media_is_included_and_to_bring_your_own_copy():
    assert "No media is included." in DISCLAIMER
    assert "Bring your own lawfully owned copy." in DISCLAIMER
