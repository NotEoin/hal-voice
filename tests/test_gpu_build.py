"""GPU handling of the voice build: GpuBusy at the deadline and `gpu-light run` for GPU stages."""

import stat
import sys

import pytest

from voicebuild import __main__ as vb_main
from voicebuild import build


class FakeBuild:
    """Stands in for `build.Build`: only `say` is used by the wait."""

    def say(self, msg: str) -> None:
        pass


def fake_gpu_light(tmp_path, monkeypatch, status_code: int) -> None:
    """Puts a `gpu-light` script on PATH that logs its argv and answers `status` with a code."""
    log = tmp_path / "calls.log"
    exe = tmp_path / "bin" / "gpu-light"
    exe.parent.mkdir()
    exe.write_text(
        f'#!/bin/sh\necho "$@" >> {log}\n[ "$1" = status ] && exit {status_code}\nexit 0\n'
    )
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{exe.parent}:/usr/bin:/bin")
    monkeypatch.delenv(build.IN_GPU_RUN_ENV, raising=False)


def test_wait_for_gpu_raises_after_deadline(monkeypatch):
    monkeypatch.setattr(build, "free_gpu_gb", lambda: 2.0)
    monkeypatch.setattr(build, "WAIT_FOR_GPU_S", 0)
    monkeypatch.delenv(build.IN_GPU_RUN_ENV, raising=False)
    with pytest.raises(build.GpuBusy):
        build.wait_for_gpu(FakeBuild())


def test_wait_for_gpu_returns_when_memory_is_free(monkeypatch):
    monkeypatch.setattr(build, "free_gpu_gb", lambda: 20.0)
    monkeypatch.delenv(build.IN_GPU_RUN_ENV, raising=False)
    build.wait_for_gpu(FakeBuild())


def test_wait_for_gpu_is_skipped_inside_gpu_light_run(monkeypatch):
    monkeypatch.setattr(build, "free_gpu_gb", lambda: 0.1)
    monkeypatch.setattr(build, "WAIT_FOR_GPU_S", 0)
    monkeypatch.setenv(build.IN_GPU_RUN_ENV, "1")
    build.wait_for_gpu(FakeBuild())


def test_gpu_light_answers_only_when_status_succeeds(tmp_path, monkeypatch):
    fake_gpu_light(tmp_path, monkeypatch, 0)
    assert build.gpu_light_answers()
    (tmp_path / "down").mkdir()
    fake_gpu_light(tmp_path / "down", monkeypatch, 3)
    assert not build.gpu_light_answers()


def test_gpu_light_is_absent_without_the_command(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert not build.gpu_light_answers()


def test_build_stage_uses_gpu_light_run_when_daemon_answers(tmp_path, monkeypatch):
    fake_gpu_light(tmp_path, monkeypatch, 0)
    monkeypatch.setattr(sys, "argv", ["make-hal-voice", "film.mkv", "--from", "clean"])
    vb_main.run_under_gpu_light("train")
    calls = (tmp_path / "calls.log").read_text().splitlines()
    run = next(c for c in calls if c.startswith("run"))
    assert "--class batch --vram 12G --preempt cuda-checkpoint --wait --" in run
    assert "-m voicebuild film.mkv --only train" in run
    assert "--from" not in run


def test_stage_argv_drops_earlier_stage_selection():
    argv = vb_main.stage_argv("select", ["--only=find", "a.mkv", "--only", "train", "--ocr"])
    assert argv[-4:] == ["a.mkv", "--ocr", "--only", "select"]
