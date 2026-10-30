"""Tests for checkpoint pins and the host-mode start gate."""

import pytest

from halvoice import cli, config, ttspins


@pytest.fixture
def dirs(tmp_path):
    gsv = tmp_path / "gsv"
    pre = gsv / ttspins.PRETRAINED
    pre.mkdir(parents=True)
    (pre / "base.ckpt").write_bytes(b"base")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "hal.pth").write_bytes(b"hal")
    return gsv, weights, tmp_path / "models.lock"


def test_record_then_verify_passes(dirs):
    gsv, weights, lock = dirs
    pins = ttspins.record(gsv, weights, lock)
    assert set(pins) == {"pretrained/base.ckpt", "weights/hal.pth"}
    ttspins.verify(gsv, weights, lock)


def test_changed_checkpoint_fails_naming_the_file(dirs):
    gsv, weights, lock = dirs
    ttspins.record(gsv, weights, lock)
    (weights / "hal.pth").write_bytes(b"swapped")
    with pytest.raises(ttspins.PinError, match=r"weights/hal\.pth"):
        ttspins.verify(gsv, weights, lock)


def test_unpinned_checkpoint_fails_naming_the_file(dirs):
    gsv, weights, lock = dirs
    ttspins.record(gsv, weights, lock)
    (weights / "new.ckpt").write_bytes(b"x")
    with pytest.raises(ttspins.PinError, match=r"weights/new\.ckpt"):
        ttspins.verify(gsv, weights, lock)


def test_missing_lock_asks_for_a_record(dirs):
    gsv, weights, lock = dirs
    with pytest.raises(ttspins.PinError, match="tts-pins record"):
        ttspins.verify(gsv, weights, lock)


def test_vanished_checkpoint_fails_naming_the_file(dirs):
    gsv, weights, lock = dirs
    ttspins.record(gsv, weights, lock)
    (weights / "hal.pth").unlink()
    with pytest.raises(ttspins.PinError, match="missing"):
        ttspins.verify(gsv, weights, lock)


def test_record_with_no_checkpoints_fails(tmp_path):
    with pytest.raises(ttspins.PinError, match=r"no \*\.ckpt"):
        ttspins.record(tmp_path / "g", tmp_path / "w", tmp_path / "l")


def host_cfg(monkeypatch, gsv, weights):
    monkeypatch.setattr(
        config,
        "load",
        lambda force=False: config.Cfg(
            {
                "tts": {
                    "server": {
                        "gptsovits_dir": str(gsv),
                        "weights_dir": str(weights),
                        "isolation": "host",
                    }
                }
            }
        ),
    )


def test_host_start_is_refused_on_a_pin_mismatch_and_names_the_file(dirs, monkeypatch, capsys):
    gsv, weights, lock = dirs
    ttspins.record(gsv, weights, lock)
    (weights / "hal.pth").write_bytes(b"swapped")
    monkeypatch.setattr(ttspins, "lock_path", lambda: lock)
    host_cfg(monkeypatch, gsv, weights)
    monkeypatch.setattr(cli.os, "execv", lambda *a: pytest.fail("must not start"))
    assert cli.main(["tts-server"]) == 1
    assert "weights/hal.pth" in capsys.readouterr().err


def test_host_start_unsets_the_weights_only_override(dirs, monkeypatch):
    gsv, weights, lock = dirs
    ttspins.record(gsv, weights, lock)
    monkeypatch.setattr(ttspins, "lock_path", lambda: lock)
    host_cfg(monkeypatch, gsv, weights)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    monkeypatch.setattr(cli.os, "execv", lambda *a: None)
    cli.main(["tts-server"])
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in cli.os.environ


def test_tts_pins_command_records_and_verifies(dirs, monkeypatch, capsys):
    gsv, weights, lock = dirs
    monkeypatch.setattr(ttspins, "lock_path", lambda: lock)
    host_cfg(monkeypatch, gsv, weights)
    assert cli.main(["tts-pins", "record"]) == 0
    assert cli.main(["tts-pins", "verify"]) == 0
    assert cli.main(["tts-pins"]) == 2
