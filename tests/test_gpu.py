"""VoiceClaims against a fake `gpu_light` module, and the loop without one."""

import sys
import time
import types
from typing import ClassVar

import pytest

from halvoice import gpu


class FakeClaim:
    """Records its lifecycle; `granted` flips when the test says so."""

    log: ClassVar[list] = []

    def __init__(self, name, **kwargs):
        self.name, self.kwargs, self.granted = name, kwargs, False
        self.released = False

    def acquire(self):
        FakeClaim.log.append(("acquire", self.name, self.kwargs))
        return self

    def release(self):
        self.released = True
        FakeClaim.log.append(("release", self.name, None))


@pytest.fixture
def fake_gl(monkeypatch):
    FakeClaim.log = []
    mod = types.ModuleType("gpu_light")
    mod.claim = FakeClaim
    monkeypatch.setitem(sys.modules, "gpu_light", mod)
    return mod


def make(linger_s=30):
    return gpu.VoiceClaims(resident_mib=3584, brain_mib=14000, linger_s=linger_s)


def test_service_claim_taken_before_models_load(fake_gl):
    order = []
    claims = make()
    claims.start_service()
    order.append("models")
    assert FakeClaim.log[0][:2] == ("acquire", "hal-voice")
    assert FakeClaim.log[0][2]["cls"] == "service"
    assert FakeClaim.log[0][2]["vram_mib"] == 3584
    assert order == ["models"]


def test_turn_claim_is_interactive_exclusive_and_uses_the_gateway(fake_gl):
    claims = make()
    claims.turn_started()
    kwargs = FakeClaim.log[0][2]
    assert (kwargs["cls"], kwargs["exclusive"], kwargs["uses"]) == (
        "interactive",
        True,
        ["hal-gateway"],
    )


def test_turn_claim_waits_at_most_turn_wait_s(fake_gl):
    claims = make()
    claims.turn_started()
    started = time.monotonic()
    assert claims.wait_for_turn(0.2) is False
    assert time.monotonic() - started < 1.0
    claims._turn.granted = True
    assert claims.wait_for_turn(0.2) is True


def test_turn_claim_released_after_linger(fake_gl):
    claims = make(linger_s=0.1)
    claims.turn_started()
    turn = claims._turn
    claims.turn_ended()
    deadline = time.monotonic() + 2
    while not turn.released and time.monotonic() < deadline:
        time.sleep(0.02)
    assert turn.released


def test_new_turn_inside_linger_keeps_the_claim(fake_gl):
    claims = make(linger_s=0.3)
    claims.turn_started()
    turn = claims._turn
    claims.turn_ended()
    claims.turn_started()
    time.sleep(0.5)
    assert not turn.released
    assert claims._turn is turn
    claims.close()
    assert turn.released


def test_loop_runs_without_gpu_light(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "gpu_light", None)  # an import raises ImportError
    caplog.set_level("INFO")
    claims = make()
    claims.start_service()
    claims.turn_started()
    assert claims.wait_for_turn(0.1) is False
    claims.turn_ended()
    claims.close()
    assert sum("gpu-light is not installed" in r.message for r in caplog.records) == 1


def test_a_failing_claim_is_logged_not_raised(monkeypatch):
    mod = types.ModuleType("gpu_light")

    def boom(*a, **k):
        raise RuntimeError("daemon exploded")

    mod.claim = boom
    monkeypatch.setitem(sys.modules, "gpu_light", mod)
    claims = make()
    claims.start_service()
    claims.turn_started()
    assert claims._service is None
    assert claims._turn is None
