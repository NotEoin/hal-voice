"""Which device speech-to-text and the TTS server run on, by `gpu.mode`.

One rule, so the answer does not depend on which file was edited last:

- `on_demand` (the default): the devices are the `stt.device`, `stt.compute_type`,
  `tts.device` and `tts.half` settings. Their defaults are the CPU, so nothing holds VRAM
  while idle. They change only when you set them in `config.yaml` or in a
  `HALVOICE_STT_DEVICE`-style environment variable. `voice.conf` is ignored: setup-voice.sh
  writes a GPU there on a 24 GB card, and that is a machine profile, not a request.
- `resident` and `off`: the `voice.conf` profile wins over the settings, as it did before
  `gpu.mode` existed. `./hal-voice` exports its values as `HALVOICE_PROFILE_*`.

`doctor` prints the result, so what is reported is what runs.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .gpu import gpu_mode

CPU = "cpu"
CPU_COMPUTE = "int8"  # faster-whisper on the CPU; float16 is not supported there
PROFILE_PREFIX = "HALVOICE_PROFILE_"


@dataclass(frozen=True)
class Placement:
    """The devices in use, and where they came from."""

    mode: str
    stt_device: str
    stt_compute: str
    tts_device: str
    tts_half: bool
    source: str  # "config" or "voice.conf profile"

    @property
    def tts_on_gpu(self) -> bool:
        """True when the TTS server is placed on a GPU."""
        return self.tts_device != CPU

    @property
    def stt_on_gpu(self) -> bool:
        """True when speech-to-text may use a GPU (`auto` tries one)."""
        return self.stt_device != CPU


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def resolve(cfg: Any, environ: Mapping[str, str] | None = None) -> Placement:
    """The devices for `cfg`, by the rule in the module docstring."""
    env = os.environ if environ is None else environ
    mode = gpu_mode(cfg)
    stt = str(cfg.get("stt.device", CPU))
    compute = str(cfg.get("stt.compute_type", CPU_COMPUTE))
    tts = str(cfg.get("tts.device", CPU))
    half = _truthy(cfg.get("tts.half", False))
    source = "config"
    if mode != "on_demand":
        profile = {
            k[len(PROFILE_PREFIX) :]: v
            for k, v in env.items()
            if k.startswith(PROFILE_PREFIX) and v
        }
        if "STT_DEVICE" in profile or "TTS_DEVICE" in profile:
            source = "voice.conf profile"
        stt = profile.get("STT_DEVICE", stt)
        compute = profile.get("STT_COMPUTE_TYPE", compute)
        tts = profile.get("TTS_DEVICE", tts)
        half = _truthy(profile["TTS_HALF"]) if "TTS_HALF" in profile else half
    if stt == CPU:
        compute = CPU_COMPUTE
    if tts == CPU:
        half = False  # torch has no fp16 on the CPU
    return Placement(mode, stt, compute, tts, half, source)
