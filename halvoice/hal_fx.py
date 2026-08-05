"""The "HAL effect" — the DSP chain applied to GPT-SoVITS output.

A clean clone gets the timbre right but still sounds like a person in a room. HAL speaks
from inside a machine, in the dry cabin of Discovery One. The chain applies, in order:

  1. pitch flattening  — ships at 0.0. The Hilbert reconstruction it uses is only valid
                         for narrowband signals and adds artifacts to broadband speech.
  2. pitch shift       — ships at 0.0. Sounded artificial; depth comes from EQ instead.
  3. EQ                — warm the low-mids, ease presence/sibilance, roll off air.
  4. subtle saturation — drive for "speaker cone" colouration. Also ships at 0.0.
  5. small-room reverb — the damped cabin tail, not a hall.
  6. loudness norm     — so every reply sits at the same level.

Stages 1, 2 and 4 are kept for experimentation but are off by default; VOICE-TUNING.md
records why. All parameters live under tts.fx in config.yaml. Everything degrades
gracefully: a missing optional library skips its stage with a warning rather than
crashing.
"""

from __future__ import annotations

import numpy as np

from . import config
from .util import get_logger

log = get_logger(__name__)


def _flatten_pitch(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Reduce pitch variation toward the median F0. Off by default (`amount` 0.0).

    Reconstructs the signal from a moving-median de-trend of the analytic signal's
    instantaneous frequency. That reconstruction assumes a narrowband signal, and
    speech is broadband, so it introduces audible artifacts at any strength. Kept as
    the record of an approach that was tried and rejected — see VOICE-TUNING.md.
    """
    if amount <= 0:
        return audio
    try:
        from scipy.signal import hilbert
    except Exception as e:  # noqa: BLE001
        log.warning("scipy missing, skipping pitch flatten (%s)", e)
        return audio
    analytic = hilbert(audio)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase)
    # de-trend toward local median
    win = max(3, int(sr * 0.02))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(inst_freq, (pad, pad), mode="edge")
    med = np.array([np.median(padded[i:i + win]) for i in range(0, len(inst_freq), max(1, win // 4))])
    med = np.interp(np.arange(len(inst_freq)), np.linspace(0, len(inst_freq), len(med)), med)
    flattened = inst_freq * (1 - amount) + med * amount
    new_phase = np.concatenate([[inst_phase[0]], inst_phase[0] + np.cumsum(flattened)])
    out = (np.abs(analytic) * np.cos(new_phase)).astype(np.float32)
    peak = np.max(np.abs(out)) or 1.0
    return out / peak * (np.max(np.abs(audio)) or 1.0)


def _pedalboard_chain(audio: np.ndarray, sr: int, fx: dict) -> np.ndarray:
    """EQ + pitch shift + drive + reverb via Spotify's pedalboard. Skipped if absent."""
    try:
        from pedalboard import (Distortion, HighShelfFilter, LowShelfFilter,
                                PeakFilter, Pedalboard, PitchShift, Reverb)
    except Exception as e:  # noqa: BLE001
        log.warning("pedalboard missing, skipping EQ/reverb/pitch (%s)", e)
        return audio

    eq = fx.get("eq", {}) or {}
    rev = fx.get("reverb", {}) or {}
    board = Pedalboard([
        PitchShift(semitones=float(fx.get("pitch_shift_semitones", 0.0))),
        LowShelfFilter(cutoff_frequency_hz=float(eq.get("low_shelf_hz", 180)),
                       gain_db=float(eq.get("low_shelf_gain_db", 2.0))),
        PeakFilter(cutoff_frequency_hz=float(eq.get("presence_hz", 3500)),
                   gain_db=float(eq.get("presence_gain_db", -2.5)), q=1.0),
        HighShelfFilter(cutoff_frequency_hz=float(eq.get("high_rolloff_hz", 9000)),
                        gain_db=float(eq.get("high_rolloff_gain_db", -3.0))),
        Distortion(drive_db=float(fx.get("drive", 0.04)) * 100.0),
    ])
    if rev.get("enabled", True):
        board.append(Reverb(room_size=float(rev.get("room", 0.18)),
                            damping=float(rev.get("damping", 0.6)),
                            wet_level=float(rev.get("wet", 0.12)),
                            dry_level=1.0 - float(rev.get("wet", 0.12)) * 0.5))
    return board(audio, sr)


def _normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    try:
        import pyloudnorm as pyln
    except Exception as e:  # noqa: BLE001
        log.warning("pyloudnorm missing, skipping loudness norm (%s)", e)
        peak = np.max(np.abs(audio)) or 1.0
        return audio / peak * 0.9
    meter = pyln.Meter(sr)
    try:
        loud = meter.integrated_loudness(audio)
        out = pyln.normalize.loudness(audio, loud, target_lufs)
    except Exception as e:  # noqa: BLE001
        log.warning("loudness measurement failed (%s); peak-normalizing instead", e)
        peak = np.max(np.abs(audio)) or 1.0
        return audio / peak * 0.9
    # guard against clipping after normalization
    peak = np.max(np.abs(out)) or 1.0
    if peak > 0.99:
        out = out / peak * 0.99
    return out.astype(np.float32)


def apply(audio: np.ndarray, sr: int) -> np.ndarray:
    """Run the full HAL effect over mono float32 audio. Returns mono float32 at the same sr."""
    cfg = config.load()
    fx = cfg.get("tts.fx", {}) or {}
    if not fx.get("enabled", True):
        return audio
    audio = np.asarray(audio, dtype=np.float32).flatten()
    audio = _flatten_pitch(audio, sr, float(fx.get("pitch_flatten", 0.0)))
    audio = _pedalboard_chain(audio, sr, fx)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    audio = _normalize_loudness(audio, sr, float(fx.get("target_lufs", -18.0)))
    return audio
