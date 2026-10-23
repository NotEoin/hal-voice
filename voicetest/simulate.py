"""Stand-ins for the takes not yet recorded, made from a recorded one.

"fans" adds noise with the spectrum of the take's own silence, 6 dB louder in all; "far" lowers
the voice by 6 dB and tops the noise back up to the room's level, as speaking from across the
room does. Neither has a real room's echo, so a recorded take beats them; they are for checking
that a change does not only work for one level of voice and noise.
"""

from __future__ import annotations

import numpy as np

FANS_DB = 6.0
FAR_DB = -6.0


def shaped_noise(silence: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """n samples of random noise with the silence's spectrum and level."""
    from scipy.signal import welch

    f, p = welch(silence, nperseg=2048)
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    spec *= np.sqrt(np.interp(np.linspace(0, 0.5, spec.size), f, p))
    out = np.fft.irfft(spec, n)
    return (out / (out.std() + 1e-12) * silence.std()).astype(np.float32)


def apply(y: np.ndarray, silence: np.ndarray, sr: int, condition: str) -> np.ndarray:
    """Returns y as it would sound in the condition."""
    seed = int(abs(y[:1000].sum()) * 1e6) % 2**32
    if condition == "fans":
        extra = np.sqrt(10 ** (FANS_DB / 10) - 1)
        return (y + extra * shaped_noise(silence, y.size, seed)).astype(np.float32)
    if condition == "far":
        g = 10 ** (FAR_DB / 20)
        return (g * y + np.sqrt(1 - g * g) * shaped_noise(silence, y.size, seed)).astype(np.float32)
    raise ValueError(f"unknown condition {condition!r}: fans or far")
