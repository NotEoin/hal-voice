"""Cleans HAL's lines for training: separates the voice, removes residual noise, matches levels.

Run in an environment with demucs and DeepFilterNet:

    python -m voicebuild.clean jobs.json

jobs.json holds {"audio": dialogue wav, "out": folder, "atten_db": float,
"items": [{"id", "start", "end", "earliest", "latest"}]}; earliest and latest bound how far a
clip may grow to take in a word the recogniser timed too short, up to the next voice. For each
item the voice is separated from the score and effects with Demucs (htdemucs), the leftover
gives a speech-to-background ratio, DeepFilterNet
removes what noise is left (limited to atten_db so the voice keeps its air), and the result is
trimmed, faded and written as 32 kHz mono PCM. Prints a JSON result per item.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PAD_S = 0.25  # read around each line so separation has context
KEEP_S = 0.12  # silence kept before the first word and after the last
FADE_S = 0.015
OUT_RATE = 32000
PEAK = 0.7
VOICE_DB = 35  # below the line's peak, the separated voice counts as silent


def clarity(seg: np.ndarray, rate: int) -> tuple[float, float]:
    """Returns the line's 4-8 kHz energy against its 0.3-3 kHz energy (dB), and the frequency
    below which 99% of its energy lies (Hz), over its louder half.

    A voice heard through a speaker or a transmission loses the high band that makes speech
    crisp, so both fall well below the film's clear lines.
    """
    import librosa

    spec = np.abs(librosa.stft(seg, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=rate, n_fft=2048)
    frames = spec.sum(axis=0)
    power = spec[:, frames >= np.percentile(frames, 50)].mean(axis=1) + 1e-12
    hf = (
        power[(freqs >= 4000) & (freqs < 8000)].sum() / power[(freqs >= 300) & (freqs < 3000)].sum()
    )
    bandwidth = freqs[np.searchsorted(np.cumsum(power) / power.sum(), 0.99)]
    return float(10 * np.log10(hf)), float(bandwidth)


def main() -> int:
    """Runs the command line."""
    import librosa
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from df.enhance import enhance, init_df

    job = json.loads(Path(sys.argv[1]).read_text())
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    separator = get_model("htdemucs").to(device).eval()
    df_model, df_state, _ = init_df(log_level="ERROR")
    info = sf.info(job["audio"])
    rate = info.samplerate
    results = {}
    for item in job["items"]:
        earliest = item.get("earliest", item["start"] - PAD_S)
        latest = item.get("latest", item["end"] + PAD_S)
        start = max(0.0, min(item["start"] - PAD_S, earliest))
        frames = int((max(item["end"] + PAD_S, latest) - start) * rate)
        mix, _ = sf.read(job["audio"], start=int(start * rate), frames=frames, dtype="float32")
        if mix.ndim > 1:
            mix = mix.mean(axis=1)
        x = librosa.resample(mix, orig_sr=rate, target_sr=separator.samplerate)
        stereo = torch.tensor(np.stack([x, x]))[None].to(device)
        with torch.no_grad():
            stems = apply_model(separator, stereo, device=device, shifts=1, overlap=0.5)[0]
        voice = stems[separator.sources.index("vocals")].mean(0).cpu().numpy()
        rest = x[: voice.size] - voice
        snr = float(10 * np.log10((np.sum(voice**2) + 1e-9) / (np.sum(rest**2) + 1e-9)))
        hf_db, bandwidth = clarity(
            mix[int((item["start"] - start) * rate) : int((item["end"] - start) * rate)], rate
        )
        y = librosa.resample(voice, orig_sr=separator.samplerate, target_sr=df_state.sr())
        y = (
            enhance(
                df_model, df_state, torch.tensor(y)[None], atten_lim_db=job.get("atten_db", 15.0)
            )[0]
            .cpu()
            .numpy()
        )
        y = librosa.resample(y, orig_sr=df_state.sr(), target_sr=OUT_RATE)
        # The recogniser often times a line's last word as an instant; follow the voice itself out
        # to where it falls silent, but never past the next speaker.
        hop = OUT_RATE // 100
        env = librosa.amplitude_to_db(
            librosa.feature.rms(y=y, frame_length=4 * hop, hop_length=hop)[0], ref=1.0
        )
        loud = env > env.max() - VOICE_DB
        first = int((item["start"] - start) * 100)
        last = int((item["end"] - start) * 100)
        lo, hi = (
            max(0, int((earliest - start) * 100)),
            min(len(loud) - 1, int((latest - start) * 100)),
        )
        while first > lo and loud[first - 1]:
            first -= 1
        while last < hi and loud[last + 1]:
            last += 1
        a = max(0, int(first * hop - KEEP_S * OUT_RATE), int((earliest - start) * OUT_RATE))
        b = min(y.size, int(last * hop + KEEP_S * OUT_RATE), int((latest - start) * OUT_RATE))
        y = y[a:b]
        n = min(int(FADE_S * OUT_RATE), y.size // 2)
        if n:
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            y[:n] *= ramp
            y[-n:] *= ramp[::-1]
        y = y * PEAK / (np.abs(y).max() or 1.0)
        path = out / f"{item['id']}.wav"
        sf.write(path, y, OUT_RATE, subtype="PCM_16")
        results[item["id"]] = {
            "hf_db": round(hf_db, 1),
            "bandwidth_hz": round(bandwidth),
            "path": str(path),
            "snr_db": round(snr, 1),
            "seconds": round(y.size / OUT_RATE, 2),
        }
        print(json.dumps({"id": item["id"], **results[item["id"]]}), flush=True)
    (out / "clean.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
