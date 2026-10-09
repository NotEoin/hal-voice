#!/usr/bin/env python3
"""Cleans the source clips in voice-data/hal-voice/ into voice-data/denoised/.

Each clip goes through Demucs (htdemucs, vocals stem) to strip score and ship
hum, then DeepFilterNet 3 to remove what noise is left, then an edge trim, short
fades and loudness matching. prepare_dataset.py --source denoised trains on the
result. Needs demucs, deepfilternet, soundfile and librosa (see train-hal.md).

    python voice-data/denoise.py [--atten-db 40] [--only STEM ...]
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "hal-voice"
OUT = ROOT / "denoised"
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
OUT_SR = 32000
EDGE_PAD = 0.12      # seconds of silence kept before and after speech
FADE = 0.015         # seconds
TARGET_PEAK = 0.7    # linear; keeps headroom for the FX chain


def separate(model, wav: np.ndarray, sr: int, device: str) -> np.ndarray:
    """Returns the vocals stem of a mono clip at the model's sample rate."""
    from demucs.apply import apply_model
    audio = librosa.resample(wav, orig_sr=sr, target_sr=model.samplerate)
    stereo = torch.tensor(np.stack([audio, audio]), dtype=torch.float32)[None].to(device)
    with torch.no_grad():
        stems = apply_model(model, stereo, device=device, shifts=2, overlap=0.5)[0]
    vocals = stems[model.sources.index("vocals")].mean(0).cpu().numpy()
    return vocals


def enhance(df_model, df_state, wav: np.ndarray, sr: int, atten_db: float) -> np.ndarray:
    """Removes residual noise with DeepFilterNet, limiting attenuation to keep the voice natural."""
    from df.enhance import enhance as df_enhance
    audio = librosa.resample(wav, orig_sr=sr, target_sr=df_state.sr())
    clean = df_enhance(df_model, df_state, torch.tensor(audio, dtype=torch.float32)[None],
                       atten_lim_db=atten_db)
    return clean[0].cpu().numpy()


def finish(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resamples to 32 kHz, trims silence at the edges, fades and normalises."""
    wav = librosa.resample(wav, orig_sr=sr, target_sr=OUT_SR)
    _, (start, end) = librosa.effects.trim(wav, top_db=40)
    pad = int(EDGE_PAD * OUT_SR)
    wav = wav[max(0, start - pad):min(len(wav), end + pad)]
    n = int(FADE * OUT_SR)
    ramp = np.linspace(0.0, 1.0, n)
    wav[:n] *= ramp
    wav[-n:] *= ramp[::-1]
    peak = np.abs(wav).max() or 1.0
    return (wav * TARGET_PEAK / peak).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atten-db", type=float, default=40.0,
                   help="most DeepFilterNet may attenuate; lower keeps more air in the voice")
    p.add_argument("--only", nargs="*", help="clip stems to process")
    args = p.parse_args()

    from demucs.pretrained import get_model
    from df.enhance import init_df

    device = "cuda" if torch.cuda.is_available() else "cpu"
    demucs_model = get_model("htdemucs").to(device).eval()
    df_model, df_state, _ = init_df()
    OUT.mkdir(exist_ok=True)

    clips = sorted(c for c in SRC.iterdir() if c.suffix in AUDIO_SUFFIXES)
    if args.only:
        clips = [c for c in clips if c.stem in args.only]
    if not clips:
        sys.exit(f"no clips in {SRC}")
    for clip in clips:
        wav, sr = librosa.load(clip, sr=None, mono=True)
        vocals = separate(demucs_model, wav, sr, device)
        clean = enhance(df_model, df_state, vocals, demucs_model.samplerate, args.atten_db)
        sf.write(OUT / f"{clip.stem}.wav", finish(clean, df_state.sr()), OUT_SR, subtype="PCM_16")
        print(f"  {clip.stem}", flush=True)
    print(f"{len(clips)} clips written to {OUT}")


if __name__ == "__main__":
    main()
