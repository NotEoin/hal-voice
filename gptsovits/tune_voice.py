#!/usr/bin/env python3
"""Finds the HAL voice whose delivery best matches a line from the film, and measures it.

Scores a synthesised line against the original clip on what makes HAL sound like HAL:
the pitch contour (where the voice rises and falls), the loudness contour, the timing
(overall length and the pause between sentences), the pitch level, the tone (spectral
distance after alignment) and the speaker. A speech recogniser checks pronunciation;
a take it cannot understand is disqualified. Every score is 0..1, higher is closer.

Run with the gptsovits Python, with the GPT-SoVITS API server up (gptsovits/run-api.sh):

    python gptsovits/tune_voice.py search --weights hal-e12.ckpt hal_e8_s288.pth
    python gptsovits/tune_voice.py search --run hal-r1          # every checkpoint a training run saved
    python gptsovits/tune_voice.py train hal-r1 --version v2 --sovits-epochs 16 --gpt-epochs 20
    python gptsovits/tune_voice.py report

The target line is held out of training (`train` drops it from the dataset), so the
score says how Hal will sound on lines it has never heard. Results append to
gptsovits/tuning/results.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GSV = Path(os.environ.get("GPTSOVITS_DIR", HERE / "GPT-SoVITS")).resolve()
CLIPS = ROOT / "voice-data" / "hal-voice"
OUT = HERE / "tuning"
RESULTS = OUT / "results.jsonl"
API = os.environ.get("HALVOICE_TTS_URL", "http://127.0.0.1:9880")
SR = 16000

TARGET = "im-sorry-dave-im-afraid-i-cant-do-that"
# Lines HAL never says in the film, for the pronunciation check.
NOVEL = ["Your dentist appointment is on Thursday at ten in the morning.",
         "I have added lunch with Sam to your calendar for Friday.",
         "The weather tomorrow will be cool and dry."]
DEFAULTS = {"speed": 0.9, "fragment_interval": 0.5, "temperature": 0.7, "top_k": 15, "top_p": 0.8}


# ---- the film ---------------------------------------------------------------------------

def transcripts() -> dict[str, str]:
    with (CLIPS / "transcripts.csv").open(newline="", encoding="utf-8") as fh:
        return {r[0].strip(): r[1].strip() for r in csv.reader(fh) if r and not r[0].startswith("#")}


def load(path: Path | str, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    return y / (np.abs(y).max() or 1.0)


def reference_clips(texts: dict[str, str]) -> dict[str, tuple[Path, str]]:
    """Film clips of 3-10 s (GPT-SoVITS's limit for a reference), as 32 kHz wavs, minus the target."""
    refdir = OUT / "refs"
    refdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for mp3 in sorted(CLIPS.glob("*.mp3")):
        stem = mp3.stem
        if stem == TARGET or stem not in texts:
            continue
        wav = refdir / f"{stem}.wav"
        if not wav.exists():
            y, _ = librosa.load(str(mp3), sr=32000, mono=True)
            sf.write(wav, y, 32000, subtype="PCM_16")
        if 3.0 <= sf.info(wav).duration <= 10.0:
            out[stem] = (wav, texts[stem])
    return out


class Judge:
    """Scores takes against the film line with analyse_voice.match, and checks pronunciation."""

    def __init__(self) -> None:
        import analyse_voice as av
        self.av = av
        self.text = transcripts()[TARGET]
        fp_path = OUT / "film-fingerprint.json"
        if fp_path.exists():
            self.fp = json.loads(fp_path.read_text())
        else:
            self.fp = av.fingerprint()
            fp_path.write_text(json.dumps(self.fp, indent=1))
        self.film_fr = av.frames(av.load(CLIPS / f"{TARGET}.mp3"))
        import torch
        from faster_whisper import WhisperModel
        gpu = torch.cuda.is_available()
        self.asr = WhisperModel("small.en", device="cuda" if gpu else "cpu", compute_type="float16" if gpu else "int8")

    def wer(self, y: np.ndarray, text: str) -> float:
        segs, _ = self.asr.transcribe(y.astype(np.float32), language="en", beam_size=1)
        said = re.findall(r"[a-z']+", " ".join(s.text for s in segs).lower())
        want = re.findall(r"[a-z']+", text.lower())
        d = np.arange(len(said) + 1)
        for w in want:                       # word-level edit distance
            prev, d[0] = d.copy(), d[0] + 1
            for k, s in enumerate(said, 1):
                d[k] = min(prev[k] + 1, d[k - 1] + 1, prev[k - 1] + (s != w))
        return float(d[-1] / max(len(want), 1))

    def score(self, takes: list[np.ndarray], novel: list[np.ndarray]) -> dict:
        parts = [self.av.match(t.astype(np.float64), self.fp, self.film_fr) for t in takes]
        avg = {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}
        avg["wer"] = float(np.mean([self.wer(t, self.text) for t in takes] +
                                   [self.wer(y, s) for y, s in zip(novel, NOVEL)]))
        # A take the recogniser can't follow is out, however close its melody.
        avg["score"] = 0.0 if avg["wer"] > 0.3 else avg["match_pct"]
        return {k: round(v, 3) for k, v in avg.items()}


# ---- synthesis --------------------------------------------------------------------------

def set_weights(gpt: str, sovits: str) -> None:
    requests.get(f"{API}/set_gpt_weights", params={"weights_path": gpt}, timeout=180).raise_for_status()
    requests.get(f"{API}/set_sovits_weights", params={"weights_path": sovits}, timeout=180).raise_for_status()


def synth(text: str, ref: tuple[Path, str], p: dict) -> np.ndarray:
    r = requests.post(f"{API}/tts", timeout=300, json={
        "text": text, "text_lang": "en", "ref_audio_path": str(ref[0]), "prompt_text": ref[1], "prompt_lang": "en",
        "text_split_method": "cut5", "batch_size": 1, "media_type": "wav", "streaming_mode": False,
        "top_k": p["top_k"], "top_p": p["top_p"], "temperature": p["temperature"], "speed_factor": p["speed"],
        "fragment_interval": p["fragment_interval"], "repetition_penalty": 1.35})
    r.raise_for_status()
    y, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y, _ = librosa.effects.trim(y, top_db=35)
    return y / (np.abs(y).max() or 1.0)


def evaluate(judge: Judge, label: str, gpt: str, sovits: str, ref: tuple[Path, str], p: dict,
             takes: int = 3, save: bool = False) -> dict:
    lines = [synth(judge.text, ref, p) for _ in range(takes)]
    novel = [synth(s, ref, p) for s in NOVEL]
    result = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "gpt": Path(gpt).name,
              "sovits": Path(sovits).name, "ref": ref[0].stem, **p, **judge.score(lines, novel)}
    if save:
        best = OUT / "samples" / f"{label}.wav"
        best.parent.mkdir(parents=True, exist_ok=True)
        sf.write(best, lines[0], SR)
        result["sample"] = str(best)
    with RESULTS.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"  match {result['score']:5.1f}%  melody {result['melody_shape']:.2f} level {result['pitch_level']:.2f} "
          f"pause {result['pause']:.2f} lengths {result['word_lengths']:.2f} loud {result['word_loudness']:.2f} "
          f"quality {result['voice_quality']:.2f} spectrum {result['spectrum']:.2f} wer {result['wer']:.2f} | "
          f"{Path(gpt).name} {Path(sovits).name} ref={ref[0].stem[:24]} "
          f"speed={p['speed']} gap={p['fragment_interval']} temp={p['temperature']}", flush=True)
    return result


def search(judge: Judge, label: str, pairs: list[tuple[str, str]]) -> dict:
    """Coordinate search: checkpoint pair, then reference clip, then pace, gap and temperature."""
    refs = reference_clips(transcripts())
    default_ref = refs.get("im-sorry-dave-i-dont-have-enough-information") or next(iter(refs.values()))
    p = dict(DEFAULTS)
    best = None
    for gpt, sov in pairs:
        set_weights(gpt, sov)
        r = evaluate(judge, label, gpt, sov, default_ref, p)
        if not best or r["score"] > best["score"]:
            best, pair = r, (gpt, sov)
    set_weights(*pair)
    ref = default_ref
    for stem, candidate in refs.items():
        r = evaluate(judge, label, *pair, candidate, p)
        if r["score"] > best["score"]:
            best, ref = r, candidate
    for key, values in (("speed", [0.8, 0.85, 0.9, 0.95, 1.0]), ("fragment_interval", [0.3, 0.5, 0.7, 0.9]),
                        ("temperature", [0.4, 0.55, 0.7, 0.9]), ("top_k", [5, 15, 30])):
        for v in values:
            if v == p[key]:
                continue
            trial = {**p, key: v}
            r = evaluate(judge, label, *pair, ref, trial)
            if r["score"] > best["score"]:
                best, p = r, trial
    final = evaluate(judge, f"{label}-best", *pair, ref, p, takes=5, save=True)
    print(f"\nBEST {label}: {json.dumps(final)}")
    return final


def run_pairs(run: str) -> list[tuple[str, str]]:
    gpts = sorted(glob.glob(str(GSV / "GPT_weights_*" / f"{run}-e*.ckpt")))
    sovs = sorted(glob.glob(str(GSV / "SoVITS_weights_*" / f"{run}_e*.pth")))
    return [(g, s) for g in gpts for s in sovs]


# ---- training ---------------------------------------------------------------------------

def train(name: str, version: str, sovits_epochs: int, gpt_epochs: int, source: str) -> None:
    """Rebuilds the dataset without the target line, then trains a new model under `name`."""
    venv_py = ROOT / ".venv" / "bin" / "python"
    where = ["--source", source] if source in ("raw", "denoised") else ["--source-dir", source]
    subprocess.run([str(venv_py if venv_py.exists() else "python3"), str(ROOT / "voice-data" / "prepare_dataset.py"),
                    *where, "--exclude", TARGET, "--keep-reference"], check=True)
    env = {**os.environ, "HAL_EXP": name, "HAL_VERSION": version,
           "HAL_SOVITS_EPOCHS": str(sovits_epochs), "HAL_GPT_EPOCHS": str(gpt_epochs)}
    subprocess.run([sys.executable, str(HERE / "train_hal.py")], check=True, env=env)


def report() -> None:
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines()] if RESULTS.exists() else []
    for r in sorted(rows, key=lambda r: -r["score"])[:15]:
        print(f"{r['score']:5.1f}% melody {r['melody_shape']:.2f} level {r['pitch_level']:.2f} pause {r['pause']:.2f} "
              f"lengths {r['word_lengths']:.2f} loud {r['word_loudness']:.2f} quality {r['voice_quality']:.2f} "
              f"spectrum {r['spectrum']:.2f} wer {r['wer']:.2f} | {r['label']} {r['gpt']} {r['sovits']} "
              f"{r['ref'][:30]} speed {r['speed']} gap {r['fragment_interval']} temp {r['temperature']} k {r['top_k']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--weights", nargs=2, metavar=("GPT", "SOVITS"), help="one pair, from gptsovits/weights/")
    s.add_argument("--run", help="every checkpoint pair a training run saved")
    s.add_argument("--label")
    t = sub.add_parser("train")
    t.add_argument("name")
    t.add_argument("--version", default="v2", choices=["v2", "v2Pro", "v2ProPlus"])
    t.add_argument("--sovits-epochs", type=int, default=16)
    t.add_argument("--gpt-epochs", type=int, default=20)
    t.add_argument("--source", default="raw", help="raw, denoised, or a folder of cleaned clips")
    sub.add_parser("report")
    sub.add_parser("self-check")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "report":
        return report()
    if args.cmd == "train":
        return train(args.name, args.version, args.sovits_epochs, args.gpt_epochs, args.source)
    judge = Judge()
    if args.cmd == "self-check":
        film = judge.av.load(CLIPS / f"{TARGET}.mp3")
        print("film vs itself:", judge.score([film], []))
        return None
    if args.weights:
        pairs = [(str(HERE / "weights" / args.weights[0]), str(HERE / "weights" / args.weights[1]))]
    else:
        pairs = run_pairs(args.run)
    if not pairs:
        sys.exit("no weights found")
    search(judge, args.label or args.run or Path(args.weights[0]).stem, pairs)


if __name__ == "__main__":
    main()
