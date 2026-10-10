#!/usr/bin/env python3
"""Fine-grained comparison of synthesised HAL against HAL in the film.

Two references:
  * a fingerprint of HAL's voice from every film clip: pitch level and range, speaking and
    articulation rate, voice quality (harmonics-to-noise ratio, jitter, shimmer, spectral
    tilt), vocal-tract resonances (formants F1-F3), the long-term spectrum in third-octave
    bands, the room (how fast sound decays after a word) and the noise floor;
  * the target line, word by word: each word's length, the gap before it, its pitch, how
    the pitch moves across it, and how loud it is.

For each candidate it prints where it departs from the film, largest gaps first, in plain
units (semitones, dB, percent). Needs praat-parselmouth and librosa.

    python gptsovits/analyse_voice.py NAME=file.wav [NAME=file.wav ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

HERE = Path(__file__).resolve().parent
CLIPS = HERE.parent / "voice-data" / "hal-voice"
TARGET = "im-sorry-dave-im-afraid-i-cant-do-that"
SR = 16000
BANDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300]

def load(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=40)
    return (y / (np.abs(y).max() or 1.0)).astype(np.float64)


# ---- whole-clip voice measures ----------------------------------------------------------

def voice(y: np.ndarray) -> dict:
    snd = parselmouth.Sound(y, SR)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=250)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    out = {}
    if f0.size > 10:
        st = 12 * np.log2(f0 / 100.0)
        out["pitch_hz"] = float(np.median(f0))
        out["pitch_range_st"] = float(np.percentile(st, 95) - np.percentile(st, 5))
        out["pitch_wobble_st"] = float(np.median(np.abs(np.diff(st))))       # frame-to-frame movement
    harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=60)
    hv = harm.values[harm.values > -100]
    out["hnr_db"] = float(np.median(hv)) if hv.size else 0.0
    pp = call(snd, "To PointProcess (periodic, cc)", 60, 250)
    out["jitter_pct"] = 100 * call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    out["shimmer_pct"] = 100 * call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5000)
    times = [pitch.xs()[i] for i in range(len(pitch.xs())) if pitch.selected_array["frequency"][i] > 0]
    for n in (1, 2, 3):
        vals = [formant.get_value_at_time(n, t) for t in times]
        vals = [v for v in vals if v == v]
        out[f"F{n}_hz"] = float(np.median(vals)) if vals else 0.0
    # Long-term spectrum relative to its own mean, so loudness doesn't count.
    spec = np.abs(librosa.stft(y.astype(np.float32), n_fft=2048, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    power = spec.mean(axis=1)
    bands = []
    for f in BANDS:
        sel = (freqs >= f / 2 ** (1 / 6)) & (freqs < f * 2 ** (1 / 6))
        bands.append(10 * np.log10(power[sel].mean() + 1e-12))
    bands = np.array(bands) - np.mean(bands)
    out["ltas_db"] = [round(float(b), 1) for b in bands]
    low = power[(freqs >= 80) & (freqs < 1000)].sum()
    high = power[(freqs >= 1000) & (freqs < 4000)].sum()
    out["tilt_db"] = float(10 * np.log10(high / low)) if low else 0.0
    # Room: how fast the level falls in the 150 ms after each burst of speech ends.
    db = librosa.amplitude_to_db(librosa.feature.rms(y=y.astype(np.float32), frame_length=512, hop_length=80)[0], ref=1.0)
    loud = db > db.max() - 25
    ends = np.flatnonzero(loud[:-1] & ~loud[1:])
    slopes = []
    for e in ends:
        seg = db[e:e + 30]                             # 30 frames * 5 ms = 150 ms
        if seg.size >= 10:
            slopes.append(np.polyfit(np.arange(seg.size) * 0.005, seg, 1)[0])
    out["decay_db_per_s"] = float(np.median(slopes)) if slopes else 0.0
    out["noise_floor_db"] = float(np.percentile(db, 5) - db.max())
    return out


# ---- the line, word by word --------------------------------------------------------------

# Word boundaries in the film line (seconds, after trimming silence at 40 dB), marked by hand
# from its pitch and loudness; a recogniser's word timings are too coarse for this.
FILM_WORDS = [("i'm", 0.06, 0.31), ("sorry", 0.33, 0.74), ("dave", 0.76, 1.10), ("i'm", 1.84, 2.06),
              ("afraid", 2.12, 2.52), ("i", 2.56, 2.80), ("can't", 2.96, 3.14), ("do", 3.16, 3.30),
              ("that", 3.30, 3.56)]
HOP = 0.01


def frames(y: np.ndarray) -> dict:
    """Per 10 ms: MFCCs, pitch in semitones above 100 Hz (NaN when unvoiced or too quiet), loudness."""
    snd = parselmouth.Sound(y, SR)
    pitch = snd.to_pitch_ac(time_step=HOP, pitch_floor=55, pitch_ceiling=160, octave_jump_cost=0.8,
                            voicing_threshold=0.5)
    inten = snd.to_intensity(minimum_pitch=60, time_step=HOP)
    n = int(snd.duration / HOP)
    ts = (np.arange(n) + 0.5) * HOP
    f0 = np.array([pitch.get_value_at_time(t) for t in ts])
    db = np.array([inten.get_value(t) for t in ts], dtype=float)
    top = np.nanmax(db[np.isfinite(db)]) if np.isfinite(db).any() else 0.0
    # Synthesised audio has true digital silence, which Praat reads as about -300 dB.
    db = np.clip(np.nan_to_num(db, nan=top - 60, neginf=top - 60), top - 60, None)
    st = 12 * np.log2(f0 / 100.0)
    st[(db < db.max() - 15) | ~np.isfinite(st)] = np.nan
    mfcc = librosa.feature.mfcc(y=y.astype(np.float32), sr=SR, n_mfcc=14, hop_length=int(HOP * SR),
                                n_fft=512)[1:, :n]
    return {"st": st, "db": db, "mfcc": mfcc, "n": min(n, mfcc.shape[1])}


def mapping(film: dict, cand: dict) -> np.ndarray:
    """For each film frame, the matching candidate frame, by dynamic time warping on MFCCs."""
    z = lambda x: (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)  # noqa: E731
    _, path = librosa.sequence.dtw(X=z(film["mfcc"][:, :film["n"]]), Y=z(cand["mfcc"][:, :cand["n"]]),
                                   metric="euclidean")
    path = path[::-1]
    out = np.zeros(film["n"], dtype=int)
    for i, j in path:
        out[i] = j
    return out


def speech_level(fr: dict) -> float:
    """Mean loudness of the speech alone, so silent stretches (room noise or digital zero) don't count."""
    db = fr["db"]
    return float(np.mean(db[db > db.max() - 25]))


def word_stats(fr: dict, start: int, end: int, line_db: float) -> dict:
    st = fr["st"][start:end]
    voiced = st[np.isfinite(st)]
    third = max(1, voiced.size // 3)
    return {"length_s": (end - start) * HOP,
            "pitch_st": float(np.median(voiced)) if voiced.size else None,
            "pitch_move_st": float(np.median(voiced[-third:]) - np.median(voiced[:third])) if voiced.size >= 6 else None,
            "loudness_db": float(np.mean(fr["db"][start:end]) - line_db)}


def melody(film: dict, cand: dict, m: np.ndarray) -> dict:
    a, b = film["st"][:len(m)], cand["st"][m]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return {"melody_error_st": None, "melody_shape_error_st": None, "pitch_offset_st": None}
    d = b[ok] - a[ok]
    return {"melody_error_st": float(np.sqrt(np.mean(d ** 2))),
            "melody_shape_error_st": float(np.sqrt(np.mean((d - np.median(d)) ** 2))),
            "pitch_offset_st": float(np.median(d))}


# ---- the film fingerprint and the comparison --------------------------------------------

def fingerprint() -> dict:
    """Mean and spread of every voice measure across the film clips."""
    rows = []
    for clip in sorted(CLIPS.glob("*.mp3")):
        y = load(clip)
        if len(y) > SR:                       # under a second is too short to measure
            rows.append(voice(y))
    keys = [k for k in rows[0] if k != "ltas_db" and all(k in r for r in rows)]
    fp = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in keys}
    ltas = np.array([r["ltas_db"] for r in rows])
    fp["ltas_db"] = (ltas.mean(0).round(1).tolist(), ltas.std(0).round(1).tolist())
    fp["clips"] = len(rows)
    return fp


UNITS = {"pitch_hz": "Hz", "pitch_range_st": "st", "pitch_wobble_st": "st", "hnr_db": "dB", "jitter_pct": "%",
         "shimmer_pct": "%", "F1_hz": "Hz", "F2_hz": "Hz", "F3_hz": "Hz", "tilt_db": "dB",
         "decay_db_per_s": "dB/s", "noise_floor_db": "dB"}
NAMES = {"pitch_hz": "pitch level", "pitch_range_st": "pitch range", "pitch_wobble_st": "pitch wobble (frame to frame)",
         "hnr_db": "clarity (harmonics-to-noise)", "jitter_pct": "jitter (pitch instability)",
         "shimmer_pct": "shimmer (loudness instability)", "F1_hz": "formant F1 (openness)",
         "F2_hz": "formant F2 (vowel front/back)", "F3_hz": "formant F3 (vocal tract length)",
         "tilt_db": "brightness (1-4 kHz vs below)", "decay_db_per_s": "room decay after words",
         "noise_floor_db": "noise floor"}


def compare(name: str, y: np.ndarray, fp: dict, film_fr: dict) -> dict:
    v = voice(y)
    fr = frames(y)
    m = mapping(film_fr, fr)
    gaps = []
    for k, ms in fp.items():
        if k in ("ltas_db", "clips") or k not in v:
            continue
        mean, sd = ms
        z = (v[k] - mean) / (sd or 1e-6)
        gaps.append((abs(z), f"{NAMES.get(k, k)}: {v[k]:.1f} {UNITS.get(k, '')} vs film {mean:.1f}±{sd:.1f} (z={z:+.1f})"))
    film_ltas, film_sd = np.array(fp["ltas_db"][0]), np.array(fp["ltas_db"][1])
    diff = np.array(v["ltas_db"]) - film_ltas
    for f, d, s in zip(BANDS, diff, film_sd):
        if abs(d) > max(2.0, 1.5 * s):
            gaps.append((abs(d) / max(s, 1.0), f"spectrum at {f} Hz: {d:+.1f} dB vs film"))
    mel = melody(film_fr, fr, m)
    film_db, cand_db = speech_level(film_fr), speech_level(fr)
    rows, prev_end, prev_cend = [], None, None
    for word, s, e in FILM_WORDS:
        fs, fe = int(s / HOP), min(int(e / HOP), len(m) - 1)
        cs, ce = int(m[fs]), int(m[max(fe - 1, fs)]) + 1
        fw, cw = word_stats(film_fr, fs, fe, film_db), word_stats(fr, cs, ce, cand_db)
        fgap = (fs - prev_end) * HOP if prev_end is not None else 0.0
        cgap = (cs - prev_cend) * HOP if prev_cend is not None else 0.0
        prev_end, prev_cend = fe, ce
        ratio = cw["length_s"] / max(fw["length_s"], 1e-3)
        fmt = lambda x: "  n/a" if x is None else f"{x:+5.1f}"  # noqa: E731
        rows.append(f"  {word:7} length {cw['length_s']:.2f}s vs {fw['length_s']:.2f}s ({(ratio - 1) * 100:+4.0f}%) | "
                    f"gap {cgap:.2f}s vs {fgap:.2f}s | pitch {fmt(cw['pitch_st'])} vs {fmt(fw['pitch_st'])} st | "
                    f"move {fmt(cw['pitch_move_st'])} vs {fmt(fw['pitch_move_st'])} st | "
                    f"loud {cw['loudness_db']:+.1f} vs {fw['loudness_db']:+.1f} dB")
        if abs(ratio - 1) > 0.25:
            gaps.append((abs(ratio - 1) * 4, f"'{word}' is {(ratio - 1) * 100:+.0f}% {'longer' if ratio > 1 else 'shorter'} than in the film"))
        if None not in (cw["pitch_move_st"], fw["pitch_move_st"]) and abs(cw["pitch_move_st"] - fw["pitch_move_st"]) > 1.5:
            gaps.append((abs(cw["pitch_move_st"] - fw["pitch_move_st"]),
                         f"'{word}' pitch moves {cw['pitch_move_st']:+.1f} st, the film {fw['pitch_move_st']:+.1f} st"))
        if abs(cgap - fgap) > 0.15:
            gaps.append((abs(cgap - fgap) * 10, f"pause before '{word}' is {cgap:.2f}s, the film {fgap:.2f}s"))
        if abs(cw["loudness_db"] - fw["loudness_db"]) > 4:
            gaps.append((abs(cw["loudness_db"] - fw["loudness_db"]) / 2,
                         f"'{word}' is {cw['loudness_db'] - fw['loudness_db']:+.1f} dB louder than the film, relative to the line"))
    gaps.sort(reverse=True)
    return {"name": name, "voice": v, "melody": mel, "length_s": fr["n"] * HOP, "gaps": [g for _, g in gaps],
            "word_rows": rows}


def match(y: np.ndarray, fp: dict, film_fr: dict) -> dict:
    """One take's distance from the film line, as named parts and a total (0 = the film itself).

    Each part is an error divided by roughly what is just noticeable, so 1.0 in any part is a
    clear difference: melody shape (1.5 st), pitch level (2 st), the pause after "Dave" (0.3 s),
    word lengths (25%), the loudness of each word against the line (4 dB), voice quality against
    HAL's fingerprint (1.5 SD) and the long-term spectrum (5 dB).
    """
    v, fr = voice(y), frames(y)
    m = mapping(film_fr, fr)
    mel = melody(film_fr, fr, m)
    film_db, cand_db = speech_level(film_fr), speech_level(fr)
    lens, louds, pause = [], [], 0.0
    prev_c = None
    for k, (word, s, e) in enumerate(FILM_WORDS):
        fs, fe = int(s / HOP), min(int(e / HOP), len(m) - 1)
        cs, ce = int(m[fs]), int(m[max(fe - 1, fs)]) + 1
        fw, cw = word_stats(film_fr, fs, fe, film_db), word_stats(fr, cs, ce, cand_db)
        lens.append(abs(np.log(max(cw["length_s"], 0.02) / max(fw["length_s"], 0.02))))
        louds.append(abs(cw["loudness_db"] - fw["loudness_db"]))
        if k == 3 and prev_c is not None:                       # the second "i'm"
            pause = abs((cs - prev_c) * HOP - (fs - int(FILM_WORDS[2][2] / HOP)) * HOP)
        prev_c = ce
    quality = [abs((v[k] - fp[k][0]) / (fp[k][1] or 1)) for k in
               ("pitch_range_st", "hnr_db", "jitter_pct", "shimmer_pct", "F1_hz", "F2_hz", "F3_hz", "tilt_db") if k in v]
    ltas = float(np.sqrt(np.mean((np.array(v["ltas_db"]) - np.array(fp["ltas_db"][0])) ** 2)))
    parts = {
        "melody_shape": (mel["melody_shape_error_st"] if mel["melody_shape_error_st"] is not None else 4.0) / 1.5,
        "pitch_level": abs(mel["pitch_offset_st"] if mel["pitch_offset_st"] is not None else 4.0) / 2.0,
        "pause": pause / 0.3,
        "word_lengths": float(np.mean(lens)) / 0.22,
        "word_loudness": float(np.mean(louds)) / 4.0,
        "voice_quality": float(np.mean(quality)) / 1.5,
        "spectrum": ltas / 5.0,
    }
    weights = {"melody_shape": 0.25, "pitch_level": 0.15, "pause": 0.1, "word_lengths": 0.1, "word_loudness": 0.1,
               "voice_quality": 0.15, "spectrum": 0.15}
    total = sum(weights[k] * parts[k] for k in parts)
    return {**{k: round(v_, 2) for k, v_ in parts.items()}, "distance": round(total, 3),
            "match_pct": round(100 * float(np.exp(-total)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", nargs="*", help="NAME=path.wav")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--fingerprint", type=Path, default=HERE / "tuning" / "film-fingerprint.json")
    args = ap.parse_args()
    if args.fingerprint.exists():
        fp = json.loads(args.fingerprint.read_text())
    else:
        fp = fingerprint()
        args.fingerprint.parent.mkdir(parents=True, exist_ok=True)
        args.fingerprint.write_text(json.dumps(fp, indent=1))
    film_fr = frames(load(CLIPS / f"{TARGET}.mp3"))
    print(f"\nThe line in the film lasts {film_fr['n'] * HOP:.2f}s")
    results = []
    for spec in args.candidates:
        name, path = spec.split("=", 1)
        r = compare(name, load(Path(path)), fp, film_fr)
        results.append(r)
        mel = r["melody"]
        print(f"\n=== {name}: {r['length_s']:.2f}s long | melody error "
              f"{mel['melody_error_st'] if mel['melody_error_st'] is None else round(mel['melody_error_st'], 2)} st, "
              f"shape error {mel['melody_shape_error_st'] if mel['melody_shape_error_st'] is None else round(mel['melody_shape_error_st'], 2)} st, "
              f"pitch offset {mel['pitch_offset_st'] if mel['pitch_offset_st'] is None else round(mel['pitch_offset_st'], 1)} st")
        print("Word by word (this voice vs the film):")
        print("\n".join(r["word_rows"]))
        print("Largest departures from the film:")
        for g in r["gaps"][:12]:
            print("  -", g)
    if args.json:
        args.json.write_text(json.dumps({"fingerprint": fp, "film_words": FILM_WORDS, "results": results}, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
