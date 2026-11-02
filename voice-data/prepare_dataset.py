#!/usr/bin/env python3
"""Build the GPT-SoVITS fine-tuning dataset from the clips in voice-data/hal-voice/.

Reads your clips plus the transcripts.csv beside them, converts each to a clean
32 kHz mono slice, writes the GPT-SoVITS label file and picks the runtime
reference clip. No dialogue is stored in this repo - see voice-data/README.md
for the CSV format and the weight column.
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "hal-voice"
DENOISED = ROOT / "denoised"
SLICES = ROOT / "clean" / "slices"
LIST_FILE = ROOT / "clean" / "hal.list"
REF_DIR = ROOT / "reference"
TRANSCRIPTS_FILE = SRC / "transcripts.csv"

# Speaker label in the .list file. GPT-SoVITS only uses it to group items.
SPEAKER = "HAL"
FFMPEG_TIMEOUT_S = 600  # one clip is seconds long; this only catches a hang

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def load_transcripts() -> dict[str, tuple[str, int]]:
    """{stem: (text, weight)} from transcripts.csv."""
    if not TRANSCRIPTS_FILE.exists():
        sys.exit(
            f"No transcripts at {TRANSCRIPTS_FILE}\n"
            "Write a two-column CSV of 'clip-stem,spoken text' next to your clips.\n"
            "See voice-data/README.md."
        )
    out: dict[str, tuple[str, int]] = {}
    with TRANSCRIPTS_FILE.open(newline="", encoding="utf-8") as fh:
        for lineno, row in enumerate(csv.reader(fh), start=1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 2:
                sys.exit(f"{TRANSCRIPTS_FILE}:{lineno}: expected at least 'stem,text'")
            stem, text = row[0].strip(), row[1].strip()
            weight = 1
            if len(row) > 2 and row[2].strip():
                try:
                    weight = int(row[2])
                except ValueError:
                    sys.exit(f"{TRANSCRIPTS_FILE}:{lineno}: weight must be an integer")
            out[stem] = (text, weight)
    return out


def pick_reference(transcripts: dict[str, tuple[str, int]]) -> str:
    """The clip used for runtime conditioning.

    GPT-SoVITS conditions on a short reference clip at synthesis time even after
    fine-tuning, so this sets the tone of every line it ever speaks. Pick the
    flattest, most unhurried one you have.

    Override by naming a clip stem in voice-data/reference.txt; otherwise the
    highest-weighted clip wins, since that is the delivery you asked for.
    """
    override = ROOT / "reference.txt"
    if override.exists():
        stem = override.read_text(encoding="utf-8").split()[0]
        if stem not in transcripts:
            sys.exit(f"reference.txt names '{stem}', which isn't in transcripts.csv")
        return stem
    return max(transcripts, key=lambda s: (transcripts[s][1], s))


def find_clip(stem: str) -> Path | None:
    """Returns the audio file for a clip name, or None."""
    for suffix in AUDIO_SUFFIXES:
        candidate = AUDIO / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def to_wav(src: Path, dst: Path) -> None:
    """32 kHz mono 16-bit PCM, light high-pass to drop rumble."""
    if src.parent != SRC:
        # denoise.py already wrote clean 32 kHz PCM; filtering again would only colour it.
        shutil.copy2(src, dst)
        return
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "32000",
            "-af",
            "highpass=f=70",
            "-sample_fmt",
            "s16",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=FFMPEG_TIMEOUT_S,
    )


AUDIO = SRC


def main() -> None:
    """Runs the command line."""
    global AUDIO, SRC, TRANSCRIPTS_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["raw", "denoised"],
        default="raw",
        help="train on the clips as fetched, or on denoise.py's output",
    )
    parser.add_argument(
        "--source-dir", type=Path, help="train on cleaned clips in this folder (overrides --source)"
    )
    parser.add_argument(
        "--clips",
        type=Path,
        help="a folder of clips with their own transcripts.csv, such as make-hal-voice's dataset",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="clip stems to hold out of training"
    )
    parser.add_argument(
        "--keep-reference",
        action="store_true",
        help="leave voice-data/reference/ (the live voice's references) untouched",
    )
    args = parser.parse_args()
    if args.clips:
        SRC = args.clips.resolve()
        TRANSCRIPTS_FILE = SRC / "transcripts.csv"
    AUDIO = (
        args.source_dir.resolve()
        if args.source_dir
        else DENOISED
        if args.source == "denoised"
        else SRC
    )
    if not AUDIO.exists():
        sys.exit(f"No clips at {AUDIO}. Run denoise.py first.")
    if not SRC.exists():
        sys.exit(f"No clips at {SRC}. Run fetch-voice-lines.py, or drop them there yourself.")

    transcripts = load_transcripts()

    # Rebuild slices from scratch so duplicates from a prior weighting don't linger.
    if SLICES.exists():
        shutil.rmtree(SLICES)
    SLICES.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    LIST_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    missing: list[str] = []
    excluded: list[str] = []
    n_clips = n_items = 0

    for stem, (text, weight) in sorted(transcripts.items()):
        clip = find_clip(stem)
        if clip is None:
            missing.append(stem)
            continue
        if weight <= 0 or stem in args.exclude:
            excluded.append(stem)
            continue

        n_clips += 1
        # Base copy + (weight - 1) duplicates under distinct names, so GPT-SoVITS
        # treats each as its own training item.
        base = SLICES / f"{stem}.wav"
        to_wav(clip, base)
        rows.append(f"{base.resolve()}|{SPEAKER}|en|{text}")
        n_items += 1
        for i in range(2, weight + 1):
            dup = SLICES / f"{stem}__x{i}.wav"
            shutil.copy2(base, dup)
            rows.append(f"{dup.resolve()}|{SPEAKER}|en|{text}")
            n_items += 1

    if not rows:
        sys.exit("Nothing to train on - no clip matched a transcript.")

    LIST_FILE.write_text("\n".join(rows) + "\n", encoding="utf-8")

    if args.keep_reference:
        print(f"  unique clips   : {n_clips}  (weighted to {n_items} training items)")
        print(f"  label file     : {LIST_FILE}")
        print(f"  held out       : {', '.join(excluded) or 'nothing'}")
        return

    ref_stem = pick_reference(transcripts)
    ref_src = find_clip(ref_stem)
    if ref_src is None:
        sys.exit(f"Reference clip '{ref_stem}' has a transcript but no audio.")
    to_wav(ref_src, REF_DIR / "hal_calm_reference.wav")
    # Further stems in reference.txt are auxiliary references, blended in for timbre.
    for old_aux in REF_DIR.glob("hal_aux_*.wav"):
        old_aux.unlink()
    aux = (
        (ROOT / "reference.txt").read_text(encoding="utf-8").split()[1:]
        if (ROOT / "reference.txt").exists()
        else []
    )
    for i, stem in enumerate(aux, start=1):
        clip = find_clip(stem)
        if clip is None:
            sys.exit(f"reference.txt names auxiliary '{stem}', which has no audio")
        to_wav(clip, REF_DIR / f"hal_aux_{i}.wav")
    if aux:
        print(f"  auxiliary refs : {len(aux)} (voice-data/reference/hal_aux_*.wav)")

    print(f"  unique clips   : {n_clips}  (weighted to {n_items} training items)")
    print(f"  label file     : {LIST_FILE}")
    print(f"  reference clip : {REF_DIR / 'hal_calm_reference.wav'}  (from {ref_stem})")
    print(f"  reference text : {transcripts[ref_stem][0]}")
    print("  -> copy that text into tts.prompt_text in config.yaml")
    if excluded:
        print(f"  excluded (weight 0): {len(excluded)}")
    if missing:
        print(f"  WARNING no audio for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
