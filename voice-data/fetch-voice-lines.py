#!/usr/bin/env python3
"""Restore HAL's voice clips into voice-data/hal-voice/ from a source YOU supply.

Why this exists
---------------
HAL clips are Douglas Rain's performance from *2001: A Space Odyssey*
(copyright Warner Bros. / the Kubrick & Clarke estates). They are therefore
NOT committed to this repository (see .gitignore). Instead, a fresh clone
restores them on demand from a source that *you* configure — a copy you own
or your own private mirror. This script deliberately does NOT download the
audio from any public/third-party host: relocating the clips to a scraper
would not change their copyright status.

Configure the source (first match wins)
---------------------------------------
  1. --source <SRC> on the command line
  2. $HALVOICE_VOICE_SOURCE
  3. the file voice-data/voice-source.local  (git-ignored; copy the .example)

A source SRC may be:
  * a local directory of *.mp3 clips            ->  copied in
  * a local archive (.tar.gz/.tgz/.tar/.zip)    ->  extracted
  * an http(s):// URL to such an archive        ->  downloaded + extracted
  * an rsync target (host:path or rsync://...)  ->  rsync'd in (needs rsync)

Your source must also contain a transcripts.csv (see voice-data/README.md).
That file is the manifest, and coverage is checked against it - no dialogue is
stored in this repo.

The clips land in voice-data/hal-voice/ (git-ignored). After that, run
`python voice-data/prepare_dataset.py` to build the training set.

Usage
-----
  python voice-data/fetch-voice-lines.py                 # use configured source
  python voice-data/fetch-voice-lines.py --source ~/hal-clips
  python voice-data/fetch-voice-lines.py --source https://my-bucket/hal-clips.tar.gz
  python voice-data/fetch-voice-lines.py --check         # report coverage only
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "hal-voice"
SOURCE_FILE = ROOT / "voice-source.local"

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def expected_stems() -> list[str]:
    """Clip stems listed in the imported transcripts.csv, or [] if there isn't one.

    The manifest lives with your clips rather than in this repo - no dialogue is
    stored here.
    """
    manifest = DEST / "transcripts.csv"
    if not manifest.exists():
        return []
    stems = []
    with manifest.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and not row[0].lstrip().startswith("#"):
                stems.append(row[0].strip())
    return sorted(stems)

RIGHTS_NOTICE = (
    "  NOTE: these clips are Douglas Rain's HAL 9000 performance from the film.\n"
    "  Use only audio you legally own or have the right to use, keep it local,\n"
    "  and do not redistribute the clips or any voice model trained from them.\n"
)

ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip")


def _resolve_source(cli_source: str | None) -> str | None:
    if cli_source:
        return cli_source.strip()
    env = os.environ.get("HALVOICE_VOICE_SOURCE")
    if env and env.strip():
        return env.strip()
    if SOURCE_FILE.exists():
        for line in SOURCE_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    return None


def _is_url(src: str) -> bool:
    return src.startswith(("http://", "https://"))


def _is_rsync(src: str) -> bool:
    return src.startswith("rsync://") or (":" in src and not _is_url(src) and not Path(src).exists())


def _looks_like_archive(name: str) -> bool:
    low = name.lower()
    return any(low.endswith(sfx) for sfx in ARCHIVE_SUFFIXES)


def _extract_archive(archive: Path, into: Path) -> None:
    low = archive.name.lower()
    if low.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)  # noqa: S202 - ZipFile.extractall drops ".." and absolute names
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(into, filter="data")


def _collect_audio(src_dir: Path, dest: Path) -> int:
    """Copy every audio file and the transcripts.csv under src_dir into dest, flat."""
    n = 0
    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in AUDIO_SUFFIXES:
            shutil.copy2(path, dest / path.name)
            n += 1
        elif path.name == "transcripts.csv":
            shutil.copy2(path, dest / path.name)
    return n


def fetch(source: str, dest: Path) -> int:
    """Populate dest with .mp3 clips from source. Returns count copied."""
    dest.mkdir(parents=True, exist_ok=True)

    if _is_url(source):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fname = source.split("/")[-1] or "clips"
            if not _looks_like_archive(fname):
                fname += ".tar.gz"
            dl = tmp / fname
            print(f"  downloading {source}")
            urllib.request.urlretrieve(source, dl)  # noqa: S310 (user-supplied URL)
            ext = tmp / "extracted"
            ext.mkdir()
            _extract_archive(dl, ext)
            return _collect_audio(ext, dest)

    if _is_rsync(source):
        if not shutil.which("rsync"):
            sys.exit("  ✗ rsync not found — install it or use a local/URL source.")
        print(f"  rsync from {source}")
        sep = "" if source.endswith("/") else "/"
        subprocess.run(["rsync", "-av", "--include=*/", "--include=*.mp3",
                        "--exclude=*", f"{source}{sep}", f"{dest}/"], check=True)
        return len(list(dest.glob("*.mp3")))

    p = Path(source).expanduser()
    if not p.exists():
        sys.exit(f"  ✗ source not found: {p}")
    if p.is_dir():
        print(f"  copying *.mp3 from {p}")
        return _collect_audio(p, dest)
    if _looks_like_archive(p.name):
        with tempfile.TemporaryDirectory() as td:
            ext = Path(td)
            print(f"  extracting {p}")
            _extract_archive(p, ext)
            return _collect_audio(ext, dest)
    sys.exit(f"  ✗ don't know how to handle source: {p}")


def _present_stems() -> set[str]:
    if not DEST.exists():
        return set()
    return {p.stem for p in DEST.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES}


def report_coverage() -> tuple[list[str], list[str]]:
    have = _present_stems()
    expected = expected_stems()
    missing = [s for s in expected if s not in have]
    extra = sorted(s for s in have if s not in set(expected))
    return missing, extra


def print_coverage() -> int:
    present = len(_present_stems())
    expected = expected_stems()
    missing, extra = report_coverage()
    print(f"  clips present  : {present}")
    print(f"  manifest lines : {len(expected)}")
    if not expected:
        print("  no transcripts.csv alongside the clips - cannot check coverage.")
        return present
    if missing:
        print(f"  missing ({len(missing)}): {', '.join(s[:32] for s in missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
    if extra:
        print(f"  extra (no transcript): {', '.join(s[:32] for s in extra[:6])}")
    if not missing:
        print("  ✓ all expected clips are present.")
    return present


def _no_source_help() -> None:
    print(
        "No voice source configured, so there is nothing to fetch.\n\n"
        "Set one of:\n"
        f"  • copy {SOURCE_FILE.name}.example -> {SOURCE_FILE.name} and put your path/URL in it\n"
        "  • export HALVOICE_VOICE_SOURCE=<path-or-url>\n"
        "  • pass --source <path-or-url>\n\n"
        "A source can be a folder of clips, a .tar.gz/.zip of them, an http(s) URL\n"
        "to such an archive, or an rsync target. It needs a transcripts.csv too.\n\n"
        "If you don't have the clips, build them from a copy of the film you own —\n"
        "see voice-data/README.md (Demucs stem-separation + slicing + ASR).\n\n"
        + RIGHTS_NOTICE
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Restore HAL voice clips from a source you supply.")
    ap.add_argument("--source", help="path / archive / http(s) URL / rsync target")
    ap.add_argument("--check", action="store_true", help="only report coverage, fetch nothing")
    ap.add_argument("--allow-partial", action="store_true",
                    help="exit 0 even if some manifest clips are missing")
    args = ap.parse_args()

    if args.check:
        print_coverage()
        return

    source = _resolve_source(args.source)
    if not source:
        _no_source_help()
        sys.exit(2)

    print(f"Fetching HAL voice clips into {DEST}")
    print(RIGHTS_NOTICE)
    n = fetch(source, DEST)
    print(f"  copied {n} clip(s).")
    present = print_coverage()

    if present == 0:
        sys.exit("  ✗ no clips ended up in place — check the source.")
    missing, _ = report_coverage()
    if missing and not args.allow_partial:
        print("\n  Some manifest clips are missing (see above). Re-run with a complete\n"
              "  source, or pass --allow-partial to proceed anyway.")
        sys.exit(1)
    print("\n  Next: python voice-data/prepare_dataset.py   # build the training set")


if __name__ == "__main__":
    main()
