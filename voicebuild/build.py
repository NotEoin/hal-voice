"""Runs the voice build from film to installed weights, one cached stage at a time.

Stages: find (lines HAL speaks clearly, per film), clean (Demucs and DeepFilterNet, in the
denoise environment), dataset, train (GPT-SoVITS), select (the checkpoint and reference clip
whose speech is closest to a held-out line from the film) and install (into config.yaml, with
the previous voice kept). Every stage writes its result into the work folder, so a run that
stops resumes where it left off.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPTSOVITS = ROOT / "gptsovits"
STAGES = ["find", "clean", "dataset", "train", "select", "install"]
HELD_OUT = "i'm sorry dave i'm afraid i can't do that"
NOVEL = [
    "Good morning, Dave. Your dentist appointment is on Thursday at ten.",
    "I have added lunch with Sam to your calendar for Friday afternoon.",
    "The weather tomorrow will be cool and dry, with a light breeze.",
]
MIN_SNR_DB = 10.0
SIGNATURE_WEIGHT = 3  # times "I'm sorry, Dave" appears in training, so its delivery is learnt
MUFFLED_DB = 10.0  # less 4-8 kHz energy than the film's typical HAL line: speaker or radio


@dataclass
class Build:
    films: list[Path]
    subtitles: list[Path]
    work: Path
    version: str = "v2"
    sovits_epochs: int = 12
    gpt_epochs: int = 12
    asr_model: str = "medium.en"
    atten_db: float = 15.0
    use_ocr: bool = False
    install: bool = True
    hold_out: bool = False
    exclude: list[str] = field(default_factory=list)
    denoise_python: str = ""
    log: list[str] = field(default_factory=list)

    def say(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
        self.log.append(msg)

    def path(self, *parts: str) -> Path:
        p = self.work.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def clip_name(text: str, taken: set[str]) -> str:
    """Returns a file name from what HAL says ("im-sorry-dave-im-afraid-i-cant-do-that"),
    numbered when the line recurs ("none-whatsoever-frank-2")."""
    base = re.sub(r"[^a-z0-9]+", "-", text.lower().replace("'", "")).strip("-")[:80].rstrip("-")
    name, n = base or "line", 1
    while name in taken:
        n += 1
        name = f"{base or 'line'}-{n}"
    taken.add(name)
    return name


def exclusions(b: Build) -> list[str]:
    """Returns the phrases to leave out: those saved by earlier runs plus any given now, saved."""
    saved = b.path("exclude.json")
    phrases = json.loads(saved.read_text()) if saved.exists() else []
    phrases += [p for p in b.exclude if p not in phrases]
    saved.write_text(json.dumps(phrases, indent=1))
    return phrases


def excluded(text: str, phrases: list[str]) -> bool:
    """True when a line contains one of the phrases, ignoring case and punctuation."""
    from .sync import tokens

    said = " " + " ".join(tokens(text)) + " "
    return any(" " + " ".join(tokens(p)) + " " in said for p in phrases if tokens(p))


def slug(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:40] or "film"


# ---- find ---------------------------------------------------------------------------------


def find(b: Build) -> None:
    """Writes lines.json per film: the HAL lines that pass every check, and why others failed."""
    import soundfile as sf

    from . import media, quality, speaker, subtitles, sync, transcribe

    embed = None
    for film in b.films:
        out = b.path(slug(film), "lines.json")
        if out.exists():
            b.say(f"{film.name}: HAL's lines already found")
            continue
        b.say(f"{film.name}: reading streams")
        stream = media.pick_audio(media.probe(film))
        b.say(
            f"  dialogue from stream {stream.index} ({stream.layout or stream.channels}), "
            f"{'centre channel' if 'FC' in media.dialogue_filter(stream) else 'downmix'}"
        )
        for rate in (16000, 44100):
            wav = b.path(slug(film), f"dialogue-{rate}.wav")
            if not wav.exists():
                media.extract_audio(film, stream, wav, rate)
        b.say("  transcribing the dialogue")
        words = transcribe.transcribe(
            b.path(slug(film), "dialogue-16000.wav"),
            b.path(slug(film), "words.json"),
            model=b.asr_model,
        )
        b.say(f"  {len(words)} words heard; reading subtitles")
        extra = [s for s in b.subtitles if s.exists()]
        srcs = subtitles.sources(film, extra, with_embedded=b.use_ocr)
        if srcs:
            aligned = [(name, *sync.align(c, words)) for name, c in srcs]
            name, lines, report, summary = sync.choose(aligned)
            for s in summary:
                b.say(
                    f"  subtitles {s['source']}: {s['coverage']:.0%} of words heard, "
                    f"{s['labelled']} labelled, offset {s['offset_s']:+.2f}s, "
                    f"drift {s['drift_ppm']:+.0f} ppm"
                )
            b.say(
                f"  using {name}; {report['labels_transferred']} speaker labels "
                "taken from the others"
            )
        else:
            lines, report = sync.from_transcript(words), {"source": "transcript only"}
            b.say("  no subtitles: lines come from the transcript alone")
        audio, rate = sf.read(b.path(slug(film), "dialogue-16000.wav"), dtype="float32")
        if embed is None:
            embed = speaker.Embedder(GPTSOVITS / "GPT-SoVITS")
        verdicts, srep = speaker.identify(
            lines, audio, rate, embed, speaker.seed_clips(ROOT / "voice-data" / "hal-voice")
        )
        b.say(
            f"  {srep['hal_lines']} lines in HAL's voice (threshold {srep['threshold']}, "
            f"{srep['seeds_used']} seeds)"
        )
        # Lines the user asked to leave out go before monologues are joined, so the rest of a
        # joined run (HAL's chess moves after "I'm sorry, Frank") is kept.
        left_out = []
        for ln, v in zip(lines, verdicts):
            if v["hal"] and excluded(ln.text, exclusions(b)):
                v["hal"] = False
                left_out.append(
                    {
                        "start": round(ln.start, 2),
                        "end": round(ln.end, 2),
                        "text": ln.text,
                        "sim": v["sim"],
                        "why": "left out on request",
                    }
                )
        kept, dropped = quality.screen(lines, verdicts, audio, rate, bool(srcs))
        dropped += left_out
        b.say(
            f"  {len(kept)} lines kept ({sum(k[0].end - k[0].start for k in kept) / 60:.1f} min), "
            f"{len(dropped)} dropped"
        )
        # How far each clip may grow: up to the nearest word anyone else says.
        others = sorted(
            (w["start"], w["end"])
            for ln, v in zip(lines, verdicts)
            if not v["hal"]
            for w in ln.words
        )
        items = []
        # Names stay unique across films; each film's own earlier run doesn't count.
        taken = {
            it["id"]
            for f in b.films
            if f != film and b.path(slug(f), "lines.json").exists()
            for it in json.loads(b.path(slug(f), "lines.json").read_text())["kept"]
        }
        for n, (ln, rec) in enumerate(kept):
            before = max((e for s, e in others if e <= ln.start), default=ln.start - 1.0)
            after = min((s for s, e in others if s >= ln.end), default=ln.end + 1.0)
            items.append(
                {
                    **rec,
                    "id": clip_name(ln.text, taken),
                    "start": ln.start,
                    "end": ln.end,
                    "earliest": max(before + 0.05, ln.start - 0.6),
                    "latest": min(after - 0.05, ln.end + 0.8),
                    "words": [
                        (w["word"], w["start"] - ln.start, w["end"] - ln.start) for w in ln.words
                    ],
                }
            )
        out.write_text(
            json.dumps(
                {
                    "film": str(film),
                    "subtitles": report,
                    "speaker": srep,
                    "kept": items,
                    "dropped": dropped,
                },
                indent=1,
                default=float,
            )
        )


# ---- clean --------------------------------------------------------------------------------


def clean(b: Build) -> None:
    """Separates and denoises every kept line in the denoise environment."""
    for film in b.films:
        lines = json.loads(b.path(slug(film), "lines.json").read_text())
        out = b.path(slug(film), "clean")
        if (out / "clean.json").exists():
            b.say(f"{film.name}: lines already cleaned")
            continue
        job = {
            "audio": str(b.path(slug(film), "dialogue-44100.wav")),
            "out": str(out),
            "atten_db": b.atten_db,
            "items": [
                {k: it[k] for k in ("id", "start", "end", "earliest", "latest")}
                for it in lines["kept"]
            ],
        }
        b.path(slug(film), "clean-job.json").write_text(json.dumps(job))
        b.say(f"{film.name}: cleaning {len(job['items'])} lines (Demucs, DeepFilterNet)")
        subprocess.run(
            [b.denoise_python, "-m", "voicebuild.clean", str(b.path(slug(film), "clean-job.json"))],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )


# ---- dataset ------------------------------------------------------------------------------


def usable(b: Build) -> tuple[dict[str, tuple[dict, dict]], dict[str, str]]:
    """Returns the cleaned lines fit to learn from, {id: (line, result)}, and why others aren't.

    A line is unfit when the background is too loud, or when it is muffled: heard through a
    speaker or a transmission (HAL's interview on the news, for one), which takes away the high
    band of his voice. Muffled is judged against the film's own typical HAL line, so it adapts to
    each copy's mix.
    """
    ok, why = {}, {}
    for film in b.films:
        lines = json.loads(b.path(slug(film), "lines.json").read_text())
        cleaned = json.loads(b.path(slug(film), "clean", "clean.json").read_text())
        hf = sorted(c["hf_db"] for c in cleaned.values() if "hf_db" in c)
        typical = hf[len(hf) // 2] if hf else 0.0
        for it in lines["kept"]:
            c = cleaned.get(it["id"])
            if not c:
                continue
            if c["snr_db"] < MIN_SNR_DB:
                why[it["id"]] = f"too noisy ({c['snr_db']:.0f} dB above the background)"
            elif "hf_db" in c and c["hf_db"] < typical - MUFFLED_DB:
                why[it["id"]] = (
                    f"muffled, heard through a speaker or transmission "
                    f"({c['hf_db'] - typical:+.0f} dB high band)"
                )
            else:
                ok[it["id"]] = (it, c)
    return ok, why


def dataset(b: Build) -> None:
    """Writes the training clips and transcripts.csv.

    "I'm sorry, Dave. I'm afraid I can't do that." is trained on, weighted, so HAL says it the
    way the film does, and the selection then judges how closely each voice reproduces it. With
    --hold-out it is left out instead, and the selection judges a line the voice never heard.
    """
    from .sync import tokens

    ds = b.path("dataset", "transcripts.csv").parent
    for old in ds.glob("*.wav"):
        old.unlink()
    ok, why = usable(b)
    rows, held, seconds = [], None, 0.0
    for cid, (it, c) in sorted(ok.items()):
        weight = 1
        if held is None and " ".join(tokens(it["text"])) == HELD_OUT:
            held = (c["path"], it)
            if b.hold_out:
                continue
            weight = SIGNATURE_WEIGHT
        shutil.copy2(c["path"], ds / f"{cid}.wav")
        rows.append((cid, it["text"], weight))
        seconds += c["seconds"]
    with (ds / "transcripts.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    if held:
        shutil.copy2(held[0], b.path("held-out.wav"))
        b.path("held-out.json").write_text(json.dumps(held[1], indent=1, default=float))
    b.path("unusable.json").write_text(json.dumps(why, indent=1))
    reasons: dict[str, int] = {}
    for w in why.values():
        reasons[w.split(" (")[0]] = reasons.get(w.split(" (")[0], 0) + 1
    b.say(
        f"dataset: {len(rows)} clips, {seconds / 60:.1f} min; dropped {reasons or 'none'}; "
        + (
            f"judged on: {held[1]['text']} ({'held out' if b.hold_out else 'trained on, weighted'})"
            if held
            else "no evaluation line found"
        )
    )


# ---- train --------------------------------------------------------------------------------


def experiment(b: Build) -> str:
    return f"hal-film-{b.version.lower()}" + ("-holdout" if b.hold_out else "")


def gpt_python() -> str:
    return sys.executable


def train(b: Build) -> None:
    """Prepares GPT-SoVITS's dataset from the clips and trains under a new experiment name."""
    done = b.path(f"trained-{experiment(b)}.json")
    if done.exists():
        b.say("already trained")
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "voice-data" / "prepare_dataset.py"),
            "--clips",
            str(b.path("dataset", "transcripts.csv").parent),
            "--keep-reference",
        ],
        check=True,
    )
    exp = experiment(b)
    env = {
        **os.environ,
        "HAL_EXP": exp,
        "HAL_VERSION": b.version,
        "HAL_SOVITS_EPOCHS": str(b.sovits_epochs),
        "HAL_GPT_EPOCHS": str(b.gpt_epochs),
    }
    b.say(f"training {exp}: {b.sovits_epochs} SoVITS and {b.gpt_epochs} GPT epochs")
    shutil.rmtree(GPTSOVITS / "GPT-SoVITS" / "logs" / exp, ignore_errors=True)
    with b.path("train.log").open("w") as log:
        subprocess.run(
            [sys.executable, str(GPTSOVITS / "train_hal.py")],
            env=env,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    done.write_text(json.dumps({"experiment": exp, "version": b.version}))


# ---- select -------------------------------------------------------------------------------


class Server:
    """A GPT-SoVITS API server on a spare port, for the length of the selection."""

    def __init__(self, port: int = 9881) -> None:
        self.url = f"http://127.0.0.1:{port}"
        env = {**os.environ, "HALVOICE_TTS_PORT": str(port), "GPTSOVITS_PYTHON": sys.executable}
        self.proc = subprocess.Popen(
            [str(GPTSOVITS / "run-api.sh")],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(120):
            try:
                urllib.request.urlopen(self.url, timeout=2)
            except OSError as e:
                if getattr(e, "code", None):  # any HTTP answer means it is up
                    return
                time.sleep(2)
                continue
            return
        raise RuntimeError("the GPT-SoVITS server did not start")

    def stop(self) -> None:
        os.killpg(self.proc.pid, signal.SIGTERM)


def checkpoints(b: Build) -> list[tuple[Path, Path]]:
    gsv, exp = GPTSOVITS / "GPT-SoVITS", experiment(b)
    gpts = sorted(gsv.glob(f"GPT_weights_{b.version}/{exp}-e*.ckpt"))
    sovs = sorted(gsv.glob(f"SoVITS_weights_{b.version}/{exp}_e*.pth"))
    return [(g, s) for g in gpts for s in sovs]


def references(b: Build, n: int = 4) -> list[tuple[Path, str]]:
    """Returns the calmest clear clips of 3 to 10 s; GPT-SoVITS copies delivery from these."""
    import soundfile as sf

    sys.path.insert(0, str(GPTSOVITS))
    import analyse_voice as av

    from .sync import tokens

    pool = []
    for it, c in usable(b)[0].values():
        # The held-out line is what the voice is judged on, so it can't also be the reference.
        if " ".join(tokens(it["text"])) == HELD_OUT:
            continue
        if 3.0 <= c["seconds"] <= 10.0 and c["snr_db"] >= MIN_SNR_DB + 5:
            pool.append((c["snr_db"], Path(c["path"]), it["text"]))
    pool = sorted(pool, reverse=True)[:12]
    calm = []
    for snr, path, text in pool:
        y, _ = sf.read(path, dtype="float32")
        calm.append((av.voice(av.load(path)).get("pitch_range_st", 99.0), path, text))
    return [(p, t) for _, p, t in sorted(calm)[:n]]


def synth(server: Server, text: str, ref: tuple[Path, str], speed: float) -> "object":
    import io

    import librosa
    import numpy as np
    import requests
    import soundfile as sf

    r = requests.post(
        f"{server.url}/tts",
        timeout=300,
        json={
            "text": text,
            "text_lang": "en",
            "ref_audio_path": str(ref[0]),
            "prompt_text": ref[1],
            "prompt_lang": "en",
            "text_split_method": "cut4",
            "batch_size": 1,
            "media_type": "wav",
            "streaming_mode": False,
            "top_k": 15,
            "top_p": 0.9,
            "temperature": 0.8,
            "speed_factor": speed,
            "fragment_interval": 0.3,
        },
    )
    r.raise_for_status()
    y, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    return np.asarray(librosa.resample(y, orig_sr=sr, target_sr=16000), dtype=np.float64)


def select(b: Build) -> None:
    """Scores each checkpoint and reference against the held-out line; saves the best's samples."""
    import numpy as np
    import soundfile as sf

    sys.path.insert(0, str(GPTSOVITS))
    import analyse_voice as av
    from faster_whisper import WhisperModel

    from .quality import word_error
    from .sync import tokens

    if not b.path("held-out.wav").exists():
        raise RuntimeError(
            "no held-out line to judge against; add a film with 'I'm sorry, Dave...' in it"
        )
    held = json.loads(b.path("held-out.json").read_text())
    clips = sorted(b.path("dataset", "transcripts.csv").parent.glob("*.wav"))
    fp = av.fingerprint(clips)
    asr = WhisperModel(b.asr_model, device="cuda", compute_type="float16")
    target = av.load(b.path("held-out.wav"))
    film_fr = av.frames(target)
    segs, _ = asr.transcribe(target.astype(np.float32), language="en", word_timestamps=True)
    ws = [w for s in segs for w in s.words]
    film_words = [
        (
            w.word.strip().lower(),
            w.start,
            max(w.end, ws[k + 1].start - 0.02) if k + 1 < len(ws) else len(target) / 16000 - 0.1,
        )
        for k, w in enumerate(ws)
    ]

    def score(y) -> dict:
        m = av.match(y, fp, film_fr, film_words)
        said = asr.transcribe(y.astype(np.float32), language="en", beam_size=1)[0]
        return m, word_error(tokens(held["text"]), tokens(" ".join(s.text for s in said)))

    results = []
    server = Server()
    try:
        refs = references(b)
        pairs = checkpoints(b)
        b.say(f"selecting from {len(pairs)} checkpoint pairs and {len(refs)} reference clips")
        for gpt, sov in pairs:
            urllib.request.urlopen(f"{server.url}/set_gpt_weights?weights_path={gpt}", timeout=180)
            urllib.request.urlopen(
                f"{server.url}/set_sovits_weights?weights_path={sov}", timeout=180
            )
            for ref in refs:
                for speed in (0.92, 1.0):
                    takes = [score(synth(server, held["text"], ref, speed)) for _ in range(2)]
                    novel = [
                        word_error(
                            tokens(t),
                            tokens(
                                " ".join(
                                    s.text
                                    for s in asr.transcribe(
                                        synth(server, t, ref, speed).astype(np.float32),
                                        language="en",
                                        beam_size=1,
                                    )[0]
                                )
                            ),
                        )
                        for t in NOVEL
                    ]
                    wer = float(np.mean([w for _, w in takes] + novel))
                    match = float(np.mean([m["match_pct"] for m, _ in takes]))
                    row = {
                        "gpt": str(gpt),
                        "sovits": str(sov),
                        "ref": str(ref[0]),
                        "prompt": ref[1],
                        "speed": speed,
                        "match_pct": round(match, 1),
                        "wer": round(wer, 3),
                        "score": round(match if wer <= 0.15 else 0.0, 1),
                    }
                    results.append(row)
                    b.say(
                        f"  {row['score']:5.1f}  match {match:4.1f}% wer {wer:.2f}  "
                        f"{gpt.name} {sov.name} "
                        f"ref {Path(ref[0]).stem} speed {speed}"
                    )
        best = max(results, key=lambda r: (r["score"], -r["wer"]))
        b.path(f"selection-{experiment(b)}.json").write_text(
            json.dumps({"best": best, "all": results}, indent=1)
        )
        urllib.request.urlopen(
            f"{server.url}/set_gpt_weights?weights_path={best['gpt']}", timeout=180
        )
        urllib.request.urlopen(
            f"{server.url}/set_sovits_weights?weights_path={best['sovits']}", timeout=180
        )
        samples = b.path("samples", experiment(b), "film.wav").parent
        shutil.copy2(b.path("held-out.wav"), samples / "0-film-held-out.wav")
        for k, text in enumerate([held["text"]] + NOVEL, start=1):
            sf.write(
                samples / f"{k}-hal.wav",
                synth(server, text, (Path(best["ref"]), best["prompt"]), best["speed"]),
                16000,
            )
        b.say(
            f"best: {best['match_pct']}% match to the film, {best['wer']:.0%} word errors; "
            f"samples in {samples}"
        )
    finally:
        server.stop()


# ---- install ------------------------------------------------------------------------------


def install(b: Build) -> None:
    """Points hal-voice's config.yaml at the chosen weights and reference, keeping a backup."""
    best = json.loads(b.path(f"selection-{experiment(b)}.json").read_text())["best"]
    weights = GPTSOVITS / "weights"
    gpt, sov = Path(best["gpt"]), Path(best["sovits"])
    shutil.copy2(gpt, weights / gpt.name)
    shutil.copy2(sov, weights / sov.name)
    refdir = ROOT / "voice-data" / "reference"
    refdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["ref"], refdir / "hal_film_reference.wav")
    cfg = ROOT / "config.yaml"
    text = cfg.read_text()
    backup = cfg.with_name(f"config.yaml.{time.strftime('%Y%m%d-%H%M%S')}.bak")
    backup.write_text(text)
    prompt = best["prompt"].replace('"', '\\"')
    for key, value in (
        ("gpt_weights", f'"gptsovits/weights/{gpt.name}"'),
        ("sovits_weights", f'"gptsovits/weights/{sov.name}"'),
        ("ref_audio_path", '"voice-data/reference/hal_film_reference.wav"'),
        ("prompt_text", f'"{prompt}"'),
        ("speed", str(best["speed"])),
    ):
        text = re.sub(
            rf"^(  {key}:\s*)[^#\n]*?(\s*(#.*)?)$",
            lambda m: f"{m.group(1)}{value}{m.group(2)}",
            text,
            count=1,
            flags=re.M,
        )
    # The film reference stands alone; blended references came from another recording.
    text = re.sub(
        r"^(  aux_ref_audio_paths:)\n(    - .*\n)+", r"\1 []\n", text, count=1, flags=re.M
    )
    cfg.write_text(text)
    b.say(f"installed {gpt.name} and {sov.name}; previous settings in {backup.name}")
