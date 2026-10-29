"""Runs the voice build from film to installed weights, one cached stage at a time.

Stages: find (lines HAL speaks clearly, per film), clean (Demucs and DeepFilterNet, in the
denoise environment), dataset, train (GPT-SoVITS), select (the checkpoints, reference clip and
pace, with samples to compare with the film) and install (settings into
~/.config/hal-voice/built.yaml, with the previous voice kept). Every stage writes its result
into the work folder, so a run that stops resumes where it left off.
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

from halvoice import config

ROOT = Path(__file__).resolve().parents[1]
GPTSOVITS = ROOT / "gptsovits"
STAGES = ["find", "clean", "dataset", "train", "select", "install"]
HELD_OUT = "i'm sorry dave i'm afraid i can't do that"
NOVEL = [
    "Good morning, Dave. Your dentist appointment is on Thursday at ten.",
    "I have added lunch with Sam to your calendar for Friday afternoon.",
    "The weather tomorrow will be cool and dry, with a light breeze.",
]
# Subprocess limits. The long ones are caps against a hang, not expected run times.
NVIDIA_SMI_TIMEOUT_S = 10  # matches hal-employment's GPU gate
CLEAN_TIMEOUT_S = 6 * 3600  # Demucs and DeepFilterNet over every kept line of a film
PREPARE_TIMEOUT_S = 3600  # slicing and resampling the clips
# train_hal.py: five short steps and two training stages of at most 6 h each (gptsovits/train_hal.py)
TRAIN_TIMEOUT_S = 14 * 3600

MIN_SNR_DB = 10.0
MAX_SPOKEN_GAP = 0.5  # share of subtitle words a line's own transcript may differ by
SIGNATURE_WEIGHT = 3  # times "I'm sorry, Dave" appears in training, so its delivery is learnt
MUFFLED_DB = 10.0  # less 4-8 kHz energy than the film's typical HAL line: speaker or radio
# How the chosen voice speaks: the pace and sampling it was chosen at by ear. With these and
# "I'm sorry, Dave" as its reference, the voice says that line as the film does.
SPEED = 0.92
SAMPLING = {"top_k": 15, "top_p": 0.9, "temperature": 0.8}


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
    verbatim: bool = False
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


def release_gpu() -> None:
    """Frees the GPU memory of models a stage has dropped, so the next stage (training runs as its
    own process) can have it; torch and CTranslate2 both keep freed memory cached otherwise."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
        for ln, v in zip(lines, verdicts, strict=False):
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
            for ln, v in zip(lines, verdicts, strict=False)
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
        for ln, rec in kept:
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
    embed = None
    release_gpu()


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
            timeout=CLEAN_TIMEOUT_S,
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

    The training text is the subtitle's, as for the chosen voice; with --verbatim it is what
    each clip is heard to say instead.
    """
    from . import quality, sync
    from .sync import tokens

    ds = b.path("dataset", "transcripts.csv").parent
    for old in ds.glob("*.wav"):
        old.unlink()
    b.path("spoken.json").unlink(missing_ok=True)
    ok, why = usable(b)
    verbatim, taken, spoken = None, set(), {}
    if b.verbatim:
        from .verbatim import Verbatim

        # Each cleaned clip is transcribed again, so the text is what HAL says in exactly the
        # audio trained on; subtitles condense speech.
        b.say("writing down each clip as spoken")
        verbatim = Verbatim(b.asr_model)
    rows, held, seconds = [], None, 0.0
    # The default order and names are those the chosen voice was trained with; the order sets
    # how GPT-SoVITS shuffles the clips, so it is kept for the same voice.
    order = (lambda kv: kv[1][0]["start"]) if b.verbatim else (lambda kv: kv[0])
    for cid, (it, c) in sorted(ok.items(), key=order):
        said = it["text"]
        if verbatim:
            said = sync.tidy(verbatim.clip(Path(c["path"]), it["text"]))
            gap = quality.word_error(tokens(it["text"]), tokens(said))
            if not said or gap > MAX_SPOKEN_GAP:
                why[cid] = f"heard as '{said}', too far from the subtitle"
                continue
        weight = 1
        if held is None and " ".join(tokens(it["text"])) == HELD_OUT:
            held = (c["path"], it)
            if b.hold_out:
                continue
            weight = SIGNATURE_WEIGHT
        name = clip_name(said, taken) if verbatim else cid
        shutil.copy2(c["path"], ds / f"{name}.wav")
        rows.append((name, said, weight))
        seconds += c["seconds"]
        if verbatim:
            spoken[name] = {"spoken": said, "subtitle": it["text"], "start": it["start"]}
    if verbatim:
        b.path("spoken.json").write_text(json.dumps(spoken, indent=1))
        verbatim = None
        release_gpu()
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


TRAIN_FREE_GB = 12.0  # GPU memory SoVITS training needs at batch size 4
WAIT_FOR_GPU_S = 1800


def free_gpu_gb() -> float | None:
    """Returns free memory on the first GPU in GB, or None without nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=NVIDIA_SMI_TIMEOUT_S,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return int(out.split()[0]) / 1024


def wait_for_gpu(b: Build) -> None:
    """Waits until enough GPU memory is free; something else on the machine may be using it."""
    deadline = time.time() + WAIT_FOR_GPU_S
    free = free_gpu_gb()
    while free is not None and free < TRAIN_FREE_GB and time.time() < deadline:
        b.say(f"waiting for GPU memory: {free:.1f} GB free, training needs {TRAIN_FREE_GB:.0f} GB")
        time.sleep(30)
        free = free_gpu_gb()


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
        timeout=PREPARE_TIMEOUT_S,
    )
    exp = experiment(b)
    env = {
        **os.environ,
        "HAL_EXP": exp,
        "HAL_VERSION": b.version,
        "HAL_SOVITS_EPOCHS": str(b.sovits_epochs),
        "HAL_GPT_EPOCHS": str(b.gpt_epochs),
    }
    wait_for_gpu(b)
    b.say(f"training {exp}: {b.sovits_epochs} SoVITS and {b.gpt_epochs} GPT epochs")
    shutil.rmtree(GPTSOVITS / "GPT-SoVITS" / "logs" / exp, ignore_errors=True)
    with b.path("train.log").open("w") as log:
        subprocess.run(
            [sys.executable, str(GPTSOVITS / "train_hal.py")],
            env=env,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=TRAIN_TIMEOUT_S,
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
                urllib.request.urlopen(self.url, timeout=2)  # noqa: S310 - the local server URL built above
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
    for _snr, path, text in pool:
        calm.append((av.voice(av.load(path)).get("pitch_range_st", 99.0), path, text))
    return [(p, t) for _, p, t in sorted(calm)[:n]]


def synth(server: Server, text: str, ref: tuple[Path, str], speed: float) -> object:
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
            # Each request is one line said whole; splitting it lets a sentence drop out.
            "text_split_method": "cut0",
            "batch_size": 1,
            "media_type": "wav",
            "streaming_mode": False,
            **SAMPLING,
            "speed_factor": speed,
            "fragment_interval": 0.3,
        },
    )
    r.raise_for_status()
    y, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    return np.asarray(librosa.resample(y, orig_sr=sr, target_sr=16000), dtype=np.float64)


def last(paths: list[Path]) -> Path:
    """Returns the checkpoint of the latest epoch; names sort e12 before e4."""
    return max(paths, key=lambda p: int(re.search(r"[-_]e(\d+)(?:_s\d+)?$", p.stem).group(1)))


def select(b: Build) -> None:
    """Chooses the checkpoints, reference clip and pace, and saves samples beside the film's line.

    By default this is the chosen voice's recipe: the last checkpoints, "I'm sorry, Dave" itself
    as the reference, and SPEED. With --hold-out the voice has never heard that line, so every
    checkpoint pair and calm reference is scored on how closely it says it instead.
    """
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
    asr = WhisperModel(b.asr_model, device="cuda", compute_type="float16")

    def errors(y, text: str) -> float:
        said = asr.transcribe(y.astype(np.float32), language="en", beam_size=1)[0]
        return word_error(tokens(text), tokens(" ".join(s.text for s in said)))

    def checked(text: str, ref: tuple[Path, str], speed: float, tries: int = 4):
        """Says a line until the recogniser hears it whole; a take can lose its place."""
        best = None
        for _ in range(tries):
            y = synth(server, text, ref, speed)
            wer = errors(y, text)
            if best is None or wer < best[1]:
                best = (y, wer)
            if wer <= 0.15:
                break
        return best

    def use(gpt, sov) -> None:
        urllib.request.urlopen(f"{server.url}/set_gpt_weights?weights_path={gpt}", timeout=180)  # noqa: S310 - local server
        urllib.request.urlopen(f"{server.url}/set_sovits_weights?weights_path={sov}", timeout=180)  # noqa: S310 - local server

    results = []
    server = Server()
    try:
        pairs = checkpoints(b)
        if not b.hold_out:
            best = {
                "gpt": str(last([g for g, _ in pairs])),
                "sovits": str(last([s for _, s in pairs])),
                "ref": str(b.path("held-out.wav")),
                "prompt": held["text"],
                "speed": SPEED,
            }
            b.say(f"using {Path(best['gpt']).name} and {Path(best['sovits']).name}")
        else:
            clips = sorted(b.path("dataset", "transcripts.csv").parent.glob("*.wav"))
            fp = av.fingerprint(clips)
            target = av.load(b.path("held-out.wav"))
            film_fr = av.frames(target)
            segs, _ = asr.transcribe(target.astype(np.float32), language="en", word_timestamps=True)
            ws = [w for s in segs for w in s.words]
            film_words = [
                (
                    w.word.strip().lower(),
                    w.start,
                    max(w.end, ws[k + 1].start - 0.02)
                    if k + 1 < len(ws)
                    else len(target) / 16000 - 0.1,
                )
                for k, w in enumerate(ws)
            ]
            refs = references(b)
            b.say(f"selecting from {len(pairs)} checkpoint pairs and {len(refs)} reference clips")
            for gpt, sov in pairs:
                use(gpt, sov)
                for ref in refs:
                    for speed in (SPEED, 1.0):
                        takes = []
                        for _ in range(2):
                            y = synth(server, held["text"], ref, speed)
                            m = av.match(y, fp, film_fr, film_words)
                            takes.append((m, errors(y, held["text"])))
                        novel = [errors(synth(server, t, ref, speed), t) for t in NOVEL]
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
        best.update(SAMPLING)
        use(best["gpt"], best["sovits"])
        samples = b.path("samples", experiment(b), "film.wav").parent
        shutil.copy2(b.path("held-out.wav"), samples / "0-film-held-out.wav")
        wers = []
        for k, text in enumerate([held["text"], *NOVEL], start=1):
            y, wer = checked(text, (Path(best["ref"]), best["prompt"]), best["speed"])
            sf.write(samples / f"{k}-hal.wav", y, 16000)
            wers.append(wer)
        best["sample_wer"] = round(float(np.mean(wers)), 3)
        b.path(f"selection-{experiment(b)}.json").write_text(
            json.dumps({"best": best, "all": results}, indent=1)
        )
        b.say(f"samples in {samples}, {best['sample_wer']:.0%} word errors")
    finally:
        server.stop()


# ---- install ------------------------------------------------------------------------------


def install(b: Build) -> None:
    """Copies the chosen weights and reference clip into place and writes their settings,
    pace and sampling to `~/.config/hal-voice/built.yaml` (the previous file kept as
    `built.yaml.prev`). No file in the repository is edited."""
    best = json.loads(b.path(f"selection-{experiment(b)}.json").read_text())["best"]
    weights = GPTSOVITS / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    gpt, sov = Path(best["gpt"]), Path(best["sovits"])
    shutil.copy2(gpt, weights / gpt.name)
    shutil.copy2(sov, weights / sov.name)
    refdir = ROOT / "voice-data" / "reference"
    refdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["ref"], refdir / "hal_film_reference.wav")
    values = {
        # Relative to the checkout, so the file stays valid if the checkout moves.
        "gpt_weights": f"gptsovits/weights/{gpt.name}",
        "sovits_weights": f"gptsovits/weights/{sov.name}",
        "ref_audio_path": "voice-data/reference/hal_film_reference.wav",
        "prompt_text": best["prompt"],
        # The film reference stands alone; blended references came from another recording.
        "aux_ref_audio_paths": [],
        "speed": best["speed"],
        **{k: best.get(k, v) for k, v in SAMPLING.items()},
    }
    path = config.write_built(values, f"voice build {experiment(b)}")
    b.say(f"installed {gpt.name} and {sov.name}; settings in {path}")
