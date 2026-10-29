"""Replays the test clips through hal-voice's own hands-free listening, without TTS or brain.

Each clip goes through handsfree.Segmenter (the voice detector and end-of-speech rule) and
Ears.decide (the name, greeting and window rules), with the transcriber the service uses and
the settings it runs with: config.yaml, then the hal-voice unit's environment and voice.conf.
A clip is played on its own, with the take's room noise before and after it; a follow-up line
is played with the follow-up window opened a second before it, as after one of HAL's replies.

Two clocks: "sim" runs the frames as fast as they come and adds each transcription's measured
time, so tuning sweeps take minutes; "live" runs the real listening threads against a
microphone that delivers the clip at real-time pace, for the final word on latency.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from halvoice import config, handsfree, stt

from . import script

LEAD_S = 1.5        # room noise before each clip
TAIL_S = 7.0        # and after it, long enough for a command window to close; and longer
                    # than the longest utterance, so one the detector never ends still is
REPLY_GAP_S = 1.0   # a follow-up starts this long after HAL stops speaking
NOISE_S = 30        # room noise alone, played to check it never wakes HAL
UNIT = Path.home() / ".config/systemd/user/hal-voice.service"


def service_env(root: Path) -> dict[str, str]:
    """The HALVOICE_ settings the service runs with: its unit's Environment= lines, then the
    voice.conf the launcher exports."""
    env = {}
    unit = UNIT if UNIT.exists() else root / "systemd" / "hal-voice.service"
    for line in unit.read_text().splitlines():
        m = re.match(r"^Environment=(HALVOICE_\w+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip('"')
    conf = root / "voice.conf"
    if conf.exists():
        for line in conf.read_text().splitlines():
            m = re.match(r'^(HALVOICE_STT_\w+)="?([^"]*)"?$', line.strip())
            if m:
                env[m.group(1)] = m.group(2)
    return env


def settings(overrides: dict | None = None) -> config.Cfg:
    """Loads hal-voice's config as the service sees it, with any tuning overrides on top."""
    os.environ.update(service_env(config.root_dir()))
    cfg = config.load(force=True)
    for dotted, value in (overrides or {}).items():
        node = cfg
        *path, leaf = dotted.split(".")
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = value
    stt.Transcriber._model = None
    return cfg


class CachedTranscriber:
    """The service's transcriber, remembering each utterance's text and how long it took."""

    def __init__(self, cache: Path) -> None:
        self.inner = stt.Transcriber()
        self.cache_path = cache
        self.cache = json.loads(cache.read_text()) if cache.exists() else {}
        cfg = config.load()
        self.key = json.dumps([cfg.get("stt"), cfg.get("audio")], sort_keys=True, default=str)

    def __call__(self, audio: np.ndarray) -> tuple[str, float]:
        k = hashlib.sha1(self.key.encode() + audio.tobytes(), usedforsecurity=False).hexdigest()
        if k not in self.cache:
            self.inner._load()
            t = time.perf_counter()
            text = self.inner.transcribe(audio)
            self.cache[k] = [text, round(time.perf_counter() - t, 4)]
        return self.cache[k][0], self.cache[k][1]

    def save(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache))


@dataclass
class Result:
    clip: dict
    events: list = field(default_factory=list)   # (seconds after end of speech, kind, text)
    heard: list = field(default_factory=list)    # every transcript, request or not


def noise_bed(silence: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n samples of the take's own room noise, looped from its silence with crossfades."""
    if silence.size == 0:
        return np.zeros(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    seg, fade = min(silence.size, 32000), 800
    pos = 0
    while pos < n:
        a = int(rng.integers(0, max(1, silence.size - seg + 1)))
        piece = silence[a: a + seg].copy()
        piece[:fade] *= np.linspace(0, 1, fade)
        piece[-fade:] *= np.linspace(1, 0, fade)
        end = min(n, pos + seg)
        out[pos:end] += piece[: end - pos]
        pos += seg - fade
    return out


def stream_for(clip_audio: np.ndarray, silence: np.ndarray, sr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tail_s = max(TAIL_S, float(config.load().get("audio.max_utterance_s", 30)) + 1.0)
    lead = noise_bed(silence, int(LEAD_S * sr), rng)
    tail = noise_bed(silence, int(tail_s * sr), rng)
    return np.concatenate([lead, clip_audio, tail])


def _int16(y: np.ndarray) -> np.ndarray:
    return np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)


def run_sim(clip: dict, audio: np.ndarray, silence: np.ndarray, sr: int,
            transcribe: CachedTranscriber) -> Result:
    """Plays one clip through Segmenter and Ears.decide on a simulated clock."""
    y = _int16(stream_for(audio, silence, sr, seed=int(re.sub(r"\D", "", clip["id"]) or 0)))
    speech_end = LEAD_S + clip["speech_end"]
    opened = LEAD_S + clip["speech_start"] - REPLY_GAP_S
    # The clock starts far from zero, as time.time() does, so no window looks open at the start.
    t0 = 1e6
    now = [t0]
    ears = handsfree.Ears(lambda a: transcribe(a)[0])
    ears.clock = lambda: now[0]
    seg = ears.segmenter()
    frame = int(sr * handsfree.FRAME_MS / 1000)
    utterances = []
    for k in range(len(y) // frame):
        now[0] = t0 + (k + 1) * frame / sr
        if clip.get("followup") and now[0] - frame / sr < t0 + opened <= now[0]:
            ears.stopped_speaking()
            ears.follow_up()
        out = seg.feed(y[k * frame: (k + 1) * frame])
        if out is not None:
            utterances.append((now[0], out, ears.overlapped(out, seg.lead_s)))
    res = Result(clip)
    busy = 0.0
    for emitted, out, overlapped in utterances:
        text, took = transcribe(out)
        now[0] = max(emitted, busy) + took
        busy = now[0]
        res.heard.append(text)
        ears.decide(text.strip(), overlapped)
        while not ears.requests.empty():
            h = ears.requests.get()
            res.events.append((round(now[0] - t0 - speech_end, 3), "turn" if h.text else "wake",
                               h.text))
    return res


class FakeMic:
    """Delivers a recording frame by frame at real-time pace, like sounddevice's InputStream."""

    def __init__(self, y: np.ndarray, sr: int, at: dict) -> None:
        self.y, self.sr, self.pos, self.at = y, sr, 0, dict(at)
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc) -> None:
        return None

    def read(self, frame: int):
        due = self.t0 + (self.pos + frame) / self.sr
        delay = due - time.time()
        if delay > 0:
            time.sleep(delay)
        block = self.y[self.pos: self.pos + frame]
        if len(block) < frame:
            block = np.concatenate([block, np.zeros(frame - len(block), dtype=np.int16)])
        self.pos += frame
        for t, fn in list(self.at.items()):
            if self.pos / self.sr >= t:
                fn()
                del self.at[t]
        return block[:, None], False


def run_live(clip: dict, audio: np.ndarray, silence: np.ndarray, sr: int,
             transcribe: CachedTranscriber) -> Result:
    """Plays one clip through the real listening threads at real-time pace."""
    y = _int16(stream_for(audio, silence, sr, seed=int(re.sub(r"\D", "", clip["id"]) or 0)))
    res = Result(clip)

    def heard(a):
        # Transcribed for real here: live mode is the check on the cached timings.
        text = transcribe.inner.transcribe(a)
        res.heard.append(text)
        return text

    ears = handsfree.Ears(heard)
    at = {}
    if clip.get("followup"):
        def opened():
            ears.stopped_speaking()
            ears.follow_up()
        at[LEAD_S + clip["speech_start"] - REPLY_GAP_S] = opened
    mic = FakeMic(y, sr, at)
    ears.open_stream = lambda frame: mic
    ears.start()
    speech_end = LEAD_S + clip["speech_end"]
    stop_at = time.time() + len(y) / sr + 1.0
    while time.time() < stop_at:
        try:
            h = ears.requests.get(timeout=0.05)
        except queue.Empty:
            continue
        t = time.time() - mic.t0
        res.events.append((round(t - speech_end, 3), "turn" if h.text else "wake", h.text))
        if h.text:
            # One turn is the answer; a second would come within the end-of-speech pause.
            stop_at = min(stop_at, time.time() + 2.0)
    ears.stop()
    time.sleep(0.6)
    return res


def run(manifest: Path, mode: str = "sim", takes: list[str] | None = None,
        lines: list[str] | None = None, overrides: dict | None = None,
        conditions: list[str] | None = None, say=print) -> list[Result]:
    """Replays every clip (or the chosen takes and lines) and returns what happened."""
    import soundfile as sf

    from . import simulate

    settings(overrides)
    base = manifest.parent
    transcribe = CachedTranscriber(base / ".stt-cache.json")
    data = json.loads(manifest.read_text())
    results = []
    try:
        for take in data["takes"]:
            if takes and take["take"] not in takes:
                continue
            silence, sr = sf.read(base / take["silence"], dtype="float32")
            for cond in conditions or [""]:
                sil = simulate.apply(silence, silence, sr, cond) if cond else silence
                play = run_live if mode == "live" else run_sim
                # Half a minute of the room alone: nothing in it may wake HAL.
                quiet = {"id": "noise", "section": script.NOT_FOR_HAL, "kind": "noise",
                         "text": "(room noise only)", "speech_start": 0.0, "speech_end": 0.0,
                         "wake": False, "turn": False, "followup": False, "form": ""}
                for clip in [*take["clips"], quiet]:
                    if clip.get("missing") or (lines and clip["id"] not in lines):
                        continue
                    if clip["id"] == "noise":
                        audio = noise_bed(sil, NOISE_S * sr, np.random.default_rng(7))
                        results.append(play({**clip, "condition": take["take"] + (
                            f"~{cond}" if cond else "")}, audio, sil, sr, transcribe))
                        continue
                    audio, _ = sf.read(base / clip["file"], dtype="float32")
                    if cond:
                        audio = simulate.apply(audio, silence, sr, cond)
                    row = {**clip, "condition": take["take"] + (f"~{cond}" if cond else "")}
                    results.append(play(row, audio, sil, sr, transcribe))
                say(f"  {take['take']}{'~' + cond if cond else ''}: replayed")
    finally:
        transcribe.save()
    return results
