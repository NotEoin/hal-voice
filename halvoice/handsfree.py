"""Hands-free listening: one microphone stream, cut into utterances, each transcribed once.

The stream never closes, so nothing said between "Hal" and the request is
lost, and each utterance keeps a pre-roll of audio from before the voice
detector fired, so the soft "H" of the name survives. The detector listens
above the hum of a PC's fans and against the room's own noise floor, or an
utterance in a noisy room never ends. An utterance becomes a request when it
starts (after a greeting or not) or ends with HAL's name, when it follows a
bare "Hal" within the command window, or when it comes during the follow-up
window after HAL has spoken. While HAL speaks, only his name counts, which
interrupts him. `hal-voice voicetest` replays recordings through all of this.
"""

from __future__ import annotations

import collections
import difflib
import queue
import re
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import config
from .util import get_logger

log = get_logger(__name__)

FRAME_MS = 30           # webrtcvad accepts 10, 20 or 30 ms frames
PRE_ROLL_MS = 360       # audio kept from before the detector fired
MIN_SPEECH_MS = 240     # shorter bursts are clicks and coughs
ECHO_TAIL_S = 0.6       # speech starting this soon after HAL stops is his own echo and reverb
ECHO_SPEECH_S = 0.5     # speech going on this long past the echo tail is the user, not echo
FLOOR_RISE_DB = 0.02    # per frame the noise floor may rise: 0.7 dB a second

# Whisper's usual spellings of a spoken "Hal", alone or after a greeting.
NAME = r"(?:hal|hall|hale|haul|howl|hull|hel|hell|al|owl|hol)"
# Said before the name. An attention word alone ("Hey Hal") waits for the request; any other
# greeting alone ("Good morning, Hal") is a turn of its own, so HAL greets back.
GREET = (r"good\s*morning|good\s*afternoon|good\s*evening|good\s*night|morning|afternoon|"
         r"evening|hello|hiya|hi|hey|okay|ok|oi|right|so|thanks|thank\s+you|cheers")
ATTENTION = {"hey", "ok", "okay", "oi", "right", "so"}
LEADING = re.compile(rf"^\W*(?:(?P<greet>{GREET})[\s,.!]+)?{NAME}\b[\s,.!?:;-]*", re.I)
TRAILING = re.compile(rf"[\s,]+{NAME}\W*$", re.I)
# In noise Whisper writes a quiet "Hal" as "How". Only where "how" cannot start a sentence is it
# taken for the name: set off by punctuation ("How, what time is it?") or before a question word.
HOW = re.compile(rf"^\W*(?:(?P<greet>{GREET})[\s,.!]+)?how(?:\s*(?:[,?!]+|\.\.\.|\u2026)\s*(?=\w)"
                 r"|\s+(?=(?:what|what's|whats|when|where|which|who|whose)\b))", re.I)
# "Hal said he'd ring back" is about someone called Hal (or Al), not to HAL.
ABOUT = re.compile(rf"^\W*{NAME}\s+(?:said|says|told|tells|thinks|thought|reckons|reckoned|"
                   r"mentioned|wants|wanted|knows|asked)\b", re.I)


def addressed(text: str, how_is_name: bool = True) -> tuple[bool, str]:
    """Returns (named, request): whether HAL was spoken to, and what was asked of him.

    The name is removed from the request; a greeting is kept, so HAL can return it.
    """
    if ABOUT.match(text):
        return False, text.strip(" ,.!?")
    lead = LEADING.match(text) or (HOW.match(text) if how_is_name else None)
    if lead:
        request, greet = text[lead.end():], lead.group("greet")
    else:
        trail = TRAILING.search(text)
        if not trail:
            return False, text.strip(" ,.!?")
        request, greet = text[:trail.start()], None
    request = request.strip(" ,.!?")
    # "Hal, Hal" is the name said twice (or noise heard as it), not a request.
    if re.fullmatch(rf"(?:\W*{NAME}\W*)*", request, re.I):
        request = ""
    if greet and re.sub(r"\s+", " ", greet.lower()) not in ATTENTION:
        greet = re.sub(r"\s+", " ", greet).capitalize()
        request = f"{greet}. {request[:1].upper()}{request[1:]}" if request else greet
    return True, request


@dataclass
class Heard:
    text: str           # the request, with the name removed
    named: bool         # HAL's name was in it
    raw: str            # the full transcript


class Segmenter:
    """Cuts a stream of FRAME_MS int16 frames into utterances.

    The microphone thread and the offline replay test (voicetest) both feed frames through
    this, so a recording is cut exactly as live speech is.
    """

    def __init__(self, sr: int, vad_level: int, end_ms: int, max_s: float,
                 highpass_hz: float = 0.0, margin_db: float = 0.0,
                 pre_roll_ms: int = PRE_ROLL_MS) -> None:
        import webrtcvad

        self.vad = webrtcvad.Vad(vad_level)
        self.sr = sr
        # A PC's fans hum and rumble below a voice; webrtcvad hears that as speech, on every
        # frame, so an utterance never ends. The detector listens above highpass_hz.
        self.sos = self.zi = None
        if highpass_hz:
            from scipy.signal import butter, sosfilt_zi

            self.sos = butter(4, highpass_hz, "highpass", fs=sr, output="sos")
            self.zi = sosfilt_zi(self.sos) * 0.0
        # And a frame is speech only when it is margin_db above the noise floor, followed as
        # the noise changes: fast down, slowly up, so speech does not raise it.
        self.margin_db = margin_db
        self.floor_db: float | None = None
        self.pre: collections.deque = collections.deque(maxlen=max(1, pre_roll_ms // FRAME_MS))
        self.end_frames = end_ms // FRAME_MS
        self.max_frames = int(max_s * 1000 / FRAME_MS)
        self.voiced: list[np.ndarray] = []
        self.speech_frames = self.silent = 0
        self.lead_s = 0.0       # pre-roll at the front of the last utterance returned

    def feed(self, chunk: np.ndarray) -> "np.ndarray | None":
        """Takes one frame; returns an utterance as float32 audio when one has just ended."""
        heard = chunk
        if self.sos is not None:
            from scipy.signal import sosfilt

            y, self.zi = sosfilt(self.sos, chunk.astype(np.float64), zi=self.zi)
            heard = np.clip(y, -32768, 32767).astype(np.int16)
        is_speech = self.vad.is_speech(heard.tobytes(), self.sr)
        if self.margin_db:
            level = 10 * np.log10(np.mean(heard.astype(np.float64) ** 2) + 1.0)
            if self.floor_db is None:
                self.floor_db = level
            elif level < self.floor_db:
                self.floor_db += 0.3 * (level - self.floor_db)
            else:
                self.floor_db += FLOOR_RISE_DB
            is_speech = is_speech and level >= self.floor_db + self.margin_db
        if not self.voiced:
            self.pre.append(chunk)
            if is_speech:
                self.voiced = list(self.pre)
                self.lead_s = (len(self.voiced) - 1) * FRAME_MS / 1000
                self.speech_frames, self.silent = 1, 0
            return None
        self.voiced.append(chunk)
        if is_speech:
            self.speech_frames += 1
            self.silent = 0
        else:
            self.silent += 1
        if self.silent < self.end_frames and len(self.voiced) < self.max_frames:
            return None
        out = None
        if self.speech_frames * FRAME_MS >= MIN_SPEECH_MS:
            out = np.concatenate(self.voiced).astype(np.float32) / 32768.0
        self.voiced, self.speech_frames, self.silent = [], 0, 0
        self.pre.clear()
        return out


class Ears:
    """Background listener that yields requests from a queue."""

    def __init__(self, transcribe) -> None:
        cfg = config.load()
        self.sr = int(cfg.get("audio.sample_rate", 16000))
        self.device = cfg.get("audio.input_device")
        self.vad_level = int(cfg.get("audio.vad_aggressiveness", 2))
        self.highpass_hz = float(cfg.get("audio.vad_highpass_hz", 0) or 0)
        self.margin_db = float(cfg.get("audio.vad_margin_db", 0) or 0)
        self.pre_roll_ms = int(cfg.get("activation.keyword.pre_roll_ms", PRE_ROLL_MS))
        self.how_is_name = bool(cfg.get("activation.keyword.how_is_name", True))
        self.end_ms = int(cfg.get("activation.keyword.end_silence_ms", 600))
        self.max_s = float(cfg.get("audio.max_utterance_s", 30))
        self.command_window = float(cfg.get("activation.keyword.command_window_s", 6))
        self.followup_s = float(cfg.get("activation.followup_s", 8))
        self.transcribe = transcribe
        self.requests: "queue.Queue[Heard]" = queue.Queue()
        self.interrupt = threading.Event()
        self.speaking = threading.Event()
        self._armed_until = 0.0         # a request without the name is accepted until then
        self._quiet_since = 0.0         # when HAL last stopped speaking
        self._said: collections.deque = collections.deque(maxlen=12)   # HAL's recent sentences
        self._pending_name_at = 0.0     # when a bare "Hal" was heard
        self._stop = threading.Event()
        self._audio: "queue.Queue[np.ndarray]" = queue.Queue()
        self.clock = time.time          # the replay test runs on the recording's own clock

    # -- windows ----------------------------------------------------------------
    def follow_up(self) -> None:
        """Accepts a request without the name for a short while (called when HAL stops talking)."""
        self._armed_until = self.clock() + self.followup_s

    def disarm(self) -> None:
        self._armed_until = 0.0

    def spoke(self, sentence: str) -> None:
        """Records a sentence HAL said, so it is not taken for the user when the mic hears it."""
        self._said.append(sentence.lower())

    def stopped_speaking(self) -> None:
        self._quiet_since = self.clock()

    def _is_echo(self, text: str) -> bool:
        """True when a transcript matches something HAL said: the mic heard the speakers."""
        low = text.lower()
        return any(difflib.SequenceMatcher(None, low, said).ratio() > 0.55 or (len(low) > 12 and low in said)
                   for said in self._said)

    # -- threads ----------------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._capture, name="ears-capture", daemon=True).start()
        threading.Thread(target=self._understand, name="ears-stt", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def overlapped(self, audio: np.ndarray, lead_s: float) -> bool:
        """True when an utterance just ended is mostly HAL's own voice: it came while he
        talked, or in the echo of his last words and no further."""
        if self.speaking.is_set():
            return True
        now = self.clock()
        # When the voice was first heard (the pre-roll is not the start) and last heard.
        began = now - len(audio) / self.sr + lead_s
        ended = now - self.end_ms / 1000
        tail = self._quiet_since + ECHO_TAIL_S
        return began < tail and ended < tail + ECHO_SPEECH_S

    def segmenter(self) -> Segmenter:
        return Segmenter(self.sr, self.vad_level, self.end_ms, self.max_s, self.highpass_hz,
                         self.margin_db, self.pre_roll_ms)

    def open_stream(self, frame: int):
        """The microphone, as a context manager whose read(frame) returns (block, overflowed)."""
        import sounddevice as sd

        return sd.InputStream(samplerate=self.sr, channels=1, dtype="int16", blocksize=frame,
                              device=self.device)

    def _capture(self) -> None:
        """Splits the microphone stream into utterances and queues them for transcription."""
        seg = self.segmenter()
        frame = int(self.sr * FRAME_MS / 1000)
        with self.open_stream(frame) as stream:
            log.info("listening for HAL's name (continuous, %d ms pre-roll)", self.pre_roll_ms)
            while not self._stop.is_set():
                block, _ = stream.read(frame)
                audio = seg.feed(block[:, 0].copy())
                if audio is not None:
                    self._audio.put((audio, self.overlapped(audio, seg.lead_s)))

    def _understand(self) -> None:
        """Transcribes utterances and decides which are requests."""
        while not self._stop.is_set():
            try:
                audio, overlapped = self._audio.get(timeout=0.5)
            except queue.Empty:
                continue
            # stt.initial_prompt helps Whisper hear a quiet "Hal"; voicetest checks that it
            # does not hear him in the room's noise alone.
            self.decide(self.transcribe(audio).strip(), overlapped)

    def decide(self, text: str, overlapped: bool = False) -> None:
        """Queues a transcript as a request, a bare "Hal", an interruption, or nothing."""
        if not text or re.fullmatch(r"\W*(you|thank you|thanks for watching)?\W*", text, re.I):
            return
        if self._is_echo(text):
            log.debug("ignored (HAL's own voice): %r", text)
            return
        named, request = addressed(text, self.how_is_name)
        now = self.clock()
        if self.speaking.is_set() or (overlapped and not named):
            if named and not self._is_echo(text):
                log.info("heard HAL's name while speaking: interrupting")
                self.interrupt.set()
                if request:
                    self.requests.put(Heard(request, True, text))
                else:
                    self._pending_name_at = now
            return
        if named and not request:
            log.info("heard HAL's name: listening for the request")
            self._pending_name_at = now
            self.requests.put(Heard("", True, text))
            return
        armed = now < self._armed_until or now - self._pending_name_at < self.command_window
        if named or armed:
            self._pending_name_at = 0.0
            self.disarm()
            log.info("you: %s  (heard %r)", request, text)
            self.requests.put(Heard(request, named, text))
        else:
            log.debug("ignored (no name): %r", text)
