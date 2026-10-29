"""The voice test script: what each line says and what hal-voice should do when it hears it.

script.txt is read aloud once per take. "Marker N" lines mark the sections; numbered lines are
the test lines, with stage directions in brackets. What each line should do follows from its
section and wording: sections 2, 3 and 5 to 20 are spoken to HAL, section 4 never is, and a
line there without his name is a follow-up, said while the follow-up window is open.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCRIPT = Path(__file__).with_name("script.txt")
NOT_FOR_HAL = 4
TAKES = {
    "desk-1": ("Desk, take one", "desk", "normal"),
    "desk-2": ("Desk, take two", "desk", "loud"),
    "room-1": ("Across the room, take one", "room", "normal"),
    "room-2": ("Across the room, take two", "room", "loud"),
}
# Said before HAL's name to get his attention, with no reply wanted; any other greeting said
# alone ("Good morning, Hal.") is a turn of its own, which HAL answers.
ATTENTION = {"hey", "ok", "okay", "right", "oi", "so"}
GREETINGS = [
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "goodnight",
    "morning",
    "afternoon",
    "evening",
    "hello",
    "hi",
    "hey",
    "okay",
    "ok",
    "right",
    "thanks",
    "thank you",
    "cheers",
]
UNITS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
# Whisper's American spellings, and words it writes as one or as two.
SPELLING = {
    "kilometers": "kilometres",
    "summarize": "summarise",
    "color": "colour",
    "a.m.": "am",
    "p.m.": "pm",
    "todo": "to do",
    "goodnight": "good night",
    "okay": "ok",
    "mom": "mum",
    "favorite": "favourite",
}


def number_words(n: int) -> str:
    """Writes 0 to 999 as words; longer numbers are read digit by digit, like a PIN."""
    if n >= 1000:
        return " ".join(UNITS[int(d)] for d in str(n))
    if n >= 100:
        rest = n % 100
        return UNITS[n // 100] + " hundred" + (" and " + number_words(rest) if rest else "")
    if n >= 20:
        return TENS[n // 10] + (" " + UNITS[n % 10] if n % 10 else "")
    return UNITS[n]


def words(text: str) -> list[str]:
    """Lower-case words with numbers spelled out, so "10am" and "ten a.m." compare equal."""
    t = text.lower().replace("%", " percent").replace("&", " and ")
    # Whisper writes "twelve pounds" as "\u00a312".
    for sym, unit in (("\u00a3", "pounds"), ("$", "dollars"), ("\u20ac", "euros")):
        t = re.sub(rf"{re.escape(sym)}\s*([\d.,]+)", rf"\1 {unit}", t)
    for a, b in SPELLING.items():
        t = re.sub(rf"(?<![\w.]){re.escape(a)}(?![\w])", b, t)
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", t)
    t = re.sub(r"\d+", lambda m: " " + number_words(int(m.group())) + " ", t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t.replace("-", " "))
    return [w.strip("'") for w in t.split() if w.strip("'")]


@dataclass
class Line:
    """One line of the script, with what hal-voice should do when it hears it."""

    id: str  # "12" for a numbered line, "m3" for a marker
    section: int
    section_title: str
    text: str  # what is said, without stage directions
    directions: list[str] = field(default_factory=list)
    kind: str = "test"  # test | marker | chat
    wake: bool = False  # HAL should notice he is spoken to
    turn: bool = False  # a request (or greeting) should reach the brain
    followup: bool = False  # said in the follow-up window, without his name
    form: str = ""  # plain ("Hal, ..."), greeting ("Hi Hal, ..."), name-last ("..., Hal")
    then_request: bool = False  # a bare "Hal", a pause, then the request
    request: str = ""  # what should reach the brain

    def as_dict(self) -> dict:
        return asdict(self)


def name_form(text: str) -> tuple[str, str, str]:
    """Returns (form, greeting, request) for a line spoken to HAL."""
    low = text.lower()
    m = re.match(r"^\W*hal\b\W*", low)
    if m:
        return "plain", "", text[m.end() :].strip(" ,.!?")
    for g in sorted(GREETINGS, key=len, reverse=True):
        m = re.match(rf"^\W*{g}\W+hal\b\W*", low)
        if m:
            return "greeting", text[: len(g)].strip(), text[m.end() :].strip(" ,.!?")
    m = re.search(r"\W+hal\W*$", low)
    if m:
        return "name-last", "", text[: m.start()].strip(" ,.!?")
    return "", "", text.strip(" ,.!?")


def load(path: Path = SCRIPT) -> list[Line]:
    """Reads the script into lines, markers included, in the order they are spoken."""
    out: list[Line] = []
    section, title = 0, ""
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        m = re.match(r"^Marker ([\w-]+)\.\s*(.*)$", raw)
        if m:
            section += 1
            title = re.sub(r"\[.*\]", "", m.group(2)).strip(" .")
            said = f"Marker {m.group(1)}. {title}" if title else f"Marker {m.group(1)}"
            out.append(Line(f"m{section}", section, title, said, kind="marker"))
            continue
        m = re.match(r"^(\d+)\.\s*(.*)$", raw)
        if not m:
            continue
        body = m.group(2)
        directions = re.findall(r"\(([^)]*)\)", body)
        text = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)|\.\.\.", " ", body)).strip()
        line = Line(m.group(1), section, title, text, directions)
        if not text:
            line.kind = "chat"
        elif section != NOT_FOR_HAL and 2 <= section <= 20:
            form, greeting, request = name_form(text)
            line.form = form
            if form:
                line.wake = True
                attention = greeting.lower() in ATTENTION
                line.turn = bool(request) or (bool(greeting) and not attention)
                line.request = request
                if greeting and not attention:
                    line.request = f"{greeting}. {request}" if request else greeting
                line.then_request = any(re.fullmatch(r"\d+ s", d) for d in directions)
            else:
                line.followup = line.turn = True
                line.request = text.strip(" ,.!?")
        out.append(line)
    return out
