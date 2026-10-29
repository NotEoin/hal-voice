"""AST scan: every subprocess call and HTTP request in the shipped code has a timeout."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNED = [
    *ROOT.glob("halvoice/**/*.py"),
    *ROOT.glob("haltranscribe/**/*.py"),
    *ROOT.glob("voicebuild/**/*.py"),
    *ROOT.glob("voicetest/**/*.py"),
    *ROOT.glob("gptsovits/*.py"),
    *ROOT.glob("voice-data/*.py"),
]
NEEDS_TIMEOUT = {
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "requests.get",
    "requests.post",
    "requests.put",
    "requests.patch",
    "requests.delete",
    "requests.head",
    "requests.request",
    "urllib.request.urlopen",
    "urlopen",
    "http.client.HTTPConnection",
    "http.client.HTTPSConnection",
}
# `urlretrieve` takes no timeout at all.
FORBIDDEN = {"urllib.request.urlretrieve", "urlretrieve"}
# Popen has no timeout argument. Each file below says why its process is bounded another way.
POPEN_ALLOWED = {
    "halvoice/brains/claude.py": "its process group is killed after brain.turn_timeout_s",
    "voicebuild/build.py": "the TTS server runs until Server.stop() ends its process group",
}


def dotted(node: ast.expr) -> str:
    """`a.b.c` for an attribute chain, else an empty string."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def problems(path: Path) -> list[str]:
    found = []
    rel = path.relative_to(ROOT).as_posix()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        where = f"{rel}:{node.lineno} {name}"
        if name in NEEDS_TIMEOUT and not any(k.arg == "timeout" for k in node.keywords):
            found.append(f"{where} has no timeout")
        if name in FORBIDDEN:
            found.append(f"{where} cannot take a timeout")
        if name.endswith("Popen") and rel not in POPEN_ALLOWED:
            found.append(f"{where} is not on the Popen allow-list")
    return found


def test_scan_covers_the_code_that_calls_out():
    names = {p.name for p in SCANNED}
    assert {"tts.py", "build.py", "train_hal.py", "fetch-voice-lines.py"} <= names


@pytest.mark.parametrize("path", SCANNED, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_blocking_call_has_a_timeout(path):
    assert problems(path) == []
