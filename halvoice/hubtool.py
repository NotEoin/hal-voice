"""One-shot worker for hal-hub's `exec` manifests (voice and transcribe packs).

python -m halvoice.hubtool FUNCTION    JSON arguments on stdin, one JSON value on stdout

Both manifests name this module, so the hub runs every tool in hal-voice's own
environment and hal-voice imports nothing from Hal (08, "exec manifests").
A failure exits non-zero with the reason on stderr; stdout stays empty.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable
from typing import Any

# function name -> (module, attribute); modules load lazily so `say` never imports whisper
FUNCTIONS: dict[str, tuple[str, str]] = {
    "say": ("halvoice.tools", "say"),
    "transcribe": ("haltranscribe.tools", "transcribe"),
    "record_start": ("haltranscribe.tools", "record_start"),
    "record_stop": ("haltranscribe.tools", "record_stop"),
    "recording": ("haltranscribe.tools", "recording"),
    "save_note": ("haltranscribe.tools", "save_note"),
}
USAGE_EXIT = 2


def resolve(name: str) -> Callable[..., Any]:
    """Returns the callable for a manifest `function`; raises KeyError if unknown."""
    if name not in FUNCTIONS:
        raise KeyError(f"no function {name!r}; known: {', '.join(sorted(FUNCTIONS))}")
    module, attr = FUNCTIONS[name]
    return getattr(importlib.import_module(module), attr)


def run(name: str, raw_args: str) -> str:
    """Runs one call and returns its JSON result; raises on a bad name or arguments."""
    args = json.loads(raw_args or "{}")
    if not isinstance(args, dict):
        raise ValueError("arguments must be a JSON object")
    return json.dumps(resolve(name)(**args), default=str)


def main(argv: list[str] | None = None) -> int:
    """Entry point: reads the arguments from stdin, prints the result."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return USAGE_EXIT
    try:
        out = run(argv[0], sys.stdin.read())
    except (KeyError, ValueError, TypeError) as exc:
        print(f"hubtool: {exc}", file=sys.stderr)
        return USAGE_EXIT
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
