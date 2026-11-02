"""hal-transcribe command line, driven by hal-module.toml.

python -m haltranscribe                      list the tools
python -m haltranscribe TOOL [key=value ...] run one (the module prefix is optional)
python -m haltranscribe --stdin FUNCTION      run a function with JSON arguments on stdin
"""

import json
import sys
import tomllib
from pathlib import Path

from . import tools

MANIFEST = Path(__file__).resolve().parent / "hal-module.toml"


def _coerce(value: str, kind: str):
    if kind == "integer":
        return int(value)
    if kind == "number":
        return float(value)
    if kind == "boolean":
        return value.lower() in ("1", "true", "yes", "on")
    if kind == "array":
        return [v.strip() for v in value.split(",") if v.strip()]
    if kind == "object":
        return json.loads(value)
    return value


def _show(result) -> None:
    print(result if isinstance(result, str) else json.dumps(result, indent=2, default=str))


def main(argv=None) -> int:
    """Runs the tool named in `argv` with `key=value` arguments, lists the tools with `--list`, or runs `--stdin NAME` with JSON arguments from standard input; returns the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    spec = tomllib.loads(MANIFEST.read_text())
    if argv[:1] == ["--stdin"]:
        _show(getattr(tools, argv[1])(**json.loads(sys.stdin.read() or "{}")))
        return 0
    prefix = spec["module"]["name"] + "_"
    by_name = {}
    for t in spec.get("tools", []):
        by_name[t["name"]] = by_name[t["name"].removeprefix(prefix)] = t
    if not argv or argv[0] in ("-h", "--help", "--list"):
        print(__doc__)
        for t in spec.get("tools", []):
            print(f"  {t['name'].removeprefix(prefix):22} {t['description'].splitlines()[0][:80]}")
        return 0
    tool = by_name.get(argv[0])
    if tool is None:
        print(f"no tool {argv[0]}", file=sys.stderr)
        return 2
    params = tool.get("params", {})
    kwargs = {}
    for a in argv[1:]:
        key, _, value = a.partition("=")
        kwargs[key] = _coerce(value, params.get(key, {}).get("type", "string"))
    _show(getattr(tools, tool.get("function", tool["name"]))(**kwargs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
