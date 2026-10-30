"""voicetest: split a recording of the test script, replay it through hands-free listening,
and send the requests heard to HAL's brain.

    hal-voice voicetest split RECORDING [--take desk-1]
    hal-voice voicetest run [--live] [--take T] [--line N] [--set key=value] [--simulate]
    hal-voice voicetest compare LABEL [LABEL ...]
    hal-voice voicetest brain [--label LABEL] [--manifest FILE ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from halvoice import config

from . import score

TESTS = config.root_dir() / "voice-data" / "tests"


def value(text: str):
    """A --set value as YAML would read it: 600 is a number, true a flag, cut0 a string."""
    import yaml

    return yaml.safe_load(text)


def cmd_split(args) -> int:
    from faster_whisper import WhisperModel

    from halvoice.stt import _preload_cuda

    from . import split

    _preload_cuda()
    try:
        model = WhisperModel("medium.en", device="cuda", compute_type="float16")
    except (RuntimeError, ValueError):
        model = WhisperModel("medium.en", device="cpu", compute_type="int8")
    takes = []
    for rec in args.recordings:
        takes += split.split(Path(rec).expanduser(), TESTS, model, take=args.take)
    print(f"manifest: {split.write_manifest(TESTS, takes)}")
    return 0


def cmd_run(args) -> int:
    import logging

    from . import replay

    # The listener's own log would interleave with the report.
    logging.getLogger("halvoice").setLevel(logging.WARNING)

    manifest = TESTS / "manifest.json"
    if not manifest.exists():
        print("no clips yet: run 'hal-voice voicetest split RECORDING' first", file=sys.stderr)
        return 1
    overrides = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        overrides[k] = value(v)
    conditions = [""] + (["fans", "far"] if args.simulate else [])
    if args.condition:
        conditions = ["" if c == "recorded" else c for c in args.condition]
    label = args.label or ("live" if args.live else "sim")
    print(
        f"replaying ({'real time' if args.live else 'simulated clock'})"
        + (f" with {overrides}" if overrides else "")
    )
    results = replay.run(
        manifest,
        "live" if args.live else "sim",
        args.take or None,
        args.line or None,
        overrides,
        conditions,
    )
    rows = [score.judge(r) for r in results]
    out = TESTS / "runs"
    out.mkdir(exist_ok=True)
    (out / f"{label}.json").write_text(
        json.dumps(
            {"overrides": overrides, "mode": "live" if args.live else "sim", "rows": rows}, indent=1
        )
    )
    note = f"Mode: {'real time' if args.live else 'simulated clock'}." + (
        f" Settings changed: {overrides}." if overrides else ""
    )
    report = score.markdown(label, rows, note)
    (out / f"{label}.md").write_text(report)
    print(report if args.verbose else report.split("## Failures")[0])
    print(f"report: {out / (label + '.md')}")
    return 0


def cmd_compare(args) -> int:
    runs = {}
    for label in args.labels:
        runs[label] = json.loads((TESTS / "runs" / f"{label}.json").read_text())["rows"]
    print(score.compare(runs))
    return 0


def cmd_brain(args) -> int:
    from . import brain

    rows = json.loads((TESTS / "runs" / f"{args.label}.json").read_text())["rows"]
    report = brain.run(
        rows, TESTS / "runs" / f"brain-{args.label}", brain.manifest_paths(args.manifest)
    )
    print(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hal-voice voicetest",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split", help="cut a recording of script.txt into clips")
    sp.add_argument("recordings", nargs="+")
    sp.add_argument("--take", help="the take, when its marker was not heard: desk-1, room-2...")
    rp = sub.add_parser("run", help="replay the clips through hands-free listening")
    rp.add_argument("--live", action="store_true", help="real-time pace through the threads")
    rp.add_argument("--take", action="append", default=[])
    rp.add_argument("--line", action="append", default=[])
    rp.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. activation.keyword.end_silence_ms=500",
    )
    rp.add_argument(
        "--simulate",
        action="store_true",
        help="also replay with louder fans and from further away (simulated)",
    )
    rp.add_argument(
        "--condition",
        action="append",
        default=[],
        help="only these conditions: recorded, fans, far",
    )
    rp.add_argument("--label", help="name for this run's report (default sim or live)")
    rp.add_argument("-v", "--verbose", action="store_true", help="print every failure")
    cp = sub.add_parser("compare", help="compare runs side by side")
    cp.add_argument("labels", nargs="+")
    bp = sub.add_parser("brain", help="send a run's requests to a model and check the tool calls")
    bp.add_argument("--label", default="sim")
    bp.add_argument(
        "--manifest", action="append", help="hal-module.toml to take tools from (repeatable)"
    )
    args = ap.parse_args(argv)
    return {"split": cmd_split, "run": cmd_run, "compare": cmd_compare, "brain": cmd_brain}[
        args.cmd
    ](args)


if __name__ == "__main__":
    sys.exit(main())
