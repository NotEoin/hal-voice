"""Sends the test script's commands to HAL's brain, in hal eval's sandbox, and checks the tools.

Each command goes in twice: as written in the script, which tests understanding alone, and as
hands-free listening heard it in a replay run, which tests recognition and understanding
together. The brain is hal-voice's own (halvoice.brain: the voice channel, HAL's persona) and
runs in the sandbox from core hal's evaluate.py: a throwaway calendar, task list, vault and
state. Extensions that act on the machine or the outside world are offered to the brain but
never run: a call to one answers that it was not carried out, which is enough to see that the
right tool was chosen.
"""

from __future__ import annotations

import json
import re
import sys
import time
import tomllib
from pathlib import Path

from . import script

EXPECT = Path(__file__).with_name("tools.toml")
SECTIONS = {3, *range(7, 21)}
ALWAYS = {"hal_find_tools"}  # looking for tools is not acting


def conversations(lines: dict[str, script.Line], expect: dict) -> list[list[str]]:
    """The command lines in order, a line and the follow-up said after it kept together."""
    followers = {v["after"] for v in expect.values() if "after" in v}
    out = []
    for lid, ln in lines.items():
        if ln.section not in SECTIONS or lid not in expect or lid in followers:
            continue
        conv, at = [lid], lid
        while "after" in expect.get(at, {}):
            at = expect[at]["after"]
            conv.append(at)
        out.append(conv)
    return out


def judge(want: dict, greetings: dict, turn: dict) -> list[str]:
    names = [t["name"] for t in turn.get("tools", []) if t["name"] not in ALWAYS]
    problems = []
    tools = want.get("tools", [])
    if tools and not (("none" in tools and not names) or set(names) & set(tools)):
        problems.append(f"called {', '.join(names) or 'nothing'}, wanted {' or '.join(tools)}")
    for f in want.get("forbid", []):
        if f in names:
            problems.append(f"called {f}")
    pattern = want.get("reply")
    if pattern:
        pattern = greetings.get(pattern, pattern)
        if not re.search(pattern, turn.get("reply", ""), re.I):
            problems.append(f"reply did not match /{pattern}/")
    return problems


def sandbox():
    """Builds hal eval's sandbox, with the outward-facing extensions loaded but stubbed."""
    from halvoice import brain as hv

    core = hv.find_core()
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    from halcore import evaluate

    real = evaluate.config.load()
    box = evaluate.make_sandbox(None)
    cfg = box / "config.toml"
    keep = sorted(set(real.get("hub", {}).get("disabled", [])) - set(evaluate.UNSAFE))
    cfg.write_text(
        re.sub(r"^disabled = .*$", f"disabled = {json.dumps(keep)}", cfg.read_text(), flags=re.M)
    )
    hub = evaluate._fresh_hub()
    real_call = hub.call

    def call(name: str, args: dict | None = None) -> dict:
        tool = hub.get(name)
        if tool is not None and tool.module.name in evaluate.UNSAFE:
            if tool.confirm and not (args or {}).get("_confirmed"):
                return {
                    "needs_confirmation": f"{name} {json.dumps(args or {})}",
                    "hint": "ask the user, then call again with _confirmed: true",
                }
            return {"result": f"(test run: {name} was not carried out)"}
        return real_call(name, args)

    hub.call = call
    return box, hv


def run(rows: list[dict], out: Path, say=print) -> str:
    """Asks the brain every command line, as scripted and as heard; returns the report."""
    import shutil

    expect = tomllib.loads(EXPECT.read_text())
    greetings, wants = expect["greeting"], expect["line"]
    lines = {ln.id: ln for ln in script.load() if ln.kind == "test"}
    take = sorted({r["condition"] for r in rows})[0]
    heard = {r["id"]: r["request"] for r in rows if r["condition"] == take}
    box, hv = sandbox()
    results = []
    try:
        for conv in conversations(lines, wants):
            for variant in ("script", "heard"):
                texts = (
                    [lines[i].request for i in conv]
                    if variant == "script"
                    else [heard.get(i, "") for i in conv]
                )
                brain = hv.direct_brain()
                for lid, text in zip(conv, texts, strict=True):
                    row = {"id": lid, "variant": variant, "said": text, "tools": [], "reply": ""}
                    if not text:
                        row["problems"] = ["not heard"]
                        results.append(row)
                        continue
                    started = time.time()
                    try:
                        brain.reply(text)
                        turn = brain.last_turn
                    except Exception as e:  # noqa: BLE001 - one broken turn fails one line
                        turn = {"reply": f"(error: {e})", "tools": []}
                    row.update(
                        tools=[t["name"] for t in turn.get("tools", [])],
                        reply=turn.get("reply", ""),
                        seconds=round(time.time() - started, 1),
                        problems=judge(wants[lid], greetings, turn),
                    )
                    results.append(row)
                    say(
                        f"  {lid:>4} {variant:6} {'ok ' if not row['problems'] else 'BAD'} "
                        f"{','.join(row['tools']) or '-':32} {text[:50]}"
                    )
    finally:
        shutil.rmtree(box, ignore_errors=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    report = markdown(take, results, lines)
    out.with_suffix(".md").write_text(report)
    return report


def _cell(r: dict) -> str:
    if r["problems"]:
        return "; ".join(r["problems"])
    return "ok: " + (", ".join(r["tools"]) or "no tool")


def markdown(take: str, results: list[dict], lines: dict) -> str:
    by = {(r["id"], r["variant"]): r for r in results}
    ids = list(dict.fromkeys(r["id"] for r in results))
    ok = {
        v: sum(not by[(i, v)]["problems"] for i in ids if (i, v) in by) for v in ("script", "heard")
    }
    greet = [i for i in ids if lines[i].section == 3]
    out = [
        f"# Brain check: {take}",
        "",
        "Understanding is the brain given the script's words; end to end is the brain given "
        "what hands-free listening heard.",
        "",
        "| | Understanding | End to end |",
        "| --- | --- | --- |",
        f"| Right tool and reply | {ok['script']}/{len(ids)} | {ok['heard']}/{len(ids)} |",
        f"| Greetings (section 3) | "
        f"{sum(not by[(i, 'script')]['problems'] for i in greet)}/{len(greet)} | "
        f"{sum(not by[(i, 'heard')]['problems'] for i in greet)}/{len(greet)} |",
        "",
        "| Line | Script | Heard as | Understanding | End to end |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i in ids:
        s, h = by[(i, "script")], by[(i, "heard")]
        out.append(f"| {i} | {lines[i].request} | {h['said']} | {_cell(s)} | {_cell(h)} |")
    return "\n".join(out) + "\n"
