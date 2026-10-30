"""Sends the test script's commands to an OpenAI-compatible endpoint and checks the tool calls.

Each command goes in twice: as written in the script, which tests understanding alone, and as
hands-free listening heard it in a replay run, which tests recognition and understanding
together. The endpoint is Hal's gateway or `brain.base_url`, and the model gets HAL's persona
and the tool schemas of the manifests named by `--manifest` or `HALVOICE_TOOL_MANIFESTS`. No
tool runs: each call is recorded and answered with "not carried out", which is enough to see
that the right tool was chosen. This module does not import Hal.
"""

from __future__ import annotations

import json
import os
import re
import time
import tomllib
from pathlib import Path

from halvoice import config, llm

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


MANIFEST_ENV = "HALVOICE_TOOL_MANIFESTS"  # os.pathsep-separated hal-module.toml files
MAX_ROUNDS = 4  # tool rounds per turn before the check gives up on a reply
NOT_RUN = "(test run: {name} was not carried out)"


def tool_schemas(manifests: list[Path]) -> list[dict]:
    """OpenAI tool schemas for every tool the manifests declare."""
    schemas = []
    for path in manifests:
        for tool in tomllib.loads(path.read_text(encoding="utf-8")).get("tools", []):
            params = tool.get("params", {})
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                k: {"type": v.get("type", "string")} for k, v in params.items()
                            },
                            "required": [k for k, v in params.items() if v.get("required")],
                        },
                    },
                }
            )
    return schemas


def manifest_paths(given: list[str] | None = None) -> list[Path]:
    """The manifests named on the command line or in `HALVOICE_TOOL_MANIFESTS`."""
    names = given or [m for m in os.environ.get(MANIFEST_ENV, "").split(os.pathsep) if m]
    paths = [Path(n).expanduser() for n in names]
    missing = [p for p in paths if not p.is_file()]
    if not paths or missing:
        raise SystemExit(
            f"voicetest brain needs tool manifests: pass --manifest FILE or set {MANIFEST_ENV}"
            + (f" (not found: {', '.join(map(str, missing))})" if missing else "")
        )
    return paths


class RecordingBrain:
    """Asks an OpenAI-compatible endpoint and records the tool calls without running them."""

    def __init__(self, persona: str, schemas: list[dict]) -> None:
        self.messages: list[dict] = [{"role": "system", "content": persona}]
        self.schemas = schemas
        self.last_turn: dict = {"reply": "", "tools": []}

    def reply(self, text: str) -> str:
        """Answers one spoken turn; the calls made are in `last_turn['tools']`."""
        self.messages.append({"role": "user", "content": text})
        called: list[dict] = []
        content = ""
        for _ in range(MAX_ROUNDS):
            msg = llm.complete(self.messages, tools=self.schemas)
            calls = msg.get("tool_calls") or []
            self.messages.append(
                {"role": "assistant", **{k: v for k, v in msg.items() if k != "role"}}
            )
            content = msg.get("content") or ""
            if not calls:
                break
            for call in calls:
                name = call["function"]["name"]
                called.append({"name": name, "arguments": call["function"].get("arguments", "")})
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": NOT_RUN.format(name=name),
                    }
                )
        self.last_turn = {"reply": content, "tools": called}
        return content


def run(rows: list[dict], out: Path, manifests: list[Path], say=print) -> str:
    """Asks the brain every command line, as scripted and as heard; returns the report."""
    expect = tomllib.loads(EXPECT.read_text())
    greetings, wants = expect["greeting"], expect["line"]
    lines = {ln.id: ln for ln in script.load() if ln.kind == "test"}
    take = sorted({r["condition"] for r in rows})[0]
    heard = {r["id"]: r["request"] for r in rows if r["condition"] == take}
    schemas = tool_schemas(manifests)
    persona = str(config.load().get("brain.persona", ""))
    results = []
    for conv in conversations(lines, wants):
        for variant in ("script", "heard"):
            texts = (
                [lines[i].request for i in conv]
                if variant == "script"
                else [heard.get(i, "") for i in conv]
            )
            brain = RecordingBrain(persona, schemas)
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
                except (llm.Unavailable, KeyError) as e:
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
