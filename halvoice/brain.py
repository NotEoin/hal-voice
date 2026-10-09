"""The direct brain: a local model that streams HAL's reply and calls Hal's tools itself.

Each turn sends the conversation to Ollama with the few hal-hub tools that fit
the request (plus search, so the model can look for more), streams the reply
text as it is generated, and runs any tool calls in between. There is no agent
start-up, so the first words arrive in well under a second. Long, many-step
work goes to Claude Code through the deep_work tool, in the background.

Needs hal-hub (found through brain.hub_path, HAL_HUB or ~/.config/hal/config.toml).
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import tomllib
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Iterator

from . import config
from .util import get_logger

log = get_logger(__name__)

HAL_CONFIG = Path(os.environ.get("HAL_CONFIG", "~/.config/hal/config.toml")).expanduser()

TOOL_RULES = """
RULE ONE, above everything below: you act through tool calls. When Dave asks you to do, remember,
schedule, book, plan, change or look up anything about his life, his plans or this computer, call
the tool FIRST and speak after it returns. Saying you did something without a tool call is a lie.
Any plan with a day or a time (lunch, a meeting, an appointment, a call) is a calendar_add.
General knowledge (facts, explanations, maths, advice) needs no tool: answer it directly.

Tools (how you act on Dave's world):
- You have tools for his calendar, tasks, notes and memory, computer, files, lights, media, web,
  weather, timers, email, jobs and more. When a request needs one, call it; do not describe
  what you would do. If none of the tools offered fits, call hal_find_tools to search for one.
- Times: pass the time words exactly as Dave said them ("thursday at ten") to calendar and timer tools.
- A result with needs_confirmation means the action is destructive: ask Dave in one short sentence,
  and only when he agrees call the tool again with _confirmed set to true.
- Anything Dave tells you about himself that will still be true next month, save with memory_fact.
  Anything he asks you to remember, save with memory_remember.
- For a long job (writing or changing code, researching and writing a document, anything with
  many steps), call deep_work with a complete description; it runs in the background and you
  will be told when it is done. Say you have started it.
- After a tool runs, tell Dave the outcome in a sentence or two. Never read raw data or JSON aloud.
"""

# A reply that says something was done, or that it cannot be done, when no tool ran,
# is held back and retried once.
SUSPECT = re.compile(
    r"\b(I(?:'ve| have|'ll| will)? (?:added|scheduled|set|saved|noted|created|booked|put|moved|cancell?ed|deleted|"
    r"removed|turned|sent|started|reminded|recorded|updated|remembered|stored|logged|written|made a note)|"
    r"(?:added|scheduled|noted|saved|stored|remembered) (?:it|that|this)|can'?not|can't|unable to|not able to|"
    r"(?:don't|do not) have (?:the )?(?:ability|capability|access|tools?|means)|I'?m sorry|I am sorry|"
    r"I apologi[sz]e|unfortunately)\b", re.I)
NO_TOOL = ("Your draft reply was not spoken: it either said something was done without a tool call, or said "
           "you cannot do it. You have tools for nearly everything. If Dave asked for an action, call the right "
           "tool now: memory_fact for a lasting fact about Dave, memory_remember for anything to remember, "
           "calendar_add for any plan with a day or time, or hal_find_tools to search. If he asked a general "
           "question, answer it from your own knowledge. Reply with exactly OK only if it is truly impossible.")

DEEP_WORK = {"type": "function", "function": {
    "name": "deep_work",
    "description": "Start a long, many-step task in the background (coding, editing files, research, writing a "
                   "document). The agent works on its own and Hal reports back when it finishes.",
    "parameters": {"type": "object", "properties": {
        "task": {"type": "string", "description": "the full task, with every detail Dave gave"}},
        "required": ["task"]}}}


LEAKED = re.compile(r"^\s*(?:\[TOOL_CALLS\])?\s*([a-z_]+)\s*(\{.*\})\s*$", re.S)


def _leaked_call(content: str, names: set[str]) -> dict | None:
    """Returns a tool call the model wrote as text ("name{...json...}"), or None."""
    m = LEAKED.match(content or "")
    if not m or m.group(1) not in names:
        return None
    try:
        return {"function": {"name": m.group(1), "arguments": json.loads(m.group(2))}}
    except json.JSONDecodeError:
        return None


def _hal_config() -> dict:
    try:
        return tomllib.loads(HAL_CONFIG.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _import_hub():
    """Imports halhub from the configured path, or returns None if it is not installed."""
    cfg = config.load()
    candidates = [os.environ.get("HAL_HUB"), cfg.get("brain.hub_path")]
    paths = _hal_config().get("hub", {}).get("module_paths", [])
    candidates += [str(Path(p).expanduser() / "hal-hub") for p in paths]
    for c in candidates:
        if c and (Path(config.resolve(c) or c) / "halhub").is_dir():
            path = str(Path(config.resolve(c) or c))
            if path not in sys.path:
                sys.path.insert(0, path)
            break
    try:
        import halhub
        return halhub
    except ImportError:
        log.warning("hal-hub not found: HAL will talk but cannot use tools")
        return None


class DirectBrain:
    """One spoken conversation with a local model and Hal's tools."""

    def __init__(self) -> None:
        cfg = config.load()
        shared = _hal_config().get("hal", {})
        self.url = (cfg.get("brain.ollama_url") or shared.get("ollama_url") or "http://localhost:11434").rstrip("/")
        self.model = cfg.get("brain.model") or shared.get("chat_model") or "qwen3.5:9b"
        self.keep_alive = str(cfg.get("brain.keep_alive", "30m"))
        self.history_turns = int(cfg.get("brain.history_turns", 10))
        self.max_rounds = int(cfg.get("brain.max_tool_rounds", 5))
        self.tool_limit = int(cfg.get("brain.tools_per_turn", 8))
        self.num_ctx = int(cfg.get("brain.num_ctx", 8192))
        self.persona = str(cfg.get("brain.persona") or cfg.get("claude.voice_system_prompt", "")).strip()
        self.hub = _import_hub()
        self.history: list[dict] = []
        self.deep_worker = None
        self.on_deep_done = None        # callback(text) when background work finishes
        log.info("brain: %s via %s, %s", self.model, self.url, "with tools" if self.hub else "no tools")

    # -- context -----------------------------------------------------------------
    def _system(self) -> str:
        facts = ""
        if self.hub:
            got = self.hub.call("memory_profile", {})
            if isinstance(got.get("result"), str) and got["result"].strip():
                facts = "\nWhat you know about Dave:\n" + got["result"][:2500]
        now = datetime.now().strftime("%A %-d %B %Y, %H:%M")
        return f"{TOOL_RULES}\n{self.persona}\nIt is {now}.{facts}"

    def _tools(self, text: str) -> list[dict]:
        if not self.hub:
            return [DEEP_WORK]
        specs = self.hub.ollama_tools(text, self.tool_limit)
        for spec in specs:
            tool = self.hub.get(spec["function"]["name"])
            if tool is not None and tool.confirm:
                spec["function"]["parameters"]["properties"]["_confirmed"] = {
                    "type": "boolean", "description": "true only after Dave has agreed"}
        return specs + [DEEP_WORK]

    def warm(self) -> None:
        """Loads the model into memory so the first turn is not slow."""
        try:
            self._post({"model": self.model, "messages": [], "keep_alive": self.keep_alive,
                        "options": {"num_ctx": self.num_ctx}}, stream=False).read()
        except OSError as e:
            log.warning("could not warm %s: %s", self.model, e)

    # -- the turn ----------------------------------------------------------------
    def _post(self, payload: dict, stream: bool = True):
        req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps({**payload, "stream": stream}).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=300)

    def _run_tool(self, name: str, args: dict) -> dict:
        if name == "deep_work":
            return self._deep_work(args.get("task", ""))
        if not self.hub:
            return {"error": "no tools are installed"}
        started = time.time()
        out = self.hub.call(name, args)
        log.info("tool %s(%s) -> %s in %.1fs", name, json.dumps(args)[:120], "error" if "error" in out else "ok",
                 time.time() - started)
        return out

    def _deep_work(self, task: str) -> dict:
        if self.deep_worker and self.deep_worker.is_alive():
            return {"error": "a background task is already running"}

        def work():
            from .claude_client import ClaudeSession
            try:
                reply = "".join(ClaudeSession().ask(task))
            except Exception as e:  # noqa: BLE001 - reported to Dave rather than raised
                reply = f"The background task failed: {e}"
            if self.hub:
                self.hub.call("memory_remember", {"text": f"Task: {task}\n\nResult:\n{reply}", "title": "Background task"})
            if self.on_deep_done:
                self.on_deep_done(reply)

        self.deep_worker = threading.Thread(target=work, name="deep-work", daemon=True)
        self.deep_worker.start()
        return {"started": True}

    def _recent(self) -> list[dict]:
        """Returns the last history_turns exchanges, starting at a user message."""
        users = [i for i, m in enumerate(self.history) if m["role"] == "user"]
        start = users[-self.history_turns] if len(users) > self.history_turns else 0
        self.history = self.history[start:]
        return self.history

    def _stream_round(self, messages: list[dict], tools: list[dict]):
        """Yields ('text', delta) and finally ('done', (content, calls)) for one model call."""
        calls, content = [], ""
        with self._post({"model": self.model, "messages": messages, "tools": tools, "think": False,
                         "keep_alive": self.keep_alive,
                         "options": {"temperature": 0.4, "num_ctx": self.num_ctx}}) as r:
            for line in r:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                if msg.get("content"):
                    content += msg["content"]
                    yield "text", msg["content"]
                calls += msg.get("tool_calls") or []
                if chunk.get("done"):
                    break
        yield "done", (content, calls)

    def ask(self, text: str) -> Iterator[str]:
        """Yields the reply's text as it streams, running tool calls along the way.

        The first sentence of each model reply is held back until it is clear the model
        is not claiming an action it never took, or refusing something a tool can do;
        such a reply is replaced by one silent retry before anything is spoken.
        """
        messages = [{"role": "system", "content": self._system()}, *self._recent(),
                    {"role": "user", "content": text}]
        tools = self._tools(text)
        reply = ""
        tools_ran = retried = False
        for _ in range(self.max_rounds + 1):
            held, released = "", False
            content, calls = "", []
            for kind, value in self._stream_round(messages, tools):
                if kind == "done":
                    content, calls = value
                    break
                if released:
                    reply += value
                    yield value
                    continue
                held += value
                if re.search(r"[.!?](\s|$)", held) and not SUSPECT.search(held):
                    released = True
                    reply += held
                    yield held
            if not calls:
                leaked = _leaked_call(content, {t["function"]["name"] for t in tools})
                if leaked:
                    calls, content, held, released = [leaked], "", "", True
            messages.append({"role": "assistant", "content": content, **({"tool_calls": calls} if calls else {})})
            if calls:
                tools_ran = True
                for call in calls:
                    fn = call.get("function", {})
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        args = json.loads(args or "{}")
                    out = self._run_tool(fn.get("name", ""), args)
                    messages.append({"role": "tool", "tool_name": fn.get("name", ""),
                                     "content": json.dumps(out, default=str)[:6000]})
                continue
            if not released and held:
                if not tools_ran and not retried and SUSPECT.search(held):
                    retried = True
                    log.info("held back %r; asking the model to use a tool", held[:80])
                    messages[-1:] = [{"role": "system", "content": NO_TOOL}]
                    original = held
                    continue
                if retried and held.strip().rstrip(".") == "OK":
                    held = original
                reply += held
                yield held
            break
        # The whole turn, tool calls and results included, so later turns see how things were done.
        turn = [m for m in messages[1:] if m.get("role") != "system"]
        self.history += turn[-12:] if turn else [{"role": "user", "content": text}]
        if self.hub and reply.strip():
            threading.Thread(target=self.hub.call, args=("memory_log_conversation",
                             {"user": text, "assistant": reply}), daemon=True).start()
