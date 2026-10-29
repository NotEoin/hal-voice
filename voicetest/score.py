"""Judges each replayed clip against what its line should do, and summarises per condition."""

from __future__ import annotations

import difflib
from collections import defaultdict

import numpy as np

from . import script

FORMS = ["plain", "greeting", "name-last"]


def wer(got: str, want: str) -> float:
    """Word error rate of got against want, after spelling numbers and units alike."""
    a, b = script.words(got), script.words(want)
    if not b:
        return 0.0 if not a else 1.0
    d = np.arange(len(b) + 1, dtype=float)
    for i in range(1, len(a) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(b) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
    return float(d[-1] / len(b))


def cut(got: str, want: str) -> str:
    """ "start" or "end" when the request lost words there, else ""."""
    a, b = script.words(got), script.words(want)
    if not a or not b:
        return ""
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    if ops[0][0] == "insert":
        return "start"
    if ops[-1][0] == "insert":
        return "end"
    return ""


def judge(r) -> dict:
    """What happened to one clip, and whether that was right."""
    c = r.clip
    turns = [e for e in r.events if e[1] == "turn"]
    first = r.events[0] if r.events else None
    row = {
        "condition": c["condition"],
        "id": c["id"],
        "section": c["section"],
        "text": c["text"],
        "form": c.get("form", ""),
        "heard": " | ".join(t for t in r.heard if t),
        "woke": bool(r.events),
        "turns": len(turns),
        "request": turns[0][2] if turns else "",
        "latency": None,
        "wer": None,
        "cut": "",
        "problems": [],
    }
    p = row["problems"]
    if c["kind"] == "chat" or (c["section"] == script.NOT_FOR_HAL):
        if r.events:
            p.append(f"false wake: {first[1]} {first[2]!r}")
        return row
    if not (c["wake"] or c["followup"]):
        return row
    if c["wake"] and not r.events:
        p.append("missed wake")
        return row
    if c["turn"]:
        if not turns:
            p.append(
                "request not caught"
                if c["then_request"] or c["followup"] or r.events
                else "missed wake"
            )
            return row
        row["latency"] = turns[0][0]
        row["wer"] = round(wer(turns[0][2], c["request"]), 3)
        row["cut"] = cut(turns[0][2], c["request"])
        if row["cut"]:
            p.append(f"words cut at the {row['cut']}")
        if len(turns) > 1:
            p.append(f"split into {len(turns)} turns")
        if row["wer"] >= 0.5:
            p.append("request misheard")
    else:
        row["latency"] = first[0]
        if turns:
            p.append(f"unwanted request {turns[0][2]!r}")
    return row


def summary(rows: list[dict]) -> dict[str, dict]:
    """Per condition: wake hit rates by form, follow-ups, false wakes, cuts, latency, WER."""
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    out = {}
    for cond, rs in by.items():
        s = {}
        for form in FORMS:
            w = [r for r in rs if r["form"] == form and r["section"] != script.NOT_FOR_HAL]
            s[f"wake {form}"] = (sum(r["woke"] for r in w), len(w))
        fu = [r for r in rs if not r["form"] and r["section"] not in (script.NOT_FOR_HAL,)]
        s["follow-ups"] = (sum(r["turns"] > 0 for r in fu), len(fu))
        nf = [r for r in rs if r["section"] == script.NOT_FOR_HAL]
        s["false wakes"] = (sum(r["woke"] for r in nf), len(nf))
        want = [r for r in rs if r["latency"] is not None or "request not caught" in r["problems"]]
        s["requests caught"] = (sum(r["turns"] > 0 for r in want), len(want))
        s["cut off"] = sum(bool(r["cut"]) for r in rs)
        lat = [r["latency"] for r in rs if r["latency"] is not None]
        s["latency mean"] = round(float(np.mean(lat)), 2) if lat else None
        s["latency worst"] = round(float(np.max(lat)), 2) if lat else None
        w = [r["wer"] for r in rs if r["wer"] is not None]
        s["wer"] = round(float(np.mean(w)), 3) if w else None
        s["failures"] = sum(bool(r["problems"]) for r in rs)
        out[cond] = s
    return out


def fmt(v) -> str:
    if isinstance(v, tuple):
        return f"{v[0]}/{v[1]} ({100 * v[0] / v[1]:.0f}%)" if v[1] else "-"
    return "-" if v is None else str(v)


def markdown(label: str, rows: list[dict], note: str = "") -> str:
    """The report: a summary per condition, then every failure with its clip."""
    s = summary(rows)
    conds = list(s)
    keys = list(next(iter(s.values()))) if s else []
    out = [f"# Hands-free replay: {label}", ""]
    if note:
        out += [note, ""]
    out += ["| | " + " | ".join(conds) + " |", "| --- |" + " --- |" * len(conds)]
    for k in keys:
        cells = []
        for c in conds:
            v = s[c][k]
            if k == "wer" and v is not None:
                cells.append(f"{v:.1%}")
            elif k.startswith("latency") and v is not None:
                cells.append(f"{v:.2f} s")
            else:
                cells.append(fmt(v))
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    out += [
        "",
        "## Failures",
        "",
        "| Condition | Line | Said | Heard | Problem |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        if r["problems"]:
            out.append(
                f"| {r['condition']} | {r['id']} | {r['text']} | {r['heard']} | "
                f"{'; '.join(r['problems'])} |"
            )
    return "\n".join(out) + "\n"


def compare(labelled: dict[str, list[dict]]) -> str:
    """Before and after: each run's summary side by side, per condition."""
    sums = {k: summary(v) for k, v in labelled.items()}
    conds = sorted({c for s in sums.values() for c in s})
    out = []
    for c in conds:
        keys = next(iter(s[c] for s in sums.values() if c in s)).keys()
        out += [f"### {c}", "", "| | " + " | ".join(sums) + " |", "| --- |" + " --- |" * len(sums)]
        for k in keys:
            cells = []
            for s in sums.values():
                v = s.get(c, {}).get(k)
                if k == "wer" and v is not None:
                    cells.append(f"{v:.1%}")
                elif k.startswith("latency") and v is not None:
                    cells.append(f"{v:.2f} s")
                else:
                    cells.append(fmt(v))
            out.append(f"| {k} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)
