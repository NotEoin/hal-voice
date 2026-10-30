"""A fake `hal acp` for the tests: speaks ACP v1 over stdio and follows the prompt text.

Prompt text starting with `slow` never answers (until cancelled), `ask` raises a broker
permission request first, `ask-other` raises a non-broker one, `die` exits. The first
argument is a log file that receives every message the client sent.
"""

import json
import sys

log = open(sys.argv[1], "a", encoding="utf-8")


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def chunk(sid, text):
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text},
                },
            },
        }
    )


def permission(meta):
    send(
        {
            "jsonrpc": "2.0",
            "id": 900,
            "method": "session/request_permission",
            "params": {
                "sessionId": "0123456789ab",
                "toolCall": {"toolCallId": "t1"},
                "options": [
                    {"optionId": "allow-1", "name": "Allow", "kind": "allow_once"},
                    {"optionId": "deny-1", "name": "Deny", "kind": "reject_once"},
                ],
                "_meta": meta,
            },
        }
    )
    return json.loads(sys.stdin.readline())


for line in sys.stdin:
    msg = json.loads(line)
    log.write(line)
    log.flush()
    method, rid = msg.get("method"), msg.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": 1}})
    elif method == "session/new":
        send({"jsonrpc": "2.0", "id": rid, "result": {"sessionId": "0123456789ab"}})
    elif method == "session/prompt":
        text = msg["params"]["prompt"][-1]["text"]
        sid = msg["params"]["sessionId"]
        if text == "die":
            sys.exit(3)
        if text == "slow":
            chunk(sid, "starting ")
            for more in sys.stdin:
                log.write(more)
                log.flush()
                if json.loads(more).get("method") == "session/cancel":
                    chunk(sid, "late chunk")
                    break
            send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "cancelled"}})
            continue
        if text in ("ask", "ask-other"):
            meta = {"hal/origin": "broker", "hal/summary": "Send the email?"}
            if text == "ask-other":
                meta = {"hal/summary": "not from the broker"}
            answer = permission(meta)
            log.write("ANSWER " + json.dumps(answer["result"]) + "\n")
            log.flush()
        chunk(sid, "Good evening, ")
        chunk(sid, "Dave.")
        send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})
