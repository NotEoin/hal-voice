"""Local model access through Ollama, and the GPU guard that keeps it polite.

Model work waits while another program (a training run, a llama.cpp server)
holds the GPU. Stdlib only.
"""
import json
import subprocess
import urllib.request

from .config import shared


class Unavailable(Exception):
    pass


# Hal's own GPU users: Ollama, the voice server and speech recognition.
OWN = ("ollama", "GPT-SoVITS", "api_v2.py", "hal-voice", "halvoice", "faster_whisper")


def gpu_busy() -> str | None:
    """Returns the name of a program other than Hal's own that is using the GPU, or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    ignore = OWN + tuple(shared().get("gpu_ignore", []))
    others = []
    for line in out.splitlines():
        pid, _, name = line.partition(",")
        try:
            cmd = open(f"/proc/{pid.strip()}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            cmd = name
        if not any(i in cmd or i in name for i in ignore):
            others.append(name.strip().rsplit("/", 1)[-1])
    return ", ".join(others) or None


def _post(path: str, payload: dict, timeout: float) -> dict:
    url = shared()["ollama_url"].rstrip("/") + path
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except OSError as e:
        raise Unavailable(f"Ollama: {e}") from e


def chat(system: str, user: str, *, model: str | None = None, json_out: bool = False,
         images: list[str] | None = None, timeout: float = 300) -> str:
    """Returns the model's reply to one prompt."""
    msg = {"role": "user", "content": user}
    if images:
        msg["images"] = images
    payload = {"model": model or shared()["chat_model"], "stream": False, "think": False,
               "messages": [{"role": "system", "content": system}, msg]}
    if json_out:
        payload["format"] = "json"
        payload["options"] = {"temperature": 0.1}
    return _post("/api/chat", payload, timeout)["message"]["content"]


def chat_json(system: str, user: str, **kw) -> dict:
    out = chat(system, user, json_out=True, **kw)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {}


def embed(texts: list[str], timeout: float = 120) -> list[list[float]]:
    """Returns one embedding per text."""
    return _post("/api/embed", {"model": shared()["embed_model"], "input": texts}, timeout)["embeddings"]
