# 🔴 hal-voice

![licence](https://img.shields.io/badge/licence-MIT-blue?style=flat-square)
![platform](https://img.shields.io/badge/platform-Linux-lightgrey?style=flat-square)
![python](https://img.shields.io/badge/python-3.10--3.12-blue?style=flat-square)
![voice](https://img.shields.io/badge/voice-100%25%20local-success?style=flat-square)

> *"I am completely operational, and all my circuits are functioning perfectly."*

Talk to [Claude Code](https://claude.ai/code) and have it answer out loud, in the voice of HAL 9000,
with a red lens watching you from the corner of the screen. Speech recognition and synthesis always
run on your own machine; the thinking is done by whatever Claude Code you already have — cloud by
default, or a local model if you point it at one.

![The eye reacting to what it's doing](./media/eye.gif)

## 🌟 Highlights

- **It starts speaking before it's finished thinking.** Replies are synthesised sentence by sentence
  as they're generated, so you hear the first words in about a second rather than waiting for the
  whole answer.
- **Interrupt it whenever you like** — speaking again cuts it off mid-sentence.
- **A red eye that reacts** — idle, listening, thinking, speaking.
- **Your voice never leaves the machine.** Speech recognition and synthesis are both local. What
  Claude Code does with the transcript is between you and your existing setup.
- **Works with the Claude Code you already have.** No model to install, nothing to configure —
  clone it and talk. Point it at a local model later if you want to.
- **Drop it into a normal session.** Run it as an MCP server and get speak/listen as tools inside a
  regular Claude Code session, without the full voice loop.
- **Bring your own voice.** Train it on any voice data you have the rights to.
- **One config file**, with environment variable overrides for everything in it.

## ℹ️ Overview

```
push-to-talk  →  faster-whisper  →  Claude Code (cloud, or your local model)
                                              ↓
                     speakers  ←  FX chain  ←  GPT-SoVITS
```

You hold a key, ask a question, and it answers aloud. Underneath, speech is transcribed with
faster-whisper, handed to Claude Code running headless, and the reply is streamed back out through
GPT-SoVITS and a small effects chain that gives it the right weight and room.

hal-voice runs **no model of its own**. It is a voice on the front of a Claude Code session, so
whatever that session can already do — your tools, your MCP servers, your project's memory — it can
now do out loud.

### ✍️ Author

Built by [@NotEoin](https://github.com/NotEoin).

### Pointing it at a project

Claude Code is launched *inside a directory*, and picks up that directory's `CLAUDE.md`, `.mcp.json`
and `.claude/` settings automatically. That is the whole integration mechanism:

```bash
./hal-voice --project ~/code/my-project   # HAL gains that project's memory and tools
./hal-voice connection                    # show what it resolved, and why
```

Set it permanently as `claude.project_dir` in [`config.yaml`](config.yaml).

It also pairs with [Hal](https://github.com/NotEoin/Hal) — a local Claude Code setup with an
Obsidian vault as its memory. Point `claude.hal_root` at a Hal checkout (or run
`./setup-voice.sh --hal auto`) and hal-voice reads that install's `hal.conf`, following its
24GB/12GB/cloud profile automatically so you only choose once. To plug into a setup that isn't Hal,
set `claude.env_file` to any `KEY=VALUE` file of `ANTHROPIC_*` settings.

## 🚀 Usage

```bash
./hal-voice doctor       # check your audio devices, models and TTS server
./hal-voice              # start the voice loop
./hal-voice tts "hello"  # speak one line, no mic or model needed
./hal-voice mcp          # run just the speak/listen MCP server
./hal-voice eye          # run the eye on its own
./hal-voice connection   # show which Claude Code setup it will drive, and why
```

**Run `doctor` first.** Almost every problem in a voice pipeline turns out to be an audio device
problem, and it'll tell you in two seconds.

Hold **space** to talk, release to send. Speaking again while it's replying interrupts it.

### Configuration

Everything lives in [`config.yaml`](config.yaml) — audio devices, activation, model choices, voice
tuning, the effects chain and the eye. Any value can be overridden with an environment variable:

```bash
HALVOICE_TTS_SPEED=0.9 ./hal-voice        # slow it down
HALVOICE_STT_MODEL=medium.en ./hal-voice  # more accurate transcription
```

## ⬇️ Installation

```bash
git clone https://github.com/NotEoin/hal-voice.git
cd hal-voice
./setup-voice.sh
```

**Requirements:**

| | |
|---|---|
| **OS** | Linux, with PulseAudio or PipeWire |
| **Python** | 3.10–3.12. **Not 3.13+** — numpy 1.x has no wheels there, and parts of the audio stack still need it. `setup-voice.sh` finds a suitable interpreter, or falls back to conda |
| **System packages** | PortAudio and FFmpeg |
| **GPU** | Optional — see the profiles below. Voice synthesis wants ~2GB of VRAM but will run on CPU |
| **Hardware** | A working microphone and speakers |
| **Also needed** | A [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) checkout with its pretrained models |

`setup-voice.sh` reads your GPU and proposes a profile; `--profile <name>` overrides it. This is
about **VRAM only** — it does not change which model does the thinking.

| Profile | Speech recognition | Voice synthesis | |
|---|---|---|---|
| `24gb` | GPU | GPU (fp16) | Fastest |
| `12gb` | CPU | GPU (fp16) | Leaves room for a local LLM beside it |
| `cloud` | CPU | CPU | No NVIDIA GPU required at all |

The `cloud` profile is not cosmetic: GPT-SoVITS ships configured for CUDA and won't start without
it, so this rewrites its device settings. Synthesis on CPU is slower than real time, so expect HAL
to pause before speaking.

Developed on an RTX 4070 (12GB); originally built on an RTX 3090.

## 🎙️ Giving it a voice

**No audio ships with this repository.** You supply the voice data.

1. Put your clips and a `transcripts.csv` somewhere, and point `voice-source.local` at it.
2. `python voice-data/fetch-voice-lines.py` — imports them.
3. `python voice-data/prepare_dataset.py` — slices, cleans and labels them.
4. Fine-tune — see **[gptsovits/train-hal.md](gptsovits/train-hal.md)**.

Around forty clean, well-transcribed clips is enough for a recognisable result. Consistency matters
more than quantity — same microphone, same room, same delivery.

**→ Full guide: [voice-data/README.md](voice-data/README.md)**

### ⚖️ Please use this responsibly

HAL was performed by the late Douglas Rain. This repository ships the *pipeline*, never the
performance — no audio, no transcripts and no trained models are included, and the import script
will not download audio from public sources.

Use voice data you recorded yourself or have clear permission to use. Keep anything you train
private and personal: don't distribute the model or its output, don't present it as someone else,
and don't use it commercially. In most places a person's voice is protected, and cloning one you
don't have rights to is not okay.

## 🧭 Limitations

- **Wake word doesn't work yet.** The code is there and "Hey HAL" is configurable, but detection is
  unreliable and it's disabled by default, so its dependencies aren't installed either — run
  `./setup-voice.sh --with-wake` if you want to experiment. **Push-to-talk is the supported way in.**
- **It's not instant.** Streaming hides a lot of the delay, but synthesis still has to happen —
  and on the `cloud` (CPU) profile it runs slower than real time, so HAL pauses before answering.
- **The voice effects are tuned to one voice.** Treat the defaults as a starting point.
- Interrupting can clip the first syllable of your next sentence.
- Linux only.

## 🔗 Related projects

- **[Hal](https://github.com/NotEoin/Hal)** — a local Claude Code setup with an Obsidian vault as
  its memory. hal-voice sits on top of it, or runs perfectly well without it.

## 💭 Feedback and contributing

Got it working with a different TTS backend, or trained a wake word that actually fires? Open an
[issue](https://github.com/NotEoin/hal-voice/issues) or start a
[discussion](https://github.com/NotEoin/hal-voice/discussions).

## 📄 Licence

MIT — see [LICENSE](LICENSE). This covers the code only, not any voice data you bring to it.
