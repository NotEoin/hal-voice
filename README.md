# 🔴 hal-voice

![licence](https://img.shields.io/badge/licence-MIT-blue?style=flat-square)
![platform](https://img.shields.io/badge/platform-Linux-lightgrey?style=flat-square)
![python](https://img.shields.io/badge/python-3.11--3.12-blue?style=flat-square)
![voice](https://img.shields.io/badge/voice-100%25%20local-success?style=flat-square)

> *"I am completely operational, and all my circuits are functioning perfectly."*

Talk to Hal and have it answer out loud, in the voice of HAL 9000, with a red lens watching you
from the corner of the screen. Speech recognition and synthesis always run on your own machine; the
thinking is done by core Hal's local brain with the tools of your installed extensions, or by
whatever [Claude Code](https://claude.ai/code) you already have.

![The eye reacting to what it's doing](./media/eye.gif)

## 🌟 Highlights

- **It starts speaking before it's finished thinking.** Replies are synthesised sentence by sentence
  as they're generated, so you hear the first words in about a second rather than waiting for the
  whole answer.
- **Talk at a normal pace.** Say "Hal" at the start or end of a sentence and the request is heard in
  the same breath; after he answers, carry on for a few seconds without his name.
- **A fast local brain with tools.** In `direct` mode (the default) core Hal's brain answers with a
  local model and calls the tools of the extensions installed through hal-hub (calendar, memory,
  computer, lights and more), with first words in about a second. Long jobs go to Claude Code in the
  background.
- **Interrupt it whenever you like**: say his name while he talks and he stops to listen.
- **A red eye that reacts** - idle, listening, thinking, speaking.
- **Your voice never leaves the machine.** Speech recognition and synthesis are both local. What
  Claude Code does with the transcript is between you and your existing setup.
- **Works with the Claude Code you already have.** No model to install, nothing to configure -
  clone it and talk. Point it at a local model later if you want to.
- **Drop it into a normal session.** Run it as an MCP server and get speak/listen as tools inside a
  regular Claude Code session, without the full voice loop.
- **Bring your own voice.** Train it on any voice data you have the rights to.
- **One config file**, with environment variable overrides for everything in it.

## ℹ️ Overview

```
"Hal, ..."  ->  faster-whisper  ->  core Hal's brain + hal-hub tools (or Claude Code)
                                              v
                     speakers  <-  FX chain  <-  GPT-SoVITS
```

You say HAL's name (or hold a key), ask a question, and it answers aloud. Underneath, speech is
transcribed with faster-whisper, handed to the brain, and the reply is streamed back out through
GPT-SoVITS and a small effects chain that gives it the right weight and room.

hal-voice runs **no language model of its own**. It is a voice on the front of a brain: core Hal's
agent loop in `direct` mode, or a headless Claude Code session in `claude` mode.

### ✍️ Author

Built by [@NotEoin](https://github.com/NotEoin).

### Pointing it at a project

In `claude` mode, Claude Code is launched *inside a directory*, and picks up that directory's
`CLAUDE.md`, `.mcp.json` and `.claude/` settings automatically. By default that is the vault named in
`~/.config/hal/config.toml` (`[hal].vault`), else the current directory. To choose another:

```bash
./hal-voice --project ~/code/my-project   # HAL gains that project's memory and tools
./hal-voice connection                    # show what it resolved, and why
```

Set it permanently as `claude.project_dir` in [`config.yaml`](halvoice/config.yaml).

### Choosing a brain

`brain.mode` decides who answers:

- `direct` (the default) uses core Hal's brain (`halcore.brain`, in `core/hal` of the ecosystem). It
  streams the reply from a local Ollama model and calls the tools of the extensions installed
  through hal-hub, so first words arrive in about a second. Tasks with many steps go to Claude Code
  through a `deep_work` tool, and HAL tells you when they finish. The model and the other brain
  settings come from `[brain]` in `~/.config/hal/config.toml`, shared with Hal's chat and Telegram;
  `brain.*` in `config.yaml` overrides them for voice only, and keeps HAL's persona.
- `claude` runs Claude Code for every turn, so HAL has that project's full toolset and memory, at the
  cost of a few seconds' start-up per turn.

```bash
HALVOICE_BRAIN_MODEL=mistral-small3.2:24b ./hal-voice   # another model, for this run only
HALVOICE_BRAIN_MODE=claude ./hal-voice                  # Claude Code for every turn
```

Core Hal is found at `HAL_CORE`, else at `core/hal` in the ecosystem folder, else at `[hal].core` in
the shared config. Without it, `direct` mode logs why and falls back to `claude` mode.

In `claude` mode it can also follow a [Hal](https://github.com/NotEoin/Hal) install's profile: point
`claude.hal_root` at a checkout with a `hal.conf` (or run `./setup-voice.sh --hal auto`). To plug into
a setup that isn't Hal, set `claude.env_file` to any `KEY=VALUE` file of `ANTHROPIC_*` settings.

## 🚀 Usage

```bash
./hal-voice doctor       # check your audio devices, models and TTS server
./hal-voice              # start the voice loop
./hal-voice tts "hello"  # speak one line, no mic or model needed
./hal-voice mcp          # run just the speak/listen MCP server
./hal-voice eye          # run the eye on its own
./hal-voice connection   # show which Claude Code setup it will drive, and why
bin/hal-say "hello"      # speak a line if the voice server is up; silent in quiet hours
```

Through hal-hub, other modules and the brain can speak with the `voice_say` tool.

**Run `doctor` first.** Almost every problem in a voice pipeline turns out to be an audio device
problem, and it'll tell you in two seconds.

Hold **space** to talk, release to send. Speaking again while it's replying interrupts it.

### Configuration

Everything lives in [`config.yaml`](halvoice/config.yaml) - audio devices, activation, model choices, voice
tuning, the effects chain and the eye. Any value can be overridden with an environment variable:

```bash
HALVOICE_TTS_SPEED=0.9 ./hal-voice        # slow it down
HALVOICE_STT_MODEL=medium.en ./hal-voice  # more accurate transcription
```

### Testing hands-free listening

`voicetest` replays a recording of you reading [`voicetest/script.txt`](voicetest/script.txt)
through the same voice detector, name rules and speech recognition the service uses, without the
voice or the brain, and reports what woke HAL, what reached him and how fast:

```bash
./hal-voice voicetest split ~/recording.wav    # cut it into lines (kept in voice-data/tests/)
./hal-voice voicetest run                      # replay on a simulated clock, in a minute or so
./hal-voice voicetest run --live               # replay at real-time pace through the threads
./hal-voice voicetest run --simulate --set activation.keyword.end_silence_ms=500 --label e500
./hal-voice voicetest compare sim e500         # two runs side by side
./hal-voice voicetest brain --label sim        # send the commands heard to the brain, sandboxed
```

Record on the microphone HAL uses, with HAL turned off, saying "Marker one" and the take ("Desk,
take one") first. `--simulate` adds louder fans and a quieter voice, made from your recording,
until you have recorded those takes. The clips are your voice, so they stay out of git.

## ⬇️ Installation

hal-voice is a Hal ecosystem extension: it lives in `extensions/hal-voice` and is installed by the
ecosystem installer, which also renders its systemd units:

```bash
hal ext add voice
cd ~/src/hal-ecosystem/extensions/hal-voice
./setup-voice.sh          # the virtualenv, GPT-SoVITS and the voice profile
```

The units in `systemd/` are templates: the installer replaces `@ROOT@` with this folder and `@CORE@`
with core Hal's, and copies them to `~/.config/systemd/user`. `hal-tts.service` runs the GPT-SoVITS
server and `hal-voice.service` the hands-free loop; both belong to `hal.target` and use the GPU, so
`hal off` stops them.

`hal-tts.service` starts `hal-voice tts-server`, which exits with an error naming
`tts.server.gptsovits_dir` when the GPT-SoVITS checkout is missing. If the checkout is on a drive
that mounts late, make the unit wait for it with a drop-in
(`systemctl --user edit hal-tts.service`, then add `RequiresMountsFor=/path/to/the/drive`).
`hal-voice.service` sets no `PATH`, so a `claude` installed under npm's global prefix needs
`claude.command` set to its absolute path. Python's CA bundle is found automatically
(`halvoice.util.ca_bundle`).

It still runs on its own (`git clone https://github.com/NotEoin/hal-voice.git`, then
`./setup-voice.sh`), in `claude` mode.

**Requirements:**

| | |
|---|---|
| **OS** | Linux, with PulseAudio or PipeWire |
| **Python** | 3.11 or 3.12. **Not 3.13+**: numpy 1.x has no wheels there, and parts of the audio stack still need it. `setup-voice.sh` finds a suitable interpreter, or falls back to conda |
| **System packages** | PortAudio and FFmpeg |
| **GPU** | Optional - see the profiles below. Voice synthesis wants ~2GB of VRAM but will run on CPU |
| **Hardware** | A working microphone and speakers |
| **Also needed** | A [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) checkout with its pretrained models |

`setup-voice.sh` reads your GPU and proposes a profile; `--profile <name>` overrides it. This is
about **VRAM only** - it does not change which model does the thinking.

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

The quickest route is your own copy of *2001: A Space Odyssey*. Put the film (and any subtitle
files) in `voice-data/film/` and run:

```bash
./make-hal-voice
```

It finds every line HAL speaks clearly, cleans and trains on them, and installs the result.
**-> How it works and what it accepts: [VOICE-BUILD.md](VOICE-BUILD.md)**

To train on clips you already have instead:

1. Put your clips and a `transcripts.csv` somewhere, and point `voice-source.local` at it.
2. `python voice-data/fetch-voice-lines.py` - imports them.
3. `python voice-data/prepare_dataset.py` - slices, cleans and labels them.
4. Fine-tune - see **[gptsovits/train-hal.md](gptsovits/train-hal.md)**.

Around forty clean, well-transcribed clips is enough for a recognisable result. Consistency matters
more than quantity - same microphone, same room, same delivery.

**-> Full guide: [voice-data/README.md](voice-data/README.md)**

## About the film voice pipeline

hal-voice includes `make-hal-voice` and the `voicebuild` package. They turn a copy of a film that
you own into training data for a voice model and train that model on your own machine. Read this
before you use them.

- **No media is included.** This repository contains source code only. It does not contain,
  link to or download any film, audio, subtitles, transcripts or trained voice weights. Short
  quotations of film dialogue appear in the default persona, configuration comments and tests.
- **Bring your own lawfully owned copy.** The pipeline reads a film file and subtitles that you
  supply. Use it only on a copy you own and have the right to use this way. In some countries
  copying a disc, or getting round its copy protection, is not allowed even for personal use.
- **Do not distribute what it makes.** The clips, transcripts and voice weights it produces are
  derived from a copyrighted performance. Keep them on your own machine. Do not share, publish
  or sell them, including as model weights on model-sharing sites.
- **A real person's voice.** HAL 9000 was voiced by the actor Douglas Rain (1927 to 2018).
  Cloning a real person's voice can engage personality, publicity and similar rights, which in
  some places last after death, and the synthetic-media rules of the platforms you use. Do not
  use a cloned voice to impersonate anyone, to mislead people, or for commercial purposes.
- **No affiliation.** "HAL 9000" and "2001: A Space Odyssey" belong to their owners. This
  project is not affiliated with or endorsed by them.
- **The licence covers the code only.** The MIT licence grants no rights in any film,
  performance, character or voice.
- **This is not legal advice.** Copyright, format shifting, text and data mining, and voice
  and likeness rights differ between countries and change over time. If you are unsure whether
  your use is lawful where you live, ask a lawyer.

## 🧭 Limitations

- **The trained wake word is unreliable.** openWakeWord detection is disabled by default, and its
  dependencies aren't installed; run `./setup-voice.sh --with-wake` to experiment. For hands-free
  use, set `activation.keyword.enabled: true` instead: one microphone stream is cut into utterances,
  each transcribed once (on the GPU with `stt.device: auto`), and a turn starts when one begins or
  ends with "Hal", greeting first or not ("Good morning, Hal"). It costs some GPU or CPU time
  whenever someone is speaking nearby, and "Hey Al" still sounds like "Hey Hal" to it.
- **The direct brain needs a model that calls tools reliably.** Smaller models tend to say they did
  something instead of doing it; the brain holds back such replies and retries once, but a 20B-class
  model with good tool use (mistral-small3.2 measured best here) is the practical minimum.
- **It's not instant.** Streaming hides a lot of the delay, but synthesis still has to happen -
  and on the `cloud` (CPU) profile it runs slower than real time, so HAL pauses before answering.
- **The voice effects are tuned to one voice.** Treat the defaults as a starting point.
- Interrupting can clip the first syllable of your next sentence.
- Linux only.

## 🔗 Related projects

- **[Hal](https://github.com/NotEoin/Hal)** - a local assistant with an Obsidian vault as its memory.
  Its brain answers in `direct` mode, and hal-hub gives it the tools of the other Hal extensions.

## 💭 Feedback and contributing

Got it working with a different TTS backend, or trained a wake word that actually fires? Open an
[issue](https://github.com/NotEoin/hal-voice/issues) or start a
[discussion](https://github.com/NotEoin/hal-voice/discussions).

## 📄 Licence

MIT - see [LICENSE](LICENSE). This covers the code only, not any voice data you bring to it.
