# hal-voice

![licence](https://img.shields.io/badge/licence-MIT-blue?style=flat-square)
![platform](https://img.shields.io/badge/platform-Linux%20%7C%20WSL2-lightgrey?style=flat-square)
![python](https://img.shields.io/badge/python-3.11--3.12-blue?style=flat-square)

> *"I am completely operational, and all my circuits are functioning perfectly."*

Talk to your computer and hear HAL 9000 answer, with a red lens in the corner of the screen that
watches you while it listens. Speech recognition (faster-whisper) and the voice (GPT-SoVITS) run on
your own machine. The reply comes from the brain you choose: Hal's local agent, any
OpenAI-compatible server, or Claude Code.

![The eye reacting to what it is doing](./media/eye.gif)

```
"Hal, ..."  ->  faster-whisper  ->  brain (Hal agent, OpenAI-compatible server or Claude Code)
                                              v
                     speakers  <-  FX chain  <-  GPT-SoVITS
```

- **It speaks before it has finished thinking.** Replies are synthesised sentence by sentence as
  they are generated.
- **Interrupt it any time.** Say "Hal" while it speaks and it stops.
- **Just talk.** Say "Hal" at the start or end of a request; for a few seconds after an answer you
  can carry on without the name.
- **Your voice stays home, and so does your GPU.** By default it listens and speaks on the CPU and
  holds no GPU memory while it waits.

## Part of the Hal suite

hal-voice is one part of [Hal](https://github.com/NotEoin/Hal), a modular AI assistant that runs on
your own machine, built privacy and security first. Around Hal,
[hal-tools](https://github.com/NotEoin/hal-tools) adds the tools of a desktop assistant,
[hal-memory](https://github.com/NotEoin/hal-memory) a memory,
[gpu-light](https://github.com/NotEoin/gpu-light) a fair share of the GPU, and
[hal-employment](https://github.com/NotEoin/hal-employment) a job-search helper. Each repo
works on its own; installed together they make one assistant. See [how it fits
together](https://github.com/NotEoin/Hal#how-it-fits-together).

## Requirements

| | |
|---|---|
| OS | Linux with PulseAudio or PipeWire, or Windows 11 through WSL2 ([how-to](docs/howto/run-on-windows-with-wsl2.md)). Native Windows and macOS are unsupported. |
| Python | 3.11 or 3.12, not 3.13 or later: numpy 1.x has no wheels there. `setup-voice.sh` finds a suitable interpreter or falls back to conda. |
| System packages | PortAudio and FFmpeg. |
| Hardware | A microphone and speakers. A GPU is optional: by default hal-voice runs on the CPU and shares the GPU with background jobs ([sharing the GPU](docs/user/voice.md#sharing-the-gpu)). |
| Also | A [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) checkout with its pretrained models, and voice data you supply (see below). |

## Install

```bash
git clone https://github.com/NotEoin/hal-voice.git && cd hal-voice
./setup-voice.sh        # the virtualenv, GPT-SoVITS and the voice profile
./hal-voice doctor      # check audio devices, models and the TTS server
```

To run it as a Hal pack with systemd units, see
[Install the voice services](docs/howto/install-the-voice-services.md).

## Run

```bash
./hal-voice              # start the voice loop; hold space to talk
./hal-voice doctor       # run this first when something is wrong
./hal-voice tts "hello"  # speak one line, no mic or model needed
./hal-voice mcp          # run only the speak/listen MCP server
./hal-voice eye          # run the eye on its own
./hal-voice connection   # show which Claude Code setup it will drive, and why
bin/hal-say "hello"      # speak a line if the voice server is up; silent in quiet hours
```

Choose who answers with `brain.mode`: see [Choose a brain](docs/howto/choose-a-brain.md). Settings
and their layers are in [Configuration](docs/user/configuration.md). Through hal-hub, other
modules and the brain can speak with the `voice_say` tool.

## Give it a voice

No audio ships with this repository. You supply the voice data. If you own a copy of
*2001: A Space Odyssey*, put the film and any subtitle files in `voice-data/film/` and run:

```bash
./make-hal-voice   # finds the lines HAL speaks, cleans and trains on them, installs the result
```

[Build the voice](docs/howto/build-the-voice.md) explains what it accepts. To train on clips you
already have, follow [voice-data/README.md](voice-data/README.md) and
[gptsovits/train-hal.md](gptsovits/train-hal.md). Around forty clean, well-transcribed clips give a
recognisable result; consistency matters more than quantity.

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

## Limitations

- The trained wake word is unreliable. openWakeWord detection is off by default and its
  dependencies are not installed; `./setup-voice.sh --with-wake` is for experiments. For hands-free
  use set `activation.keyword.enabled: true`: one microphone stream is cut into utterances, each
  transcribed once, and a turn starts when one begins or ends with "Hal". It costs CPU or GPU time
  whenever someone is speaking nearby, and "Hey Al" still sounds like "Hey Hal" to it.
- The brain needs a model that calls tools reliably. Smaller models tend to say they did something
  instead of doing it.
- Synthesis takes time. Streaming hides much of the delay, but on the `cloud` (CPU) profile it runs
  slower than real time, so there is a pause before the voice speaks.
- The voice effects are tuned to one voice. Treat the defaults as a starting point.
- Interrupting can clip the first syllable of your next sentence.
- Windows through WSL2 has not been run on a Windows PC yet (unverified; `hal-voice doctor` probes
  each part).

## Test

```bash
uv run --frozen --extra test pytest -q   # hermetic tests: no GPU, network or audio
ruff check . && ruff format --check .
```

[Test hands-free listening](docs/howto/test-hands-free-listening.md) covers `voicetest`, which
replays a recording of you through the detector.

## Documentation

The [docs index](docs/README.md) lists every page. Report a problem or share a result in the
[issues](https://github.com/NotEoin/hal-voice/issues) or
[discussions](https://github.com/NotEoin/hal-voice/discussions).

## Licence

MIT, see [LICENSE](LICENSE). This covers the code only, not any voice data you bring to it.
