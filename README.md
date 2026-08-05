# hal-voice

A HAL 9000 voice interface for Claude Code. Hold a key or say the wake word, talk to it, and it
answers out loud — calm, unhurried and slightly too pleased with itself — while a red lens watches
you from the corner of the screen.

Fan work, built for my own machine. Everything runs locally.

```
wake word / push-to-talk  ->  faster-whisper  ->  Claude Code (headless)
                                                        |
                       speakers  <-  HAL FX  <-  GPT-SoVITS
```

It started as an add-on inside my [hal](../Hal) repo — a local Claude Code setup pointed at Ollama —
and got pulled out into its own project once it stopped being a weekend experiment.

## The interesting part is the latency

A naive version waits for the whole reply, synthesises it, then plays it. On a long answer that's
several seconds of dead air, and dead air kills the illusion completely — HAL doesn't buffer.

So the reply is streamed out of Claude Code, split into sentences as they arrive, and sentence one
goes to the synthesiser while the model is still writing sentence two. First audio lands about as
fast as the model can finish a sentence. Barge-in works too: talking over a long answer stops
playback, which you need, because a machine that can't be interrupted is genuinely irritating.

## What's in it

| Stage | Default | Notes |
|---|---|---|
| Activation | push-to-talk | or openWakeWord for "Hello, HAL" |
| STT | faster-whisper `small.en` | on CPU, so the GPU stays free |
| LLM | Claude Code, headless | pointed at local Ollama |
| TTS | GPT-SoVITS | fine-tuned on voice data you supply |
| FX | pedalboard | EQ voicing, cabin reverb, loudness matching |
| UI | pygame | the red eye, reacting to state |

Everything lives in `config.yaml`. Any value takes an environment override of the form
`HALVOICE_<SECTION>_<KEY>`, e.g. `HALVOICE_TTS_SPEED=0.9`, which is how most of the tuning got done.

## Install

```bash
./setup-voice.sh
```

You need PortAudio and FFmpeg from your package manager, a working mic and speakers, and a
GPT-SoVITS checkout with its pretrained models. The script walks through it.

## Use

```bash
./hal-voice              # the full voice loop
./hal-voice tts "hello"  # one-shot synthesis, no mic, no model
./hal-voice mcp          # just the speak/listen MCP server
./hal-voice doctor       # check audio devices, models and the TTS server
```

Run `doctor` first. Nearly every problem in a voice pipeline turns out to be an audio device
problem, and it'll tell you in two seconds what would otherwise cost an hour.

MCP mode is the one worth knowing about: rather than running the whole loop, register hal-voice as
an MCP server inside a normal Claude Code session and you get speak/listen as tools. You keep the
TUI and drop into voice only when you want it.

## The voice

**There is no audio in this repository, and no dialogue either.** You supply both.

The clips are gitignored, the trained weights are gitignored, and the transcripts live with your
clips rather than in the repo. A fresh clone starts empty:

1. Put your clips and a `transcripts.csv` somewhere, and point `voice-source.local` at it.
2. `python voice-data/fetch-voice-lines.py` imports them.
3. `python voice-data/prepare_dataset.py` slices, normalises and writes the label file.
4. Fine-tune — see [gptsovits/train-hal.md](gptsovits/train-hal.md).

About 20–40 clean, well-transcribed clips is enough for a recognisable clone. Consistency matters
much more than volume.

`transcripts.csv` takes an optional weight column, and it turned out to be the highest-leverage
thing in the whole project. Oversampling the takes with the delivery you want and zeroing the ones
you don't moved the result further than any amount of sampler tuning.

### On the voice itself

HAL was performed by the late Douglas Rain. This repo is a homage: it ships the *pipeline*, not the
performance. No audio, no transcripts and no trained weights are committed, and
`fetch-voice-lines.py` deliberately won't pull audio from a public host — moving a recording to a
different URL doesn't change who owns it.

If you point this at Rain's performance, keep the result local and personal. Don't distribute the
model or its output, don't pass it off as him, and don't use it commercially — that's where homage
stops and likeness rights start. The same applies to anyone else's voice: use audio you recorded
yourself or have clear permission for.

## Tuning

The clone gets you timbre. Pace, weight and how present it sounds in the room all come from the FX
chain and the sampler. [VOICE-TUNING.md](VOICE-TUNING.md) is the log of what worked and what didn't.

The short version: EQ does more than pitch shifting, and three of the FX parameters ended up at
zero because they sounded worse than doing nothing. The one that caused the most trouble —
`pitch_flatten` — was reconstructing from the Hilbert analytic signal, which is only valid for
narrowband signals. Speech is broadband, so it was smearing artefacts across the spectrum and I
spent a while blaming the clone for it.

## Hardware

Built on an RTX 3090 with STT on CPU, leaving the card for the LLM and TTS. It runs on less — use a
smaller Whisper model and a smaller LLM. GPT-SoVITS inference is the expensive part; the rest of the
pipeline is cheap.

## Status

Works end to end. Known rough edges:

- Wake-word mode uses a stock openWakeWord model unless you train your own, and the stock ones are
  noticeably worse at rejecting false positives.
- The FX chain was tuned by ear against one voice. The defaults are a starting point, not a preset.
- Barge-in can clip the first syllable of the new turn.
- Linux only, tested with PulseAudio/PipeWire.

## Licence

MIT — see [LICENSE](LICENSE). Covers the code. It does not cover any voice data you bring to it.
