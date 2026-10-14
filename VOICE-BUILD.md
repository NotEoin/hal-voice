# Build HAL's voice from the film

`./make-hal-voice` turns your copy of *2001: A Space Odyssey* into HAL's voice for hal-voice. It
finds every line HAL speaks clearly, cleans it, trains GPT-SoVITS on the result and installs the
best checkpoint. This page covers what it needs, how to run it, and what each stage does.

## Set it up

The build needs ffmpeg, GPT-SoVITS with its pretrained base models, and the extra packages the
following command installs:

```bash
./setup-voice.sh --with-training
```

That adds parselmouth, PyAV and faster-whisper to GPT-SoVITS's environment, and creates a
separate environment for Demucs and DeepFilterNet, which need an older torch. Reading image
subtitles (DVD and Blu-ray) also needs tesseract:

```bash
sudo apt install tesseract-ocr
```

## Run it

Put the film, and any subtitle files you have, in `voice-data/film/`, then run:

```bash
./make-hal-voice
```

To leave the film where it is, name the file or its folder instead:

```bash
./make-hal-voice "/media/films/2001 - A Space Odyssey/"
./make-hal-voice film.mkv --subs film.en.srt
```

A full build takes about half an hour on a 24GB card. Hal is turned off while the build uses
the GPU and back on at the end. Every stage saves its result in `voice-data/build/` (or
`HALVOICE_BUILD_DIR`), so a run that stops picks up where it left off. The following options
change what runs:

| Option | Effect |
| --- | --- |
| `--from STAGE` | Run from this stage on: `find`, `clean`, `dataset`, `train`, `select`, `install` |
| `--only STAGE` | Run one stage, for example `--only find` to see which lines it picks |
| `--version v2ProPlus` | Train GPT-SoVITS v2ProPlus instead of v2 |
| `--sovits-epochs N`, `--gpt-epochs N` | Train longer or shorter (12 each by default) |
| `--ocr` | Also read image subtitles inside the film when there are text ones too |
| `--no-install` | Build and choose a checkpoint, but leave the voice as it is |

Each run writes `report.md` into the build folder, listing every line kept and every line in
HAL's voice that was dropped, with the reason.

## What it accepts

The film can be any file ffmpeg reads: MKV, MP4, M2TS and others. The build takes the main
English dialogue track, skips commentary, and uses the centre channel of a surround mix, where
film dialogue is mixed apart from the score and effects.

Subtitles help but are optional. The build reads, in this order of preference:

1. Subtitle files beside the film, in `voice-data/film/`, or given with `--subs`: SRT, WebVTT,
   ASS/SSA, VobSub (`.idx` with `.sub`) and PGS (`.sup`).
2. Text subtitles inside the film.
3. Image subtitles inside the film, read with tesseract, when nothing else is available or
   `--ocr` is given.

With several sources, each one is scored by how much of its text is heard in the film. The best
one gives the text, and speaker labels from every source are combined. With no subtitles at all,
lines come from the transcript.

## How it works

The following stages run in order:

1. **Transcribe.** faster-whisper transcribes the dialogue track with a time for every word.
2. **Sync.** Each subtitle source's text is aligned to the transcript across the whole film.
   The timing differences give a robust fit of a constant offset and a drift (subtitles made for
   another frame rate), with a running correction for local jumps at cuts. Each line's times
   then come from its own recognised words, so subtitles out by seconds still cut cleanly.
3. **Find HAL.** Every line is embedded with GPT-SoVITS's speaker-verification model. Lines the
   subtitles label `HAL` (as `HAL:` or `[HAL]`), and a few phrases only HAL says, seed his voice;
   every other labelled character is a counter-example. A line counts as HAL when it is closer to
   HAL than the other characters' lines are, and clearly closer to HAL than to any one of them.
4. **Screen.** A HAL line is kept when no other voice overlaps it, the transcript agrees with
   the subtitle, the recogniser is confident, and its pitch and pace are HAL's usual. That rules
   out the singing and the slowed shutdown. Long lines are split at their pauses.
5. **Clean.** Demucs separates the voice from the score and effects, giving a
   speech-to-background ratio; DeepFilterNet removes what noise is left, limited to 15 dB so the
   voice keeps its air. Each clip is extended to where the voice stops, never into the next
   speaker, and its level is matched.
6. **Train.** GPT-SoVITS is fine-tuned on the clips. "I'm sorry, Dave. I'm afraid I can't do
   that." is held out of training.
7. **Select.** Every saved checkpoint pair, with the calmest clean clips as reference, says the
   held-out line and three lines HAL never says in the film. Each is scored against the film's
   delivery with `gptsovits/analyse_voice.py` (melody, timing, word lengths and loudness, voice
   quality, spectrum) and checked for pronunciation with the recogniser. The best is kept, with
   samples beside the film's line in `voice-data/build/samples/` for you to compare.
8. **Install.** The chosen weights and reference clip are copied into place and `config.yaml`
   points at them. The previous `config.yaml` is kept beside it as `config.yaml.<time>.bak`.

## Your copy stays yours

The film, the clips cut from it and the weights trained on them stay on your machine. The build
folders are ignored by git, and nothing from the film is part of this repository.
