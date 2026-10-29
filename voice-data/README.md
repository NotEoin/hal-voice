# Building HAL's voice dataset

The single biggest lever on how much this sounds like HAL-9000 is **the quality of the
training audio**, not the model. This guide gets you a clean, well-labelled dataset that
GPT-SoVITS can fine-tune into an uncanny HAL.

> **Rights note.** HAL's voice was performed by the late Douglas Rain. Cloning it for your
> own private, local use is a personal project. Do **not** publish or distribute the trained
> voice model or generated audio as if it were him, and don't use it commercially - that
> strays into personality/likeness-rights territory. The *code* in this repo is yours to ship;
> the *voice model* you train from his performance is the sensitive part. Keep it local.

## What you need

- **5-15 minutes** of clean HAL dialogue is plenty for a strong fine-tune (GPT-SoVITS can do
  a passable clone from ~1 minute, but HAL's narrow, controlled delivery rewards more).
- Audio that is **HAL's voice only** - no Strauss/Ligeti score, no ship hums, no other
  characters. This is the hard part: in the film his lines sit over music and room tone.
- **16 kHz or higher, mono, WAV.** GPT-SoVITS resamples internally, but feed it clean PCM.

## Step 1 -- Source the audio

You have 2 viable options:

1. **Isolated/looped dialogue collections (Recommended)** Fan and sound-design communities have extracted
   clean HAL line packs. If you have a source of dry HAL lines, that's ideal - skip to Step 3.
2. **The film's audio**, then isolate the voice (Step 2). Use a copy you legally own.

Drop whatever you gather into `voice-data/raw/`.

## Step 2 -- Isolate the voice (stem separation)

Strip music and effects with Demucs (best-in-class open-source separation):

```bash
# in the hal-voice venv (or any venv)
pip install demucs
# htdemucs has a 'vocals' stem that captures dialogue well
demucs --two-stems vocals -o voice-data/separated voice-data/raw/*.wav
# vocals end up in voice-data/separated/htdemucs/<name>/vocals.wav
```

Then high-pass out remaining rumble and de-ess lightly if needed (ffmpeg):

```bash
ffmpeg -i vocals.wav -af "highpass=f=85, afftdn=nf=-25" voice-data/clean/hal_clean.wav
```

Listen critically. If you can still hear music under his voice, the clone will learn the
music too. Cut those regions.

## Step 3 -- Slice into utterances

GPT-SoVITS trains on short clips (roughly 3-10 s each) paired with transcripts.

- Use GPT-SoVITS's built-in **UVR5 / slicer** tools (in its WebUI, the "0-Fetch dataset"
  tab) - it slices on silence and can also run separation/ASR for you, or
- slice by hand in Audacity, exporting one clip per line.

Aim for clips that are **representative of the tone you want**: slow, level, unhurried.
Avoid the few moments where HAL's pitch rises ("I'm afraid, Dave") *unless* you specifically
want that colour - for the default unnerving-calm HAL, weight the dataset toward the flat,
polite lines.

## Step 4 -- Transcribe (labels)

Each clip needs its text. Write a `transcripts.csv` and keep it **with your clips**, not in this
repo - nothing about the dialogue is stored here. Two columns, no header:

```csv
clip-filename-without-extension,the exact words spoken in that clip
```

An optional third column is a weight: `0` drops a clip from training, `n > 1` oversamples it
`n` times.

```csv
a-calm-level-line,Everything is running smoothly.,3
a-strained-line,I really think I'm entitled to an answer.,0
```

This is the most useful knob in the pipeline. The clone learns whatever is most common in the
set, so weight the takes with the delivery you want and zero the outliers.

GPT-SoVITS's WebUI has a one-click ASR (Faster-Whisper) step, or transcribe with the hal-voice
STT - either way you still need to fold the result into `transcripts.csv`.

**Proof-read the transcripts.** Wrong labels are the most common cause of a mushy clone.

## Step 5 -- Pick the runtime reference clip

Separately from training, GPT-SoVITS conditions each synthesis on a short **reference clip**
(3-10 s). It sets the tone of every line HAL ever speaks, even after fine-tuning.

`prepare_dataset.py` picks the highest-weighted clip and writes it to
`voice-data/reference/hal_calm_reference.wav`. To choose explicitly, put a clip stem in
`voice-data/reference.txt`.

Then set `tts.prompt_text` in `config.yaml` to the **exact** words spoken in that clip -
`prepare_dataset.py` prints it for you to copy. A mismatch here costs noticeable quality and is
easy to miss. Audition several and keep the flattest.

## Folder layout

```
voice-data/
  raw/                      # whatever you gathered (music+voice ok here)
  separated/                # Demucs output
  hal-voice/                # imported clips + transcripts.csv
  clean/                    # voice-only, denoised
    slices/                 # 3-10 s training clips
    hal.list                # GPT-SoVITS label file
  reference/
    hal_calm_reference.wav  # the inference-time tone reference
  reference.txt             # optional: name the reference clip stem
```

Once `clean/slices/` + `hal.list` exist, continue to **`../gptsovits/train-hal.md`**.
