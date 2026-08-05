# Fine-tuning GPT-SoVITS into HAL-9000

Once `voice-data/clean/slices/` and `hal.list` exist, GPT-SoVITS can be fine-tuned on
them. On a 24GB card the run takes roughly **20–60 minutes**.

GPT-SoVITS trains **two** small models that work together:

- **SoVITS** (the decoder/timbre) — learns *what HAL sounds like*.
- **GPT** (the autoregressive text→token model) — learns *how HAL phrases and paces speech*.

Training produces a `.pth` (SoVITS) and a `.ckpt` (GPT). hal-voice loads both at runtime.

## 0. Prerequisites

```bash
cd hal-voice/gptsovits/GPT-SoVITS
bash install.sh            # one-time: torch + deps (large). Use its conda/venv.
# Download the pretrained BASE models per its README into GPT_SoVITS/pretrained_models/
```

> **VRAM:** training needs the GPU to itself. A local LLM served alongside it will not
> co-exist with the run on a 24GB card — stop it for the duration (`ollama stop <model>`)
> and restart it afterwards. Inference is far lighter (~2–4 GB), so the contention is a
> one-time, training-only concern.

## 1. Launch the WebUI

```bash
python webui.py     # opens http://localhost:9874
```

The WebUI walks the whole pipeline. Its tabs cover dataset preparation and training:

1. **0-Fetch dataset** — (optional) run UVR5 separation, slicing, and ASR here if you didn't
   do Step 2–4 yourself. If you already have `hal.list`, skip to formatting.
2. **1-GPT-SoVITS-TTS → 1A Dataset formatting** — point it at your `hal.list` and audio dir.
   Run the three "one-click" formatting steps (text, SSL/Hubert, semantic token extraction).
3. **1B Fine-tuned training:**
   - **SoVITS training:** ~8 epochs is a good start for a single-speaker character voice.
     Batch size: as high as VRAM allows once the GPU is otherwise free.
   - **GPT training:** ~15 epochs is a sensible default. More overfits a small dataset and
     makes the delivery stiff rather than calm.
4. **1C Inference** — load your new weights + a reference clip and audition lines.

## 2. Tuning for the HAL tone

Timbre comes from the fine-tune. The flat, unhurried delivery comes from these:

- **Reference clip selection** dominates prosody. Audition several 3–10 s references and keep
  the one that sounds most level and unhurried. Lock it into `config.yaml: tts.ref_audio_path`
  (+ exact `prompt_text`).
- **Don't over-train the GPT model.** Too many epochs → clipped, robotic delivery. HAL is
  smooth and legato. If he sounds choppy, train the GPT model *fewer* epochs.
- **Weight the dataset toward flat lines.** If half your clips are the rare emotional HAL
  ("I'm afraid"), the model learns to emote. For the classic HAL, feed mostly the polite,
  affectless lines.
- **Speed.** HAL is slow. Keep `tts.speed ≈ 0.90` in config; bake the rest into reference
  choice rather than slowing so far it sounds drugged.
- The remaining character comes from **hal-voice's FX chain** (`tts.fx` in config.yaml):
  low-shelf body, presence de-emphasis and the cabin reverb. Tune those *after* the clone
  sounds right on its own — see [VOICE-TUNING.md](../VOICE-TUNING.md).

## 3. Install the weights

Copy the two checkpoints GPT-SoVITS produced into `gptsovits/weights/` and name them to
match `config.yaml` (or edit config to match your filenames):

```
gptsovits/weights/hal-e15.ckpt        # GPT  (tts.gpt_weights)
gptsovits/weights/hal_e8_s200.pth     # SoVITS (tts.sovits_weights)
```

`run-api.sh` writes the resolved weight paths into GPT-SoVITS's `tts_infer.yaml`, and
hal-voice hot-swaps them over the API on first synthesis (`/set_gpt_weights`,
`/set_sovits_weights`).

## 4. Verify end-to-end

```bash
# terminal 1: the voice engine
./gptsovits/run-api.sh
# terminal 2:
./hal-voice doctor          # should now tick "weights" and "server"
./hal-voice tts "Good evening, Dave. Everything is running smoothly."
```

If that line sounds like HAL, you're done — `./hal-voice` for the full conversational loop.

## Optional: RVC for timbre

If the timbre is still not locked after fine-tuning, a **Retrieval-based Voice Conversion**
pass over GPT-SoVITS's output is the next lever: train an RVC model on the same clean
dataset and run synthesised audio through it before the FX chain. It costs an extra model
and some latency. It would sit in `halvoice/tts.py`, after `synth_raw` and before
`hal_fx.apply`. Not implemented here — the fine-tune and FX chain were enough.
