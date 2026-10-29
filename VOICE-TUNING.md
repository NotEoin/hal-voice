# HAL Voice -- Tuning Notes

A log of how the HAL voice was dialled in, and why each setting ended up where it did.
All settings live in [`config.yaml`](halvoice/config.yaml) under `tts:` (sampling + `fx:` chain) and
`claude.voice_system_prompt` (persona). The voice = **GPT-SoVITS clone -> FX chain (client-side)**.

## Current settings

- **Clone:** GPT-SoVITS v2, weighted-dataset retrain -> `gptsovits/weights/hal-e12.ckpt` + `hal_e8_s288.pth`.
- **Sampling (`tts`):** `top_k 30`, `top_p 0.85`, `temperature 0.95`, `speed 0.97`.
- **FX (`tts.fx`):** `pitch_flatten 0`, `pitch_shift 0`, `drive 0`; EQ = low-shelf **+5.0 dB @250 Hz**,
  presence **-0.5 dB @3200**, high-shelf **-3.0 dB @7500** (the "3e" voicing); reverb `room 0.18 / wet 0.10`.
- **Persona:** full HAL-9000 character prompt (identity locked to HAL, addresses user as "Dave", film-style calm menace).

## How it got there

### 1. "Fried" / distorted  -> it was the FX, not the clone
- **Diagnosis:** synthesized the *same* line with FX off vs on. Raw clone was clean (peak ~0.49);
  the FX chain made it harsh.
- **Cause:** `fx.pitch_flatten` (was `0.35`). Its Hilbert-analytic-signal reconstruction
  (`|analytic|*cos(cumsum(inst_freq))`) is only valid for *narrowband* signals - on broadband
  speech it adds artifacts. [ ] **Disabled it (`0.0`).**
- Also turned **`drive` 0.04 -> 0** (saturation was adding grit on top).
- [x] **Worked:** distortion gone. HAL's monotone now comes from the clone + slow speed, not DSP.

### 2. "Stiff and flat" -> sampling was too tight + EQ too dark
- **Cause (stiff):** sampling was `top_k 10 / top_p 0.7 / temp 0.7` - over-constrained -> robotic.
  Loosening it (-> `30 / 0.85 / 0.95`, speed `0.92->0.97`) added natural cadence. [x]
- **Cause (flat/dull):** a **hardcoded -6 dB** high-shelf in `hal_fx.py`. Made it
  configurable (`eq.high_rolloff_gain_db`) and eased it. [x]
- A/B'd three presets (A current / B lively / **C natural**). **C won.**

### 3. "Not full enough / should be deeper"
- Tried `pitch_shift_semitones -1.5`. [ ] **Didn't work - sounded artificial. Removed (`0.0`).**
- [x] **Depth via EQ instead:** boost low-shelf, roll off the top. Ran a 6-way EQ menu
  (neutral / warm / **deepdark** / clear / cabin / grit). **"deepdark" (#3) won.**
- Refinement round around #3 -> **"3e" won** (#3 but presence brought back to -0.5 and top a touch
  more open at -3 @7500, so it's deep but not muffled).

### 4. Dataset retrain -- weighted toward the flat, polite delivery
- `prepare_dataset.py` reads a weight column from `transcripts.csv`: **0 drops a clip**, **n
  oversamples it n times** (copies under distinct names = real oversampling).
- Dropped the four most strained/emotional takes and tripled fourteen of the calmest. 42 unique
  clips -> 70 training items. This did more for consistency than any FX change.
- **GPT epochs 15 -> 12** - fewer epochs = smoother/legato (more -> stiff, per `gptsovits/train-hal.md`).

## What worked vs what didn't

| Worked [x] | Didn't [ ] |
|---|---|
| Disabling `pitch_flatten` (main fix for "fried") | `pitch_flatten` at any strength - flawed DSP |
| Looser sampling for natural cadence | Tight sampling (`temp 0.7`) -> stiff/flat |
| EQ for depth (low-shelf boost + high roll-off) | `pitch_shift` for depth -> artificial |
| Weighting the set toward calm takes; GPT <=12 epochs | Leaving outlier deliveries in the set |
| A/B from one raw synth (isolates FX from clone) | - |

## How to re-tune later (no retrain needed for FX)
- Generate comparison clips by synthesizing **one** raw line and applying FX variants to that
  same audio offline. Comparing across separate generations tells you nothing.
- Quick dials in `config.yaml tts.fx.eq`: `low_shelf_gain_db` (body), `high_rolloff_gain_db`
  (darkness), `presence_gain_db` (clarity/sibilance), `reverb.wet` (space).
- The remaining gap is most likely the clone itself, from a small 47-clip MP3 dataset.
  Levers: cleaner or longer source audio, a different reference clip (`tts.ref_audio_path`),
  or more SoVITS epochs for timbre.

## Retrain command (for reference)
```bash
python voice-data/prepare_dataset.py                       # rebuild the weighted dataset
conda activate gptsovits && python gptsovits/train_hal.py  # ~15 min on a 24GB card
# then point config.yaml tts.gpt_weights/sovits_weights at the new gptsovits/weights/ files
```
