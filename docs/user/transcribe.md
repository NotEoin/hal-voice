# hal-transcribe

hal-transcribe turns speech into text: any audio or video file, or a meeting, lecture or voice memo
recorded from the microphone. A recording becomes a note in the vault with a summary, the decisions
made and a checklist of actions.

## Highlights

- Any format ffmpeg reads, transcribed with faster-whisper on the GPU (or the CPU without one).
- "Hal, record this meeting" and "stop recording": recorded with PipeWire in the background.
- Notes saved to `Transcripts/` with a summary, decisions and `- [ ]` actions from the local model.
- Timestamps on request.

## Install

hal-transcribe needs ffmpeg, PipeWire's `pw-record`, and a Python environment with faster-whisper:

```bash
git clone <url> ~/src/hal-ecosystem/hal-transcribe
cd ~/src/hal-ecosystem/hal-transcribe
python3 -m venv .venv && .venv/bin/pip install faster-whisper
```

`bin/run` uses this `.venv`, or `HAL_TRANSCRIBE_PYTHON`, or hal-voice's environment.

## Use

Through hal-hub as `hal-hub transcribe_<tool>`, or `python -m halvoice.hubtool FUNCTION` (JSON arguments on stdin):

```bash
hal-hub transcribe_file path=~/Downloads/voice-memo.m4a
hal-hub transcribe_record_start title="Team standup"
hal-hub transcribe_record_stop
```

## Configuration

Settings live in the `transcribe:` section of hal-voice's config (`~/.config/hal-voice/config.yaml`):

```yaml
transcribe:
  model: medium.en     # Whisper model; small.en is used on the CPU fallback
  folder: Transcripts  # note folder inside the vault
  vault: ""            # empty = [hal].vault of ~/.config/hal/config.toml
```

A `[transcribe]` table in `~/.config/hal/config.toml` is still read, with a deprecation warning, for
one minor version. Summaries come from Hal's gateway when it runs, else from `brain.base_url`;
without either, notes are saved without a summary. With gpu-light installed, Whisper waits up to
120 s for the GPU and then runs on the CPU.

From a shell: `hal-voice transcribe-file PATH [--json]`.

## Licence

MIT. See [LICENSE](../../LICENSE).
