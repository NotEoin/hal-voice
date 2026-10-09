# hal-transcribe

hal-transcribe turns speech into text: any audio or video file, or a meeting, lecture or voice memo
recorded from the microphone. A recording becomes a note in the vault with a summary, the decisions
made and a checklist of actions.

## Highlights

- Any format ffmpeg reads, transcribed with faster-whisper on the GPU (or the CPU without one).
- "Hal, record this meeting" and "stop recording": recorded with PipeWire in the background.
- Notes saved to `Areas/Work/Meetings/` with a summary, decisions and `- [ ]` actions from the local model.
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

Through hal-hub as `hal-hub transcribe_<tool>`, or `bin/run` from the module folder:

```bash
hal-hub transcribe_file path=~/Downloads/voice-memo.m4a
hal-hub transcribe_record_start title="Team standup"
hal-hub transcribe_record_stop
```

## Configuration

Settings live in the `[transcribe]` table of `~/.config/hal/config.toml`:

```toml
[transcribe]
model = "medium.en"
folder = "Areas/Work/Meetings"
```

## Licence

MIT. See [LICENSE](LICENSE).
