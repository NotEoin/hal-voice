# Talk to Hal out loud

hal-voice listens for a wake word, turns your speech into text, sends it to a brain and speaks
the answer. The README covers install and first run; this page is the short map.

| I want to | Do this |
| --- | --- |
| Start the loop | `hal-voice` (or enable `hal-voice.service`) |
| Check my set-up | `hal-voice doctor` |
| Pick a microphone or speaker | `hal-voice devices`, then set `audio.*` in `~/.config/hal-voice/config.yaml` |
| Choose the brain | `brain.mode` in the same file; the default is never Claude Code |
| Hear one line | `hal-voice tts --out line.wav "Good morning"` |
| Transcribe a file | `hal-voice transcribe-file PATH`; see [transcribe.md](transcribe.md) |
| Build HAL's voice | [build-the-voice](../howto/build-the-voice.md) |
| Choose container or host TTS | [tts-isolation](../howto/tts-isolation.md) |
