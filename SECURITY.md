# Security

## Reporting a problem

Report a vulnerability privately to the maintainer to the maintainer (contact
details are on the repository page), not in a public issue. Include the version (`git describe` or the
`pyproject.toml` version), what you did and what you expected. Expect an answer within 7 days.

## What hal-voice does and does not protect you from

- **Microphone.** Audio is captured only while the voice loop runs. Speech recognition runs
  on your machine (faster-whisper); no audio is sent anywhere by hal-voice itself.
- **Brain modes.** The mode sets where your words go.
  - `hal` (the default): Hal's boxed agent; stays on your machine, tools run through Hal's broker.
  - `openai`: your words go to the server at `brain.base_url`. Use a local one for privacy.
  - `claude`: runs Claude Code on the host. **Not private, not sandboxed.** It is chosen only
    by setting `brain.mode: claude`; hal-voice never falls back to it on its own.
- **No telemetry.** hal-voice sends no usage data, crash reports or update checks.
- **TTS server.** The GPT-SoVITS server listens on 127.0.0.1 unless you set `HALVOICE_TTS_HOST`. Its weights are pickled
  PyTorch checkpoints, which can run code when loaded: load only weights you built or trust.
- **Film pipeline.** `make-hal-voice` trains on a film you own, on your machine. Read the
  disclaimer in `README.md` before you use it.

## Scope

Reports about the dependencies (GPT-SoVITS, faster-whisper, PyTorch) belong to those
projects; tell us when hal-voice's use of them makes the problem worse.
