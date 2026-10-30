# Changelog

All notable changes to hal-voice are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Removed

- The `direct` brain mode, `halvoice/brain.py` and the keys `brain.ollama_url`, `keep_alive`,
  `max_tool_rounds`, `tools_per_turn`, `num_ctx`, `temperature`, `hub_path` and `ui.rgb_follow`.
  A config that still sets one stops with an error naming the replacement. A missing `hal` never
  falls back to Claude Code.

### Added

- `brain.mode: hal` (the new default): the voice loop is an ACP client of `hal acp --channel voice
  --toolset assistant`, with spoken approvals for broker requests (`brain.voice_approvals`).
- Host-mode TTS (`tts.server.isolation: host`) is pinned: `hal-voice tts-pins record|verify` hash the
  checkpoints, `tts-server` refuses to start on a mismatch, and `run-api.sh` no longer sets
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`.
- TTS container mode (`tts.server.isolation: container`, the default): `hal-voice tts-server` runs
  GPT-SoVITS in a GPU container with no network and bridges its socket to `127.0.0.1:9880`;
  `hal-voice tts-image build` makes the image. Docker tests: `pytest -m docker`.
- Schema 2 `exec` manifests for the voice and transcribe tools (`halvoice/hal-module.toml`,
  `haltranscribe/hal-module.toml`), run by the one-shot worker `python -m halvoice.hubtool`.
- `hal-voice transcribe-file PATH [--json]` and `hal-voice tts --out FILE.wav TEXT`.
- `haltranscribe`, the hal-transcribe package, merged with its history. Settings are now the
  `transcribe:` section of hal-voice's config; a `[transcribe]` table in
  `~/.config/hal/config.toml` still works for one minor version, with a warning.
- `halvoice.llm`, a stdlib chat client for Hal's gateway or `brain.base_url`.
- gpu-light support, optional: the voice loop claims the GPU for its models and for each turn
  (`gpu:` keys in the config), and Whisper in `haltranscribe` waits up to 120 s for the card.
  The voice build runs each GPU stage as a `gpu-light run` batch job when the daemon answers.

### Changed

- Docs are split into `docs/user`, `docs/howto` and `docs/dev` with an index in `docs/README.md`;
  `VOICE-BUILD.md` and `VOICE-TUNING.md` moved there.
- The voice build stops with `GpuBusy` when the GPU stays busy for 30 minutes; it used to carry
  on and run out of memory.
- A missing core Hal no longer starts Claude Code silently; `brain.mode: claude` is the only way.
- `voicetest brain` asks any OpenAI-compatible endpoint and no longer needs core Hal.
