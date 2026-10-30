# Changelog

All notable changes to hal-voice are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `hal-voice transcribe-file PATH [--json]` and `hal-voice tts --out FILE.wav TEXT`.
- `haltranscribe`, the hal-transcribe package, merged with its history. Settings are now the
  `transcribe:` section of hal-voice's config; a `[transcribe]` table in
  `~/.config/hal/config.toml` still works for one minor version, with a warning.
- `halvoice.llm`, a stdlib chat client for Hal's gateway or `brain.base_url`.
- gpu-light support, optional: the voice loop claims the GPU for its models and for each turn
  (`gpu:` keys in the config), and Whisper in `haltranscribe` waits up to 120 s for the card.
  The voice build runs each GPU stage as a `gpu-light run` batch job when the daemon answers.

### Changed

- The voice build stops with `GpuBusy` when the GPU stays busy for 30 minutes; it used to carry
  on and run out of memory.
- A missing core Hal no longer starts Claude Code silently; `brain.mode: claude` is the only way.
- `voicetest brain` asks any OpenAI-compatible endpoint and no longer needs core Hal.
