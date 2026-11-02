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

- GPU sharing talks to the gpu-light daemon with a small stdlib client; hal-voice no longer
  imports `gpu_light` (which was never installed here, so every claim was a silent no-op).
  `gpu.mode` picks the behaviour: `on_demand` (the default) holds no VRAM while idle and asks
  the daemon for room when you start speaking, `resident` keeps the old service and turn
  claims, `off` never contacts gpu-light. `gpu.turn_wait_s` is now 12 s (it was 3 s): parking
  a 24 GB batch job takes about 9 s. A request that ends with no words now gives its room back.
- With `gpu.mode: on_demand` (the default) speech-to-text and the TTS server run on the CPU
  and hold no VRAM, in host and container mode alike. The new `tts.device` and `tts.half`
  settings, like `stt.device`, ask for a GPU; the `voice.conf` profile now applies only in
  `resident` and `off` mode (`./hal-voice` exports it as `HALVOICE_PROFILE_*`). A TTS container
  on the CPU has no `--gpus` and no gpu-light claim marker. See `halvoice/placement.py`.
- Docs are split into `docs/user`, `docs/howto` and `docs/dev` with an index in `docs/README.md`;
  `VOICE-BUILD.md` and `VOICE-TUNING.md` moved there.
- The voice build stops with `GpuBusy` when the GPU stays busy for 30 minutes; it used to carry
  on and run out of memory.
- A missing core Hal no longer starts Claude Code silently; `brain.mode: claude` is the only way.
- `voicetest brain` asks any OpenAI-compatible endpoint and no longer needs core Hal.
