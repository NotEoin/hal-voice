# Changelog

All notable changes to hal-voice are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- The README is shorter. The brain, configuration, service-install and `voicetest` details moved to
  [Choose a brain](docs/howto/choose-a-brain.md), [Configuration](docs/user/configuration.md),
  [Install the voice services](docs/howto/install-the-voice-services.md) and
  [Test hands-free listening](docs/howto/test-hands-free-listening.md).
- The transcribe pack's fallback vault is `~/notes`, the default of Hal's shared config, when no vault is set.

### Removed

- The `direct` brain mode, `halvoice/brain.py` and the keys `brain.ollama_url`, `keep_alive`,
  `max_tool_rounds`, `tools_per_turn`, `num_ctx`, `temperature`, `hub_path` and `ui.rgb_follow`.
  A config that still sets one stops with an error naming the replacement. A missing `hal` never
  falls back to Claude Code.

### Added

- WSL2 support: `halvoice/hostplatform.py` detects the platform (`HALVOICE_OSRELEASE` fakes it in
  tests), the commands refuse to start on native Windows and macOS with a message naming WSL2,
  and `hal-voice doctor` adds `platform` and, under WSL2, audio server, display and systemd lines.
  New how-to: `docs/howto/run-on-windows-with-wsl2.md`. A `wsl` CI job runs the test suite.
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

- `hal-voice doctor` has a "GPU sharing" line: the `gpu.mode`, whether the gpu-light daemon
  answers, and the devices speech-to-text and the voice will use. A missing daemon is a warning.

- "Sharing the GPU" in `docs/user/voice.md`: the modes, what happens on "Hal", the CPU voice
  speed (measured 2026-11-02, i9-12900K) and a symptom table.

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
- `hal-voice.service` and `hal-tts.service` no longer carry `X-HalGPU=yes` or
  `GPU_LIGHT_CLAIM=hal-voice`: in `on_demand` mode they use no GPU while idle, so `hal off` leaves
  them running and they start on a busy card.
- Docs are split into `docs/user`, `docs/howto` and `docs/dev` with an index in `docs/README.md`;
  `VOICE-BUILD.md` and `VOICE-TUNING.md` moved there.
- The voice build stops with `GpuBusy` when the GPU stays busy for 30 minutes; it used to carry
  on and run out of memory.
- A missing core Hal no longer starts Claude Code silently; `brain.mode: claude` is the only way.
- `voicetest brain` asks any OpenAI-compatible endpoint and no longer needs core Hal.

### Fixed

- `brain.user_name` sets the name HAL calls you: each `{user}` in `brain.persona` becomes it
  (Dave by default). Before, the setting was read nowhere and the persona always said Dave.
- `voice_say` declares network: it speaks through hal-tts on 127.0.0.1:9880, which a tool
  without network cannot reach under Hal's tool sandbox, so it always said the server was down.
- Hands-free listening no longer hears a request in background noise. Whisper repeats its
  prompt back on noise, and the prompt was two requests ("Hal, read me the headlines"), so Hal
  read the news unasked every few minutes. The prompt now holds no request, and a transcript made
  only of prompt sentences, close variants of them ("Hal, one minute") and calls of his name is
  dropped, as is one sentence repeated three or more times. A bare "Hal" still wakes him.
- The voice and TTS units carry Hal's `X-HalOff=yes`, so `hal off` still stops them and keeps
  them off now that they are no longer GPU units.

- `hal-voice say` and the MCP server send HAL's checkpoints to the TTS server once per
  process. Each whole-text synthesis used to swap both checkpoints again.
