# How hal-voice is put together

| Package | Responsibility |
| --- | --- |
| `halvoice` | The voice loop, speech in and out, settings (`config.yaml`), GPU sharing (`gpu.py`, `placement.py`), brains, the TTS launchers |
| `halvoice.brains` | The brain interface; modes are `hal` (ACP client, `brains/acp.py`), `openai` and `claude`, and a missing brain raises `BrainUnavailable` |
| `halvoice.ttsbox`, `halvoice.ttspins` | Container mode and checkpoint pins for the TTS server |
| `halvoice.hubtool` | One-shot worker that hal-hub's `exec` manifests call |
| `haltranscribe` | Transcribe files and meetings; its own `hal-module.toml` |
| `voicebuild` | `make-hal-voice`: film to trained voice, stage by stage |
| `voicetest` | Replays commands against a brain and records the tool calls |

Design points:

- Heavy models run in their own processes. GPT-SoVITS is a separate server so its torch stays out
  of the voice loop's environment.
- The GPU is shared through gpu-light, over its daemon socket with a stdlib client
  (`halvoice/gpu.py`); hal-voice does not import `gpu_light`. `gpu.mode` is `on_demand` (default:
  no VRAM held while idle, one `room.request` per conversation, requested when the ears hear a
  request and awaited for at most `gpu.turn_wait_s`), `resident` or `off`. A room lives as long
  as its connection, so a worker thread holds the connection and closes it `gpu.linger_s` after
  the last reply. Any scheduler failure is logged once and never stops the loop.
- Devices come from `halvoice/placement.py`: in `on_demand` the settings (CPU by default) and
  never `voice.conf`; in `resident` and `off` the `voice.conf` profile first. `hal-voice tts-server`
  passes the result to host mode (`HALVOICE_RESOLVED_TTS_*`) and container mode (no `--gpus` on
  the CPU).
- Settings: packaged `config.yaml` holds every key; a user file may only set keys that exist.
- Tool manifests (`halvoice/hal-module.toml`, `haltranscribe/hal-module.toml`) are schema 2
  `exec` manifests; hal-voice imports nothing from Hal.
- Voice tuning history: [voice-tuning.md](voice-tuning.md).

## Test

```bash
uv run --frozen --extra test pytest -q            # hermetic suite
uv run --frozen --extra test pytest -q -m docker  # small CPU containers, on request
```
