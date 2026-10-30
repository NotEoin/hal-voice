# How hal-voice is put together

| Package | Responsibility |
| --- | --- |
| `halvoice` | The voice loop, speech in and out, settings (`config.yaml`), brains, the TTS launchers |
| `halvoice.brains` | The brain interface; modes are `hal` (ACP client, `brains/acp.py`), `openai` and `claude`, and a missing brain raises `BrainUnavailable` |
| `halvoice.ttsbox`, `halvoice.ttspins` | Container mode and checkpoint pins for the TTS server |
| `halvoice.hubtool` | One-shot worker that hal-hub's `exec` manifests call |
| `haltranscribe` | Transcribe files and meetings; its own `hal-module.toml` |
| `voicebuild` | `make-hal-voice`: film to trained voice, stage by stage |
| `voicetest` | Replays commands against a brain and records the tool calls |

Design points:

- Heavy models run in their own processes. GPT-SoVITS is a separate server so its torch stays out
  of the voice loop's environment.
- The GPU is claimed through gpu-light when its daemon answers, and everything works without it.
- Settings: packaged `config.yaml` holds every key; a user file may only set keys that exist.
- Tool manifests (`halvoice/hal-module.toml`, `haltranscribe/hal-module.toml`) are schema 2
  `exec` manifests; hal-voice imports nothing from Hal.
- Voice tuning history: [voice-tuning.md](voice-tuning.md).

## Test

```bash
uv run --frozen --extra test pytest -q            # hermetic suite
uv run --frozen --extra test pytest -q -m docker  # small CPU containers, on request
```
