# Install the voice services

hal-voice is a Hal pack with its own environment. This page installs it, renders its systemd user
units and starts them. Without Hal it still runs from a checkout in `claude` or `openai` mode.

## Steps

1. Install the pack and its environment.

   ```bash
   hal packs add voice        # from the catalogue; or: hal packs add --local PATH
   cd PATH/TO/hal-voice
   ./setup-voice.sh           # the virtualenv, GPT-SoVITS and the voice profile
   ```

   `setup-voice.sh` reads your GPU and proposes a profile; `--profile NAME` overrides it. The
   profile covers VRAM only and applies with `gpu.mode: resident` or `off`. The default,
   `on_demand`, keeps speech recognition and the voice on the CPU unless you set `stt.device` or
   `tts.device`. See [sharing the GPU](../user/voice.md#sharing-the-gpu).

   | Profile | Speech recognition | Voice synthesis |
   | --- | --- | --- |
   | `24gb` | GPU | GPU (fp16) |
   | `12gb` | CPU | GPU (fp16) |
   | `cloud` | CPU | CPU |

   The `cloud` profile rewrites GPT-SoVITS's device settings, because it ships configured for CUDA.
   CPU synthesis is slower than real time, so expect a pause before the voice speaks.

2. Render the units.

   ```bash
   hal units install   # renders systemd/ and copies the units to the user config
   ```

   `hal-tts.service` runs the GPT-SoVITS server and `hal-voice.service` the hands-free loop. Both
   belong to `hal.target`. They carry no `X-HalGPU` mark, because in `on_demand` mode they use no
   GPU while idle: `hal off` leaves them running. With `resident` mode, or a GPU `tts.device`, add
   `X-HalGPU=yes` under `[Unit]` with `systemctl --user edit`.

3. Pick how the TTS server runs. By default (`tts.server.isolation: container`) it runs in a GPU
   container with no network. Build the image once.

   ```bash
   hal-voice tts-image build   # needs docker with the nvidia runtime
   ```

   For host mode and pinned checkpoints, see [tts-isolation](tts-isolation.md).

4. Check, then start.

   ```bash
   hal-voice doctor                        # audio devices, models and the TTS server
   systemctl --user start hal-voice.service
   ```

## Symptoms and fixes

| Symptom | Fix |
| --- | --- |
| `tts-server` exits naming `tts.server.gptsovits_dir` | Set it to your GPT-SoVITS checkout. |
| The checkout is on a drive that mounts late | Add `RequiresMountsFor=/path/to/drive` to `hal-tts.service` with `systemctl --user edit`. |
| `tts-server` says docker or its `nvidia` runtime is missing | Install it. The server does not fall back to the host. |
| `claude` is not found by the unit | The unit sets no `PATH`; set `claude.command` to its absolute path. |
