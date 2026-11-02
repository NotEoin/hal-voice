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

## Sharing the GPU

hal-voice can run all day without holding VRAM, so a background GPU job (a vLLM batch run, say)
keeps the card. `gpu.mode` in `~/.config/hal-voice/config.yaml` picks how:

| `gpu.mode` | What it does |
| --- | --- |
| `on_demand` (default) | Holds no VRAM while idle. Speech-to-text (`stt.cpu_model`, small.en, int8), wake listening and the TTS server run on the CPU. Each turn asks the gpu-light daemon for room |
| `resident` | The earlier behaviour: a gpu-light service claim for the models for the life of the process, plus a claim for each turn. Devices come from the `voice.conf` profile |
| `off` | Never contacts gpu-light. Devices come from the `voice.conf` profile |

### What happens when you say "Hal"

1. As soon as the loop hears a request (or you hold push-to-talk), it sends `room.request`
   (name `hal-voice-turn`, `gpu.brain_mib` of VRAM) to the gpu-light daemon. This overlaps with
   transcription.
2. The daemon parks the background job in place with cuda-checkpoint (about 9 s for 24 GB).
3. The loop waits up to `gpu.turn_wait_s` (12 s) for the answer, then calls the brain
   whether or not the room was made. A slower answer than that costs a slow first reply, not
   a stuck loop.
4. The room is kept across follow-up turns and given back `gpu.linger_s` (30 s) after the last
   reply. The background job then resumes.

Every gpu-light failure (no daemon, refused, timeout, socket error) is logged once and the loop
carries on. The socket is `$GPU_LIGHT_SOCKET`, else `$XDG_RUNTIME_DIR/gpu-light/daemon.sock`.
hal-voice does not import gpu-light.

In `resident` mode only the basic claims are made: a restart of the daemon drops them, and
hal-voice does not make them again until it is restarted.

### Which device runs what

In `on_demand` mode the devices are `stt.device`, `stt.compute_type`, `tts.device` and
`tts.half`, which default to the CPU. Set them in `config.yaml` (or as `HALVOICE_STT_DEVICE`,
`HALVOICE_TTS_DEVICE`, `HALVOICE_TTS_HALF`) to ask for a GPU; that holds VRAM while idle. The
`voice.conf` profile written by `setup-voice.sh` is ignored in this mode, because it records
what the machine can do, not what you asked for. In `resident` and `off` mode the profile wins
over the settings. Host and container TTS follow the same device. `hal-voice doctor` prints
the mode, whether the daemon answers and the devices that will be used.

### Speed of the CPU voice

GPT-SoVITS on the CPU, warmed up, ran at 1.1x to 1.7x realtime: 4 s for a 4 s line and 6 s for
a 10 s line (measured 2026-11-02 on an i9-12900K). The first request after the server starts
takes about 8 s. hal-voice speaks sentence groups as they arrive, so this is usable; other CPUs
are unmeasured.

### Symptoms

| Symptom | Fix |
| --- | --- |
| The first reply after a long pause is slow | The room was still being made; raise `gpu.turn_wait_s`, or check `gpu-light status` to see whether the park took longer than 12 s |
| Log says `no gpu-light daemon answered` | Start the daemon, or set `gpu.mode: off` to stop trying; the voice works either way |
| The background job never resumes | The room is still held: wait `gpu.linger_s`, or stop hal-voice |
| The first spoken line takes about 8 s | The CPU voice is warming up; later lines are faster |
| The voice is slower than the speech | The CPU is busy; set `tts.device: cuda` and `tts.half: true` to use the card, which then stays in use |
| `hal off` does not stop the voice | Expected: the units use no GPU in `on_demand` mode |
| The voice still uses the GPU | `doctor` shows the devices; look for `tts.device`, `stt.device` or `gpu.mode: resident` |
