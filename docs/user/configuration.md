# Configuration

Every setting and its default is in [`halvoice/config.yaml`](../../halvoice/config.yaml): audio
devices, activation, model choices, voice tuning, the effects chain and the eye. Leave that file
alone and put only the keys you change in your own file, with the same layout.

```yaml
# ~/.config/hal-voice/config.yaml
audio:
  input_device: 3
stt:
  model: small.en
```

Settings load in this order, and later wins:

1. `halvoice/config.yaml`, the packaged defaults.
2. `~/.config/hal-voice/built.yaml`, written by the voice build (`make-hal-voice`) or by
   `hal-voice config import-built OLDFILE`.
3. `~/.config/hal-voice/config.yaml`, or the file `HALVOICE_CONFIG` names.
4. Environment variables `HALVOICE_<SECTION>_<KEY>`.

`~/.config` stands for `$XDG_CONFIG_HOME` when that is set. A key that the defaults do not have
stops hal-voice with an error naming the file and the key, so a typo is not ignored.
`python -m halvoice.config` prints the merged result and the files it read.

```bash
HALVOICE_TTS_SPEED=0.9 ./hal-voice        # slower speech, this run only
HALVOICE_STT_MODEL=medium.en ./hal-voice  # more accurate transcription
```

Related pages: [choose a brain](../howto/choose-a-brain.md), [sharing the GPU](voice.md#sharing-the-gpu).
