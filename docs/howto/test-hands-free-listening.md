# Test hands-free listening

`voicetest` replays a recording of you reading [`voicetest/script.txt`](../../voicetest/script.txt)
through the same voice detector, name rules and speech recognition the service uses. It does not
use the voice or the brain. It reports what woke the assistant, what reached it and how fast.

## Steps

1. Record yourself reading the script on the microphone hal-voice uses, with the voice loop turned
   off. Say "Marker one" and the take ("Desk, take one") first.

2. Cut the recording into lines. They are kept in `voice-data/tests/` and stay out of git, because
   they are your voice.

   ```bash
   ./hal-voice voicetest split ~/recording.wav   # use --take desk-1 if the marker was not heard
   ```

3. Replay the lines.

   ```bash
   ./hal-voice voicetest run          # simulated clock, about a minute
   ./hal-voice voicetest run --live   # real-time pace through the threads
   ```

   Until you have recorded fan-noise and quiet-voice takes, `--simulate` builds them from your
   recording.

4. Change one setting, replay and compare.

   ```bash
   ./hal-voice voicetest run --simulate --set activation.keyword.end_silence_ms=500 --label e500
   ./hal-voice voicetest compare sim e500   # two runs side by side
   ```

5. Optionally send the commands that were heard to a model. Tool calls are recorded, never run.

   ```bash
   ./hal-voice voicetest brain --label sim   # see --manifest for the tool list
   ```

## Symptoms and fixes

| Symptom | Fix |
| --- | --- |
| `split` finds no takes | Say the marker clearly, or pass `--take NAME`. |
| Many lines never wake the assistant | Change one `activation.keyword.*` setting with `--set`, replay with a new `--label` and compare. |
| `run` finds no clips | Run `split` first. |
