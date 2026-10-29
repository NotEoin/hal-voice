#!/usr/bin/env bash
# Fails when audio, model weights, transcripts or subtitles are tracked (REQ-VOICE-14). They are
# derived from a copyrighted film performance and must stay on the user's machine.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
found="$(git ls-files | grep -iE '\.(wav|mp3|flac|m4a|ogg|ckpt|pth|onnx|pt|safetensors|srt|sup)$|(^|/)transcripts\.csv$' || true)"
if [ -n "$found" ]; then
    echo "$found"
    echo "audio-guard: audio, weights or subtitle files are tracked" >&2
    exit 1
fi
echo "audio-guard: ok"
