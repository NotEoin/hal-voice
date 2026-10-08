#!/usr/bin/env bash
#
# Start the GPT-SoVITS api_v2 inference server that hal-voice talks to over HTTP.
# Keep this running in its own terminal (or as a systemd/user service) while you use HAL.
#
# It loads HAL's fine-tuned checkpoints if they exist (gptsovits/weights/), otherwise it
# starts with the base models and hal-voice will hot-swap weights at runtime via the API.
#
set -uo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
GPTSOVITS_DIR="${GPTSOVITS_DIR:-$HERE/GPT-SoVITS}"
HOST="${HALVOICE_TTS_HOST:-127.0.0.1}"
PORT="${HALVOICE_TTS_PORT:-9880}"

# Newest trained checkpoints (epoch/step numbers vary by run, so resolve by glob).
GPT_W="$(ls -t "$HERE"/weights/hal-*.ckpt 2>/dev/null | head -1)"
SOVITS_W="$(ls -t "$HERE"/weights/hal_*.pth 2>/dev/null | head -1)"

if [ ! -d "$GPTSOVITS_DIR" ]; then
  echo "GPT-SoVITS not found at $GPTSOVITS_DIR. Run ../setup-voice.sh first (it clones it)."
  exit 1
fi

# Voice profile (written by ../setup-voice.sh). Decides whether the engine runs on the
# GPU or the CPU — GPT-SoVITS ships configured for CUDA and cannot start without it.
VOICE_CONF="${HALVOICE_CONF:-$HERE/../voice.conf}"
if [ -f "$VOICE_CONF" ]; then
  # shellcheck disable=SC1090
  . "$VOICE_CONF"
fi
TTS_DEVICE="${HALVOICE_TTS_DEVICE:-cuda}"
TTS_HALF="${HALVOICE_TTS_HALF:-true}"

cd "$GPTSOVITS_DIR" || exit 1

# torch >= 2.6 defaults torch.load(weights_only=True); the checkpoints pickle a
# pathlib.PosixPath. They are trusted local files — load them fully.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Prefer the repo's own venv/conda env if present; otherwise activate the conda
# env created by setup (install.sh), falling back to whatever 'python' is active.
if [ -n "${GPTSOVITS_PYTHON:-}" ]; then
  PY="$GPTSOVITS_PYTHON"
elif [ -d "$GPTSOVITS_DIR/runtime" ]; then
  PY="$GPTSOVITS_DIR/runtime/python"
elif [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "gptsovits" ]; then
  # Not already inside the gptsovits env: activate it. A systemd service has no
  # conda on its PATH, so also look in the usual install locations.
  CONDA_BASE="$(conda info --base 2>/dev/null)"
  for dir in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge"; do
    [ -n "$CONDA_BASE" ] && break
    [ -f "$dir/etc/profile.d/conda.sh" ] && CONDA_BASE="$dir"
  done
  CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
  if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH" && conda activate gptsovits 2>/dev/null
  fi
  PY="python"
else
  PY="python"
fi

# torchcodec (the audio backend api_v2 uses) needs the CUDA libs that ship as pip
# wheels (libnppicc etc.) on the dynamic-loader path. Add nvidia/*/lib to it.
NVLIBS="$("$PY" - <<'PYEOF' 2>/dev/null
import os, glob
try:
    import nvidia
    base = os.path.dirname(nvidia.__file__)
    print(":".join(sorted(glob.glob(os.path.join(base, "*", "lib")))))
except Exception:
    pass
PYEOF
)"
[ -n "$NVLIBS" ] && export LD_LIBRARY_PATH="$NVLIBS:${LD_LIBRARY_PATH:-}"

# This api_v2.py starts with its base/config weights; HAL's fine-tuned weights are
# applied at runtime by hal-voice via /set_gpt_weights and /set_sovits_weights
# (halvoice/tts.py: ensure_weights), so no preload flags are needed here.
ARGS=(api_v2.py -a "$HOST" -p "$PORT")

# api_v2 reads device, precision AND its startup weights from the `custom:` block of
# tts_infer.yaml, with no CLI flag for any of them — so apply them here before starting.
# Idempotent. The weight paths matter because that block stores ABSOLUTE paths: a
# GPT-SoVITS checkout that has been copied or moved will otherwise load the weights of
# whatever install wrote it last. (hal-voice hot-swaps the correct ones on first synth,
# so this is about the server being self-consistent from the moment it starts.)
TTS_DEVICE="$TTS_DEVICE" TTS_HALF="$TTS_HALF" GPT_W="$GPT_W" SOVITS_W="$SOVITS_W" \
  "$PY" - <<'PYEOF' || echo "  ! could not update tts_infer.yaml — check it by hand"
import os, pathlib
try:
    import yaml
except ImportError:
    raise SystemExit(0)          # not fatal; the file's existing setting stands
p = pathlib.Path("GPT_SoVITS/configs/tts_infer.yaml")
if not p.exists():
    raise SystemExit(0)
cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
custom = cfg.setdefault("custom", {})
device = os.environ["TTS_DEVICE"]
half = os.environ["TTS_HALF"].lower() == "true"
# fp16 on the CPU is not supported by torch and fails at load time.
if device == "cpu":
    half = False
wanted = {"device": device, "is_half": half}
# Only point at trained weights that actually exist here; otherwise leave whatever the
# file has, so an untrained clone still starts on the pretrained base models.
for key, env in (("t2s_weights_path", "GPT_W"), ("vits_weights_path", "SOVITS_W")):
    path = os.environ.get(env, "")
    if path and pathlib.Path(path).is_file():
        wanted[key] = str(pathlib.Path(path).resolve())

if any(custom.get(k) != v for k, v in wanted.items()):
    custom.update(wanted)
    p.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    print(f"  config: device={device} is_half={half} (updated)")
else:
    print(f"  config: device={device} is_half={half}")
PYEOF

echo "Starting GPT-SoVITS api_v2 on http://$HOST:$PORT"
[ -f "$GPT_W" ] && echo "  GPT weights:    $GPT_W (hot-swapped by hal-voice on first synth)" \
                || echo "  GPT weights:    (not trained yet — see gptsovits/train-hal.md)"
[ -f "$SOVITS_W" ] && echo "  SoVITS weights: $SOVITS_W (hot-swapped by hal-voice on first synth)" \
                   || echo "  SoVITS weights: (not trained yet)"
exec "$PY" "${ARGS[@]}"
