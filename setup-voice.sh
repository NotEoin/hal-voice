#!/usr/bin/env bash
#
# hal-voice — setup (Ubuntu / Linux)
# Installs the system audio libs, a Python venv with hal-voice's deps, and (optionally)
# GPT-SoVITS as a separate inference server. Safe to re-run (idempotent).
#
#   ./setup-voice.sh                  # detect your GPU, propose a profile, ask
#   ./setup-voice.sh --profile cloud  # force: 24gb | 12gb | cloud   (cloud = no GPU)
#   ./setup-voice.sh --yes            # accept the detected profile without prompting
#   ./setup-voice.sh --no-gptsovits   # skip cloning/installing GPT-SoVITS (do it yourself)
#   ./setup-voice.sh --with-wake      # also install the optional wake-word deps
#   ./setup-voice.sh --with-training  # also what ./make-hal-voice needs to build the voice from the film
#   ./setup-voice.sh --hal ../Hal     # also wire it to a Hal checkout (see below)
#   ./setup-voice.sh --hal auto       # ...or find a sibling Hal automatically
#
# Needs Python 3.10–3.12 (not 3.13+ — numpy 1.x has no wheels there). It looks for one
# and falls back to conda; override with HALVOICE_PYTHON=/path/to/python3.11.
#
# TWO SEPARATE QUESTIONS, and it matters that they're separate:
#
# 1. WHO THINKS?  By default (brain.mode: direct) core Hal's brain answers: a local
#    model that calls the tools of your extensions, set in ~/.config/hal/config.toml.
#    brain.mode: claude drives Claude Code instead; `--hal <path>` then reads that
#    install's hal.conf so Claude Code uses the same local model, and claude.env_file
#    in config.yaml takes any KEY=VALUE file of ANTHROPIC_* settings for other setups.
#    Full order: halvoice/connect.py.
#
# 2. WHO SPEAKS?  That is what --profile below controls, and it is about VRAM:
#
#      24gb   TTS on GPU (fp16) + Whisper on GPU     — fastest
#      12gb   TTS on GPU (fp16) + Whisper on CPU     — leaves room for a local LLM
#      cloud  everything on CPU                      — no NVIDIA GPU needed at all
#
#    The 'cloud' profile is not cosmetic: GPT-SoVITS ships configured for CUDA and
#    will fail on a machine without it, so this rewrites its device settings.
#
set -uo pipefail

VOICE_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENV="$VOICE_ROOT/.venv"
GPTSOVITS_DIR="$VOICE_ROOT/gptsovits/GPT-SoVITS"
# The GPT-SoVITS commit HAL's voice was trained and chosen with; a later one may train differently.
GPTSOVITS_COMMIT="08d627c3338173c3229286d8787060d6559fe0f8"
CONF_FILE="$VOICE_ROOT/voice.conf"   # git-ignored; written here, sourced by ./hal-voice
DO_GPTSOVITS=1
DO_WAKE=0
DO_TRAINING=0
HAL_ROOT_ARG=""
PROFILE=""
ASSUME_YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-gptsovits) DO_GPTSOVITS=0; shift ;;
    --with-wake) DO_WAKE=1; shift ;;
    --with-training) DO_TRAINING=1; shift ;;
    --hal) HAL_ROOT_ARG="${2:-}"; shift 2 || true ;;
    --hal=*) HAL_ROOT_ARG="${1#*=}"; shift ;;
    --profile) PROFILE="${2:-}"; shift 2 || true ;;
    --profile=*) PROFILE="${1#*=}"; shift ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)"; exit 1 ;;
  esac
done

c_blue=$'\033[1;36m'; c_yellow=$'\033[1;33m'; c_green=$'\033[1;32m'; c_red=$'\033[1;31m'; c_off=$'\033[0m'
log()  { printf '\n%s==> %s%s\n' "$c_blue" "$*" "$c_off"; }
ok()   { printf '%s  ✓ %s%s\n' "$c_green" "$*" "$c_off"; }
warn() { printf '%s  ! %s%s\n' "$c_yellow" "$*" "$c_off"; }
err()  { printf '%s  ✗ %s%s\n' "$c_red" "$*" "$c_off"; }
have() { command -v "$1" >/dev/null 2>&1; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then have sudo && SUDO="sudo"; fi

printf '%s\n' "============================================================"
printf '   hal-voice — HAL 9000 voice for Claude Code\n'
printf '   Root: %s\n' "$VOICE_ROOT"
printf '%s\n' "============================================================"

# --- 0. Choose the voice profile (this is about VRAM, not about the model) ---
detect_vram_mib() {
  have nvidia-smi || return 0
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
    | sort -nr | head -n1
}

# TTS needs ~2GB. The 12gb tier keeps Whisper off the GPU so a local LLM still fits
# beside it; the 24gb tier has room for both.
pick_voice_profile() {
  case "$1" in
    24gb)        TTS_DEVICE="cuda"; TTS_HALF="true";  STT_DEVICE="cuda"; STT_COMPUTE="float16" ;;
    12gb)        TTS_DEVICE="cuda"; TTS_HALF="true";  STT_DEVICE="cpu";  STT_COMPUTE="int8" ;;
    cloud|cpu)   TTS_DEVICE="cpu";  TTS_HALF="false"; STT_DEVICE="cpu";  STT_COMPUTE="int8"
                 PROFILE="cloud" ;;
    *) err "Unknown profile: $1 (expected 24gb, 12gb or cloud)"; exit 1 ;;
  esac
}

VRAM_MIB="$(detect_vram_mib)"
if [ -n "$PROFILE" ]; then
  log "Voice profile: $PROFILE (forced with --profile)"
else
  if   [ -z "$VRAM_MIB" ];        then PROFILE="cloud"
  elif [ "$VRAM_MIB" -ge 22000 ]; then PROFILE="24gb"
  elif [ "$VRAM_MIB" -ge 11000 ]; then PROFILE="12gb"
  else                                 PROFILE="12gb"   # any CUDA card can host the ~2GB TTS
  fi
  if [ -n "$VRAM_MIB" ]; then
    log "Detected $(( VRAM_MIB / 1024 ))GB of VRAM → suggesting the '$PROFILE' voice profile"
  else
    log "No NVIDIA GPU detected → suggesting the 'cloud' voice profile (everything on CPU)"
  fi
  if [ "$ASSUME_YES" -ne 1 ]; then
    printf '\n  24gb   voice + speech recognition both on the GPU\n'
    printf '  12gb   voice on the GPU, speech recognition on the CPU\n'
    printf '  cloud  everything on the CPU (no NVIDIA GPU required)\n\n'
    printf 'Voice profile [%s]: ' "$PROFILE"
    read -r reply || reply=""
    [ -n "$reply" ] && PROFILE="$reply"
  fi
fi
pick_voice_profile "$PROFILE"
ok "Voice profile: $PROFILE (TTS on $TTS_DEVICE, speech recognition on $STT_DEVICE)"

# --- 1. System audio + build deps -------------------------------------------
log "Installing system audio libraries (PortAudio, ffmpeg, libsndfile)"
if have apt-get; then
  $SUDO apt-get update -y >/dev/null 2>&1
  $SUDO apt-get install -y portaudio19-dev libsndfile1 ffmpeg python3-venv python3-dev build-essential \
    && ok "system audio deps installed" || warn "some system deps failed — install manually if audio errors appear"
  # SDL bits so the pygame red-eye window can open
  $SUDO apt-get install -y libsdl2-2.0-0 libsdl2-image-2.0-0 >/dev/null 2>&1 && ok "SDL (pygame) libs present" || true
else
  warn "apt-get not found — install portaudio19-dev, libsndfile1, ffmpeg yourself"
fi

# --- 2. Python venv + deps ---------------------------------------------------
# The stack needs Python 3.10–3.12. On 3.13+ the install fails deep inside pip with a
# confusing wall of version constraints, because numpy 1.x publishes no wheels past
# cp312 and several audio libraries here still require numpy 1.x. Many current distros
# ship 3.13 as `python3`, so pick an interpreter deliberately rather than assuming.
find_python() {
  local candidate
  # An explicit choice always wins.
  if [ -n "${HALVOICE_PYTHON:-}" ]; then echo "$HALVOICE_PYTHON"; return; fi
  for candidate in python3.12 python3.11 python3.10; do
    if have "$candidate"; then echo "$candidate"; return; fi
  done
  # Nothing suitable on PATH — is the system python itself in range?
  if have python3; then
    if python3 -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)' 2>/dev/null; then
      echo "python3"; return
    fi
  fi
  echo ""
}

PY_BIN="$(find_python)"
USE_CONDA=0
if [ -z "$PY_BIN" ]; then
  # Fall back to conda, which can materialise a suitable interpreter on demand.
  if have conda; then
    warn "No Python 3.10–3.12 found ($(python3 -V 2>&1) is too new for this stack)"
    log "Falling back to conda to create a Python 3.11 environment"
    USE_CONDA=1
  else
    err "hal-voice needs Python 3.10, 3.11 or 3.12 — found $(python3 -V 2>&1)."
    err "Install one (e.g. 'sudo apt install python3.12 python3.12-venv'), or install"
    err "conda and re-run. To force a specific interpreter: HALVOICE_PYTHON=/path/to/python3.11 ./setup-voice.sh"
    exit 1
  fi
fi

# Reuse an existing environment if it is still usable — re-running this script should be
# cheap and safe. (`conda create -p` errors out on an existing prefix, so this is not
# merely an optimisation.)
REUSED=0
if [ -x "$VENV/bin/python" ]; then
  if "$VENV/bin/python" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)' 2>/dev/null; then
    ok "Reusing existing environment at $VENV ($("$VENV/bin/python" -V 2>&1))"
    REUSED=1
  else
    warn "Existing environment is $("$VENV/bin/python" -V 2>&1), which is out of range — rebuilding"
    rm -rf "$VENV"
  fi
fi

if [ "$REUSED" -eq 0 ]; then
  log "Creating Python environment at $VENV"
  if [ "$USE_CONDA" -eq 1 ]; then
    conda create -y -p "$VENV" python=3.11 >/dev/null 2>&1 \
      && ok "conda env ready (Python 3.11)" || { err "conda env creation failed"; exit 1; }
  else
    ok "Using $PY_BIN ($("$PY_BIN" -V 2>&1))"
    "$PY_BIN" -m venv "$VENV" && ok "venv ready" || { err "venv creation failed"; exit 1; }
  fi
fi

VENV_PY="$VENV/bin/python"
"$VENV_PY" -m pip install --upgrade pip >/dev/null 2>&1
log "Installing hal-voice Python dependencies"
"$VENV_PY" -m pip install -r "$VOICE_ROOT/requirements.txt" \
  && ok "Python deps installed" || { err "pip install failed"; exit 1; }

if [ "$DO_WAKE" -eq 1 ]; then
  log "Installing optional wake-word dependencies"
  "$VENV_PY" -m pip install -r "$VOICE_ROOT/requirements-wake.txt" \
    && ok "wake-word deps installed" \
    || warn "wake-word deps failed (tflite-runtime has limited platform support) — push-to-talk still works"
fi

# --- 3. GPT-SoVITS inference server (separate stack) ------------------------
if [ "$DO_GPTSOVITS" -eq 1 ]; then
  log "Setting up GPT-SoVITS (the voice engine) in gptsovits/GPT-SoVITS"
  mkdir -p "$VOICE_ROOT/gptsovits"
  if [ ! -d "$GPTSOVITS_DIR/.git" ]; then
    git clone https://github.com/RVC-Boss/GPT-SoVITS "$GPTSOVITS_DIR" \
      && git -C "$GPTSOVITS_DIR" checkout --quiet "$GPTSOVITS_COMMIT" \
      && ok "cloned GPT-SoVITS at ${GPTSOVITS_COMMIT:0:7}" \
      || warn "git clone failed — clone it manually into $GPTSOVITS_DIR"
  elif [ "$(git -C "$GPTSOVITS_DIR" rev-parse HEAD 2>/dev/null)" = "$GPTSOVITS_COMMIT" ]; then
    ok "GPT-SoVITS already cloned at ${GPTSOVITS_COMMIT:0:7}"
  else
    warn "GPT-SoVITS is not at ${GPTSOVITS_COMMIT:0:7}, the version HAL's voice was built with:"
    printf '       git -C %s checkout %s\n' "$GPTSOVITS_DIR" "$GPTSOVITS_COMMIT"
  fi
  if [ -f "$GPTSOVITS_DIR/install.sh" ]; then
    warn "Run GPT-SoVITS's own installer once (it pulls a large torch+CUDA stack):"
    printf '       cd %s && bash install.sh\n' "$GPTSOVITS_DIR"
    warn "Then download its pretrained base models per its README (GPT_SoVITS/pretrained_models)."
  fi
else
  warn "Skipping GPT-SoVITS (per --no-gptsovits). Point tts.api_url at your own server."
fi

# --- 3b. Optionally wire this to an existing Claude Code setup ---------------
# Writes claude.hal_root into config.yaml. From then on hal-voice reads that install's
# hal.conf for its model + endpoint, and uses its Vault as the working directory — so
# switching Hal between 24gb/12gb/cloud switches hal-voice too, with nothing to remember.
if [ -n "$HAL_ROOT_ARG" ]; then
  log "Wiring hal-voice to the Hal checkout at '$HAL_ROOT_ARG'"
  if have python3; then
    HAL_ROOT_ARG="$HAL_ROOT_ARG" python3 - "$VOICE_ROOT/config.yaml" <<'PY' && ok "config.yaml: claude.hal_root set" || warn "Could not update config.yaml — set claude.hal_root by hand"
import os, pathlib, re, sys
p = pathlib.Path(sys.argv[1])
text = p.read_text(encoding="utf-8")
value = os.environ["HAL_ROOT_ARG"]
new, n = re.subn(r'^(\s*)hal_root:.*$', lambda m: f'{m.group(1)}hal_root: "{value}"',
                 text, count=1, flags=re.M)
if not n:
    sys.exit("no hal_root key found in config.yaml")
p.write_text(new, encoding="utf-8")
PY
  else
    warn "python3 not found — set claude.hal_root in config.yaml by hand"
  fi
fi

# --- 3c. Write voice.conf (the profile the launcher and run-api.sh read) -----
log "Writing $CONF_FILE"
cat > "$CONF_FILE" <<EOF
# hal-voice — resolved voice profile. Generated by setup-voice.sh on $(date -u +%Y-%m-%dT%H:%MZ).
# Safe to edit by hand; re-running setup-voice.sh overwrites it. Git-ignored: it records
# what YOUR machine can do, not anything about the project.
#
# NOTE this is only about the voice engine (VRAM). Which model does the *thinking* is a
# separate question, answered by the claude: block in config.yaml.

# 24gb | 12gb | cloud
HALVOICE_PROFILE="$PROFILE"

# Speech-to-text placement. These override stt.device / stt.compute_type in config.yaml.
HALVOICE_STT_DEVICE="$STT_DEVICE"
HALVOICE_STT_COMPUTE_TYPE="$STT_COMPUTE"

# GPT-SoVITS placement, applied to its tts_infer.yaml by gptsovits/run-api.sh.
# cpu + half=false is what lets the voice run on a machine with no NVIDIA GPU.
HALVOICE_TTS_DEVICE="$TTS_DEVICE"
HALVOICE_TTS_HALF="$TTS_HALF"
EOF
ok "Voice profile saved to voice.conf"

# --- 3d. What make-hal-voice needs to build the voice from the film --------
# The GPT-SoVITS environment gains the analysis packages; demucs and DeepFilterNet get an
# environment of their own, since DeepFilterNet needs an older torch than GPT-SoVITS uses.
if [ "$DO_TRAINING" -eq 1 ]; then
  log "Setting up the voice build (./make-hal-voice)"
  have tesseract && ok "tesseract present (reads image subtitles)" \
    || warn "tesseract missing; image subtitles need it: $SUDO apt-get install -y tesseract-ocr"
  GSV_PY=""
  for root in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
    [ -x "$root/envs/gptsovits/bin/python" ] && GSV_PY="$root/envs/gptsovits/bin/python" && break
  done
  [ -z "$GSV_PY" ] && [ -x "$GPTSOVITS_DIR/runtime/python" ] && GSV_PY="$GPTSOVITS_DIR/runtime/python"
  if [ -n "$GSV_PY" ]; then
    "$GSV_PY" -m pip install --quiet praat-parselmouth av faster-whisper \
      && ok "analysis packages in the GPT-SoVITS environment" || warn "pip install in $GSV_PY failed"
  else
    warn "no GPT-SoVITS environment yet: run its install.sh first, then this again"
  fi
  if have conda; then
    if ! conda env list | grep -q '^hal-voice-denoise '; then
      conda create -y -n hal-voice-denoise python=3.11 >/dev/null && ok "created hal-voice-denoise"
    fi
    DN_PY="$(conda run -n hal-voice-denoise which python)"
  else
    python3.11 -m venv "$VOICE_ROOT/.venv-denoise" && DN_PY="$VOICE_ROOT/.venv-denoise/bin/python"
  fi
  "$DN_PY" -m pip install --quiet "torch==2.5.1" "torchaudio==2.5.1" demucs deepfilternet librosa soundfile \
    && ok "demucs and DeepFilterNet in $DN_PY" || warn "denoise install failed; see $DN_PY"
  [ -n "${DN_PY:-}" ] && printf '\nHALVOICE_DENOISE_PYTHON="%s"\n' "$DN_PY" >> "$CONF_FILE"
fi

# --- 4. Make 'hal-voice' executable + optional symlink ----------------------
chmod +x "$VOICE_ROOT/hal-voice" "$VOICE_ROOT"/*.sh 2>/dev/null || true
chmod +x "$VOICE_ROOT/gptsovits/"*.sh 2>/dev/null || true
if [ -d "$HOME/.local/bin" ]; then
  ln -sf "$VOICE_ROOT/hal-voice" "$HOME/.local/bin/hal-voice" && ok "linked 'hal-voice' into ~/.local/bin"
fi

cat <<EOF

${c_green}============================================================${c_off}
${c_green}  hal-voice base install complete.${c_off}
${c_green}============================================================${c_off}

Remaining one-time steps (the voice itself):

  1) Build HAL's voice from your copy of the film (needs --with-training):
       put the film, and any .srt, in voice-data/film/, then run
       ./make-hal-voice
     (VOICE-BUILD.md explains each step; voice-data/README.md and
     gptsovits/train-hal.md cover doing it by hand.)

  2) Start the GPT-SoVITS API server (keep it running):
       ./gptsovits/run-api.sh

  3) Check everything is wired up:
       ./hal-voice doctor

  4) Talk to HAL:
       ./hal-voice           # wake word + push-to-talk -> spoken replies
     or, inside a normal Claude Code session, register the MCP server (see README)
     and ask HAL to "say" things.
EOF
