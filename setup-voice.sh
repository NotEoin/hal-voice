#!/usr/bin/env bash
#
# hal-voice — setup (Ubuntu / Linux)
# Installs the system audio libs, a Python venv with hal-voice's deps, and (optionally)
# GPT-SoVITS as a separate inference server. Safe to re-run (idempotent).
#
#   ./setup-voice.sh            # full setup
#   ./setup-voice.sh --no-gptsovits   # skip cloning/installing GPT-SoVITS (do it yourself)
#
set -uo pipefail

VOICE_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENV="$VOICE_ROOT/.venv"
GPTSOVITS_DIR="$VOICE_ROOT/gptsovits/GPT-SoVITS"
DO_GPTSOVITS=1
[ "${1:-}" = "--no-gptsovits" ] && DO_GPTSOVITS=0

c_blue=$'\033[1;36m'; c_yellow=$'\033[1;33m'; c_green=$'\033[1;32m'; c_red=$'\033[1;31m'; c_off=$'\033[0m'
log()  { printf '\n%s==> %s%s\n' "$c_blue" "$*" "$c_off"; }
ok()   { printf '%s  ✓ %s%s\n' "$c_green" "$*" "$c_off"; }
warn() { printf '%s  ! %s%s\n' "$c_yellow" "$*" "$c_off"; }
err()  { printf '%s  ✗ %s%s\n' "$c_red" "$*" "$c_off"; }
have() { command -v "$1" >/dev/null 2>&1; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then have sudo && SUDO="sudo"; fi

printf '%s\n' "============================================================"
printf '   hal-voice — HAL 9000 voice add-on setup\n'
printf '   Root: %s\n' "$VOICE_ROOT"
printf '%s\n' "============================================================"

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
log "Creating Python venv at $VENV"
python3 -m venv "$VENV" && ok "venv ready" || { err "venv creation failed"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null 2>&1
log "Installing hal-voice Python dependencies"
pip install -r "$VOICE_ROOT/requirements.txt" && ok "Python deps installed" || { err "pip install failed"; exit 1; }

# --- 3. GPT-SoVITS inference server (separate stack) ------------------------
if [ "$DO_GPTSOVITS" -eq 1 ]; then
  log "Setting up GPT-SoVITS (the voice engine) in gptsovits/GPT-SoVITS"
  mkdir -p "$VOICE_ROOT/gptsovits"
  if [ ! -d "$GPTSOVITS_DIR/.git" ]; then
    git clone https://github.com/RVC-Boss/GPT-SoVITS "$GPTSOVITS_DIR" \
      && ok "cloned GPT-SoVITS" || warn "git clone failed — clone it manually into $GPTSOVITS_DIR"
  else
    ok "GPT-SoVITS already cloned"
  fi
  if [ -f "$GPTSOVITS_DIR/install.sh" ]; then
    warn "Run GPT-SoVITS's own installer once (it pulls a large torch+CUDA stack):"
    printf '       cd %s && bash install.sh\n' "$GPTSOVITS_DIR"
    warn "Then download its pretrained base models per its README (GPT_SoVITS/pretrained_models)."
  fi
else
  warn "Skipping GPT-SoVITS (per --no-gptsovits). Point tts.api_url at your own server."
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

  1) Build HAL's voice dataset + fine-tune GPT-SoVITS:
       see  voice-data/README.md   (sourcing & cleaning HAL audio)
       then gptsovits/train-hal.md  (the fine-tune run)

  2) Start the GPT-SoVITS API server (keep it running):
       ./gptsovits/run-api.sh

  3) Check everything is wired up:
       ./hal-voice doctor

  4) Talk to HAL:
       ./hal-voice           # wake word + push-to-talk -> spoken replies
     or, inside a normal Claude Code session, register the MCP server (see README)
     and ask HAL to "say" things.
EOF
