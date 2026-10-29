#!/usr/bin/env bash
# Fails when a tracked file names a path that only exists on one machine: a removable-drive
# mount, a home directory, or an npm global prefix (REQ-VOICE-09). Use `~`, `%h` or a
# placeholder such as <home> instead.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# This script and the CI workflow hold the pattern itself.
if git ls-files -z -- . ':!scripts/portability-check.sh' \
    | xargs -0 grep -nIE '/run/media/|/home/[A-Za-z0-9_.-]+/|\.npm-global' --; then
    echo "portability-check: machine-specific paths found" >&2
    exit 1
fi
echo "portability-check: ok"
