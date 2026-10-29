#!/usr/bin/env bash
# Fails when a tracked text file contains a non-ASCII character, and prints each offending line.
# README.md may keep emoji in headings and the badge image lines (REQ-VOICE-13); every other
# line of it, and every other file, must be ASCII so diffs, grep and terminals stay predictable.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

status=0
# -z keeps odd file names intact; -I skips binary files.
while IFS= read -r -d '' file; do
    hits="$(grep -nIP '[^\x00-\x7F]' -- "$file" || true)"
    if [ "$file" = "README.md" ]; then
        # Drop headings ("# ...") and badge lines ("![...](...)").
        hits="$(grep -vP '^\d+:(#|!\[)' <<<"$hits" || true)"
    fi
    if [ -n "$hits" ]; then
        sed "s|^|$file:|" <<<"$hits"
        status=1
    fi
done < <(git ls-files -z)

if [ "$status" -ne 0 ]; then
    echo "ascii-check: non-ASCII characters found" >&2
    exit 1
fi
echo "ascii-check: ok"
