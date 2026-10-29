#!/usr/bin/env bash
# Fails when a lint or type suppression names no rule or gives no reason (REQ-VOICE-12):
#   # noqa                      -> name the rule
#   # noqa: BLE001              -> add " - reason"
#   # type: ignore              -> name the error code
#   # type: ignore[code]        -> add "  # reason"
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
pattern='# noqa([^:]|$)|# noqa: [A-Z0-9, ]+$|# type: ignore([^[]|$)|# type: ignore\[[^]]+\] *$'
# This script holds the patterns itself.
if git ls-files -z -- '*.py' | xargs -0 grep -nE "$pattern" --; then
    echo "lint-reasons-check: suppressions without a rule or a reason" >&2
    exit 1
fi
echo "lint-reasons-check: ok"
