#!/usr/bin/env bash
set -euo pipefail

DAYS="${1:-7}"

if [[ "$DAYS" =~ ^[0-9]+$ ]]; then
  shift || true
else
  DAYS=7
fi

python r2_sync_and_build.py --last-days "$DAYS" "$@"
