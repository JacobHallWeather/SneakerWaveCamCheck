#!/usr/bin/env bash
set -euo pipefail

DATE_ARG="${1:-}"

if [[ -n "$DATE_ARG" ]]; then
  shift
  python r2_sync_and_build.py --date "$DATE_ARG" "$@"
else
  UTC_TODAY="$(date -u +%F)"
  python r2_sync_and_build.py --date "$UTC_TODAY" "$@"
fi
