#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .venv/bin/activate ]] || { echo 'Run bash scripts/setup.sh first.' >&2; exit 1; }
source .venv/bin/activate
exec python scripts/prepare.py "$@"
