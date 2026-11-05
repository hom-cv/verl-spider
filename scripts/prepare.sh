#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/conda_env.sh
exec python scripts/prepare.py "$@"
