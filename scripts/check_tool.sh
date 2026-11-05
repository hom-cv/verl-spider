#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/conda_env.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
exec python scripts/check_tool.py
