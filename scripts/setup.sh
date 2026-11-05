#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PREPARE=false
case "${1:-}" in
  --prepare) PREPARE=true; shift ;;
  --help|-h) echo 'Usage: bash scripts/setup.sh [--prepare [prepare.py arguments]]'; exit 0 ;;
  '') ;;
  *) echo 'Usage: bash scripts/setup.sh [--prepare [prepare.py arguments]]' >&2; exit 2 ;;
esac
[[ $(uname -s) == Linux ]] || { echo 'GPU installation requires Linux (run on Perlmutter).'; exit 1; }
source scripts/conda_env.sh --create
python -m pip install --upgrade 'pip==25.1.1' 'setuptools==80.9.0' wheel
python -m pip install -r requirements-gpu.txt
python -m pip install -e .
python -m pip check
mkdir -p runs
python -m pip freeze > runs/environment.freeze.txt
conda env export > runs/environment.conda.yaml
python -m unittest discover -s tests -v
if [[ "$PREPARE" == true ]]; then
  python scripts/prepare.py "$@"
  echo 'Ready. Inside a GPU allocation: srun --ntasks=1 --cpu-bind=none bash scripts/run.sh trace'
else
  echo 'Installed. Next: bash scripts/prepare.sh'
fi
