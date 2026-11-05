#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $(uname -s) == Linux ]] || { echo 'GPU installation requires Linux (run on Perlmutter).'; exit 1; }
python3 -c 'import sys; assert sys.version_info[:2] == (3, 10), "Activate a Python 3.10 environment first"'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade 'pip==25.1.1' 'setuptools==80.9.0' wheel
python -m pip install -r requirements-gpu.txt
python -m pip install -e .
python -m pip check
mkdir -p runs
python -m pip freeze > runs/environment.freeze.txt
python -m unittest discover -s tests -v
echo 'Installed. Next: source .venv/bin/activate; python scripts/prepare.py'
