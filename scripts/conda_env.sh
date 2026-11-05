#!/usr/bin/env bash
# Source from the project scripts. Pass --create only during setup.
SPIDER_CONDA_ENV="${SPIDER_CONDA_ENV:-verl-spider}"
if ! type module >/dev/null 2>&1; then
  echo 'The module command is unavailable. Run from a Perlmutter shell with environment modules initialized.' >&2
  return 1
fi
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda activate "$SPIDER_CONDA_ENV"; then
  if [[ "${1:-}" != --create ]]; then
    echo "Conda environment $SPIDER_CONDA_ENV is unavailable. Run bash scripts/setup.sh first." >&2
    return 1
  fi
  conda create --name "$SPIDER_CONDA_ENV" python=3.10 pip -y
  conda activate "$SPIDER_CONDA_ENV"
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "This pinned stack requires Python 3.10; choose a new SPIDER_CONDA_ENV name if the existing environment uses another version"'
