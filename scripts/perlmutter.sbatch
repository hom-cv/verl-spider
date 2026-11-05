#!/usr/bin/env bash
#SBATCH --job-name=spider-trace
#SBATCH --constraint=gpu&hbm80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --qos=regular
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
srun --ntasks=1 --cpu-bind=none bash scripts/run.sh "${MODE:-trace}"
