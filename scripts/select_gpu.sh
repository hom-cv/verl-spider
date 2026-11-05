#!/usr/bin/env bash
# Source before any CUDA imports. Preserve an existing single-device binding.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]]; then
  :
elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  # Job GPU IDs use the node's physical numbering.
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS%%,*}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi
# If no binding is available, preflight accepts a single visible GPU, but
# requires an explicit allocation-specific binding when multiple are visible.
