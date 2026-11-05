#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
MODE="${1:-trace}"
[[ "$MODE" == trace || "$MODE" == grpo ]] || { echo 'Usage: bash scripts/run.sh [trace|grpo]'; exit 1; }
source scripts/conda_env.sh
source scripts/select_gpu.sh
RUN_LABEL="${SLURM_JOB_ID:-direct}-${MODE}"
for required in data/sources.json data/mini.parquet models/qwen/config.json; do
  [[ -f "$required" ]] || { echo "Missing $required. Run bash scripts/prepare.sh first." >&2; exit 1; }
done
python - <<'PY'
import json
from pathlib import Path
import torch
from spider_trace.sql import database_path
assert torch.cuda.is_available(), 'No CUDA GPU visible. Run on your allocated GPU node, not the login node.'
assert torch.cuda.device_count() == 1, (
    'Multiple GPUs visible and no usable allocation binding. Set CUDA_VISIBLE_DEVICES '
    'to the GPU assigned to your allocation, then rerun. Do not guess GPU 0.'
)
sources = json.loads(Path('data/sources.json').read_text())
database_path(sources['database_root'], sources['db_id'])
gpu = torch.cuda.get_device_properties(0)
print(f'GPU preflight: {gpu.name}, {gpu.total_memory / 2**30:.1f} GiB')
PY
export RUN_DIR="$REPO_ROOT/runs/${RUN_LABEL}-$(date +%Y%m%d-%H%M%S)"
export TRACE_DIR="$RUN_DIR/events"
export SPIDER_DB_ROOT
SPIDER_DB_ROOT=$(python -c 'import json; print(json.load(open("data/sources.json"))["database_root"])')
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export OMP_NUM_THREADS=1
RAY_CPUS=$(python -c 'import os; print(min(4, len(os.sched_getaffinity(0))))')
mkdir -p "$TRACE_DIR"
cp data/sources.json "$RUN_DIR/sources.json"
python -m pip freeze > "$RUN_DIR/environment.freeze.txt"
conda env export > "$RUN_DIR/environment.conda.yaml"
git rev-parse HEAD > "$RUN_DIR/repo-revision.txt"
nvidia-smi > "$RUN_DIR/gpus.txt"
VAL_ONLY=True
[[ "$MODE" == trace ]] || VAL_ONLY=False
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$REPO_ROOT/data/mini.parquet" \
  data.val_files="$REPO_ROOT/data/mini.parquet" \
  data.train_batch_size=1 data.val_batch_size=1 \
  data.max_prompt_length=2048 data.max_response_length=8192 \
  data.val_max_samples="${SPIDER_VAL_SAMPLES:--1}" \
  data.return_raw_chat=True data.filter_overlong_prompts=False data.truncation=error \
  actor_rollout_ref.model.path="$REPO_ROOT/models/qwen" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_model_len=10240 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.n=2 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO_ROOT/config/agents.yaml" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=12 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=6000 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$REPO_ROOT/config/tools.yaml" \
  custom_reward_function.path="$REPO_ROOT/spider_trace/reward.py" \
  custom_reward_function.name=compute_score \
  trainer.logger='[console]' trainer.project_name=verl-spider \
  trainer.experiment_name="$RUN_LABEL" \
  trainer.nnodes=1 trainer.n_gpus_per_node=1 \
  trainer.val_only="$VAL_ONLY" trainer.val_before_train=True \
  trainer.total_epochs=1 trainer.total_training_steps=1 \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  trainer.validation_data_dir="$RUN_DIR/validation" \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  ray_kwargs.ray_init.num_cpus="$RAY_CPUS" \
  2>&1 | tee "$RUN_DIR/job.log"
python scripts/inspect_trace.py "$TRACE_DIR"
