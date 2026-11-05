#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
source .venv/bin/activate
: "${SLURM_JOB_ID:?Run inside a GPU Slurm allocation}"
MODE="${1:-trace}"
[[ "$MODE" == trace || "$MODE" == grpo ]] || { echo 'Usage: bash scripts/run.sh [trace|grpo]'; exit 1; }
export RUN_DIR="$REPO_ROOT/runs/${SLURM_JOB_ID}-${MODE}-$(date +%Y%m%d-%H%M%S)"
export TRACE_DIR="$RUN_DIR/events"
export SPIDER_DB_ROOT
SPIDER_DB_ROOT=$(python -c 'import json; print(json.load(open("data/sources.json"))["database_root"])')
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export OMP_NUM_THREADS=4
mkdir -p "$TRACE_DIR"
cp data/sources.json "$RUN_DIR/sources.json"
python -m pip freeze > "$RUN_DIR/environment.freeze.txt"
git rev-parse HEAD > "$RUN_DIR/repo-revision.txt"
nvidia-smi > "$RUN_DIR/gpus.txt"
VAL_ONLY=True
[[ "$MODE" == trace ]] || VAL_ONLY=False
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$REPO_ROOT/data/mini.parquet" \
  data.val_files="$REPO_ROOT/data/mini.parquet" \
  data.train_batch_size=4 data.val_batch_size=10 \
  data.max_prompt_length=4096 data.max_response_length=4096 \
  data.return_raw_chat=True data.filter_overlong_prompts=False data.truncation=error \
  actor_rollout_ref.model.path="$REPO_ROOT/models/qwen" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO_ROOT/config/agents.yaml" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=12000 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$REPO_ROOT/config/tools.yaml" \
  custom_reward_function.path="$REPO_ROOT/spider_trace/reward.py" \
  custom_reward_function.name=compute_score \
  trainer.logger='[console]' trainer.project_name=verl-spider \
  trainer.experiment_name="${SLURM_JOB_ID}-${MODE}" \
  trainer.nnodes=1 trainer.n_gpus_per_node=4 \
  trainer.val_only="$VAL_ONLY" trainer.val_before_train=True \
  trainer.total_epochs=1 trainer.total_training_steps=1 \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  trainer.validation_data_dir="$RUN_DIR/validation" \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  ray_kwargs.ray_init.num_cpus=32 \
  2>&1 | tee "$RUN_DIR/job.log"
python scripts/inspect_trace.py "$TRACE_DIR"
