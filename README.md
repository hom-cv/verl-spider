# veRL Spider traces on Perlmutter

Run Qwen2.5-Coder-7B-Instruct through veRL's asynchronous tool agent loop on a small
Spider subset. The model can explain its approach, execute SQL, inspect results,
revise SQL, and submit a final query. JSONL files preserve every assistant turn,
executed SQL, returned result, elapsed time, and the complete token trajectory.

Default: 10 validation questions from `concert_singer`, two sampled rollouts per
question, up to eleven tool calls and a twelfth assistant turn for the answer.
Turn and token limits can terminate a trajectory earlier. Multiple calls are
allowed, not forced. Qwen2.5 has no Qwen3 `enable_thinking` mode: any explanation
is ordinary generated assistant text.

The single prompt source is `spider_trace/prompts.py` (`sql-paper-v5`). It supplies
the database schema and question, adapting the interaction structure of the
[VerlTool SkySQL prompt](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/examples/data_preprocess/skysql/sql.py).
The model uses `<think>` for brief explanations, `<sql>` to execute a query, and
`<solution>` for its final query. The SQL-tag parser dispatches only explicit SQL
actions to `execute_sql`, bound to the sample's read-only database. Results appear
as text tables inside `<observation>`, capped at 50 rows and a character limit.
Generation stops at `</sql>` so the tool can return a real observation. This does
not force a tool call. Example observations are labeled demonstrations.

The agent rebuilds the prompt at runtime, including for existing `mini.parquet`
files; no re-download or preparation is needed. Start events record the effective
prompt and version. The previous exploration prompt and discovery tool registrations
have been removed. The inspector still requires real SQL execution; no model calls
or repeated cache hits are manufactured.

For a small first check, inside your GPU allocation:

```bash
bash scripts/check_tool.sh
SPIDER_CACHE_DIR=$(mktemp -d /tmp/verl-spider-cache.XXXXXX)
export HF_DATASETS_CACHE="$SPIDER_CACHE_DIR"
SPIDER_VAL_SAMPLES=1 bash scripts/run.sh trace
```

This runs one question with two sampled rollouts. Omit `SPIDER_VAL_SAMPLES` for
all prepared questions. The temporary dataset cache avoids the shared-filesystem
locking error observed on Perlmutter; recreate it in each new allocation.

## Quick start

SSH to Perlmutter and clone into scratch. The interactive allocation below uses
account `m4999` and requests one GPU for one hour.
Use a dedicated Conda environment named `verl-spider` for this project; your
existing `sysml` environment is not modified. The scripts load the Conda module
and activate the project environment automatically.

```bash
cd "$PSCRATCH"
git clone https://github.com/hom-cv/verl-spider.git
cd verl-spider

# One-time setup and downloads on the login node, as in CS5470 HW1.
# Creates the verl-spider Conda environment with Python 3.10 if missing.
bash scripts/setup.sh --prepare

# Request an interactive allocation.
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account m4999

# Launch the single-GPU experiment.
bash scripts/run.sh trace
```

No `srun` or batch script is needed. This follows the
[CS5470 homework workflow](https://github.com/jamalh110/CS5470_Assignment1).
`salloc` still uses Slurm to reserve resources; do not run GPU work on a login node.
The launcher preserves an existing single-GPU `CUDA_VISIBLE_DEVICES` binding;
otherwise it selects the first GPU in `SLURM_JOB_GPUS`, or the first device in an
existing visibility list. If neither is set and multiple GPUs are visible (for
example after a separate SSH), set `CUDA_VISIBLE_DEVICES` to your assigned GPU
before launching; do not assume GPU 0 belongs to your allocation.

`run.sh` uses one GPU and tensor parallelism of one. It
processes one question per batch with two sampled rollouts, one agent worker,
and at most two active inference sequences. Prompt and response budgets are 2048
and 8192 tokens respectively (10240 total); tool results shown to the model are capped at 6000
characters by the pinned agent loop. The response budget includes tool messages
and can end a trajectory before the eleven-call limit. The full bounded SQL result
is still recorded separately in the tool event.

`setup.sh --prepare` also runs `prepare.py`; omitting `--prepare` installs only.
Use `bash scripts/prepare.sh` to download/prepare later without manually activating
the environment. On subsequent sessions, return to `$PSCRATCH/verl-spider`, request
an allocation, and run `bash scripts/run.sh trace`. `run.sh` activates `verl-spider` itself.
The interactive allocation remains reserved after the script finishes; exit its
shell when finished. No `sbatch` file is needed for this workflow.

Complete setup and downloads before reserving the GPU. To activate the
environment manually on the cluster:

```bash
module load conda
conda activate verl-spider
```

Set `SPIDER_CONDA_ENV` to use a different dedicated environment name; use the same
value for setup, preparation and launch. The default named environment location
is controlled by your Conda configuration. No nested Python virtualenv is created.

Alternatively submit a batch job from the repository root:

```bash
sbatch --account=m4999 scripts/perlmutter.sbatch
```

Setup and preparation need network access; GPU computation runs inside Slurm.
Install on a node permitted by your site's installation policy; use a CPU
allocation for expensive installs if necessary. No Docker daemon is required.
The GPU stack ships its CUDA runtime. Do not mix arbitrary CUDA libraries or
an unrelated PyTorch installation into this environment. Driver compatibility
must be checked on the allocated node if vLLM fails to initialize.

The optional batch job requests **one GPU**, 45 minutes, and 32 logical CPUs using
the shared QOS. The interactive command above uses your requested interactive QOS.
Ray uses at most four CPUs, bounded by the process CPU affinity. CPU parameter
and optimizer offload, eager inference, and gradient checkpointing remain enabled.
Validation still initializes veRL's hybrid trainer/actor; it is not a lightweight
standalone inference server. GPU memory use has not yet been measured; a 40GB
allocation may need further tuning, especially for GRPO.

`setup.sh` pins veRL to release `v0.6.1`, vLLM 0.10.0 and PyTorch 2.7.1, installs
this package, checks dependencies, and records resolved versions. It does not
install FlashAttention; the launcher disables remove-padding and requests SDPA.
This combination and its GPU memory use must still be validated on Perlmutter.
The veRL release tag is a release pin rather than an immutable SHA; pip's direct
URL metadata records the resolved Git commit in the environment freeze.

## Data and model

`prepare.py` resolves model, question dataset, and database archive revisions once
and saves them in `data/sources.json`; subsequent runs reuse those revisions.
It downloads the official `Qwen/Qwen2.5-Coder-7B-Instruct` model and
`xlangai/spider` questions plus the `HAL-9001/spider-databases` SQLite rehost.
Review the dataset cards and retain their attribution/license when redistributing
data. Model weights and datasets are excluded from this repository.

If you already have the official Spider database directory:

```bash
bash scripts/prepare.sh --database-root /absolute/path/to/database --limit 10
```

Paths are recorded for the current machine, so run preparation on Perlmutter.
The schema is discovered through tools reading the SQLite file. Gold SQL goes only to the reward
metadata, never into the model prompt or tool response. Preparation verifies that
gold queries execute within the tool's restrictions. It selects the first usable
questions for one database; this is a debugging subset, not a medium/hard sample
or an unbiased benchmark. Choose another database with `--db-id`.

## Inspect traces

After the job, its run directory contains:

```text
runs/<job-id>-trace-<timestamp>/
  events/events-<host>-<pid>.jsonl   # assistant, tool, start, complete, error events
  validation/                     # veRL validation generations and scores
  job.log
  sources.json                    # exact data/model revisions and database hash
  environment.freeze.txt
  environment.conda.yaml
  repo-revision.txt
  gpus.txt
```

```bash
python scripts/inspect_trace.py runs/<run>/events
python scripts/inspect_trace.py runs/<run>/events --trajectory <id>
```

All events in one rollout share `trajectory_id`, including SQL events dispatched
as asynchronous tasks. The `start` event includes the Spider sample index and
sampling settings. `assistant` events preserve each generated turn, including
tool-call markup. `complete` preserves prompt/response text, generated token IDs,
and the response mask (1 = model token, 0 = tool/environment token). Tool events
contain the full bounded tool output; veRL may further truncate what the model
sees, which can be inspected in the completed token stream. Errors and unfinished
trajectories remain distinguishable.

By default the inspector checks trace integrity: empty runs, incomplete
trajectories, error events, missing assistant events, and invalid token masks fail.
SQL tool records must include their tool/database identity, consistent arguments,
result or error details, and finite nonnegative execution time. Broken traces
are marked protocol **not evaluated** and excluded from the model-protocol rate.
Missing final answers, failed SQL, and no tool calls are reported as unsuccessful
model protocol outcomes, not infrastructure failures. All events are preserved;
the inspector never deletes or rewrites them. Real tool calls from unsuccessful
rollouts remain available for cache evaluation. Protocol success is not SQL answer
correctness, and reward scores are reported separately by veRL.
Use `--strict` to also fail on unsuccessful protocol outcomes (for wiring checks).
Copy durable traces off scratch to your project's CFS.

JSONL is the default trace format so no tracking server is required. veRL's
optional MLflow tracing can be enabled by adding
`actor_rollout_ref.rollout.trace.backend=mlflow` and
`actor_rollout_ref.rollout.trace.token2text=True` to `scripts/run.sh`, with an
`MLFLOW_TRACKING_URI` accessible to the workers. MLflow is installed but not
required by the default run.

## SQL behavior and determinism

Keep `actor_rollout_ref.rollout.free_cache_engine=True` in this hybrid setup,
including trace-only runs. vLLM starts with dummy weights and the manager's
wake-up path transfers the real actor weights before generation. Disabling this
flag to avoid sleep/wake transitions can skip that transfer and produce nonsense.
The earlier between-batch CUDA failure remains unresolved; first verify coherent
output with `SPIDER_VAL_SAMPLES=1 bash scripts/run.sh trace` on an 80 GB GPU.

The tool opens an existing database read-only and uses SQLite's authorizer to
allow reads and a short function allowlist. Writes, ATTACH, PRAGMA, multiple
statements, random/time functions, and extension loading are denied. It applies a
fixed VM instruction budget and caps rows/output size. SQL errors are returned to
the model so it can revise its query. Each call opens and closes its connection.

This constrains nondeterminism; it does **not** prove arbitrary SQL deterministic.
Unordered LIMIT, ties in ORDER BY, and SQLite bare-column aggregation can still
be ambiguous. The prompt asks for ordering with unique tie breakers; inspect the
generated queries before using these traces for cache-correctness research.
Record the SQLite/environment version and database hash for comparisons.

The launcher now verifies the database hash against `data/sources.json` and rejects
WAL sidecars before starting the run. Do not modify the database during a run.
Each run captures `manifest.json` (database, dataset and source hashes plus SQLite
version), Git status/diff, and a `source/` snapshot including new/uncommitted runtime
code. This records the actual local implementation, not just its last commit.

Limits live in `spider_trace/limits.py` and are shared by the prompt, tool and
launcher. This is not a guarantee of arbitrary SQL determinism; cache evaluation
still requires a fixed database and an explicit policy for unordered results.

The trace inspector validates **each trajectory**: start/complete records, no error
event, a successful untruncated SQL execution, aligned response masks, and a final
solution matching a prior successful SQL call exactly (outer whitespace ignored).
Equivalent rewrites are intentionally not inferred. An empty answer now fails,
even after a successful tool call. These are protocol checks, not Spider accuracy.

## Optional one-step GRPO test

```bash
# Inside your interactive allocation:
bash scripts/run.sh grpo

# Or submit a batch job:
sbatch --account=m4999 --export=ALL,MODE=grpo scripts/perlmutter.sbatch
```

This validates first, then runs one GRPO step with one prompt and two rollouts
on one GPU. Training requires more memory than trace collection, and this mode
has not been GPU-tested. The same mini dataset supplies both training and validation solely
for debugging: scores must not be reported as held-out evaluation. Prepare at
least one question. No checkpoints are saved by default.

The reward extracts `<solution>...</solution>` (also accepting legacy `<final_sql>`)
and compares execution results
against gold SQL: 1 for equality, 0 otherwise. Unordered results use multisets,
preserving duplicates; ordered gold results use sequence equality. Truncated or
failed results receive zero. This is a deliberately limited smoke-test reward,
not the official Spider evaluator: column permutations, floating-point tolerance,
equivalent ordering ties, and nested ORDER BY need a stronger evaluator before
scientific evaluation. Rewards do not currently require a prior successful tool
call. Groups with identical rewards provide no relative learning signal.

## Local checks and status

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/setup.sh scripts/run.sh scripts/perlmutter.sbatch
```

The local tests use Python's standard library (provenance tests also use Git). They cover repeated
queries, writes, nondeterministic functions, multiple statements, recursion bounds,
truncation, path traversal, duplicate rows, and ordering in rewards.
They also cover missing final answers, failed tools, trace masks, database hashes,
dirty source snapshots, and inserted message boundaries.

`bash scripts/check_tool.sh` additionally exercises the real pinned veRL state
machine with the saved Qwen tokenizer and real SQLite tool. Only model generation
is scripted: it tests a successful two-turn exchange and rejection of an empty
final answer. It needs the installed environment and prepared files, but no model
weights are loaded and no GPU allocation is required. Diagnostic events remain
under `runs/diagnostics`, separate from model-generated evaluation traces.
The full diagnostic and GPU rollout must be run on Perlmutter; the development
workspace does not have the veRL/model environment.

## Transfer without GitHub

From the local repository, create a portable Git bundle and transfer it:

```bash
git bundle create ../verl-spider-traces.bundle --all
scp ../verl-spider-traces.bundle <username>@perlmutter.nersc.gov:verl-spider-traces.bundle
```

On Perlmutter, clone the bundle into scratch, then follow the setup commands above:

```bash
git clone "$HOME/verl-spider-traces.bundle" "$PSCRATCH/verl-spider-traces"
cd "$PSCRATCH/verl-spider-traces"
```

The bundle includes committed code only. It excludes model weights, datasets,
environments, and traces, which the preparation and job scripts create locally.

## References

- [veRL v0.6.1 agent loop](https://github.com/verl-project/verl/blob/v0.6.1/verl/experimental/agent_loop/tool_agent_loop.py)
- [veRL v0.6.1 rollout configuration](https://github.com/verl-project/verl/blob/v0.6.1/verl/trainer/config/rollout/rollout.yaml)
- [Spider](https://yale-lily.github.io/spider)
- [Question dataset](https://huggingface.co/datasets/xlangai/spider)
- [SQLite database rehost](https://huggingface.co/datasets/HAL-9001/spider-databases)
- [Qwen model](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [Perlmutter jobs](https://docs.nersc.gov/systems/perlmutter/running-jobs/)
