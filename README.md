# veRL Spider traces on Perlmutter

Run Qwen2.5-Coder-7B-Instruct through veRL's asynchronous tool agent loop on a small
Spider subset. The model can explain its approach, execute SQL, inspect results,
revise SQL, and submit a final query. JSONL files preserve every assistant turn,
executed SQL, returned result, elapsed time, and the complete token trajectory.

Default: 10 validation questions from `concert_singer`, four sampled rollouts per
question, up to five executed calls and a sixth assistant turn for the answer.
Turn and token limits can terminate a trajectory earlier. Multiple calls are
allowed, not forced. Qwen2.5 has no Qwen3 `enable_thinking` mode: any explanation
is ordinary generated assistant text.

## Quick start

SSH to Perlmutter and clone into scratch. Replace `<nersc-account>` below.
Use a dedicated Conda environment named `verl-spider` for this project; your
existing `sysml` environment is not modified. The scripts load the Conda module
and activate the project environment automatically.

```bash
cd "$PSCRATCH"
git clone https://github.com/hom-cv/verl-spider.git
cd verl-spider

# Request an interactive allocation.
salloc --account=<nersc-account> --constraint='gpu&hbm80g' \
  --nodes=1 --ntasks=1 --cpus-per-task=128 --gpus-per-node=4 \
  --qos=interactive --time=01:00:00

# One-time setup and downloads on the allocated node.
# Creates the verl-spider Conda environment with Python 3.10 if missing.
srun --ntasks=1 --cpu-bind=none bash scripts/setup.sh --prepare

# Launch the experiment.
srun --ntasks=1 --cpu-bind=none bash scripts/run.sh trace
```

`setup.sh --prepare` also runs `prepare.py`; omitting `--prepare` installs only.
Use `bash scripts/prepare.sh` to download/prepare later without manually activating
the environment. On subsequent sessions, return to `$PSCRATCH/verl-spider`, request
an allocation, and run the same `srun` command. `run.sh` activates `verl-spider` itself.
The interactive allocation remains reserved after the script finishes; exit its
shell when finished. No `sbatch` file is needed for this workflow.

The first setup and downloads may consume most of the allocation. If it expires,
request another allocation and rerun setup; completed package installations and
downloads can be reused. To activate the environment manually on the cluster:

```bash
module load conda
conda activate verl-spider
```

Set `SPIDER_CONDA_ENV` to use a different dedicated environment name; use the same
value for setup, preparation and launch. The default named environment location
is controlled by your Conda configuration. No nested Python virtualenv is created.

Alternatively submit a batch job from the repository root:

```bash
sbatch --account=<nersc-account> scripts/perlmutter.sbatch
```

Setup and preparation need network access; GPU computation runs inside Slurm.
Install on a node permitted by your site's installation policy; use a CPU
allocation for expensive installs if necessary. No Docker daemon is required.
The GPU stack ships its CUDA runtime. Do not mix arbitrary CUDA libraries or
an unrelated PyTorch installation into this environment. Driver compatibility
must be checked on the allocated node if vLLM fails to initialize.

The batch job requests **one node with four A100 80GB GPUs**, 45 minutes, and 128 logical
CPUs. Adjust account/QOS/time to your allocation. This is a conservative starting
configuration, not a measured minimum. Validation still initializes veRL's hybrid
trainer/actor; it is not a lightweight standalone inference server.

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
The schema comes from the actual SQLite file. Gold SQL goes only to the reward
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

The inspector fails if there are no completed trajectories or no executed tools;
inspect per-trajectory call counts too. It does not assert that every model
response uses tools. Copy durable traces off scratch to your project's CFS.

JSONL is the default trace format so no tracking server is required. veRL's
optional MLflow tracing can be enabled by adding
`actor_rollout_ref.rollout.trace.backend=mlflow` and
`actor_rollout_ref.rollout.trace.token2text=True` to `scripts/run.sh`, with an
`MLFLOW_TRACKING_URI` accessible to the workers. MLflow is installed but not
required by the default run.

## SQL behavior and determinism

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

## Optional one-step GRPO test

```bash
# Inside your interactive allocation:
srun --ntasks=1 --cpu-bind=none bash scripts/run.sh grpo

# Or submit a batch job:
sbatch --account=<nersc-account> --export=ALL,MODE=grpo scripts/perlmutter.sbatch
```

This validates first, then runs one GRPO step with four prompts and four rollouts
per prompt. The same mini dataset supplies both training and validation solely
for debugging: scores must not be reported as held-out evaluation. Prepare at
least four questions. No checkpoints are saved by default.

The reward extracts `<final_sql>...</final_sql>` and compares execution results
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

The SQL/reward tests require only Python's standard library. They cover repeated
queries, writes, nondeterministic functions, multiple statements, recursion bounds,
truncation, path traversal, duplicate rows, and ordering in rewards.
GPU installation and the end-to-end Slurm run have not been executed in the
development workspace. The first Perlmutter run is an integration smoke test.

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
