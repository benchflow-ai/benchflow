# BenchFlow + TRL GRPO Pipeline

This runbook shows the intended end-to-end workflow for training
`Qwen/Qwen3-4B` on a BenchFlow-compatible task suite with TRL GRPO, then
measuring held-out lift with BenchFlow eval artifacts.

The example uses the public data-agent task suites:

- training: `benchflow/data_agent_rl_environment_train`
  at `34ff63c91731df6b3670bfcd7e3d44e6790ddc48` (`2,238` tasks)
- held-out eval: `benchflow/data_agent_rl_environment_eval`
  at `0ea976c79e3248c85737c4f7363484e4d47ce287` (`366` tasks)

Both repositories contain BenchFlow-native `task.md` packages with
`environment/` and `verifier/` directories. They contain no legacy
`task.toml`, `instruction.md`, or `tests/` paths. For difficulty analysis, use
the published `manifest.parquet`; the source task-level difficulty metadata is
not a reliable distribution field.

BenchFlow owns task loading, Daytona/Docker sandbox lifecycle, verifier
execution, artifacts, and lift reporting. TRL owns optimization.

## 1. Install Extras And Credentials

```bash
uv sync --locked --extra dev --extra trl --extra sandbox-daytona

# Load credentials from your shell, secret manager, or local untracked .env.
# Keep this file outside commits and logs.
source .env
```

Required environment variables:

- `HF_TOKEN` for Hugging Face dataset snapshots and Kaggle bucket fetches
- `DAYTONA_API_KEY` for Daytona-backed runs
- model/provider credentials for the OpenAI-compatible model server used by
  the eval agent

## 2. Materialize Train And Eval Task Trees

Resolve and pin exact dataset commit SHAs before spending on training or eval.
The command writes a `.benchflow-source.json` sidecar so downstream artifacts
can retain dataset provenance.

```bash
bench tasks snapshot-hf \
  benchflow/data_agent_rl_environment_train \
  .local/data-agent-train \
  --path tasks \
  --revision 34ff63c91731df6b3670bfcd7e3d44e6790ddc48 \
  --overwrite

bench tasks snapshot-hf \
  benchflow/data_agent_rl_environment_eval \
  .local/data-agent-eval \
  --path tasks \
  --revision 0ea976c79e3248c85737c4f7363484e4d47ce287 \
  --overwrite
```

Smoke-check one held-out task:

```bash
bench tasks check .local/data-agent-eval/0000_369_369503_qa_1
```

## 3. Run Baseline Eval

Serve the base model however your eval agent expects it, then run the held-out
eval suite through BenchFlow. Keep this command identical for the final
checkpoint eval except for the model endpoint.

```bash
bench eval run \
  --tasks-dir .local/data-agent-eval \
  --agent openhands \
  --model vllm/Qwen-Qwen3-4B \
  --sandbox daytona \
  --concurrency 32 \
  --usage-tracking required \
  --health-summary-out experiments/qwen3-4b-grpo/reports/baseline-health.json \
  --canonicalize one-healthy-per-task \
  --canonical-selection-out experiments/qwen3-4b-grpo/reports/baseline-canonical.json \
  --jobs-dir experiments/qwen3-4b-grpo/jobs/baseline
```

Use a small subset first if this is a new machine or a new Daytona account:

```bash
bench eval run \
  --tasks-dir .local/data-agent-eval \
  --include 0000_369_369503_qa_1 \
  --agent openhands \
  --model vllm/Qwen-Qwen3-4B \
  --sandbox daytona \
  --concurrency 1 \
  --jobs-dir experiments/qwen3-4b-grpo/jobs/baseline-smoke
```

## 4. Run GRPO Training

The training script should use `BenchFlowSpec` as the adapter into TRL:

```python
from trl import GRPOConfig, GRPOTrainer

from benchflow.integrations.trl import BashHarnessConfig, BenchFlowSpec

spec = BenchFlowSpec(
    tasks_dir=".local/data-agent-train",
    bash_harness=BashHarnessConfig(
        environment="daytona",
        jobs_dir="experiments/qwen3-4b-grpo/jobs/grpo-train",
        submit_path="/workdir/answer.txt",
    ),
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-4B",
    args=GRPOConfig(
        learning_rate=1e-6,
        num_generations=8,
        max_completion_length=2048,
        max_steps=300,
        max_tool_calling_iterations=25,
        log_completions=True,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url="http://localhost:8000",
    ),
    train_dataset=spec.train_dataset,
    environment_factory=spec.environment_factory,
    reward_funcs=spec.reward_funcs,
)

trainer.train()
```

Save the Python snippet above as a run-local script, for example
`experiments/qwen3-4b-grpo/scripts/train_grpo.py`. The script path and
Accelerate config are experiment-owned files, not bundled BenchFlow assets.

Recommended production layout:

```bash
# Terminal 1: generation server
CUDA_VISIBLE_DEVICES=2,3 trl vllm-serve \
  --model Qwen/Qwen3-4B \
  --tensor-parallel-size 2 \
  --port 8000

# Terminal 2: training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file <accelerate-deepspeed-zero2.yaml> \
  --num_processes 2 \
  experiments/qwen3-4b-grpo/scripts/train_grpo.py
```

Start with a tiny run before the full run:

- `max_steps=2`
- `num_generations=2`
- `BenchFlowSpec(..., max_tasks=4)`
- `BashHarnessConfig(environment="docker" or "daytona", jobs_dir=".local/grpo-smoke")`

## 5. Run Final Eval

Evaluate the trained checkpoint on the exact same held-out task tree and eval
settings used for the baseline.

```bash
bench eval run \
  --tasks-dir .local/data-agent-eval \
  --agent openhands \
  --model vllm/qwen3-4b-grpo-checkpoint \
  --sandbox daytona \
  --concurrency 32 \
  --usage-tracking required \
  --health-summary-out experiments/qwen3-4b-grpo/reports/grpo-health.json \
  --canonicalize one-healthy-per-task \
  --canonical-selection-out experiments/qwen3-4b-grpo/reports/grpo-canonical.json \
  --jobs-dir experiments/qwen3-4b-grpo/jobs/grpo
```

## 6. Compare Held-Out Lift

Only healthy, scored tasks present in both jobs are included in paired lift.
Infra failures remain visible in the coverage section instead of being counted
as model failures.

```bash
bench eval compare-lift \
  --baseline experiments/qwen3-4b-grpo/jobs/baseline \
  --trained experiments/qwen3-4b-grpo/jobs/grpo \
  --out experiments/qwen3-4b-grpo/reports/EVAL_LIFT.md \
  --json-out experiments/qwen3-4b-grpo/reports/eval_lift.json
```

The report includes:

- pass-rate and mean-reward deltas
- paired bootstrap confidence intervals
- healthy coverage and excluded/error counts
- metadata breakdowns when result metadata contains fields such as
  `difficulty_level` or `reward_mode_initial`

## Production Acceptance Gate

Treat the run as complete only when all of these are true:

- baseline and final eval both cover at least 95% healthy verifier outputs
- GRPO training logs show non-constant reward signal
- final `compare-lift` report is generated from paired healthy tasks
- model, dataset revisions, commands, and Daytona settings are recorded
- trajectory artifacts are preserved for representative pass and failure cases
