# Baseten: evaluate BenchFlow agents on Model APIs, and serve and evaluate a trained checkpoint

This cookbook covers two things Baseten does for an RL workflow:

1. **Model APIs for agents.** Run a BenchFlow agent (`opencode`) on Baseten-hosted models (GLM-5.3, Kimi K3) with `bench eval run`, through BenchFlow's model proxy, with tokens and cost recorded per trial.
2. **Serving a trained checkpoint.** Deploy a base model and a LoRA checkpoint with Truss on one H100, then compare them on the held-out split with the shared evaluator, [`../common/evaluate.py`](../common/evaluate.py), the same harness every RL cookbook trains with.

It uses public BenchFlow primitives only: `bench eval run`, the `fireworks/` and `baseten/` provider routes, `evaluate.py` (`TaskRuntime` sandboxes and the attribution-aware reward helper). Training on Baseten is covered in [Baseten Training](#baseten-training), which was not run.

## Keys: where they go, and what never sees them

| Key | Where it lives | Who reads it | Never sees it |
|---|---|---|---|
| `BASETEN_API_KEY` | a mode-600 file you `source`, for example `~/.config/benchflow/baseten.env` containing `BASETEN_API_KEY=...` | `bench` and its model proxy (on this machine, for Docker sandboxes); `evaluate.py`; `baseten_deploy.py`; `truss` (through its environment) | the agent, the sandbox, any command line, any log |
| `DAYTONA_API_KEY` | a mode-600 file you `source` | BenchFlow's sandbox client | the agent |

Load keys into the environment of the one command that needs them, and never pass one as an argument:

```bash
set -a; . ~/.config/benchflow/baseten.env; set +a
```

Two places deserve care:

- **The agent evaluation uses Docker sandboxes.** BenchFlow's model proxy runs beside the sandbox: with `--sandbox docker` it runs on this machine, and the container sees only the proxy's URL and a per-run master key. With `--sandbox daytona` the proxy runs inside the sandbox: the key arrives in a mode-600 launch file that is deleted as soon as the proxy has read it, and then lives in the environment of the proxy process, which runs as root. The agent, running as the `agent` user, cannot read it, but anything with root in that sandbox could. The agent never receives the key either way; keep provider keys off remote sandboxes anyway.
- **`evaluate.py` calls the model from this machine,** so its sandboxes (Daytona or Docker) never see the key.

`baseten_deploy.py push` hands the key to `truss` as `BASETEN_TRUSS_AUTH_API_KEY` in its environment. Avoid `truss login --api-key ...`: it puts the key in the process list, and `truss login` stores it in plain text in `~/.trussrc`.

## Prerequisites

- A BenchFlow checkout with `uv sync --extra sandbox-daytona` (for `evaluate.py` on Daytona), and Docker for the agent evaluation.
- A Baseten API key with access to Model APIs and to deploying models, and a Daytona key.
- The RL cookbooks' task family (`TASKS=<folder with train/ and test/>`), from [`../tasks/`](../tasks/).

## Steps

Every step is one command of [`run.sh`](run.sh), run from the repository root. The knobs are environment variables at the top of the script.

### 1. Agents on Model APIs

```bash
TASKS=~/tasks/v1 docs/examples/rl/baseten/run.sh agent-eval
```

This runs, for each model in `MODELS`:

```bash
uv run bench eval run --tasks-dir $TASKS/test --include sql-900000 --include log-900001 --include bugfix-900003 \
  --agent opencode --model baseten/zai-org/GLM-5.3 --sandbox docker --concurrency 3 --usage-tracking required
```

`baseten/<slug>` is a registered provider: BenchFlow sends it to `https://inference.baseten.co/v1` with `BASETEN_API_KEY`, through the generic OpenAI route (so tool calls pass through unchanged), and prices it from an exact-id table, including the cached-input price. Agents resend the whole conversation every turn, so most prompt tokens are cached; billing them at the full input price would overstate the cost several times over.

Check the record: every trial's `result.json` has `agent_result.n_input_tokens`, `n_cache_read_tokens`, `n_output_tokens`, `cost_usd`, `usage_source: provider_response` and `price_source: litellm`.

### 2. Deploy a base model and a checkpoint

```bash
docs/examples/rl/baseten/run.sh deploy
export MODEL_ID=... DEPLOYMENT_ID=...     # from the push output
uv run python docs/examples/rl/baseten/baseten_deploy.py wait $MODEL_ID $DEPLOYMENT_ID
uv run python docs/examples/rl/baseten/baseten_deploy.py smoke $MODEL_ID $DEPLOYMENT_ID --model benchflow-sft
```

[`truss/qwen35-9b-lora/config.yaml`](truss/qwen35-9b-lora/config.yaml) serves `Qwen/Qwen3.5-9B` and a LoRA, `benchflow-sft`, from one vLLM server on one H100. It follows Baseten's own Qwen3.5-9B recipe (vLLM 0.22.0, the `qwen3` reasoning parser) and adds the `qwen3_coder` tool parser, which `evaluate.py` needs: it sends `tools` with `tool_choice: "auto"`, and vLLM refuses that without `--enable-auto-tool-choice`. The image and both weights are pinned. `smoke` makes one plain call and one tool call; both must succeed before you evaluate.

**Serving your own checkpoint.** Point the second `weights` entry at it. The config's comments show a private GCS source, with either Baseten's GCP workload identity (`GCP_OIDC`, no stored key) or a service-account secret. For a full fine-tuned model rather than an adapter, point the first entry at it and drop the `--enable-lora` flags. The deployment is private to your Baseten workspace: a request without a valid key is refused before it reaches the model.

**Waking an existing deployment.** `baseten_deploy.py list` shows every model and deployment with its status. An `INACTIVE` deployment needs `activate` (it redeploys); a `SCALED_TO_ZERO` one needs `wake` (or any request, which waits through the cold start). Then `wait` until `ACTIVE`, and `config` prints its Truss config, including the served model names. Check that the server was started with a tool parser before you point `evaluate.py` at it.

### 3. Evaluate both on the held-out split

```bash
TASKS=~/tasks/v1 docs/examples/rl/baseten/run.sh evaluate
```

This runs `evaluate.py` twice against the same deployment URL (`https://model-<MODEL_ID>.api.baseten.co/deployment/<DEPLOYMENT_ID>/sync/v1`), once with `--model Qwen/Qwen3.5-9B` and once with `--model benchflow-sft`, with the shared harness's defaults (10 turns, 1024 tokens per call, temperature 0.7) and `--samples 2`. Each writes `summary.json` (solve rate with a 95% Wilson interval, drops by reason, zeros by reason, results per task kind, token totals), `episodes.jsonl`, and the rollout folders.

### 4. Clean up

```bash
docs/examples/rl/baseten/run.sh cleanup
```

`deactivate` stops the deployment within seconds; requests to it then get HTTP 400 until you `activate` it again. A deployment with `min_replica: 0` also scales to zero on its own after `scale_down_delay` (900 s by default), but a replica still bills until then. Run `baseten_deploy.py list` at the end and look for anything `ACTIVE` you did not mean to leave.

## Results (2026-09-30)

All runs on the RL cookbooks' task family v1, from a GCP VM.

**Agents on Model APIs** (`opencode`, Docker sandboxes, test tasks `sql-900000`, `log-900001`, `bugfix-900003`):

| Model | Solved | Tokens (cached) | Cost | Wall time |
|---|---|---|---|---|
| `baseten/zai-org/GLM-5.3` | 3/3 | 83,700 (79,232) | $0.0186 | 1.7 min |
| `baseten/moonshotai/Kimi-K3` | 3/3 | 105,924 (69,760) | $0.1480 | 1.6 min |

For comparison, the same three tasks on Fireworks' GLM-5.3 cost $0.0624 (Fireworks charges $0.26 per million cached input tokens, Baseten $0.14).

**Base versus checkpoint on one deployment** (`evaluate.py`, 50 test tasks x 2 samples, Docker sandboxes, both runs at once):

| Model | Solved | 95% interval | Dropped | Mean turns |
|---|---|---|---|---|
| `Qwen/Qwen3.5-9B` (base) | 99/99 kept | 0.963-1.000 | 1 (`sandbox_start`) | 4.9 |
| `benchflow-sft` (the team's env-0 SFT adapter) | 98/100 | 0.930-0.994 | 0 | 5.0 |

The base solved every kept episode in every task kind and level: **task family v1 has no headroom for a 9B model under the shared harness**, so it cannot show a training gain. The adapter was trained for a different environment (env-0), and it is no better here; its two failures ended without a tool call. Use a harder family (the RL cookbooks' v2 adds partial credit and messier data) to compare checkpoints.

**Time and cost.** The push built and loaded in about 6 minutes; both evaluations together took about 9 minutes; the deployment was up about 16 minutes in total (06:55 to 07:11 UTC), about $1.73 at $6.50 per H100-hour (Baseten bills builds and model loads too). The agent evaluations cost $0.17. Total Baseten spend: about $1.90.

## Baseten Training

Baseten has two training products (checked 2026-09-30, docs.baseten.co):

- **Training Jobs** run any container on H100 or H200 GPUs, billed per GPU-minute ($6.50 per H100-hour). Checkpoints saved to `$BT_CHECKPOINT_DIR` sync automatically, and `baseten train checkpoint deploy` serves one on vLLM. The RL examples (`basetenlabs/ml-cookbook`) are VeRL GRPO recipes on 8 H100s. A BenchFlow reward would be a custom reward function that reaches BenchFlow sandboxes over the network, which puts a sandbox provider's key into the training job.
- **Loops** (early access, LoRA only) is the closer fit: your script drives a Baseten trainer and sampler, and the environment and reward run on your machine, the same shape as Fireworks' serverless Training API and Tinker. It supports `Qwen/Qwen3.5-9B`. The Fireworks cookbook's loop ([`../fireworks/fireworks_rl.py`](../fireworks/fireworks_rl.py)) is the pattern to port, and `baseten loops checkpoint deploy` would replace step 2.

Neither was run here: Loops access was not confirmed for this workspace, and a Training Jobs run needs a container with BenchFlow and a GRPO trainer, which is a larger piece of work than this cookbook. A small Training Jobs run on one H100 for two hours would cost about $13.

## Troubleshooting

- **HTTP 400 `"auto" tool choice requires --enable-auto-tool-choice`**, or every episode ending `no_tool_call` on a dedicated deployment: the vLLM server has no tool parser. The account's older `env0-qwen35-9b-*-baseten` deployments start vLLM without one, which is why this cookbook deploys a new one. `baseten_deploy.py config` shows the flags.
- **HTTP 400 "Model version ... is deactivated":** `activate` it, then `wait`.
- **A slow first request:** a scaled-to-zero deployment holds requests while a replica starts (up to 20 minutes). `wake` and `wait` before evaluating.
- **`DEPLOY_FAILED`:** open the logs URL from `push`; an unpinned image (`vllm/vllm-openai:nightly`) is a common cause, which is why this config pins one.
- **`cost_usd` is null** for a Model API: the model is newer than LiteLLM's pinned price table and not in BenchFlow's exact-id table (`HOSTED_MODEL_COST_PER_TOKEN` in `src/benchflow/providers/litellm_config.py`). Add it with the price from baseten.co/pricing, input, output and cached input.
- **The model name:** a Model API takes the slug (`zai-org/GLM-5.3`); a dedicated deployment takes the served name from its Truss config (`--served-model-name`, or a `--lora-modules` name).

## Files

| File | What |
|---|---|
| `run.sh` | The steps |
| `baseten_deploy.py` | List, push, wait, smoke-test, activate, wake, rescale and deactivate deployments; the key only in the environment |
| `truss/qwen35-9b-lora/config.yaml` | A base model and a LoRA on one H100, with tool calling on |
| [`../../../../tests/test_rl_baseten_example.py`](../../../../tests/test_rl_baseten_example.py) | The key never reaches `truss`'s argv; the config keeps tool parsing on and pins its image and weights |
