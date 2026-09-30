# Fireworks: BenchFlow agents on Fireworks models, and RL with BenchFlow rewards

This cookbook covers two things on Fireworks:

1. **Agents on Fireworks models.** Run a BenchFlow agent (`opencode`) on a Fireworks-hosted model with `bench eval run`, through BenchFlow's model proxy, with tokens and cost recorded per trial.
2. **Reinforcement learning with BenchFlow rewards.** Train a LoRA on Fireworks' serverless Training API, where every episode is a multi-turn rollout in a BenchFlow sandbox and the reward is the task's own BenchFlow verifier. Then serve the result on a dedicated deployment and compare it with the base model on the held-out split with the shared evaluator, [`../common/evaluate.py`](../common/evaluate.py).

It uses public BenchFlow primitives only: `bench eval run` and the `fireworks/` provider route, the TRL adapter's `BenchFlowRuntimeEnvironment` (a `TaskRuntime` sandbox with `run_bash` and `submit`), the shared harness ([`../common/harness.py`](../common/harness.py)), and the attribution-aware reward helper (`benchflow.integrations.rewards`).

## Which Fireworks product, and why

Fireworks has had two ways to train with your own reward:

- **Managed RFT** (`firectl rftj`, with an Eval Protocol evaluator that Fireworks builds and runs). Fireworks' own agent skills now say it "is deprecated and accepts no new jobs. Route new work to the Training API" (`fw-ai/cookbook`, `skills/fireworks-training/references/choose-method.md`, added 2026-09-22); the public docs did not say so yet on 2026-09-30. Its multi-turn mode (Eval Protocol's `RemoteRolloutProcessor`) calls an `/init` endpoint on your server over the internet and sends it a Fireworks key in the request body; its single-turn mode scores one completion in Fireworks' own sandbox, so a BenchFlow verifier would have to be reachable from there.
- **The serverless Training API** (generally available, billed per token). Your loop runs on your machine and calls Fireworks for the LoRA's forward, backward and optimizer steps and for sampling from the current weights. Nothing has to reach your machine, and sampling returns the exact tokens and their logprobs.

This cookbook uses the Training API. It is the tightest integration Fireworks allows: the policy acts in a real BenchFlow sandbox, turn by turn, and the BenchFlow verifier's reward is the training reward, with nothing exposed to the internet.

## The training rule

The reward is the verifier's, under the rule every RL cookbook here uses (proposed, not yet a formal BenchFlow decision): a failure is infrastructure only when the policy could not have caused it. Infrastructure failures are dropped and counted; every other failure scores 0, timeouts included.

| Outcome | Reward |
|---|---|
| The verifier scored the episode | the verifier's reward (v2 tasks give partial credit) |
| The policy ran out of turns, called no tool, ran out of context, or its command timed out | 0 |
| The verifier crashed after the policy acted (it could have broken the sandbox) | 0 |
| The sandbox never started; the sampler failed after retries; the verifier crashed on a sandbox the policy never touched | dropped: left out of its group, counted by reason |

In each GRPO group (several episodes of one task), a dropped episode is left out of the mean and the standard deviation and gets no advantage; a group whose kept rewards are all equal is skipped.

## How an episode is sampled

The Training API samples tokens, not chat messages, so the loop does the chat formatting itself ([`fireworks_chat.py`](fireworks_chat.py)): it renders the conversation with the model's own chat template (pinned tokenizer revision), samples the next assistant turn with its per-token logprobs, parses the Qwen3.8 completion (thinking, content, `<tool_call>` blocks) into an OpenAI-style message, and runs the tool calls in the sandbox exactly as `evaluate.py` does.

Training needs the exact tokens the policy sampled. An episode's turns are joined into one training sequence only where the next turn's prompt provably extends the previous prompt and completion, token for token; otherwise the next turn starts a new sequence. Every sampled token is trained exactly once, with the logprob it was sampled with, and prompt and tool-result tokens carry no loss. (With Qwen3.8's template, all of the 192 screened episodes of task family v1 joined into one sequence each.)

## Keys: where they go, and what never sees them

| Key | Where it lives | Who reads it | Never sees it |
|---|---|---|---|
| `FIREWORKS_API_KEY` | a mode-600 file you `source`, for example `~/.config/benchflow/fireworks.env` | `bench` and its model proxy (on this machine, for Docker sandboxes); `fireworks_rl.py`, `fireworks_deploy.py` and `evaluate.py` on this machine | the agent, every sandbox, any command line, any log |
| `DAYTONA_API_KEY` | a mode-600 file you `source` | BenchFlow's sandbox client | the agent and the policy |

```bash
set -a; . ~/.config/benchflow/fireworks.env; . ~/.config/benchflow/daytona.env; set +a
export FIREWORKS_ACCOUNT_ID=<your account id>
```

- **RL and evaluation sample from this machine.** The policy's model calls go from `fireworks_rl.py` or `evaluate.py` straight to Fireworks; a sandbox only ever receives the commands the policy runs.
- **The agent evaluation uses Docker sandboxes.** BenchFlow's model proxy runs beside the sandbox: on Docker that is this machine, and the container sees only the proxy's URL and a per-run master key. On Daytona the proxy runs inside the sandbox, so the key (read from a launch file that is deleted at once) lives in the environment of the root-owned proxy process there. The agent never receives it either way.
- **Do not use the generic `vllm/` route for Fireworks.** It needs the key on the command line (`--agent-env`) or disguised as `OPENAI_API_KEY`, and a harness that ignores base-URL overrides would send an `OPENAI_API_KEY` to OpenAI. `fireworks/` reads `FIREWORKS_API_KEY`, which BenchFlow strips from the agent's environment like every registered provider key.

## Prerequisites

- A BenchFlow checkout with `uv sync --extra sandbox-daytona`, Docker for the agent evaluation, and `uv` for the training dependencies (`fireworks-ai[training-sdk]` and `transformers`, installed on the fly by `run.sh`; no PyTorch is needed).
- A Fireworks account with serverless training (the default quota is 8 concurrent runs) and a payment method, and a Daytona key.
- The RL cookbooks' task family (`TASKS=<folder with train/ and test/>`), from [`../tasks/`](../tasks/).
- An always-on machine for training. A training session expires after 10 minutes without activity; a laptop that sleeps loses its run (Fireworks answers every later call with HTTP 503).

## Steps

Every step is one command of [`run.sh`](run.sh), run from the repository root. The knobs are environment variables at the top of the script.

| Step | Command | What it does |
|---|---|---|
| 1 | `run.sh agent-eval` | `bench eval run --agent opencode --model fireworks/accounts/fireworks/models/glm-5p3 --sandbox docker` on three test tasks |
| 2 | `run.sh screen` | Samples each of 48 train tasks 4 times with the untrained weights; writes `learnable.txt`, the tasks whose episodes disagree |
| 3 | `run.sh train` | GRPO on the learnable tasks; at the end, promotes the final checkpoint to a private model, `MODEL_ID` |
| 4 | `run.sh deploy` | One dedicated deployment for the base model and one for the trained LoRA (live merge) |
| 5 | `run.sh evaluate` | `evaluate.py` on the test split, base and tuned, with the same harness and turn budget |
| 6 | `run.sh cleanup` | Deletes both deployments and prints the GPU time Fireworks billed each |

### Step 1: agents on a Fireworks model

`fireworks/<model id>` is a registered provider: BenchFlow sends it to `https://api.fireworks.ai/inference/v1` with `FIREWORKS_API_KEY`, through the generic OpenAI route, so tool calls pass through unchanged for any model, including your own fine-tuned ones. (LiteLLM's native Fireworks route drops `tools` for any model its price table does not list as supporting function calling.) The price comes from LiteLLM's Fireworks table by exact id, or from BenchFlow's exact-id table for newer models, including the cached-input price: agents resend the conversation every turn, so most prompt tokens are cached. Every trial's `result.json` has `agent_result.n_input_tokens`, `n_cache_read_tokens`, `n_output_tokens`, `cost_usd`, `usage_source: provider_response` and `price_source: litellm`.

Check that a model is actually served before a batch: `/inference/v1/models` lists models that answer HTTP 404 (on 2026-09-30, `kimi-k2p6`, `deepseek-v4-pro` and `qwen3-8b`). BenchFlow treats a provider 404 as permanent and does not retry it.

### Step 2: screen

With a pass/fail verifier, a task every episode solves (or fails) gives GRPO nothing to learn from; Fireworks' own PostTrain pilot on 2026-09-25 stopped for exactly this reason. `screen` measures that before any training: it opens a training session, saves the untrained LoRA (identical to the base model) as a sampler checkpoint, runs `--group-size` episodes of each task, and keeps the tasks whose kept rewards differ.

### Step 3: train

Each step saves the current weights as a sampler checkpoint (names stay within Fireworks' 17-character limit), runs `GROUPS_PER_STEP` groups of `GROUP` episodes with it (`CONCURRENCY` sandboxes at a time), builds importance-sampling datums from the groups with a reward spread, and takes one `forward_backward` plus `optim_step`. `--max-usd` stops before a step that would pass it, from the token meters (an upper bound: prefill is counted as if never cached). At the end the final checkpoint is promoted to a model; promotion needs the session to be alive, so it happens in the same run.

Records in `$OUT/train/`: `metrics.jsonl` (per step: mean reward, drops and zeros by reason, groups with signal, datums, trained tokens, the forward-backward metrics, cost, timing), `episodes.jsonl` (every episode, with how many training sequences its turns became), `train.json` (the session, run, config, final checkpoint and promoted model), and `jobs/` (every rollout folder, plus `rollouts.jsonl` with each episode's messages and reward decision, dropped ones included).

### Steps 4 to 6: serve, evaluate, clean up

A LoRA trained on Fireworks can only be served on a dedicated deployment: Fireworks has no serverless LoRA, and `qwen3p8-27b` has no multi-LoRA shape, so the base model and the trained LoRA each get a single-GPU deployment on a validated shape (`fireworks_deploy.py shapes MODEL` lists them). `create` pins one replica and sets an expiry time (3 hours by default) after which Fireworks deletes the deployment even if you forget. `evaluate` calls them as `MODEL#accounts/ACCOUNT/deployments/ID` through the OpenAI-compatible endpoint.

## Results (2026-09-30)

To be filled in with this run's numbers.

## Troubleshooting

- **Every group scores the same:** the task family is too easy (or too hard) for the model. On family v1, `qwen3p8-27b` solved 192 of 192 screened episodes under the shared harness. Use a harder family, or a tighter budget applied to training and evaluation alike (`MAX_TURNS`, passed to both).
- **HTTP 503 on every call after a pause:** the training session expired (10 minutes without activity). Keep the driver awake; a new run starts a new session.
- **`save_weights_for_sampler` names:** at most 17 characters of `a-z`, `0-9` and `-`; a longer name fails only at promotion.
- **Promotion fails with NOT_FOUND:** the session is gone; promote before the run ends (`train` does).
- **`cost_usd` is null** in step 1: the model is newer than LiteLLM's pinned price table and not in `HOSTED_MODEL_COST_PER_TOKEN` (`src/benchflow/providers/litellm_config.py`); add input, output and cached-input prices from Fireworks' serverless pricing page.
- **Tokenizer:** the tokenizer must match the base model (`--tokenizer`, `--tokenizer-revision`); it renders prompts and decodes samples on your machine, and a mismatch silently corrupts rewards.

## Files

| File | What |
|---|---|
| `run.sh` | The steps |
| `fireworks_rl.py` | `screen` and `train`: sessions, sampling, episodes in BenchFlow sandboxes, GRPO datums, promotion |
| `fireworks_chat.py` | Qwen3.8's chat format for token-level sampling, and exact-token training sequences (no Fireworks or BenchFlow imports) |
| `fireworks_deploy.py` | Validated shapes, dedicated deployments with an expiry, smoke tests, deletion, billed GPU time |
| [`../../../../tests/test_rl_fireworks_example.py`](../../../../tests/test_rl_fireworks_example.py) | Parsing, token-exact sequences, group advantages with drops, datum alignment, checkpoint names, the deployment helper |
