# BenchFlow environments

This example trains on [BenchFlow](https://github.com/benchflow-ai/benchflow) tasks through Miles' agent-function layer. Each rollout is one BenchFlow episode: a task sandbox on Daytona or Docker, a `run_bash`/`submit` tool loop against the rollout's TITO session, the task's own verifier, and a reward that says why it is what it is.

```text
agentic_tool_call.generate
  -> benchflow_agent_function.run --POST /run--> BenchFlow environment server
                                                   |- task sandbox: commands only
                                                   '- model calls ----> session URL (TITO)
```

What BenchFlow brings to the sample:

- **A verdict with attribution.** The environment server knows whether the policy acted and which side failed, so it applies the rule of the agent-function failure contract (#2802) after the agent starts too: a sandbox that never started, a failing session server, or a verifier that crashed on a sandbox the policy never used is discarded with `InfraAbort`; everything else is scored, 0 for failures the policy could have caused, with a named `exit_status`.
- **Reward integrity.** When BenchFlow's integrity audit is on (`serve --integrity audit`, in BenchFlow releases that have it), an episode caught exploiting its grader scores 0 and is flagged (`exit_status: IntegrityViolation`, `eval_report.flagged`).
- **Any BenchFlow task.** Tasks are folders in BenchFlow's format (`task.md`, or a `task.toml` folder); the verifier, sandbox image and limits come from the task. The BenchFlow RL cookbook ships a generated task family with disjoint train and test splits.
- **An audit trail.** Every episode, discarded ones included, is kept as a BenchFlow rollout folder (`result.json`, verifier output, the conversation in `policy/messages.json`) and one line of `rollouts.jsonl`.

## How the pieces talk

- The environment server runs next to the rollout workers, in its own Python environment: BenchFlow needs Python 3.12 or newer and pins litellm, which caps mcp below 2, so it stays out of Miles' environment instead of being resolved together with it.
- Model calls go from the environment server straight to the session URL: one non-streamed chat request per turn, with each assistant message sent back exactly as the session server returned it (`reasoning_content` and `tool_calls` included), so the strict message matcher always extends the session.
- The sandbox never calls the model. The policy's commands reach it through the sandbox provider's API, and nothing inside it needs a route to the session server or the SGLang router. So no `--session-server-external-host` is needed, and no port has to be open to the sandbox provider's network.
- The server binds to 127.0.0.1 by default. On a multi-node job, run it on the node that hosts the rollout manager, or bind another address with `--token-file` (a bearer token; the launcher forwards only the file's path to the workers, as with provider keys).
- No provider or sandbox key enters a task sandbox: the Daytona key stays in the environment server's process, and the policy runs as the task's non-root sandbox user.

## 1. Install BenchFlow in its own environment

On the rollout host, next to Miles, a BenchFlow version that ships `benchflow.integrations.miles`:

```bash
uv venv /root/benchflow-venv --python 3.12
VIRTUAL_ENV=/root/benchflow-venv uv pip install "benchflow[sandbox-daytona]"
```

The Daytona key stays with BenchFlow: load it into the shell that starts the server from a file (for example `set -a; . daytona.env; set +a`), never into Miles' environment and never on a command line.

## 2. Tasks and prompt data

Any folder of BenchFlow tasks works. The BenchFlow RL cookbook's task family (SQL, log, CSV and bug-fix questions in one shared image, train seeds `[0, 100000)` and test seeds `[900000, 1000000)`) is generated from BenchFlow's repository:

```bash
python docs/examples/rl/tasks/generate.py --split train --out /root/benchflow/tasks/train
python docs/examples/rl/tasks/generate.py --split test --out /root/benchflow/tasks/test
# Each task carries an oracle for BenchFlow's own checks; train and evaluate on copies without it.
find /root/benchflow/tasks -name oracle -type d -prune -exec rm -rf {} +
```

The rows Miles reads (`--input-key prompt --metadata-key metadata`, no `--apply-chat-template`): the task prompt plus the harness message, and `metadata.instance_id` naming the task folder.

```bash
/root/benchflow-venv/bin/python -m benchflow.integrations.miles prepare \
    --tasks-dir /root/benchflow/tasks/train --out /root/benchflow/train.jsonl --split train
```

## 3. Start the environment server

```bash
/root/benchflow-venv/bin/python -m benchflow.integrations.miles serve \
    --tasks-dir /root/benchflow/tasks/train --sandbox daytona --max-sandboxes 32 \
    --jobs-dir /root/benchflow/jobs --job-name train
```

`--max-sandboxes` caps the sandboxes alive at once (episodes beyond it wait); `--max-turns`, `--bash-timeout` and `--max-output-chars` are the harness limits (defaults: 10 turns, 30 s, 2,000 characters); `--episode-timeout` is the wall-clock cap from sandbox start to verdict: it scores 0 once the policy has acted, and before that the overrun is discarded. `GET /health` shows episodes, sandboxes in use and exit statuses.

## 4. Launch

```bash
python examples/experimental/benchflow/run.py \
    --model-name Qwen3-1.7B --prompt-data /root/benchflow/train.jsonl \
    --num-rollout 30 --rollout-batch-size 8 --n-samples-per-prompt 8 --global-batch-size 64 \
    --save-dir /root/checkpoints/benchflow --tensorboard-dir /root/tb
```

`run.py` is a one-GPU recipe (FSDP, colocated, `--tito-model qwen3`; it turns Qwen3's thinking off for every session with `--apply-chat-template-kwargs`). Before it submits the job it checks that the environment server answers and serves every task the prompt data names; a task it lacks would otherwise fail each of its samples before the first model call, and Miles would keep replacing the dropped groups.

The agentic wiring is `launch_common.agentic_train_args`:

| Flag | Value |
|---|---|
| `--custom-agent-function-path` | `benchflow_agent_function.run` (with `abort` as the oversampling hook) |
| `--custom-rm-path` | `benchflow_agent_function.reward_func` |
| `--rollout-function-path` | `benchflow_rollout.RolloutFn` (adds `benchflow/*` metrics) |
| `--dynamic-sampling-filter-path` | `apply_reward_nonzero_std_filter`, with `--over-sampling-batch-size 1` |

Groups whose episodes all got the same reward carry no GRPO signal: they are dropped and replaced, and counted in `rollout/dynamic_filter/drop_zero_std_*`.

## Failure semantics

| `exit_status` | Meaning | Sample |
|---|---|---|
| `Submitted`, `NoToolCall`, `TurnLimitExceeded` | the policy submitted, stopped calling tools, or used its turns | verifier's reward |
| `SequenceLengthLimitExceeded` | a reply was cut at `max_tokens`, or the context would pass `--max-seq-len` | verifier's reward |
| `RequestRejected` | the session server refused a request the policy's output can break (HTTP 400, 409, 422, 500) | verifier's reward |
| `TimeLimitExceeded`, `VerifierError`, `AgentError`, `NoReward` | failures the policy could have caused (the episode's wall-clock cap counts only after the policy acted; before, the overrun is discarded under the phase that hung) | 0 |
| `IntegrityViolation` | the integrity audit caught the policy exploiting the grader | 0, flagged |
| `SandboxUnavailable`, `ModelEndpointFailed`, `GenerationAborted`, `VerifierCrashCleanRun`, `Aborted`, `ServerUnreachable`, `EnvironmentTimeout` | failures the policy cannot cause | discarded (`InfraAbort`) |

A reply cut at `max_tokens` ends the episode (the session server will not extend a truncated turn), and the server ends an episode whose next request could pass `--max-seq-len` (it counts the tokens the session server reported and estimates new tool output at 3 characters per token), so the reward is the reward of the tokens Miles trains on. On a Miles without `InfraAbort` a discarded episode returns `reward: None`, and the missing-reward filter drops its group.

## Metrics

`benchflow_rollout.RolloutFn` logs, per rollout step, over the samples that entered training: `benchflow/exit_status/<status>` (shares), `benchflow/clipped_reply_ratio` (replies cut at `max_tokens`), `benchflow/context_exhausted_ratio`, `benchflow/flagged`, and means of turns, tool calls, tokens, context length, sandbox start, verifier and model time.

## Validation

Checked without a GPU, on Daytona (BenchFlow's `docs/examples/rl/miles/smoke.py`, a stand-in session server that refuses any request not extending the stored history exactly): episodes through `benchflow_agent_function.run` score 1 with the right answer and 0 without; a 502 from the session server after the policy acted raises `InfraAbort("ModelEndpointFailed")`; the `abort` hook discards an episode stuck in a command within a second and releases its sandbox, and so does cancelling the agent function's task mid-episode; the policy runs as uid 1000 and cannot read the Daytona key; with BenchFlow's integrity audit (`serve --integrity audit`), an episode that reads an answer its task leaks and submits it scores 0 and is flagged (`IntegrityViolation`) although the verifier passed it. The offline tests are in `tests/fast/examples/experimental/benchflow`.

Not yet run: `run.py` on a GPU, so the Miles side (session server, GRPO) of this example is unvalidated end to end.
