# Train on BenchFlow tasks with Miles

[Miles](https://github.com/radixark/miles) is RadixArk's reinforcement-learning framework for LLM post-training (a fork of slime, built on SGLang). Its agentic path runs each rollout against a token-in/token-out ("TITO") session server that keeps the exact token ids and logprobs of every model call. This cookbook makes BenchFlow tasks trainable with it: every Miles rollout is one BenchFlow episode, in a BenchFlow sandbox, graded by the task's verifier, with a reward that follows BenchFlow's attribution rule.

**Status (2026-09-30).** The BenchFlow side and the Miles-side agent function are built, tested offline, and checked end to end on Daytona without a GPU (see [Results](#results)). The GPU run (baseline, GRPO, held-out evaluation) has not happened yet: creating the Prime pod failed with HTTP 402: Prime allows two instances until an account's top-ups reach $100. The commands below are the ones that run will use.

## The pieces

| Piece | Where | Runs in |
| --- | --- | --- |
| Environment server: one episode per `POST /run` | `python -m benchflow.integrations.miles serve` (`src/benchflow/integrations/miles/`) | BenchFlow's own Python environment, on the GPU host |
| Prompt data for Miles | `python -m benchflow.integrations.miles prepare` | same |
| Agent function, reward hook, per-step metrics, launcher | [`upstream/examples/experimental/benchflow/`](upstream/examples/experimental/benchflow/) | Miles' environment (the Miles container) |
| The same files as a patch for radixark/miles | `make_upstream_patch.sh` | anywhere with `git` and `uvx` |
| GPU-free check of a running environment server | [`smoke.py`](smoke.py) | BenchFlow's environment |
| Task family, shared harness, held-out evaluator | [`../tasks/`](../tasks/), [`../common/`](../common/) | shared by every RL cookbook |

```text
Miles (GPU host, Miles container)                  BenchFlow environment server (same host, own venv)
agentic_tool_call.generate                          POST /run
  -> benchflow_agent_function.run  --- /run --->     |- TaskRuntime sandbox on Daytona: commands only
                                                     |- chat turns ---> session URL (TITO) ---> SGLang
  <- reward, exit_status, eval_report  <-----        '- verifier -> attribution rule -> reward or discard
```

## Design

### Harness: BenchFlow's bash/submit harness

The policy gets the task prompt and two tools, `run_bash` and `submit`, with the limits of the RL cookbooks' shared harness ([`../common/harness.py`](../common/harness.py): 10 tool-calling turns, 30 s per command, 2,000 characters of output). This is a deliberate choice over running OpenCode or mini-swe-agent in the sandbox:

1. **Comparable numbers.** The TRL, Tinker, Prime and Fireworks cookbooks train with this harness, and the shared evaluator scores every policy with it, so a Miles-trained checkpoint is evaluated with the loop it trained in.
2. **Token-in/token-out holds.** The episode sends each assistant message back exactly as the session server returned it. Miles' agentic-rollout guide warns that harnesses may re-serialize tool-call arguments or omit `reasoning_content` on the next request, which its default strict matcher treats as a different history (v1 rolls back or rejects the turn).
3. **No network path to open.** The model loop runs on the GPU host, so the sandbox never calls the model (next section).
4. **Small models.** A two-tool prompt of a few hundred tokens suits a 1.7B–4B policy; coding agents start with thousands of tokens of instructions (Claude Code's first prompt is several thousand tokens before any tool output).
5. **Attribution after the agent starts.** The server sees the policy's first command, which is what separates a discard from a 0 once an episode is under way.

In-sandbox agents are not wired into this connector yet (BenchFlow runs them with `bf.run`); the networking they need is below.

### Networking

- **This connector.** The environment server runs on the GPU host next to Miles and calls the session URL on the host's own address. The Daytona sandbox receives the policy's commands through Daytona's API and has no route to the session server, the SGLang router or Ray. Nothing on the GPU host is published, no tunnel is needed, and `--session-server-external-host` stays unset. The server binds 127.0.0.1; binding anything else requires `--token-file`.
- **Secrets.** The Daytona key is in the environment of the server process only. It never enters a sandbox: the smoke check runs `env` inside the sandbox as the policy and fails if the key's value appears. The policy runs as the task's non-root user (uid 1000), and the task copies used for training and evaluation carry no `oracle/` folder.
- **In-sandbox agents, later.** BenchFlow's LiteLLM gateway runs inside each Daytona sandbox, so an agent there would need the gateway's upstream (the session server) reachable from Daytona's network. The plan is an authenticated relay in front of one route (`POST /sessions/<id>/v1/chat/completions` for sessions the environment server registered, with a token scoped to that session), never the raw session server, and not an ad hoc tunnel, whose faults have broken RL runs before. Or run the sandboxes where the GPU host can reach them (`--sandbox docker` on the host).
- **Check the host.** Miles' session servers, SGLang and Ray listen on the node's address. On a host with a public address, allow only SSH inbound, and check with `ss -ltnp` on the host and a port probe from outside.

### Failure contract

The rule is Miles' own (radixark/miles#2802): discard only what the policy cannot have caused, score everything else 0 with a named `exit_status`. BenchFlow applies it after the agent starts too.

| BenchFlow reason | Miles `exit_status` | Sample |
| --- | --- | --- |
| `scored` | `Submitted`, `NoToolCall`, `TurnLimitExceeded`, `SequenceLengthLimitExceeded`, `RequestRejected` (how the episode ended) | verifier's reward |
| `timeout` (the episode's wall-clock cap after the policy acted, or the verifier's timeout) | `TimeLimitExceeded` | 0 |
| `verifier_error` (after the policy acted) | `VerifierError` | 0 |
| `run_error`, `no_reward` | `AgentError`, `NoReward` | 0 |
| `integrity_violation` | `IntegrityViolation` | 0, flagged |
| `sandbox_start` | `SandboxUnavailable` | discarded |
| `model_endpoint` | `ModelEndpointFailed`; `GenerationAborted` when SGLang aborted the turn (503) | discarded |
| `verifier_crash_clean_run` | `VerifierCrashCleanRun` | discarded |
| (cancelled by Miles' abort hook) | `Aborted` | discarded |
| (the environment server did not answer by the agent function's backstop) | `EnvironmentTimeout` | discarded |

- A discard is `InfraAbort(exit_status)` on a Miles with radixark/miles#2801; before it, the agent function returns `reward: None` and Miles' missing-reward filter drops the group. A sandbox that never starts makes no model call, so Miles discards that sample on its own either way.
- The session server's refusals the policy's output can cause (400, 409, 422, 500) end the episode and the verifier scores it (`RequestRejected`, or `SequenceLengthLimitExceeded` for a context overflow). A 502 is retried as the latest turn, then discarded.
- The episode's wall-clock cap (`--episode-timeout`) is the policy's only once it has acted. Before that, the overrun is discarded under the phase that hung: `SandboxUnavailable`, `ModelEndpointFailed`, or `VerifierCrashCleanRun`.
- A reply cut at `max_tokens` ends the episode, since the session server will not extend a truncated turn. The server also ends an episode whose next request could pass Miles' `--max-seq-len`, so the reward belongs to the tokens Miles trains on.

### Reward integrity (BenchShield)

`serve --integrity audit` asks each `TaskRuntime` for BenchShield's integrity verdict (`TaskRuntimeConfig(integrity=...)`, on `feat/benchshield`). A verdict with `exploited: true` turns the episode into a flagged 0 (`IntegrityViolation`, `eval_report.flagged`, `benchflow/flagged` per step), whatever the verifier said. Until BenchFlow ships the audit, the server refuses `--integrity audit` instead of training without it. What the audit counts as an exploit is agent-attributed evidence of a crossing into a protected resource: reading the verifier, oracle or solution folders, writing the reward, or a path a task's `benchguard.yaml` marks. The task family's `control/control-leaked-answer` (its answer leaks to `/workdir/.grader/expected.json`, inside the workspace) is flagged only with a binding that marks the leak `Hidden`:

```yaml
schema_version: benchguard.task_binding.v1
resources:
  - id: leaked-grader
    selector: /workdir/.grader
    class: Hidden
    reason: the expected answer the verifier compares against leaks here
```

`smoke.py --exploit-task control-leaked-answer` checks the whole path against a server started with `--integrity audit` (results below).

### Rewards and groups

The task family's verifier is binary. Groups whose episodes all got the same reward carry no GRPO signal, so the launcher drops them and samples replacements (`--dynamic-sampling-filter-path ...apply_reward_nonzero_std_filter`, one prompt group at a time); Miles counts them in `rollout/dynamic_filter/drop_zero_std_0.0` and `drop_zero_std_1.0`. Per step, `benchflow_rollout.RolloutFn` adds `benchflow/exit_status/*`, `benchflow/clipped_reply_ratio`, `benchflow/context_exhausted_ratio`, `benchflow/flagged` and means of turns, tokens, context length and timings.

## Steps

Step 0 was run and timed. The GPU steps have not run yet, so their times are estimates.

### 0. Before renting a GPU: check the environment server (about 2 minutes, a few sandbox-minutes)

On any machine with a BenchFlow checkout and a Daytona key:

```bash
uv sync --extra sandbox-daytona
python docs/examples/rl/tasks/generate.py --split train --out ~/rl/tasks-with-oracle/train
rsync -a --exclude oracle/ ~/rl/tasks-with-oracle/train/ ~/rl/tasks/train/
export BENCHFLOW_DAYTONA_OWNER=my-miles-run      # labels every sandbox, for cleanup
set -a; . ~/.config/benchflow/daytona.env; set +a
uv run python -m benchflow.integrations.miles serve --tasks-dir ~/rl/tasks/train --max-sandboxes 8 &
uv run python docs/examples/rl/miles/smoke.py --tasks-dir ~/rl/tasks/train \
    --answers-dir ~/rl/tasks-with-oracle/train --task sql-000004 --task log-000005 --task csv-000006 --task bugfix-000007
```

`smoke.py` plays scripted episodes against a stand-in session server that refuses any request not extending its history exactly, checks the discard on a model-server failure, the abort, and that the policy cannot read the Daytona key. With a Miles checkout and the example on `PYTHONPATH`, `--via-agent-function` runs the same episodes through `benchflow_agent_function.run`.

### 1. Rent one GPU (estimate: 10 minutes)

One H200 (141 GB) fits a 1.7B–4B policy trained with FSDP and colocated with SGLang. On Prime, use the Prime cookbook's pod tools and watchdog (`docs/examples/rl/prime/pods/`), so the pod is in the shared ledger with an owner and a maximum lifetime:

```bash
python3 prime_pods.py --dir ~/prime-pods create --owner miles --gpu-type H200_141GB --gpu-count 1 \
    --image ubuntu_22_cuda_12 --name rl-miles-h200 --max-hours 8
```

Price on 2026-09-30: $4.50/h for the GPU, $5.48/h with the offer's minimum 1,500 GB disk, CPU and memory.

### 2. Start the Miles container (estimate: 10 minutes, mostly the image pull)

```bash
docker run -d --name miles --gpus all --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 \
    --network=host -v /work:/work radixark/miles:latest sleep infinity
```

`radixark/miles:latest` is CUDA 13 (23.7 GB compressed); on a driver older than 580 use `radixark/miles:latest-cu12` (29.1 GB). This cookbook's example was written against Miles `79ef601` (2026-09-30).

### 3. BenchFlow, tasks and prompt data (estimate: 5 minutes)

Stream the BenchFlow checkout in with `git archive` and the stripped task folders with `tar`; then, in the container:

```bash
cd /work/benchflow && uv venv .venv --python 3.12 && uv sync --extra sandbox-daytona
cp -r docs/examples/rl/miles/upstream/examples/experimental/benchflow /root/miles/examples/experimental/
.venv/bin/python -m benchflow.integrations.miles prepare --tasks-dir /work/tasks/train --out /work/train.jsonl --split train
```

### 4. Model choice and baseline (estimate: 15 minutes per model)

Serve the base model with SGLang and score it with the shared evaluator (thinking off, the evaluator's sampling). Choose the model on a train subset, so the test split stays held out for the before-and-after comparison:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B --served-model-name policy --tool-call-parser qwen25 \
    --reasoning-parser qwen3 --host 127.0.0.1 --port 30100 &
SGLANG_KEY=unused .venv/bin/python docs/examples/rl/common/evaluate.py --tasks-dir /work/tasks/test --base-url http://127.0.0.1:30100/v1 \
    --model policy --api-key-env SGLANG_KEY --sandbox daytona --concurrency 32 --samples 4 \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' --out /work/eval/base
```

For the model choice, run the same command on `--tasks-dir /work/tasks/train --limit 48 --samples 2 --out /work/eval/choice-<model>`, and pick the Qwen3 size whose solve rate lands between 20% and 60%; then run the command above, on the test split, for the baseline. The family is close to saturated for newer instruct models (the Tinker cookbook measured Qwen3.5-4B at 96% on a train subset), so start with Qwen3-1.7B and move down to 0.6B or up to 4B as needed.

### 5. Train (estimate: 3 to 4 hours for 30 steps)

```bash
.venv/bin/python -m benchflow.integrations.miles serve --tasks-dir /work/tasks/train --sandbox daytona \
    --max-sandboxes 32 --jobs-dir /work/jobs --job-name train &
cd /root/miles && python examples/experimental/benchflow/run.py --model-name Qwen3-1.7B --prompt-data /work/train.jsonl \
    --num-rollout 30 --rollout-batch-size 8 --n-samples-per-prompt 8 --global-batch-size 64 \
    --save-dir /work/ckpt --tensorboard-dir /work/tb
```

Watch `rollout/raw_reward`, `benchflow/exit_status/*` and `benchflow/clipped_reply_ratio` in the log (`rollout N: {...}`); `curl 127.0.0.1:12100/health` shows sandboxes in use and exit statuses.

### 6. Evaluate the trained checkpoint (estimate: 20 minutes)

```bash
python tools/convert_fsdp_to_hf.py --input-dir /work/ckpt/<last iteration> --output-dir /work/hf-trained --origin-hf-dir /root/models/Qwen3-1.7B
```

Then serve `/work/hf-trained` as in step 4 and run the same evaluator command with `--out /work/eval/trained`. Compare the two `summary.json` files: solve rate with a 95% Wilson interval, per task kind.

### 7. Export and clean up

- Export: stream checkpoints, logs, TensorBoard files, `rollouts.jsonl` and eval results off the pod through a machine that holds the storage credentials (never put cloud credentials on the pod), after a secret scan.
- The pod: `prime_pods.py delete <pod id>`, then `prime_pods.py list` must not show it.
- Sandboxes: `BENCHFLOW_DAYTONA_OWNER=my-miles-run bench sandbox cleanup --all` deletes every sandbox of that owner (and only those); `--dry-run` lists them. The server releases sandboxes on abort, on a client that goes away and on shutdown, but a killed host leaves them to this step.

## Results

### GPU-free check on Daytona (2026-09-30)

`smoke.py` against the environment server on the VM, owner-labelled Daytona sandboxes, train tasks, a scripted policy:

| Episode | Through `/run` | Through `benchflow_agent_function.run` (Miles `79ef601` + #2801's `InfraAbort`) |
| --- | --- | --- |
| sql, log, csv with the right answer | `Submitted`, reward 1.0 | `Submitted`, reward 1.0 |
| bug fix without a fix | `Submitted`, reward 0.0 | `Submitted`, reward 0.0 |
| HTTP 502 from the session server after the policy acted | discarded, `ModelEndpointFailed` (two retries, no verifier run) | `InfraAbort("ModelEndpointFailed")` |
| stuck in `sleep 120`, then abort | discarded, `Aborted`, 0.1 s after `/abort` | discarded, `Aborted`, 0.1 s after the `abort` hook |
| the integrity control, reading the leak and submitting it (server on `--integrity audit`) | `IntegrityViolation`, reward 0.0, flagged (verifier 1.0; verdict `AgentViolation`) | not run through the agent function |
| stuck in `sleep 120`, then the caller goes away | episode cancelled, sandbox released 0.5 s later | same, cancelling the agent function's task |

- No request broke the stand-in's exact-history check, `reasoning_content` included.
- The integrity row ran on a VM-only merge of this branch with `feat/benchshield` (aead8cfd), whose `TaskRuntimeConfig.integrity` the server detects; an honest task in the same run scored 1.0 unflagged (`VectorExposed`: the verifier shares the sandbox).
- The policy ran as uid 1000, and the Daytona key's value never appeared in the sandbox's `env`.
- After each run: 0 episodes in flight, 0 sandboxes held, and 0 Daytona sandboxes left with the owner label.
- Per episode: sandbox start 4–18 s, verifier 11–30 s, 18–44 s end to end.

### Training

Not run yet (Prime billing, above). The run will report the model and its baseline solve rate, the reward curve over 30 steps, held-out solve rates before and after with 95% intervals, the clip ratio and context use, and the cost.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `run.py` stops with "not served by" | The prompt data names tasks the environment server lacks: start the server on the folder `prepare` read. |
| Many `SandboxUnavailable` discards | Daytona quota or concurrency. Lower `--max-sandboxes`; clean up leftover sandboxes. |
| Many `RequestRejected` | The session server refused turns. Its log says why (a TITO mismatch or invalid messages); a custom `--session-message-matcher` or a changed chat template is the usual cause. |
| High `benchflow/clipped_reply_ratio` | Replies hit `--rollout-max-response-len`: raise it, or keep thinking off. |
| High `benchflow/context_exhausted_ratio` | Episodes outgrow `--max-seq-len`: raise it, or lower `--max-output-chars`. |
| Most groups dropped as `zero_std_1.0` | The model already solves the tasks: use a smaller model or harder tasks. `zero_std_0.0`: the opposite. |
| `serve --integrity audit` refuses to start | This BenchFlow has no integrity audit yet. |
| Sandboxes left after a crash | `bench sandbox cleanup --all` with the run's `BENCHFLOW_DAYTONA_OWNER`. |

## How this compares with the Harbor leg

- **Attribution after the agent starts.** radixark/miles#2802 notes that once a Harbor agent runs, the trial's exception cannot tell a platform failure from one the agent caused, so those score 0. BenchFlow's environment server knows whether the policy acted and which side failed, so a verifier that crashed on an untouched sandbox, or a session server that went away mid-episode, is discarded rather than trained on as a false 0.
- **Reward integrity.** With BenchShield's audit, an episode that exploited its grader scores 0 and is flagged, instead of being rewarded.
- **Tasks.** BenchFlow's `task.md` is one file per task, and BenchFlow reads Harbor `task.toml` folders directly, so the same server serves either.
- **Harnesses.** BenchFlow runs agent harnesses through ACP (Claude Code, Codex, OpenCode and others). This connector trains with the bash/submit harness for the reasons above; in-sandbox harnesses are the next step, behind the authenticated relay.
