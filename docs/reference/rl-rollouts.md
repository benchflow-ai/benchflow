# Rollouts for a trainer: `bf.rollout_group`

A trainer that serves its own policy (vLLM, SGLang, a TRL `vllm-serve`) asks BenchFlow for N rollouts of a task, a GRPO group, and gets typed results back as each one finishes: the reward, whether to score or mask it, the policy version, the verifier's per-test results, and exact token segments with action masks. BenchFlow runs each rollout as an ordinary rollout (sandbox, agent, verifier, files under `jobs_dir`); it does not train, sync weights or host the policy.

```python
import asyncio
import benchflow as bf

async def main():
    policy = bf.Policy(
        "vllm/my-policy",                     # route and served model name
        base_url="http://127.0.0.1:8000/v1",  # your server; loopback is fine
        api_key_env="POLICY_KEY",             # read here, never sent into a sandbox
        version=0,
    )
    config = bf.RolloutConfig(task_path="tasks/sql-000004", agent="opencode",
                              environment="docker", jobs_dir="jobs/rl")
    async with policy:
        for step in range(100):
            group = bf.rollout_group(config, n=8, policy=policy, group_id=f"step{step}")
            async for rollout in group:           # as each finishes
                print(rollout.index, rollout.reward, rollout.attribution)
            result = await group.wait()           # advantages are set now
            batch = [(seg, r.advantage) for r in result.trainable() for seg in r.segments]
            ...                                   # your update on seg.prompt_ids,
                                                  # seg.completion_ids, seg.action_mask,
                                                  # seg.logprobs
            policy.version = step + 1             # after the weights reach the server

asyncio.run(main())
```

Several groups can run at once (one task each), and groups for the next step can start while the trainer updates: every result records the policy version that served each of its calls. `Policy(max_concurrency=N)` caps the rollouts running against the server across all groups; `rollout_group(concurrency=)` caps one group.

## `bf.Policy`

| Field | Meaning |
| --- | --- |
| `model` | Route and served model name: `vllm/<name>` or `sglang/<name>` (the gateway asks these servers for token ids); other OpenAI-compatible routes work but give logprobs only unless `BENCHFLOW_CAPTURE_TOKEN_IDS=1`. |
| `base_url`, `api_key` / `api_key_env` | The server's `/v1` URL as this machine sees it, and its key. |
| `version` | The current policy version; set it after each weight update. |
| `max_concurrency` | Cap on rollouts against this policy at once, across groups. |
| `relay_bind`, `relay_public_url`, `relay_ssl` | Only for sandboxes on another machine (Daytona): where the relay listens and the address those sandboxes use (see below). |
| `capture_routed_experts` | Ask SGLang for MoE routing (`return_routed_experts`); vLLM returns it when the server runs with `--enable-return-routed-experts`. |

Rollouts never see the server's address or key. Each rollout's model gateway talks to BenchFlow's *policy relay*, a reverse proxy in the trainer's process that accepts only `POST /v1/chat/completions`, `POST /v1/completions` and `GET /v1/models`, each with a credential valid for that rollout only, and adds the server's key itself. The relay listens on loopback. When a gateway runs inside a Docker sandbox (tasks with `network_mode: no-network`), the relay also listens on the Docker host address. On Daytona the gateway always runs inside the sandbox, so the relay must be reachable from Daytona: `relay_bind="0.0.0.0:8443"` and `relay_public_url="https://relay.example.com"` behind TLS (or `relay_ssl=`). Without them, a Daytona group refuses to start.

## `bf.rollout_group(config, *, n, policy, ...)`

| Argument | Default | Meaning |
| --- | --- | --- |
| `config` | | A `bf.RolloutConfig` (task, agent, sandbox, timeouts, `jobs_dir`); leave `model` unset. |
| `n` | | Group size. |
| `attempts` | 2 | Tries per member. Only masked (infrastructure) failures are retried. |
| `on_failure` | `"mask"` | A member still masked after its last attempt: `"mask"`, `"zero"` (score 0.0) or `"raise"` (`RolloutGroupError`). |
| `advantage` | `"grpo"` | `(r - mean) / (std + 1e-4)` over the group's scored members; `"loo"`; or None. |
| `drop_zero_variance` | False | A group whose scored rewards are all equal gets no advantages and `dropped == "zero_variance"`. |
| `reward` | `"verifier"` | `"tests"` trains on the share of the verifier's tests that passed (its CTRF report). |
| `trainable_kinds` | agent, subagent, chat | Segment kinds returned in `segments`; the rest go to `excluded_segments`. |
| `integrity` | `"auto"` | Apply a BenchShield verdict the rollout wrote (`integrity/claim_verdict.json`), `"off"`, or a callable returning `{"exploited": bool, "reason": str}`. |
| `startup_timeouts` | defaults | `StartupTimeouts(sandbox_sec=, acp_handshake_sec=, gateway_sec=)` for every member. |
| `sandbox_lifetime` | 60 idle min, delete when stopped | `SandboxLifetime(lease_sec=, auto_stop_min=, auto_delete_min=)`, for Daytona. |

`async for rollout in group` yields each member once, when final. `await group.wait()` returns a `GroupResult` (`rollouts` in member order, `trainable()`, `zero_variance`, `dropped`, counts), and writes it to `<jobs_dir>/<job>/groups/<group_id>.json`. `await group.cancel()`, or leaving `async with group:` early, stops the members still running and deletes their sandboxes.

## `bf.TrainingRollout`

| Field | Meaning |
| --- | --- |
| `reward` | The training reward; None when masked. Partial credit from the verifier passes through. |
| `verifier_reward`, `rewards`, `reward_source` | What the verifier wrote, and where `reward` came from (`verifier`, `tests`, `integrity`, `failure_policy`). |
| `passed`, `outcome` | `outcome` is `passed`, `failed`, `masked` or `cancelled`. |
| `attribution`, `attribution_reason`, `failure` | `score` or `mask`, why, and whether the failure was the policy's or the infrastructure's (below). |
| `error`, `error_category`, `verifier_error`, `verifier_error_category` | As in `result.json`. |
| `policy_version`, `policy_versions` | `policy.version` when the rollout started, and every version that served one of its calls. |
| `tests` | Per-test results: `name`, `status`, `duration_ms`, `message`. |
| `segments`, `excluded_segments` | Exact token segments (below). |
| `tokens` | The account of every call: `status` (`exact`, `partial`, `none`), `dropped` calls with reasons, `failed_attempts`, `attestation`. |
| `startup` | The startup timeouts used, per-phase seconds, and `failed_phase` when startup failed. |
| `attempts` | Earlier attempts of this member that a retry replaced. |
| `flagged`, `integrity` | An integrity audit found the policy exploited the grader: `reward` is 0.0. |
| `advantage` | Set when the group completes. |

### Score or mask

A failure is masked only when the policy cannot have caused it; everything else scores, 0 when the verifier gave nothing:

| Case | Attribution | `attribution_reason` |
| --- | --- | --- |
| The verifier scored the rollout, including after the agent ran out of time | score | `scored` |
| The agent timed out and the verifier gave no reward | score (0) | `timeout` |
| The agent crashed, or the sandbox was lost, after the policy acted | score (0) | `run_error` |
| The verifier failed after the policy acted | score (0) | `verifier_error` |
| The sandbox did not start | mask | `sandbox_start` |
| The agent's install or startup failed (ACP handshake, PTY, gateway start) | mask | `agent_setup` |
| The model endpoint failed (5xx, 429, connection, auth; or its last answer to the rollout failed and the rollout did not pass) | mask | `model_endpoint` |
| The verifier's own infrastructure failed (dependency install, lost verifier sandbox) | mask | `verifier_infra` |
| The verifier crashed on a sandbox the policy never touched | mask | `verifier_crash_clean_run` |
| Any other failure before the policy answered a call | mask | `infrastructure` |
| An integrity audit found an exploit | score (0) | `integrity_violation` |

A crash or lost sandbox after the policy acted scores 0 because the policy can kill its own sandbox; masking it would teach the policy to crash when it is failing.

### Token segments

Each `bf.TokenSegment` is one span of one conversation in which every prompt extends the previous prompt and sampled tokens exactly, so it can be trained on as one sequence:

- `prompt_ids`: the first call's prompt, as the server tokenized it;
- `completion_ids`: everything after it, each call's sampled tokens and the tokens the next prompt adds (tool results, the next turn's template);
- `action_mask`: 1 for a token the policy sampled, 0 for one the environment added;
- `logprobs`: the server's logprob per sampled token, 0.0 where the mask is 0;
- `policy_versions`, aligned with `calls`; `call_spans`, where each call's sampled tokens sit;
- `start_reason`: why a segment after the first began (`rerender` when the next prompt rewrote the sampled turn, `compaction` when it rewrote the history before it, `after_dropped_call`);
- `routing`: MoE routing per call when the server returned it, passed through as returned (`source`, `encoding`, `data`, `start`, `sequence_length`);
- `digest`, and `verify()` to check the lists are the ones BenchFlow built.

Calls that offer tools are grouped by tool set: the first tool set is the agent loop (`agent`), another is a subagent (`subagent`). A tool-less call in a run that uses tools, or a call the gateway labelled `title`, `summary` or `helper`, is a `helper` (Claude Code's side prompts, OpenCode's title generator); a `compaction` call summarises the agent's history. A failed provider call is never in a segment; when the agent sent the same request again, `tokens.calls` links the failure and its retry (`retried_by`, `retry_of`). A call without prompt ids, sampled ids or logprobs, or with a different number of ids and logprobs, is listed in `tokens.dropped` with its reason, never dropped silently.

`tokens.attestation` compares each stored call with the relay's record of the server's raw answer by token digest: `attested` when every stored call matches, `mismatch` when a stored call holds tokens the server never sent (its segment is not trainable), `partial` when the relay served calls the store does not have.

## Sandboxes

- **Kill-safe.** While groups run, SIGTERM, SIGINT and SIGHUP delete every sandbox the process started before the previous handler runs. A process killed outright (SIGKILL, out of memory, a lost machine) leaves a lease file; the next group on the machine, or `bench sandbox cleanup`, deletes what it lists. Daytona sandboxes also carry `benchflow.lease` (the process) and, with `SandboxLifetime(lease_sec=)`, `benchflow.expires`; the Daytona reaper deletes a sandbox whose process is gone or whose lease expired, and a group's Daytona sandboxes stop after 60 idle minutes and are deleted when stopped.
- **No oracle.** A task's `oracle/` (or `solution/`) is uploaded only for the `oracle` agent: path lockdown hides it only from a non-root sandbox user, and Terminal-Bench style tasks need a root agent (`sandbox_user=None`).
- **Startup.** `StartupTimeouts` sets the sandbox, ACP handshake and gateway timeouts per group; a startup failure is masked and retried, and `rollout.startup` names the phase.

## Files

Each attempt is a normal rollout folder: `<jobs_dir>/<job>/<task>__<group_id>-rNN-aK/` with `result.json`, the verifier's output, `trajectory/llm_trajectory.jsonl` (the gateway's store, with a digest per call) and `trajectory/policy_relay.jsonl` (the relay's record: status, policy version, digest per call). `bench train stream` and `bf.stream_rollouts` read the same folders and carry the same `segments` and `tokens` ([rollout stream](rollout-stream.md)).
