# AgentBeats A2A Adapter Audit

Date: 2026-05-21

Branch: `codex/agentbeats-a2a-adapter-audit`

## Decision

BenchFlow should add AgentBeats A2A as a sibling participant adapter, not as an
ACP rename or an ACP transport flag.

The reusable surfaces are `Role`, `Scene`, `Rollout`, sandbox lifecycle, skill
deployment, trajectory/artifact directories, and verifier handoff. The
non-reusable surface is the current participant execution path:
`Rollout.connect_as()` -> `connect_acp()` -> `ACPClient` ->
`execute_prompts()`. That path is ACP-specific from process launch through
session updates.

## Evidence

- `src/benchflow/_types.py` has protocol-neutral `Role`, `Scene`, and `Turn`
  declarations, but `Role.agent` currently names a local registered ACP agent.
- `src/benchflow/rollout.py` owns setup, skill deployment, sandbox lockdown,
  provider credential mapping, trajectory publication, verifier execution, and
  result serialization.
- `Rollout.connect_as()` installs a local agent binary, writes credentials, and
  calls `connect_acp()` for every role turn.
- `src/benchflow/acp/runtime.py` and `src/benchflow/acp/client.py` implement ACP
  JSON-RPC session setup, prompt execution, cancellation, and update capture.
- Multi-role scenes currently exchange messages through `/app/.outbox` prompt
  injection. They do not target endpoint-based participants.
- `Rollout.verify()` already publishes captured trajectory to
  `/logs/agent/acp_trajectory.jsonl` and delegates scoring to the existing
  verifier path.

## Contract And Initial Implementation

The future A2A participant adapter should live under
`src/benchflow/agents/a2a.py` and expose a small contract independent from ACP:

- `A2AParticipantRequest`: endpoint URL, role name, visible prompt, workspace,
  optional skills directory, role timeout, role idle timeout, and redacted
  metadata.
- `A2ATaskHandle`: opaque started-task handle with task id, endpoint URL, and
  role name.
- `A2AParticipantResult`: terminal status, normalized A2A trajectory events,
  artifact references, optional final response, and redacted error type.
- `A2AParticipantAdapter`: `start(request)`, `wait(handle)`, and
  `cancel(handle, reason)`.

The module now also includes `A2AClientParticipantAdapter`, a small
`a2a-sdk==0.3.20` backed implementation that sends one visible prompt to an A2A
endpoint, normalizes task/message events, records artifact references, and
returns a terminal `A2AParticipantResult`. The runtime dependency remains the
client SDK; the `http-server` extra is a dev dependency for the toy endpoint
smoke test.

Rollout wiring now has a first implementation. `Role.transport` defaults to
`"acp"`, and explicit A2A roles use `Role.transport="a2a"` plus
`Role.endpoint_url`. ACP roles still use the existing `connect_acp()` and
`execute_prompts()` path.

## Done Signal

The adapter must not infer completion from free-form text. A purple participant
is done only when the A2A task reaches a terminal state:

- `completed`: final response or artifact is available for verifier handoff.
- `failed`: participant task failed without trustworthy output.
- `cancelled`: BenchFlow or the caller cancelled the participant task.
- `timeout`: BenchFlow timeout handling cancelled or abandoned the task.

Only `completed` proceeds as a normal model attempt. Other terminal states
should still persist trajectory and produce a classified run result.

## Timeout And Cancellation

- Use `Role.timeout_sec` when present, otherwise the task `[agent].timeout_sec`.
- Use `Role.idle_timeout_sec` when present, otherwise rollout idle timeout.
- On timeout, call `A2AParticipantAdapter.cancel()` before verifier handoff.
- Persist timeout/cancel as participant communication failures, not verifier
  failures.
- Preserve ACP timeout behavior; do not route ACP roles through A2A timeout code.

## Trajectory And Artifacts

A2A updates are written separately from ACP trajectory data:

- `trajectory/a2a_trajectory.jsonl` for normalized A2A task updates.
- `artifacts/a2a_artifacts.json` for A2A artifact references.
- A2A final responses may include file artifacts under a `files[]` payload;
  BenchFlow materializes those files under the rollout workspace only, then
  records them as `sandbox://...` refs in `artifacts/a2a_artifacts.json`.
- Existing `trajectory/acp_trajectory.jsonl` remains ACP-only.
- Public rows may reference redacted artifact ids or digests, not private paths,
  raw provider payloads, credentials, or verifier logs.

## Verifier Handoff

The A2A adapter is not a verifier. It only drives the purple participant to a
terminal state and materializes the participant output into the same sandbox or
artifact layout that existing verifiers already consume.

After a `completed` A2A result:

1. Persist normalized A2A trajectory.
2. Persist or reference participant artifacts.
3. Publish any verifier-visible trajectory/artifact data required by the task.
4. Call the existing `_verify_rollout()` path.
5. Serialize result fields without changing ACP result semantics.

AgentBeats/Amber worker runs add one Docker-sandbox caveat: the task container
can run through Amber's Docker gateway while its bind-mounted verifier log path
is not visible from inside the worker container. Set
`BENCHFLOW_DOCKER_LOGS_HOST_MOUNTED=false` in that worker environment so
BenchFlow downloads `/logs/verifier` from the task container before parsing
`reward.txt`. The default remains `true` for normal local Docker runs where the
host mount is visible.

## Runtime File Changes

Phase 4 implementation has stayed scoped to these files:

- `src/benchflow/agents/a2a.py`: concrete AgentBeats A2A participant
  client/adapter.
- `src/benchflow/_types.py`: explicit participant transport discriminator and
  endpoint field on `Role`, with ACP as the default.
- `src/benchflow/_utils/yaml_loader.py`: YAML role parsing for transport,
  endpoint, role timeouts, role skills, and capabilities.
- `src/benchflow/rollout.py`: branch role execution to ACP or A2A while keeping
  setup, verifier, cleanup, and result serialization shared.
- `src/benchflow/sandbox/docker.py`: allow worker environments to disable the
  verifier-log host-mount fast path and avoid `--rmi all` cleanup for prebuilt
  task images.
- `src/benchflow/models.py`: add `a2a` as a trajectory source.
- `tests/test_a2a_participant_adapter_contract.py`: convert pending runtime
  tests into passing implementation tests.
- `tests/test_docker_uploads.py`: cover the Docker log-mount flag and prebuilt
  cleanup behavior.

## Runtime Tests

`tests/test_a2a_participant_adapter_contract.py` now contains:

- passing contract-shape tests for request/result/artifact fields
- passing client tests for message normalization, cancellation, unknown
  handles, and a toy SDK-backed A2A endpoint smoke
- passing runtime tests for endpoint invocation, timeout cancellation, artifact
  capture, file materialization, verifier handoff, and A2A trajectory
  persistence

These tests deliberately keep ACP behavior untouched while documenting the
implementation surface.

## Verification

Baseline before edits:

```bash
uv sync --extra dev
uv run pytest tests/test_acp.py tests/test_agent_registry.py tests/test_scene.py tests/test_runtime.py -q
```

Result: `69 passed, 3 skipped`.

Post-edit focused verification:

```bash
uv run pytest tests/test_a2a_participant_adapter_contract.py tests/test_acp.py tests/test_agent_registry.py tests/test_scene.py tests/test_runtime.py -q
uv run ruff check src/benchflow/agents/a2a.py tests/test_a2a_participant_adapter_contract.py
uv run ruff format --check src/benchflow/agents/a2a.py tests/test_a2a_participant_adapter_contract.py
```

Result: `71 passed, 3 skipped, 5 xfailed`; ruff check passed; ruff format
reported `2 files already formatted`.

Adapter implementation verification:

```bash
uv run pytest tests/test_a2a_participant_client.py tests/test_a2a_participant_adapter_contract.py tests/test_acp.py tests/test_agent_registry.py tests/test_scene.py tests/test_runtime.py -q
uv run ruff check src/benchflow/agents/a2a.py tests/test_a2a_participant_client.py tests/test_a2a_participant_adapter_contract.py
uv run ruff format --check src/benchflow/agents/a2a.py tests/test_a2a_participant_client.py tests/test_a2a_participant_adapter_contract.py
uv run ty check src/benchflow/agents/a2a.py
```

Result: `74 passed, 3 skipped, 5 xfailed`; ruff check passed; ruff format
reported `3 files already formatted`; `ty` passed.

Rollout wiring verification:

```bash
uv run pytest tests/test_a2a_participant_client.py tests/test_a2a_participant_adapter_contract.py tests/test_scene_outbox_trial.py tests/test_acp.py tests/test_agent_registry.py tests/test_scene.py tests/test_runtime.py -q
uv run pytest tests/test_yaml_config.py tests/test_adapters.py tests/test_connect_as_env.py tests/test_internet_policy.py -q
uv run ruff check src/benchflow/agents/a2a.py src/benchflow/_types.py src/benchflow/_utils/yaml_loader.py src/benchflow/rollout.py src/benchflow/models.py tests/test_a2a_participant_client.py tests/test_a2a_participant_adapter_contract.py
uv run ruff format --check src/benchflow/agents/a2a.py src/benchflow/_types.py src/benchflow/_utils/yaml_loader.py src/benchflow/rollout.py src/benchflow/models.py tests/test_a2a_participant_client.py tests/test_a2a_participant_adapter_contract.py
uv run ty check src/benchflow/agents/a2a.py src/benchflow/_types.py src/benchflow/_utils/yaml_loader.py src/benchflow/rollout.py src/benchflow/models.py
```

Results: `95 passed, 3 skipped, 1 warning`; `59 passed, 6 warnings`; ruff
check passed; ruff format reported `7 files already formatted`; `ty` passed.

After adding file materialization:

```bash
uv run pytest tests/test_a2a_participant_client.py tests/test_a2a_participant_adapter_contract.py tests/test_scene_outbox_trial.py tests/test_acp.py tests/test_agent_registry.py tests/test_scene.py tests/test_runtime.py -q
uv run pytest tests/test_yaml_config.py tests/test_adapters.py tests/test_connect_as_env.py tests/test_internet_policy.py -q
uv run ruff check src/benchflow/agents/a2a.py src/benchflow/_types.py src/benchflow/_utils/yaml_loader.py src/benchflow/rollout.py src/benchflow/models.py tests/test_a2a_participant_client.py tests/test_a2a_participant_adapter_contract.py
uv run ruff format --check src/benchflow/agents/a2a.py src/benchflow/_types.py src/benchflow/_utils/yaml_loader.py src/benchflow/rollout.py src/benchflow/models.py tests/test_a2a_participant_client.py tests/test_a2a_participant_adapter_contract.py
uv run ty check src/benchflow/agents/a2a.py src/benchflow/_types.py src/benchflow/_utils/yaml_loader.py src/benchflow/rollout.py src/benchflow/models.py
```

Results: `96 passed, 3 skipped, 1 warning`; `59 passed, 6 warnings`; ruff
check passed; ruff format reported `7 files already formatted`; `ty` passed.

Docker sandbox worker-gateway verification:

```bash
uv run pytest tests/test_docker_uploads.py tests/test_verifier_multi_container.py -q
```

Result: `19 passed, 20 warnings`.

Real SkillsBench smoke:

- Task:
  `/Users/liu.10379/Documents/work/skillsbench/tasks/perf-cycle-optimization`.
- Participant:
  toy SDK-backed A2A endpoint through `A2AClientParticipantAdapter`.
- Result:
  success `true`, reward `0.13043478260869565`, trajectory source `a2a`,
  verifier error `null`.
- Important environment note:
  the smoke failed when `jobs_dir` was under `/tmp` because Docker did not
  surface the verifier reward file through the bind mount. The same smoke
  passed when `jobs_dir` was under the `/Users/...` worktree path.

Remaining implementation proof:

- deployed or public-packaged green-agent worker integration using this A2A role
  path
