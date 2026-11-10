# Python API

Each public name in the `benchflow` package (`benchflow.__all__`) has its signature or fields, the methods BenchFlow defines on it, and the first paragraph of its docstring. A name, field, argument or method new since 0.9.2 has `(0.10)` after it when 0.10.0rc3 has it, or `(unreleased)` when no release has it yet.

<!-- This page is generated from the code by tools/docs_reference.py. To change it, change the docstring or signature in the code, then run `uv run python tools/docs_reference.py --write`. -->

Import the package as `import benchflow as bf`. [Run a benchmark](../evaluate/run-a-benchmark.md) and [Read the results](../evaluate/read-results.md) have worked examples, and [CLI and Python](./cli-python.md) gives the Python argument of each `bench` flag.

## Functions

### `abranch`

```python
bf.abranch(task_path: 'str | Path | None' = None, *, agent: 'str | None' = None, children: 'ChildrenArg', model: 'str | None' = None, prompts: 'Iterable[str] | None' = None, checkpoint_after: 'int | None' = None, sandbox: 'str | None' = None, parent: "Literal['continue', 'discard'] | None" = None, snapshot_layers: 'Iterable[str]' = ('sandbox',), concurrency: 'int' = 1, isolate_children: 'bool' = False, resume_session: 'bool' = False, child_retries: 'int' = 1, continue_after_child_failure: 'bool' = True, retain_snapshots: 'bool' = False, from_checkpoint: 'str | Path | None' = None, fork: 'str | None' = None, checkpoint: 'str | None' = None, checkpoints: 'CheckpointPolicy | None' = None, jobs_dir: 'str | Path' = 'jobs', job_name: 'str | None' = None, agent_env: 'Mapping[str, str] | None' = None, reasoning_effort: 'str | None' = None, harness: 'str' = 'acp', on_event: 'Callable[[dict[str, Any]], None] | None' = None) -> 'BranchResult'
```

A coroutine function: `await` it, or run it with `asyncio.run(...)`.

Branch one task's run at a checkpoint into labelled, verifier-scored children.

### `aregrade`

```python
bf.aregrade(path: 'str | Path', *, tasks_dir: 'str | Path | None' = None, sandbox: 'str | None' = None, concurrency: 'int' = 4, reason: 'str | None' = None, runner: 'Runner | None' = None, review: 'bool' = False, reviewer: 'ReviewerConfig | None' = None) -> 'RegradeSummary'
```

A coroutine function: `await` it, or run it with `asyncio.run(...)`.

New arguments in 0.10: `review`, `reviewer`.

Async `regrade`.

### `arun`

```python
bf.arun(subject: 'Agent | TrialConfig | str', env: 'Environment | str | None' = None, config: 'RuntimeConfig | None' = None, *, task_path: 'str | Path | None' = None, model: 'str | None' = None) -> 'RolloutResult'
```

A coroutine function: `await` it, or run it with `asyncio.run(...)`.

Primary user-facing API; multiple calling conventions.

### `arun_batch`

```python
bf.arun_batch(configs: 'Iterable[RolloutConfig]', *, concurrency: 'int' = 4, on_result: 'Callable[[Completed], None] | None' = None) -> 'Results'
```

A coroutine function: `await` it, or run it with `asyncio.run(...)`.

Run rollouts with bounded concurrency; return their results in input order.

### `as_completed`

```python
bf.as_completed(configs: 'Iterable[RolloutConfig]', *, concurrency: 'int' = 4) -> 'AsyncIterator[Completed]'
```

Run rollouts with at most ``concurrency`` at a time, yielding each as it finishes.

### `astream_rollouts`

```python
bf.astream_rollouts(job_dir: 'str | Path', *, follow: 'bool' = True, poll_interval: 'float' = 1.0, timeout: 'float | None' = None, group_by: 'str | Sequence[str] | None' = None, group_size: 'int | None' = None, on_warning: 'Callable[[str], None] | None' = None, on_withheld: 'Callable[[list[StreamedRollout]], None] | None' = None) -> 'AsyncIterator[StreamedRollout]'
```

New arguments in 0.10: `on_withheld`.

Async form of `stream_rollouts`; folder scans run off the event loop.

### `branch`

```python
bf.branch(task_path: 'str | Path | None' = None, *, agent: 'str | None' = None, children: 'ChildrenArg', model: 'str | None' = None, prompts: 'Iterable[str] | None' = None, checkpoint_after: 'int | None' = None, sandbox: 'str | None' = None, parent: "Literal['continue', 'discard'] | None" = None, snapshot_layers: 'Iterable[str]' = ('sandbox',), concurrency: 'int' = 1, isolate_children: 'bool' = False, resume_session: 'bool' = False, child_retries: 'int' = 1, continue_after_child_failure: 'bool' = True, retain_snapshots: 'bool' = False, from_checkpoint: 'str | Path | None' = None, fork: 'str | None' = None, checkpoint: 'str | None' = None, checkpoints: 'CheckpointPolicy | None' = None, jobs_dir: 'str | Path' = 'jobs', job_name: 'str | None' = None, agent_env: 'Mapping[str, str] | None' = None, reasoning_effort: 'str | None' = None, harness: 'str' = 'acp', on_event: 'Callable[[dict[str, Any]], None] | None' = None) -> 'BranchResult'
```

Blocking form of `abranch`; takes exactly the same arguments.

### `build_service_hooks`

```python
bf.build_service_hooks(services: list[benchflow.sandbox.services.ServiceConfig]) -> list[collections.abc.Callable[[typing.Any], collections.abc.Coroutine[typing.Any, typing.Any, None]]]
```

Build pre_agent_hooks that start the given services.

### `collect_metrics`

```python
bf.collect_metrics(results_dir: str | pathlib.Path, benchmark: str = '', agent: str = '', model: str = '') -> benchflow.metrics.BenchmarkMetrics
```

Collect metrics from a results directory.

### `compare`

```python
bf.compare(job_a: 'Job | str | Path', job_b: 'Job | str | Path', *, include_controls: 'bool' = False, labels: 'tuple[str, str] | None' = None, vary: 'Iterable[str]' = (), on_mismatch: "Literal['warn', 'raise', 'ignore']" = 'warn', ks: 'Iterable[int] | None' = None, solve_threshold: 'float | None' = None, by: 'Iterable[str]' = ()) -> 'Comparison'
```

Pair two jobs' trials by task name and summarise the differences.

### `compile_scenes_to_steps`

```python
bf.compile_scenes_to_steps(scenes: 'list[Scene]', *, default_prompt: 'str | None' = None) -> 'list[Step]'
```

Lower declarative scenes into explicit rollout Steps.

### `detect_services_from_dockerfile`

```python
bf.detect_services_from_dockerfile(task_path: pathlib.Path | str) -> list[benchflow.sandbox.services.ServiceConfig]
```

Auto-detect which services a task needs from its Dockerfile.

### `discover_skills`

```python
bf.discover_skills(*search_dirs: pathlib.Path | str) -> list[benchflow.skills.SkillInfo]
```

Discover skills from one or more directories.

### `get_agent`

```python
bf.get_agent(name: str) -> tuple[benchflow.agents.registry.AgentConfig, str]
```

Get agent config by name.

### `infer_env_key_for_model`

```python
bf.infer_env_key_for_model(model: str) -> str | None
```

Infer the required API key environment variable from a model ID.

### `install_skill`

```python
bf.install_skill(spec: str, target_dir: pathlib.Path | None = None) -> pathlib.Path | None
```

Install a skill from skills.sh.

### `is_vertex_model`

```python
bf.is_vertex_model(model: str) -> bool
```

True if the model uses Vertex AI (GCP ADC auth, not API keys).

### `list_agents`

```python
bf.list_agents() -> list[benchflow.agents.registry.AgentConfig]
```

List all registered agents.

### `list_workspace_snapshots`

```python
bf.list_workspace_snapshots(env: Any) -> list[str]
```

A coroutine function: `await` it, or run it with `asyncio.run(...)`.

List available *workspace* snapshot names.

### `load_job`

```python
bf.load_job(path: 'str | Path | Iterable[str | Path]', *, attempts: "Literal['best', 'all']" = 'best') -> 'Job'
```

Read every trial under ``path`` into a `Job`.

### `load_manifest`

```python
bf.load_manifest(path: 'str | Path') -> 'EnvironmentManifest'
```

Load and validate an environment manifest.

### `load_results_jsonl`

```python
bf.load_results_jsonl(path: 'str | Path') -> 'list[Trial]'
```

Read every row of a ``results.jsonl`` file as a `Trial`.

### `load_review_rubric`

```python
bf.load_review_rubric(path: 'Path | None' = None) -> 'Rubric'
```

Load the parsed rubric using the canonical snapshot reader.

### `load_rubric`

```python
bf.load_rubric(path: 'Path') -> 'RubricConfig'
```

Load a rubric from either a ``.toml`` or ``.json`` file.

### `load_rubric_json`

```python
bf.load_rubric_json(path: 'Path') -> 'RubricConfig'
```

Load and parse a ``rubric.json`` file (Harvey LAB style).

### `load_rubric_toml`

```python
bf.load_rubric_toml(path: 'Path') -> 'RubricConfig'
```

Load and parse a ``rubric.toml`` file.

### `load_trial`

```python
bf.load_trial(path: 'str | Path') -> 'Trial'
```

Read one trial directory (or its ``result.json``) into a `Trial`.

### `ors_tool_outputs_to_reward_events`

```python
bf.ors_tool_outputs_to_reward_events(outputs: 'list[dict[str, Any]]', *, source: 'str' = 'ors-tool-output') -> 'list[dict[str, Any]]'
```

Convenience wrapper for ORS tool-output reward event normalization.

### `parse_skill`

```python
bf.parse_skill(skill_md: pathlib.Path) -> benchflow.skills.SkillInfo | None
```

Parse a SKILL.md file and return SkillInfo.

### `register_agent`

```python
bf.register_agent(name: str, install_cmd: str, launch_cmd: str, *, protocol: str = 'acp', session_factory: str = '', requires_env: list[str] | None = None, description: str = '', skill_paths: list[str] | None = None, install_timeout: int = 900, default_model: str = '', model_route: str = 'proxy', api_protocol: str = '', env_mapping: dict[str, str] | None = None, credential_files: list[benchflow.agents.registry.CredentialFile] | None = None, launcher_credential_files: list[str] | None = None, home_dirs: list[str] | None = None, subscription_auth: benchflow.agents.registry.SubscriptionAuth | None = None, acp_model_format: str = 'bare', supports_acp_set_model: bool = True, acp_model_config_id: str = '', acp_effort_config_id: str = '', disallow_web_tools_setup_cmd: str = '', disallow_web_tools_owned_paths: list[str] | None = None, disallow_web_tools_launch_suffix: str = '', disallow_hosted_search_setup_cmd: str = '', disallow_hosted_search_launch_suffix: str = '', cacheable_version: str = '') -> benchflow.agents.registry.AgentConfig
```

New arguments in 0.10: `model_route`.

Register a custom agent at runtime.

### `register_service`

```python
bf.register_service(name: str, cli_name: str, port: int, db_path: str, **kwargs: Any) -> benchflow.sandbox.services.ServiceConfig
```

Register a custom service at runtime.

### `regrade`

```python
bf.regrade(path: 'str | Path', *, tasks_dir: 'str | Path | None' = None, sandbox: 'str | None' = None, concurrency: 'int' = 4, reason: 'str | None' = None, runner: 'Runner | None' = None, review: 'bool' = False, reviewer: 'ReviewerConfig | None' = None) -> 'RegradeSummary'
```

New arguments in 0.10: `review`, `reviewer`.

Re-run each trial's task verifier (from ``tasks_dir`` when given) on the trial's frozen final workspace, in a fresh sandbox.

### `render_task_md_from_legacy`

```python
bf.render_task_md_from_legacy(task_dir: 'str | Path') -> 'str'
```

Render a legacy ``task.toml`` + ``instruction.md`` task as ``task.md``.

### `resolve_source`

```python
bf.resolve_source(repo: str, path: str | None = None, ref: str | None = None) -> pathlib.Path
```

Resolve a dataset source to a local filesystem path.

### `rollout_config_from_yaml`

```python
bf.rollout_config_from_yaml(path: 'str | Path', task_path: 'Path | None' = None) -> 'RolloutConfig'
```

Parse a YAML file into a RolloutConfig.

### `run`

```python
bf.run(subject: 'Agent | TrialConfig | str', env: 'Environment | str | None' = None, config: 'RuntimeConfig | None' = None, *, task_path: 'str | Path | None' = None, model: 'str | None' = None) -> 'RolloutResult'
```

A coroutine function: `await` it, or run it with `asyncio.run(...)`.

Primary user-facing API; multiple calling conventions.

### `run_batch`

```python
bf.run_batch(configs: 'Iterable[RolloutConfig]', *, concurrency: 'int' = 4, on_result: 'Callable[[Completed], None] | None' = None) -> 'Results'
```

Blocking form of `arun_batch`.

### `run_reviews`

```python
bf.run_reviews(path: 'Path', *, agent: 'str', model: 'str | None' = None, environment: 'str' = 'docker', rubric_path: 'Path | None' = None, prompt_path: 'Path | None' = None, agent_env: 'dict[str, str] | None' = None, concurrency: 'int' = 4, timeout_sec: 'int' = 1800, image: 'str' = 'python@sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91', open_network: 'bool' = False, tasks_root: 'Path | None' = None, filter_passing: 'bool | None' = None, out_dir: 'Path | None' = None) -> 'tuple[ReviewReport, Path]'
```

A coroutine function: `await` it, or run it with `asyncio.run(...)`.

Review rollout(s) at ``path`` and return the report plus its location.

### `run_sync`

```python
bf.run_sync(subject: 'Agent | RolloutConfig | str', env: 'Environment | str | None' = None, config: 'RuntimeConfig | None' = None, *, task_path: 'str | Path | None' = None, model: 'str | None' = None) -> 'RolloutResult'
```

Blocking form of `arun`; takes exactly the same arguments.

### `score_weighted_review`

```python
bf.score_weighted_review(rubric: 'Rubric', checks: 'Mapping[str, Mapping[str, Any]]', *, deterministic_pass: 'bool') -> 'ReviewScoring'
```

Aggregate one structurally valid v0.2 review.

### `stage_dockerfile_deps`

```python
bf.stage_dockerfile_deps(task_path: pathlib.Path, context_root: pathlib.Path) -> None
```

Copy Dockerfile COPY sources into environment/_deps/ and rewrite paths.

### `stream_rollouts`

```python
bf.stream_rollouts(job_dir: 'str | Path', *, follow: 'bool' = True, poll_interval: 'float' = 1.0, timeout: 'float | None' = None, group_by: 'str | Sequence[str] | None' = None, group_size: 'int | None' = None, on_warning: 'Callable[[str], None] | None' = None, on_withheld: 'Callable[[list[StreamedRollout]], None] | None' = None) -> 'Iterator[StreamedRollout]'
```

New arguments in 0.10: `on_withheld`.

Yield each finished rollout of a job as it finishes.

### `to_inspect_task`

```python
bf.to_inspect_task(scene: 'Scene', rubric: 'Rubric | None' = None) -> 'dict[str, Any]'
```

Convenience function to convert a Scene to Inspect AI format.

### `to_ors_reward`

```python
bf.to_ors_reward(result: 'VerifyResult') -> 'dict[str, Any]'
```

Convenience function to convert ``VerifyResult`` to ORS format.

### `workspace_restore`

```python
bf.workspace_restore(env: Any, ref: str, workspace: str = '/app') -> None
```

A coroutine function: `await` it, or run it with `asyncio.run(...)`.

Restore ``workspace`` to a named workspace snapshot.

### `workspace_snapshot`

```python
bf.workspace_snapshot(env: Any, name: str, workspace: str = '/app') -> str
```

A coroutine function: `await` it, or run it with `asyncio.run(...)`.

Create a named *workspace-only* snapshot; tar of ``workspace``.

### `write_ors_tool_outputs_jsonl`

```python
bf.write_ors_tool_outputs_jsonl(outputs: 'list[dict[str, Any]]', path: 'str | Path', *, source: 'str' = 'ors-tool-output') -> 'list[dict[str, Any]]'
```

Convenience wrapper for writing ORS reward JSONL evidence.

## Classes

### `ACPClient`

Defined in `benchflow.acp.client`.

Client that speaks ACP to an agent process.

```python
bf.ACPClient(transport: benchflow.acp.transport.Transport, *, subagent_transcript: bool = False, typed_failures: bool = False, session_meta: dict[str, typing.Any] | None = None)
```

New arguments in 0.10: `session_meta`.

- `.authenticate(method_id: str) -> dict`: Async: `await` it. Authenticate with the agent using one of its advertised auth methods.
- `.cancel() -> None`: Async: `await` it. Cancel the current prompt.
- `.close() -> None`: Async: `await` it. Close the transport and clean up.
- `.connect() -> None`: Async: `await` it. Start the transport.
- `.expect_silence(seconds: float) -> None`: Let the transport's read guard wait at least *seconds* (#1143).
- `.from_config(command: str | None = None, transport_type: str = 'stdio', url: str | None = None, env: dict[str, str] | None = None, cwd: str | None = None) -> 'ACPClient'`: Create an ACPClient from agent configuration.
- `.initialize() -> acp.schema.InitializeResponse`: Async: `await` it. Send initialize handshake.
- `.on_ask_user(handler: collections.abc.Callable[[dict[str, typing.Any]], collections.abc.Awaitable[str]] | None) -> None`: Register the agent-initiated ``session/request_permission`` handler.
- `.prompt(text: str) -> acp.schema.PromptResponse`: Async: `await` it. Send a prompt to the agent and wait for completion.
- `.session`
- `.session_load(session_id: str, cwd: str = '/app', mcp_servers: list[benchflow.acp.types.McpServerSpec] | None = None) -> benchflow.acp.session.ACPSession`: Async: `await` it. Load an existing pre-created session.
- `.session_new(cwd: str = '/app', mcp_servers: list[benchflow.acp.types.McpServerSpec] | None = None) -> benchflow.acp.session.ACPSession`: Async: `await` it. Create a new agent session.
- `.set_config_option(config_id: str, value: str) -> dict`: Async: `await` it. Set a session configuration option.
- `.set_model(model_id: str) -> dict`: Async: `await` it. Set the model for the current session.

### `ACPSession`

Defined in `benchflow.acp.session`.

Tracks mutable state for one ACP session.

```python
bf.ACPSession(session_id: str)
```

- `.distinct_tool_titles`: Count of distinct display titles across recorded tool calls.
- `.full_message`: Concatenated root-agent text, excluding explicitly attributed child chunks.
- `.full_thought`: Concatenated root-agent reasoning, excluding attributed child chunks.
- `.handle_update(update: dict, *, _receipt: dict[str, str] | None = None) -> None`: Process an update with an optional trusted, host-captured replay receipt.
- `.latest_usage_totals() -> dict[str, int | None] | None`: Return the latest cumulative ACP usage snapshot, if any.
- `.mark_prompt_end() -> None`: Flush pending agent text after a prompt completes.
- `.pending_tool_call_ids() -> list[str]`: Return tool calls that have not reached a terminal status.
- `.pending_tool_call_state() -> tuple[tuple[str, int], ...]`: Return stable pending-call identities with progress versions.
- `.progress_snapshot() -> tuple[int, str]`: (tool-call count, last tool title); the console heartbeat's counters, also polled by the live eval dashboard's activity cell.
- `.record_agent_timeout(*, timeout_sec: float, pending_tool_call_ids: list[str], terminal_trajectory_complete: bool) -> None`: Append BenchFlow's terminal timeout marker to the ACP event stream.
- `.record_prompt_usage(usage: object | None) -> None`: Record cumulative ACP token usage returned by session/prompt.
- `.record_user_prompt(text: str) -> None`: Record a user prompt. Call before sending each ACP prompt.
- `.stop_reason` (0.10): How the last prompt ended, as the agent reported it.

### `Agent`

Defined in `benchflow.runtime`.

Thin wrapper around a registered agent + model + credentials.

| Field | Type | Default |
|---|---|---|
| `name` | `str` | `required` |
| `model` | `str` | `required` |
| `env` | `dict[str, str]` | `(computed)` |

- `.config`: The registry entry for this agent, or None for a raw command.
- `.launch_cmd`: The command that starts the agent; raises KeyError for an unknown agent.

### `AgentInstallError`

Defined in `benchflow.models`.

Agent installation failed in the sandbox.

### `AgentTimeoutError`

Defined in `benchflow.models`.

Agent execution exceeded the allowed wall-clock time.

### `BaseUser`

Defined in `benchflow.contracts.user`.

Abstract user that drives a progressive-disclosure rollout loop.

```python
bf.BaseUser()
```

- `.run(round: 'int', instruction: 'str', round_result: 'RoundResult | None' = None) -> 'str | None'`: Async: `await` it. Produce the next prompt for the agent, or ``None`` to stop.
- `.setup(instruction: 'str', solution: 'str | None' = None) -> 'None'`: Async: `await` it. Called once before the first round.

### `BashToolResult`

Defined in `benchflow.rollout.task_runtime`.

Result from a runtime bash tool call.

| Field | Type | Default |
|---|---|---|
| `command` | `str` | `required` |
| `return_code` | `int` | `required` |
| `stdout` | `str` | `required` |
| `stderr` | `str` | `required` |
| `elapsed_sec` | `float` | `required` |

### `BenchmarkMetrics`

Defined in `benchflow.metrics`.

Aggregated metrics for a benchmark run.

| Field | Type | Default |
|---|---|---|
| `benchmark` | `<class 'str'>` | `required` |
| `agent` | `<class 'str'>` | `required` |
| `model` | `<class 'str'>` | `required` |
| `tasks` | `list[benchflow.metrics.TaskMetrics]` | `(computed)` |

- `.avg_cost_per_trial_usd`: Mean USD over the trials that reported a cost; None when none did.
- `.avg_duration`: Average duration per completed task (seconds).
- `.avg_skill_invocations`: Average structured skill invocations per completed task.
- `.avg_tool_calls`: Average tool calls per completed task.
- `.error_breakdown`: Categorize errors.
- `.errored`: Trials that ended with an agent error and no score.
- `.failed`: Trials scored below a pass.
- `.integration_failures`: Count the unscored trials whose agent integration broke, by cause.
- `.mean_reward`: Mean reward over every trial, an unscored one counting as 0.
- `.mean_reward_scored_only`: Mean reward over scored trials only (BenchFlow 0.7's mean_reward); None when none.
- `.mean_reward_scored_only_n`: The n of `mean_reward_scored_only`: trials with a reward.
- `.memory_score`: Mean memory score over the tasks that have one; None when none do.
- `.memory_scores`: Memory score per task, for trials that have one.
- `.memory_summary`: Memory scores in summary form: count, mean and formatted mean.
- `.passed`: Trials whose score outcome is a pass.
- `.score`: Pass rate over all tasks.
- `.score_excl_errors`: Pass rate excluding errored tasks.
- `.score_verifier_errored`: Count of tasks whose disjoint score bucket is verifier_errored.
- `.summary() -> dict[str, typing.Any]`: Export as summary dict.
- `.telemetry_completed_tasks`: Completed tasks included in telemetry coverage denominator.
- `.telemetry_coverage`: Fraction of completed trials with trusted token usage.
- `.telemetry_tasks`: Completed tasks with provider telemetry.
- `.total`: Trials counted (one per trial; see collect_metrics).
- `.total_cache_creation_tokens`: Prompt-cache write tokens over completed trials with trusted usage.
- `.total_cache_read_tokens`: Prompt-cache read tokens over completed trials with trusted usage.
- `.total_cost_usd`: USD over trials that reported a cost; None when none did (a subscription login reports none; unknown is not $0).
- `.total_input_tokens`: Input tokens over completed trials with trusted usage.
- `.total_output_tokens`: Output tokens over completed trials with trusted usage.
- `.total_tokens`: All tokens over completed trials with trusted usage.
- `.verifier_error_breakdown`: Categorize verifier errors.
- `.verifier_errored`: Backward-compatible alias for score-verifier-error count.

### `BranchChildResult`

Defined in `benchflow.branch_api`.

One child of a branched run, as recorded in ``tree.json``.

| Field | Type | Default |
|---|---|---|
| `label` | `str` | `required` |
| `status` | `str` | `required` |
| `reward` | `float \| None` | `required` |
| `reward_source` | `str \| None` | `required` |
| `path` | `str \| None` | `required` |
| `fork_id` | `str \| None` | `required` |
| `parent_label` | `str \| None` | `None` |
| `cost` | `dict[str, Any] \| None` | `None` |
| `advantage` | `float \| None` | `None` |

- `.to_record() -> 'dict[str, Any]'`: The child as a flat dict.

### `BranchPlanError`

Defined in `benchflow.branch_run`.

The branch request is inconsistent; nothing was started.

### `BranchResult`

Defined in `benchflow.branch_api`.

What one branched run produced.

| Field | Type | Default |
|---|---|---|
| `task` | `str` | `required` |
| `job_dir` | `Path` | `required` |
| `rollout_dir` | `Path \| None` | `required` |
| `value` | `float \| None` | `required` |
| `children` | `list[BranchChildResult]` | `required` |
| `parent` | `RolloutResult \| None` | `required` |
| `parent_reward` | `float \| None` | `required` |
| `fork_status` | `str \| None` | `required` |
| `parent_restore` | `str \| None` | `required` |
| `kept_snapshot` | `str \| None` | `None` |
| `error` | `str \| None` | `None` |
| `timing_sec` | `dict[str, Any] \| None` | `None` |
| `cost` | `dict[str, Any] \| None` | `None` |
| `forks` | `list[dict[str, Any]]` | `(computed)` |

- `.child(label: 'str') -> 'BranchChildResult'`: The child with this label (KeyError naming the labels otherwise).
- `.ok`: True when the fork completed and the trial raised no error.
- `.to_csv(path: 'str | Path') -> 'Path'`: Write one CSV row per child and return the path.
- `.to_jsonl(path: 'str | Path') -> 'Path'`: Write one JSON line per child and return the path.
- `.to_records() -> 'list[dict[str, Any]]'`: Return one flat dict per child; each holds the task and V(checkpoint).

### `Budget`

Defined in `benchflow.budget`.

Caps for one job; set at least one (see the module docstring).

| Field | Type | Default |
|---|---|---|
| `max_cost_usd` | `float \| None` | `None` |
| `max_sandbox_seconds` | `float \| None` | `None` |
| `max_tokens` | `int \| None` | `None` |
| `max_rollouts` | `int \| None` | `None` |

- `.coerce(value: 'Budget | Mapping[str, Any] | None') -> 'Budget | None'`: A Budget from itself, a mapping (``to_dict``'s shape) or None.
- `.exceeded(spent: 'Mapping[str, Any]') -> 'str | None'`: The reason a spend cap (USD, sandbox-seconds, tokens) is reached by ``spent``, or None. The rollout cap limits starts instead: see `BudgetGuard`.
- `.to_dict() -> 'dict[str, Any]'`: The caps as a mapping (``summary.json``'s ``budget.caps``).

### `ChildSpec`

Defined in `benchflow.branch_run`.

One ``--child`` option: a label and the child's own prompt.

| Field | Type | Default |
|---|---|---|
| `label` | `str` | `required` |
| `prompt` | `str \| None` | `None` |
| `parent` | `str \| None` | `None` |

### `CodeExecRewardFunc`

Defined in `benchflow.rewards.builtins`.

Arbitrary Python scoring function.

```python
bf.CodeExecRewardFunc(func: 'Callable[[Path], float]') -> 'None'
```

- `.score(rollout_dir: 'Path') -> 'float'`: Async: `await` it. The wrapped function's value for ``rollout_dir``, as a float.

### `Comparison`

Defined in `benchflow.jobs`.

Two jobs paired by task name (``a`` and ``b`` are their denominators).

| Field | Type | Default |
|---|---|---|
| `job_a` | `Job` | `required` |
| `job_b` | `Job` | `required` |
| `rows` | `list[ComparisonRow]` | `required` |
| `summary` | `ComparisonSummary` | `required` |
| `a` | `Denominators` | `required` |
| `b` | `Denominators` | `required` |
| `caveats` | `list[str]` | `required` |
| `labels` | `tuple[str, str]` | `('A', 'B')` |
| `mismatches` | `list[SettingMismatch]` | `(computed)` |
| `vary` | `tuple[str, ...]` | `()` |
| `include_controls` | `bool` | `False` |
| `solve_rates_a` | `SolveRates \| None` | `None` |
| `solve_rates_b` | `SolveRates \| None` | `None` |
| `a_paired` | `Denominators \| None` | `None` |
| `b_paired` | `Denominators \| None` | `None` |
| `by` | `tuple[str, ...]` | `()` |
| `notes` | `list[str]` | `(computed)` |
| `scoring` (0.10) | `dict[str, Any] \| None` | `None` |

- `.to_json(path: 'str | Path | None' = None, *, indent: 'int' = 2) -> 'Any'`: `to_json_dict` as a JSON string, or written to ``path`` (returned).
- `.to_json_dict() -> 'dict[str, Any]'`: The comparison as a ``benchflow.comparison`` document.
- `.to_markdown() -> 'str'`: A short report: denominators, the per-task table and the caveats.
- `.to_records() -> 'list[dict[str, Any]]'`: One flat dict per row, with its delta and the settings that differ.

### `ComparisonRow`

Defined in `benchflow.jobs`.

One task in a comparison; rewards are means over each side's scored trials.

| Field | Type | Default |
|---|---|---|
| `task` | `str` | `required` |
| `status` | `Literal['paired', 'only_a', 'only_b']` | `required` |
| `n_a` | `int` | `required` |
| `n_b` | `int` | `required` |
| `scored_a` | `int` | `required` |
| `scored_b` | `int` | `required` |
| `reward_a` | `float \| None` | `required` |
| `reward_b` | `float \| None` | `required` |
| `checks` | `list[SettingCheck]` | `(computed)` |
| `group` | `dict[str, Any]` | `(computed)` |

- `.delta`: reward_b - reward_a, or None unless both sides were scored.

### `ComparisonSummary`

Defined in `benchflow.jobs`.

ComparisonSummary(tasks: 'int', paired: 'int', only_a: 'int', only_b: 'int', both_scored: 'int', same_reward: 'int', b_higher: 'int', b_lower: 'int', mean_delta: 'float \| None', one_run_per_side: 'bool', setting_mismatches: 'int' = 0)

| Field | Type | Default |
|---|---|---|
| `tasks` | `int` | `required` |
| `paired` | `int` | `required` |
| `only_a` | `int` | `required` |
| `only_b` | `int` | `required` |
| `both_scored` | `int` | `required` |
| `same_reward` | `int` | `required` |
| `b_higher` | `int` | `required` |
| `b_lower` | `int` | `required` |
| `mean_delta` | `float \| None` | `required` |
| `one_run_per_side` | `bool` | `required` |
| `setting_mismatches` | `int` | `0` |

### `Completed`

Defined in `benchflow.batch`.

One finished rollout from `as_completed`.

```python
bf.Completed(index: ForwardRef('int'), config: ForwardRef('RolloutConfig'), result: ForwardRef('RolloutResult'))
```

### `Criterion`

Defined in `benchflow.rewards.rubric_config`.

A single evaluation criterion.

| Field | Type | Default |
|---|---|---|
| `description` | `str` | `required` |
| `type` | `Literal['binary', 'likert', 'numeric']` | `'binary'` |
| `name` | `str \| None` | `None` |
| `points` | `int` | `5` |
| `min` | `float` | `0.0` |
| `max` | `float` | `100.0` |
| `weight` | `float` | `1.0` |
| `files` | `list[str]` | `(computed)` |
| `space` | `Space` | `'output'` |

- `.id`: The criterion's name, or the start of its description.
- `.normalize(raw: 'float') -> 'float'`: Normalize a raw score to [0, 1].

### `Denominators`

Defined in `benchflow.jobs`.

Counts behind a pass rate.

| Field | Type | Default |
|---|---|---|
| `attempted` | `int` | `0` |
| `scored` | `int` | `0` |
| `assessment_errors` | `int` | `0` |
| `unscored` | `int` | `0` |
| `execution_errors` | `int` | `0` |
| `passed` | `int` | `0` |
| `mean_reward` | `float \| None` | `None` |
| `clean_scored` | `int` | `0` |
| `clean_passed` | `int` | `0` |
| `controls_excluded` | `int` | `0` |
| `integration_failures` | `int` | `0` |
| `mean_reward_scored_only` | `float \| None` | `None` |
| `mean_reward_caveat` | `str \| None` | `None` |
| `outcomes` (0.10) | `dict[str, int]` | `(computed)` |

- `.of(trials: 'Iterable[Trial]', *, controls_excluded: 'int' = 0) -> 'Denominators'`: Count ``trials`` (see the class docstring for each count).
- `.pass_rate_attempted`: Passed over attempted trials; None when none was attempted.
- `.pass_rate_clean`: Passed over scored trials whose execution completed; None when none.
- `.pass_rate_scored`: Passed over scored trials; None when none was scored.

### `DocumentNudgeUser`

Defined in `benchflow.contracts.user`.

Deterministic user compiled from a task document.

```python
bf.DocumentNudgeUser(*, persona: 'str | None' = None, private_facts: 'dict[str, str] | None' = None, branchable: 'bool' = False, branch_execution: 'str' = 'none', confirmation_policy: 'str | None' = None, handoff_kind: 'str' = 'none', handoff_team: 'str | None' = None) -> 'None'
```

- `.run(round: 'int', instruction: 'str', round_result: 'RoundResult | None' = None) -> 'str | None'`: Async: `await` it. Round 0 sends the instruction; later rounds nudge the agent from the task document until it passes.
- `.setup(instruction: 'str', solution: 'str | None' = None) -> 'None'`: Async: `await` it. Called once before the first round.

### `Environment`

Defined in `benchflow.runtime`.

Wraps a Docker/Daytona sandbox environment, owns lifecycle.

```python
bf.Environment(inner: 'Any', task_path: 'Path', sandbox: 'str') -> 'None'
```

- `.download_dir(src: 'str', dst: 'str | Path', service: 'str' = 'main') -> 'None'`: Async: `await` it. Download a directory from the sandbox.
- `.download_file(src: 'str', dst: 'str | Path') -> 'None'`: Async: `await` it. Copy the sandbox file ``src`` to the host path ``dst``.
- `.exec(cmd: 'str', **kwargs: 'Any') -> 'Any'`: Async: `await` it. Run a command in the sandbox.
- `.exec_in_service(service: 'str', cmd: 'str', **kwargs: 'Any') -> 'Any'`: Async: `await` it. Run a command in a named compose service container (#248).
- `.from_task(task_path: 'str | Path', sandbox: 'str' = 'daytona', rollout_name: 'str | None' = None, planes: 'Any | None' = None, *, jobs_dir: 'str | Path' = 'jobs') -> 'Environment'`: Create an environment and allocate its trial's artifact paths.
- `.inner`: The underlying sandbox environment (Docker/Daytona). Use for Scene-based shared sandbox access.
- `.prepare_kept_services(agent: 'str', *, sandbox_user: 'str | None' = 'agent') -> 'None'`: Tell the sandbox, before `start`, what will run in it, so a task that keeps services (``x-keep-services``) gets the kept-service freezer it needs: the oracle, or an agent run as the non-root ``sandbox_user``. ``Runtime.execute`` does this for an environment it starts; call it yourself for one you start.
- `.start(force_build: 'bool' = False) -> 'None'`: Async: `await` it. Start the sandbox (``force_build=True`` rebuilds its image first).
- `.stop(delete: 'bool' = True) -> 'None'`: Async: `await` it. Stop the sandbox if it was started (``delete=False`` keeps it).
- `.task`: The task this environment runs (a `benchflow.Task`).
- `.upload_dir(src: 'str | Path', dst: 'str', service: 'str' = 'main') -> 'None'`: Async: `await` it. Upload a directory into the sandbox.
- `.upload_file(src: 'str | Path', dst: 'str') -> 'None'`: Async: `await` it. Copy a host file into the sandbox at ``dst``.

### `EnvironmentManifest`

Defined in `benchflow.environment.manifest`.

A benchmark's self-describing Environment-plane declaration.

| Field | Type | Default |
|---|---|---|
| `name` | `str` | `required` |
| `image` | `str \| None` | `None` |
| `base_image` | `str \| None` | `None` |
| `ports` | `list` | `(computed)` |
| `services` | `list` | `(computed)` |
| `owns_lifecycle` | `bool` | `True` |
| `keep_alive` | `bool` | `True` |
| `isolation` | `Literal` | `'per_task'` |
| `task_selection` | `TaskSelection` | `(computed)` |
| `readiness` | `Readiness` | `(computed)` |
| `forward_env` | `ForwardEnv` | `(computed)` |
| `state` | `benchflow.environment.manifest.StateSpec \| None` | `None` |

- `.all_ports`: Every port the environment exposes; declared + service-derived.
- `.effective_http`: The HTTP readiness probes to gate on.
- `.model_validate_path(path: 'str | Path') -> 'EnvironmentManifest'`: Load + validate a manifest file, picking the parser by extension.
- `.model_validate_toml(toml_data: 'str') -> 'EnvironmentManifest'`: Parse a manifest from a TOML string.
- `.model_validate_yaml(yaml_data: 'str') -> 'EnvironmentManifest'`: Parse a manifest from a YAML string; the canonical format.

### `Evaluation`

Defined in `benchflow.evaluation`.

Run a benchmark job across multiple tasks.

```python
bf.Evaluation(tasks_dir: 'str | Path', jobs_dir: 'str | Path', config: 'EvaluationConfig | None' = None, job_name: 'str | None' = None, on_result: 'Callable[[str, RunResult], None] | None' = None, on_task_start: 'Callable[[str], None] | None' = None, on_plan: 'Callable[[int, int, int, tuple[int, int, int]], None] | None' = None, preflight: 'bool' = True, budget: 'Budget | dict[str, Any] | None' = None, *, restart_changed: 'bool' = False)
```

New arguments in 0.10: `restart_changed`.

- `.from_dict(raw: 'dict[str, Any]', *, warn_default_agent: 'bool' = True, **kwargs: 'Any') -> 'Evaluation'`: Build an Evaluation from a native config mapping (``to_dict``'s shape).
- `.from_yaml(path: 'str | Path', *, warn_default_agent: 'bool' = True, job_cell: 'int | None' = None, **kwargs: 'Any') -> 'Evaluation'`: Create a Job from a YAML config file.
- `.job_dir`: Where this job writes its rollouts, known before it runs.
- `.resume(job_dir: 'str | Path', *, tasks_dir: 'str | Path | None' = None, agent_env: 'dict[str, str] | None' = None, restart_changed: 'bool' = False, **config_overrides: 'Any') -> 'Evaluation'`: Rebuild the Evaluation that created ``job_dir``, ready to finish it.
- `.run() -> 'EvaluationResult'`: Async: `await` it. Execute the job.
- `.run_sync() -> 'EvaluationResult'`: Blocking form of `run` (also works inside a running event loop).
- `.stream() -> 'AsyncIterator[tuple[str, RolloutResult]]'`: Run the job, yielding ``(task_name, RolloutResult)`` as each task finishes.
- `.to_dict(*, include_agent_env: 'bool' = False) -> 'dict[str, Any]'`: This job as the native config mapping that ``from_yaml`` and ``bench eval run --config`` read.
- `.to_yaml(path: 'str | Path', *, include_agent_env: 'bool' = False) -> 'Path'`: Write `to_dict` as YAML (``bench eval run --config`` reads it).
- `.warn_unmatched_selection() -> 'None'`: Warn once per pattern that matched no task, though others did.

### `EvaluationConfig`

Defined in `benchflow.evaluation`.

Configuration for a benchmark job.

| Field | Type | Default |
|---|---|---|
| `agent` | `str` | `'claude-agent-acp'` |
| `model` | `str \| None` | `None` |
| `reasoning_effort` | `str \| None` | `None` |
| `harness` | `str` | `'acp'` |
| `environment` | `str` | `'docker'` |
| `concurrency` | `int` | `4` |
| `build_concurrency` | `int \| None` | `None` |
| `warm_pool` | `int \| None` | `None` |
| `prompts` | `list[str \| None] \| None` | `None` |
| `agent_env` | `dict[str, str]` | `(computed)` |
| `retry` | `RetryConfig` | `(computed)` |
| `reviewer` | `ReviewerConfig` | `(computed)` |
| `skills_dir` | `str \| None` | `None` |
| `codex_apps_policy` | `str \| None` | `None` |
| `sandbox_user` | `str \| None` | `'agent'` |
| `sandbox_locked_paths` | `list[str] \| None` | `None` |
| `sandbox_setup_timeout` | `int` | `120` |
| `skip_agent_install` | `bool` | `False` |
| `fresh_install` | `bool` | `False` |
| `allow_cloud_metadata` | `bool` | `False` |
| `agent_idle_timeout` | `int \| None` | `600` |
| `agent_idle_timeout_given` (0.10) | `bool \| None` | `None` |
| `context_root` | `str \| None` | `None` |
| `base_image_override` | `str \| None` | `None` |
| `separate_verifier` | `bool` | `False` |
| `separate_verifier_include` | `tuple[str, ...]` | `()` |
| `exclude_tasks` | `set[str]` | `(computed)` |
| `include_tasks` | `set[str]` | `(computed)` |
| `n_tasks` | `int \| None` | `None` |
| `sample_seed` | `int \| None` | `None` |
| `timeout_multiplier` | `float \| None` | `None` |
| `extra_instruction` | `str \| None` | `None` |
| `skill_mode` | `str` | `'no-skill'` |
| `skill_creator_dir` | `str \| None` | `None` |
| `self_gen_no_internet` | `bool` | `False` |
| `job_mode` | `str` | `'parallel-independent'` |
| `source_provenance` | `dict[str, Any] \| None` | `None` |
| `dataset_name` | `str \| None` | `None` |
| `dataset_version` | `str \| None` | `None` |
| `dataset_task_digests` | `dict[str, str]` | `(computed)` |
| `usage_tracking` | `UsageTrackingConfig` | `(computed)` |
| `environment_manifest` | `EnvironmentManifest \| None` | `None` |
| `config_override` | `dict \| None` | `None` |
| `loop_strategy` | `LoopStrategySpec \| str \| None` | `None` |
| `checkpoints` | `str \| None` | `None` |
| `checkpoint_keep` | `int` | `3` |
| `freeze_workspace` | `bool` | `False` |
| `keep_sandbox` (0.10) | `str` | `'never'` |
| `integrity` | `str` | `'off'` |
| `verifier_sandbox` | `str` | `'task'` |
| `retry_from_checkpoint` | `str \| None` | `None` |
| `retry_prompt` | `str \| None` | `None` |
| `retry_resume_session` | `bool` | `False` |
| `budget` | `Budget \| None` | `None` |
| `trial_budget` (0.10) | `TrialBudget \| None` | `None` |
| `seeds` | `list[int] \| None` | `None` |
| `pre_agent_hooks` | `list[Callable[..., Any]] \| None` | `None` |
| `agent_defaulted` | `bool` | `False` |
| `accept_deviations` (0.10) | `str \| None` | `None` |
| `planned_trials` (0.10) | `int` | `1` |
| `trial_number` (0.10) | `int \| None` | `None` |
| `model_given` (0.10) | `bool \| None` | `None` |

- `.checkpoint_policy() -> 'CheckpointPolicy | None'`: The parsed checkpoint policy, or None when not requested.
- `.idle_timeout_source() -> 'str'` (0.10): Where the idle limit came from: ``given``, or BenchFlow's ``default`` (``effort_max`` and ``effort_xhigh`` for its longer one).
- `.retry_policy() -> 'RetryPolicy | None'`: The parsed retry policy, or None when not requested.
- `.settle_idle_timeout() -> 'None'` (0.10): Give a run that set no idle limit BenchFlow's default for its effort; call again after the effort changes.

### `EvaluationResult`

Defined in `benchflow.evaluation`.

Aggregated results for a job.

| Field | Type | Default |
|---|---|---|
| `job_name` | `str` | `required` |
| `config` | `EvaluationConfig` | `required` |
| `total` | `int` | `0` |
| `passed` | `int` | `0` |
| `failed` | `int` | `0` |
| `errored` | `int` | `0` |
| `verifier_errored` | `int` | `0` |
| `guard_refused` | `int` | `0` |
| `unusable_rewards` (0.10) | `int` | `0` |
| `elapsed_sec` | `float` | `0.0` |
| `memory_score` | `float \| None` | `None` |
| `memory_scores` | `dict[str, float]` | `(computed)` |
| `task_failures` | `list[TaskFailure]` | `(computed)` |
| `mean_reward` | `float \| None` | `None` |
| `mean_reward_scored_only` | `float \| None` | `None` |
| `mean_reward_n` | `int` | `0` |
| `mean_reward_scored_only_n` | `int` | `0` |
| `mean_reward_caveat` | `str \| None` | `None` |
| `job_dir` | `Path \| None` | `None` |
| `results` | `dict[str, RolloutResult]` | `(computed)` |
| `budget` | `dict[str, Any] \| None` | `None` |
| `reused` | `int` | `0` |
| `ran` | `int` | `0` |
| `not_run` | `int` | `0` |
| `stopped` | `bool` | `False` |
| `interrupted` | `int` | `0` |
| `run_checks` (0.10) | `dict[str, Any] \| None` | `None` |
| `scoring` (0.10) | `dict[str, Any] \| None` | `None` |

- `.planned`: The tasks the job set out to run: ``total`` plus ``not_run``.
- `.score`: Pass rate over all tasks.
- `.score_excl_errors`: Pass rate excluding errored tasks.
- `.score_of_plan`: Pass rate over the planned tasks (a task not run counts as not passed); the same as ``score`` when every planned task ran.
- `.to_csv(path: 'str | Path') -> 'Path'`: Write one CSV row per task and return the path.
- `.to_jsonl(path: 'str | Path') -> 'Path'`: Write one JSON line per task and return the path.
- `.to_records() -> 'list[dict[str, Any]]'`: Return one flat dict per task (``RolloutResult.to_record``), sorted by task name.

### `ExecResult`

Defined in `benchflow.sandbox.protocol`.

ExecResult(return_code: 'int', stdout: 'str', stderr: 'str')

| Field | Type | Default |
|---|---|---|
| `return_code` | `int` | `required` |
| `stdout` | `str` | `required` |
| `stderr` | `str` | `required` |

### `Fork`

Defined in `benchflow.jobs`.

One branch point recorded in the trial's ``tree.json``.

| Field | Type | Default |
|---|---|---|
| `id` | `str` | `required` |
| `status` | `str \| None` | `required` |
| `value` | `float \| None` | `required` |
| `parent_restore` | `str \| None` | `required` |
| `children` | `list[BranchChildResult]` | `required` |
| `snapshot` | `dict[str, Any]` | `(computed)` |
| `timing_sec` | `dict[str, Any] \| None` | `None` |
| `kind` | `str` | `'fork'` |
| `parent_node` | `str \| None` | `None` |
| `cost` | `dict[str, Any] \| None` | `None` |

### `FunctionUser`

Defined in `benchflow.contracts.user`.

Wrap a sync or async function as a ``BaseUser``.

```python
bf.FunctionUser(fn: 'Callable[[int, str, RoundResult | None], str | Awaitable[str | None] | None]') -> 'None'
```

- `.run(round: 'int', instruction: 'str', round_result: 'RoundResult | None' = None) -> 'str | None'`: Async: `await` it. Call the wrapped function (sync or async) for this round's prompt; None ends the loop.
- `.setup(instruction: 'str', solution: 'str | None' = None) -> 'None'`: Async: `await` it. Called once before the first round.

### `GroupDenominators`

Defined in `benchflow.jobs`.

One group's denominators, e.g. key=&#123;"agent": …, "model": …&#125;.

| Field | Type | Default |
|---|---|---|
| `key` | `dict[str, Any]` | `required` |
| `denominators` | `Denominators` | `required` |

### `ImageBuilder`

Defined in `benchflow.sandbox.protocol`.

Build-only: produces image refs from Dockerfiles/configs.

```python
bf.ImageBuilder(*args, **kwargs)
```

- `.build(config: 'ImageConfig') -> 'ImageRef'`: Async: `await` it. Build the image ``config`` describes and return its reference.
- `.cached(config: 'ImageConfig') -> 'ImageRef | None'`: Async: `await` it. The reference of an already-built image for ``config``, or None.

### `ImageConfig`

Defined in `benchflow.sandbox.protocol`.

ImageConfig(dockerfile: 'Path', context_dir: 'Path', build_args: 'dict[str, str] \| None' = None, cache_key: 'str \| None' = None)

| Field | Type | Default |
|---|---|---|
| `dockerfile` | `Path` | `required` |
| `context_dir` | `Path` | `required` |
| `build_args` | `dict[str, str] \| None` | `None` |
| `cache_key` | `str \| None` | `None` |

### `ImageRef`

Defined in `benchflow.sandbox.protocol`.

ImageRef(tag: 'str', digest: 'str \| None' = None)

| Field | Type | Default |
|---|---|---|
| `tag` | `str` | `required` |
| `digest` | `str \| None` | `None` |

### `InspectAdapter`

Defined in `benchflow.adapters.inspect_ai`.

Wraps a BenchFlow Scene + Rubric as an Inspect AI-compatible task.

```python
bf.InspectAdapter(scene: 'Scene', rubric: 'Rubric | None' = None) -> 'None'
```

- `.to_inspect_task() -> 'dict[str, Any]'`: Convert to Inspect AI task format.

### `Job`

Defined in `benchflow.jobs`.

Every trial under a job directory (or any folder of trials).

| Field | Type | Default |
|---|---|---|
| `path` | `Path` | `required` |
| `trials` | `list[Trial]` | `required` |
| `summary` | `dict[str, Any] \| None` | `None` |
| `evaluation` | `dict[str, Any] \| None` | `None` |
| `interrupted` | `list[Path]` | `(computed)` |
| `unreadable` (0.10) | `list[Path]` | `(computed)` |

- `.agents() -> 'list[Trial]'`: Agent runs (control runs left out).
- `.branch_views() -> 'list[dict[str, Any]]'`: The ``benchflow.branch-view/1`` document of every trial that branched (see docs/train/environment.md).
- `.by_task(*, include_controls: 'bool' = False) -> 'dict[str, list[Trial]]'`: Trials grouped by task (control runs left out unless ``include_controls``).
- `.controls() -> 'list[Trial]'`: Control runs (oracle, empty/nop).
- `.cost_usd`: USD over the job's trials that reported one (one attempt each; ``trial.attempts`` has the rest); None when none did.
- `.denominators(*, include_controls: 'bool' = False) -> 'Denominators'`: Attempted, scored, errors and pass rates (control runs left out unless ``include_controls``).
- `.denominators_by(by: 'Iterable[str]' = ('agent', 'model'), *, include_controls: 'bool' = False) -> 'list[GroupDenominators]'`: Denominators per group, e.g. per agent and model (see `GROUP_KEYS`).
- `.integrity() -> 'Any'`: Integrity verdicts for every trial (an ``IntegrityReport``).
- `.kind`: evaluation (an Evaluation job), branch (a branch job) or directory.
- `.planned_tasks`: The tasks the job set out to run, as its evaluation.json records them; None when it does not (a job from before BenchFlow 0.8.3, a folder of trials or of several jobs).
- `.pooled_differences() -> 'list[PooledDifference]'`: What differs among the trials this job pools (DF-p3-eval-runner-8).
- `.running() -> 'int | None'`: The pid of the process on this host that is running this job (it holds the job's lock), or None when the job is not running.
- `.solve_rates(*, ks: 'Iterable[int] | None' = None, solve_threshold: 'float | None' = None, include_controls: 'bool' = False) -> 'SolveRates'`: pass@k, pass^k and the solve rate over this job's trials.
- `.to_csv(path: 'str | Path') -> 'Path'`: Write `to_records` as CSV (one row per trial) and return the path.
- `.to_json(path: 'str | Path | None' = None, *, include_trajectories: 'bool' = False, include_verifier: 'bool' = True, indent: 'int' = 2) -> 'Any'`: `to_json_dict` as a JSON string, or written to ``path`` (returned).
- `.to_json_dict(*, include_trajectories: 'bool' = False, include_verifier: 'bool' = True) -> 'dict[str, Any]'`: The job as a ``benchflow.job`` document (see ``benchflow.job_export``).
- `.to_jsonl(path: 'str | Path') -> 'Path'`: Write `to_records` as JSON Lines and return the path.
- `.to_markdown() -> 'str'`: A short report of the job, also what ``print(job)`` shows.
- `.to_records() -> 'list[dict[str, Any]]'`: Return one flat dict per trial, sorted by task. Each has a ``job`` key with this job's path, so rows from several jobs stay distinguishable.
- `.unfinished() -> 'tuple[list[str], list[str]]'`: ``(interrupted, not_started)``: tasks whose trial started but wrote no result (a kill, Ctrl-C, a cancelled run), and planned tasks with no trial at all. ``not_started`` needs `planned_tasks`; without it it is empty. A task finished since (resumed) is in neither. While `running`, the first list holds the trials still running.

### `JudgeConfig`

Defined in `benchflow.rewards.rubric_config`.

The ``[judge]`` section of a rubric.

| Field | Type | Default |
|---|---|---|
| `model` | `str` | `'claude-sonnet-5-5'` (0.9.2: `claude-sonnet-4-6`) |
| `mode` | `Literal['batched', 'individual']` | `'individual'` |
| `files` | `list[str]` | `(computed)` |
| `timeout` | `int` | `120` |

### `LLMJudgeRewardFunc`

Defined in `benchflow.rewards.builtins`.

LLM-as-judge for subjective quality scoring.

```python
bf.LLMJudgeRewardFunc(prompt: 'str' = '', model: 'str' = 'claude-sonnet-5-5', *, rubric_path: 'Path | None' = None, criteria: 'list[dict] | None' = None, mode: "Literal['batched', 'individual']" = 'individual', judge_model: 'str | None' = None, judge_env: 'Mapping[str, str] | None' = None, judge_errors_are_infra: 'bool' = False) -> 'None'
```

In 0.9.2, `model` defaults to `claude-sonnet-4-6`.

- `.events`: Dense reward events from the last ``score()`` call.
- `.score(rollout_dir: 'Path') -> 'float'`: Async: `await` it. Score the rollout, returning a float in [0, 1].

### `ModelDocumentNudgeUser`

Defined in `benchflow.contracts.user`.

Model-backed user compiled from a task document.

```python
bf.ModelDocumentNudgeUser(*, model: 'str', persona: 'str | None' = None, private_facts: 'dict[str, str] | None' = None, branchable: 'bool' = False, branch_execution: 'str' = 'none', confirmation_policy: 'str | None' = None, handoff_kind: 'str' = 'none', handoff_team: 'str | None' = None, call_model: 'UserModelCaller | None' = None) -> 'None'
```

- `.run(round: 'int', instruction: 'str', round_result: 'RoundResult | None' = None) -> 'str | None'`: Async: `await` it. Round 0 sends the instruction; later rounds ask a model for the next nudge.
- `.setup(instruction: 'str', solution: 'str | None' = None) -> 'None'`: Async: `await` it. Called once before the first round.

### `Monitor`

Defined in `benchflow.monitor`.

Score a rollout in production (Mode 3 of the architecture).

```python
bf.Monitor(config: 'MonitorConfig') -> 'None'
```

- `.replay(trajectory_path: 'str | Path') -> 'MonitorResult'`: Async: `await` it. Re-score a persisted rollout under monitor semantics.
- `.run() -> 'MonitorResult'`: Async: `await` it. Score a single persisted trajectory against the monitor rubric.
- `.watch() -> 'NoReturn'`: Async: `await` it. Stream-score live production events.

### `MonitorConfig`

Defined in `benchflow.monitor`.

Configuration for a monitor run.

| Field | Type | Default |
|---|---|---|
| `source` | `str \| Path` | `required` |
| `rubric_path` | `str \| Path \| None` | `None` |
| `jobs_dir` | `str \| Path` | `'jobs/monitor'` |
| `run_name` | `str \| None` | `None` |
| `metadata` | `dict[str, Any]` | `(computed)` |

### `MonitorNotImplementedError`

Defined in `benchflow.monitor`.

Raised by Monitor stubs until the runtime lands (#386).

### `MonitorResult`

Defined in `benchflow.monitor`.

Result of a monitor run.

| Field | Type | Default |
|---|---|---|
| `run_name` | `str` | `required` |
| `source` | `str` | `required` |
| `rewards` | `dict[str, Any] \| None` | `None` |
| `trajectory` | `list[dict]` | `(computed)` |
| `error` | `str \| None` | `None` |
| `verifier_error` | `str \| None` | `None` |
| `alert` | `bool` | `False` |
| `artifact_dir` | `Path \| None` | `None` |
| `metadata` | `dict[str, Any]` | `(computed)` |
| `started_at` | `datetime \| None` | `None` |
| `finished_at` | `datetime \| None` | `None` |

### `ORSAdapter`

Defined in `benchflow.adapters.ors`.

Converts BenchFlow reward types to ORS-compatible format.

```python
bf.ORSAdapter()
```

- `.reward_event_to_ors(event: 'RewardEvent') -> 'dict[str, Any]'`: Convert a single ``RewardEvent`` to ORS event format.
- `.tool_outputs_to_reward_events(outputs: 'list[dict[str, Any]]', *, source: 'str' = 'ors-tool-output') -> 'list[dict[str, Any]]'`: Normalize ORS-style tool outputs into verifier evidence records.
- `.verify_result_to_ors(result: 'VerifyResult') -> 'dict[str, Any]'`: Convert ``VerifyResult`` to ORS reward response format.
- `.write_tool_outputs_jsonl(outputs: 'list[dict[str, Any]]', path: 'str | Path', *, source: 'str' = 'ors-tool-output') -> 'list[dict[str, Any]]'`: Write ORS tool-output rewards to ``trajectory/ors-rewards.jsonl``.

### `PassthroughUser`

Defined in `benchflow.contracts.user`.

Sends the original instruction unchanged for one round.

```python
bf.PassthroughUser()
```

- `.run(round: 'int', instruction: 'str', round_result: 'RoundResult | None' = None) -> 'str | None'`: Async: `await` it. Round 0 sends the instruction; later rounds end the loop.
- `.setup(instruction: 'str', solution: 'str | None' = None) -> 'None'`: Async: `await` it. Called once before the first round.

### `PathReward`

Defined in `benchflow.rewards.node`.

Adapter; lift a legacy path-based ``RewardFunc`` into ``Reward``.

```python
bf.PathReward(func: 'RewardFunc', source: 'str | None' = None) -> 'None'
```

- `.score(node: 'RolloutNode') -> 'VerifyResult'`: Async: `await` it. Adapt a path-based RewardFunc: score the rollout folder the node records.

### `PublicationDecision`

Defined in `benchflow.review.scoring`.

Publication band produced after deterministic and blocker gates.

```python
bf.PublicationDecision(*values)
```

### `RegradeSummary`

Defined in `benchflow.eval_regrade`.

What ``bf.regrade`` did to a job or trial.

| Field | Type | Default |
|---|---|---|
| `path` | `str` | `required` |
| `regrade_id` | `str` | `required` |
| `tasks_dir` | `str \| None` | `required` |
| `reason` | `str \| None` | `required` |
| `trials` | `list[TrialRegrade]` | `(computed)` |

- `.changed`: Regraded trials whose verdict changed.
- `.counts() -> 'dict[str, int]'`: How many trials were regraded, changed (and which way), failed or not regradable.
- `.failed`: Trials whose regrade failed.
- `.not_regradable`: Trials that kept no frozen workspace to score again.
- `.regraded`: Trials whose stored workspace was scored again.
- `.to_dict() -> 'dict[str, Any]'`: The summary as the JSON document ``bench eval regrade --json`` prints.

### `Results`

Defined in `benchflow.batch`.

A list of `RolloutResult` with summary numbers and exports.

```python
bf.Results(iterable=(), /)
```

- `.denominators(*, include_controls: 'bool' = False) -> 'Denominators'`: Attempted, scored, errors and pass rates, counted as ``Job.denominators``.
- `.mean_reward`: Mean ``reward`` over every result, an unscored one counting as 0; None when there is no result.
- `.mean_reward_caveat`: Set when a reward is below 0 and some result is unscored: the every-trial `mean_reward` then counts an error above it.
- `.mean_reward_scored_only`: Mean ``reward`` over scored results only (BenchFlow 0.7's mean); None when nothing was scored.
- `.n_errored`: How many results ended with an agent-side error (errored or timed out).
- `.n_passed`: How many results passed.
- `.n_scored`: How many results have a reward.
- `.to_csv(path: 'str | Path') -> 'Path'`: Write the records to a CSV file and return its path.
- `.to_jsonl(path: 'str | Path') -> 'Path'`: Write the records to a JSONL file and return its path.
- `.to_records() -> 'list[dict[str, Any]]'`: One flat, JSON-safe dict per result, e.g. for ``pandas.DataFrame(…)``.

### `RetryConfig`

Defined in `benchflow.evaluation`.

Configuration for retry behavior.

| Field | Type | Default |
|---|---|---|
| `max_retries` | `int` | `2` |
| `retry_on_install` | `bool` | `True` |
| `retry_on_pipe` | `bool` | `True` |
| `retry_on_acp` | `bool` | `True` |
| `retry_on_idle_timeout` | `bool` | `True` |
| `retry_on_infra` | `bool` | `True` |
| `retry_on_verifier_infra` | `bool` | `True` |
| `retry_on_api_error` | `bool` | `True` |
| `wait_multiplier` | `float` | `2.0` |
| `min_wait_sec` | `float` | `1.0` |
| `max_wait_sec` | `float` | `30.0` |
| `exclude_categories` | `set[str]` | `(computed)` |

- `.backoff_delay(attempt: 'int') -> 'float'`: Exponential backoff delay for retry attempt.
- `.from_mapping(raw: 'dict | None') -> 'RetryConfig'`: Build from a serialized ``retry`` payload (e.g. a worker config).
- `.reruns_on_resume(error: 'str | None', *, category: 'str | None' = None) -> 'bool'`: Whether a resumed job runs an unscored trial again.
- `.reruns_unjudged_solver(scoring: 'ScoringResult | None', error: 'str | None', *, category: 'str | None' = None, on_resume: 'bool' = False) -> 'bool'`: Whether a rubric trial's solver runs again despite its scoring block.
- `.reruns_verifier_error_on_resume(verifier_error: 'str | None') -> 'bool'`: Whether a resumed job runs a scoreless verifier-errored trial again.
- `.should_retry(error: 'str | None', *, category: 'str | None' = None) -> 'bool'`: Check if an error is retryable.
- `.should_retry_verifier_error(verifier_error: 'str | None') -> 'bool'`: Check if a verifier error is infrastructure-retryable.

### `ReviewerConfig`

Defined in `benchflow.review.options`.

One reviewer runtime, independent of solver credentials and budgets.

| Field | Type | Default |
|---|---|---|
| `agent` | `str` | `'opencode'` |
| `model` | `str \| None` | `None` |
| `reasoning_effort` | `str \| None` | `None` |
| `environment` | `str` | `'docker'` |
| `timeout_sec` | `int` | `1800` |
| `idle_timeout_sec` | `int` | `600` |
| `storage_mb` (0.10) | `int \| None` | `None` |
| `memory_mb` (0.10) | `int` | `2048` |
| `setup_timeout_sec` (0.10) | `int \| None` | `None` |
| `evidence_timeout_sec` (0.10) | `int \| None` | `None` |
| `concurrency` | `int` | `4` |
| `image` | `str` | `'python@sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91'` |
| `agent_env` | `dict` | `(computed)` |
| `open_network` | `bool` | `False` |

- `.coerce(value: 'object' = None) -> 'Self'`: A ReviewerConfig from itself, a mapping, or None (the defaults).
- `.normalize_agent(value: 'str') -> 'str'`: Resolve an agent alias to its registry name.
- `.normalize_effort(value: 'str | None') -> 'str | None'`: Normalize the reasoning effort label.
- `.to_config_artifact() -> 'dict'`: The config as recorded in a trial's config.json: agent_env keys, and only values safe to record.
- `.to_dict() -> 'dict'`: The config as a JSON-ready mapping (agent_env values included).
- `.validate_environment(value: 'str') -> 'str'`: Refuse an unknown sandbox provider.
- `.validate_image(value: 'str') -> 'str'`: Refuse an empty image reference or one with whitespace.

### `ReviewReport`

Defined in `benchflow.review.runner`.

Everything one ``bench review`` invocation produced.

| Field | Type | Default |
|---|---|---|
| `path` | `str` | `required` |
| `rubric_path` | `str` | `required` |
| `criteria` | `list[str]` | `required` |
| `agent` | `str` | `required` |
| `model` | `str \| None` | `required` |
| `environment` | `str` | `required` |
| `network` | `str` | `'no-internet'` |
| `job_summary` | `str \| None` | `None` |
| `trials` | `list[TrialReview]` | `(computed)` |

- `.to_dict() -> 'dict[str, Any]'`: The report as ``review_report.json``'s mapping.

### `ReviewRubric`

Defined in `benchflow.review.config`.

A parsed review rubric.

| Field | Type | Default |
|---|---|---|
| `criteria` | `list` | `required` |

- `.contract`: The rubric contract: weighted (blockers and scored criteria) or legacy.
- `.is_weighted`: Whether this is a weighted rubric (not the legacy pass/fail one).
- `.metadata() -> 'dict[str, Any]'`: Return the rubric contract carried into wrapper/report artifacts.

### `ReviewRubricCriterion`

Defined in `benchflow.review.config`.

One criterion the reviewer grades.

| Field | Type | Default |
|---|---|---|
| `name` | `str` | `required` |
| `description` | `str` | `required` |
| `guidance` | `str` | `required` |
| `blocker` | `int \| None` | `None` |
| `weight` | `int \| None` | `None` |

- `.is_blocker`: Whether a fail on this criterion zeroes the reward.
- `.is_legacy`: Whether the criterion is from a legacy rubric (no blocker flag).
- `.metadata() -> 'dict[str, str | int | None]'`: Return the non-prompt criterion contract recorded in artifacts.

### `ReviewRubricError`

Defined in `benchflow.review.config`.

Raised when a rubric file cannot be loaded or is not a valid rubric.

### `ReviewScoring`

Defined in `benchflow.review.scoring`.

Auditable host-side aggregation of one valid v0.2 review.

| Field | Type | Default |
|---|---|---|
| `deterministic_pass` | `bool` | `required` |
| `all_blockers_pass` | `bool` | `required` |
| `failed_blockers` | `tuple[str, ...]` | `required` |
| `weighted_points` | `int` | `required` |
| `max_weighted_points` | `int` | `required` |
| `raw_quality` | `float` | `required` |
| `gated_quality` | `float` | `required` |
| `decision` | `PublicationDecision` | `required` |

- `.to_dict() -> 'dict[str, Any]'`: The scoring as a JSON-ready mapping.

### `Reward`

Defined in `benchflow.rewards.protocol`.

The canonical Reward-plane contract; ``score(node) -&gt; VerifyResult``.

```python
bf.Reward(*args, **kwargs)
```

- `.score(node: 'RolloutNode') -> 'VerifyResult'`: Async: `await` it. Score one rollout node: its reward and events.

### `RewardEvent`

Defined in `benchflow.rewards.events`.

A single reward signal emitted during or after a trial.

| Field | Type | Default |
|---|---|---|
| `type` | `str` | `required` |
| `reward` | `float` | `required` |
| `source` | `str` | `required` |
| `step` | `int \| None` | `None` |
| `space` | `Space` | `'output'` |
| `granularity` | `Granularity` | `'terminal'` |
| `ts` | `str` | `(computed)` |

### `RewardFunc`

Defined in `benchflow.rewards.protocol`.

Legacy path-based scoring shape; ``score(rollout_dir) -&gt; float``.

```python
bf.RewardFunc(*args, **kwargs)
```

- `.score(rollout_dir: 'Path') -> 'float'`: Async: `await` it. Score a rollout folder: a reward in [0, 1].

### `Role`

Defined in `benchflow._types`.

One agent participant in a scene.

| Field | Type | Default |
|---|---|---|
| `name` | `str` | `required` |
| `agent` | `str` | `required` |
| `model` | `str \| None` | `None` |
| `reasoning_effort` | `str \| None` | `None` |
| `env` | `dict[str, str]` | `(computed)` |
| `timeout_sec` | `int \| None` | `None` |
| `idle_timeout_sec` | `int \| None` | `None` |
| `skills_dir` | `str \| Path \| None` | `None` |
| `capabilities` | `list[str] \| None` | `None` |

### `Rollout`

Defined in `benchflow.rollout`.

Decomposed trial lifecycle with independently-callable phases.

```python
bf.Rollout(config: 'RolloutConfig') -> 'None'
```

- `.acp_client`: The agent's ACP client while connected, else None.
- `.activity_snapshot() -> 'ActivitySnapshot'`: The eval dashboard's per-task `ActivitySnapshot`: the current lifecycle phase plus the live `SessionCounters` (tool calls, last tool title, total tokens, distinct tool titles). ``counters`` is None before the agent session exists; the phase then carries the cell ("creating sandbox…", "verifying…"), so a 90s sandbox create is not indistinguishable from a hang.
- `.adopted`: Whether run()'s lifecycle began (it then owns the sandbox's clean-up).
- `.begin_prestart() -> 'asyncio.Task[None]'`: Run setup(), start() and the install now, in a task of their own.
- `.branch(n: 'int', run_child: 'ChildRunner | IdentifiedChildRunner | None' = None, *, require_sandbox_snapshot: 'bool' = False, snapshot_layers: 'frozenset[str] | set[str] | None' = None, child_labels: 'list[str | None] | None' = None, retain_snapshots: 'bool' = False, restore_parent: 'bool' = True, child_requests: 'list[str | None] | None' = None, isolate_children: 'bool' = False, concurrency: 'int' = 1, resume_session: 'bool' = False, child_retries: 'int' = 0, continue_after_child_failure: 'bool' = False, resume_session_id: 'str | None' = None, reuse_snapshot: 'Any' = None) -> 'float'`: Async: `await` it. Branch the rollout at the cursor into ``n`` child continuations.
- `.cleanup() -> 'None'`: Async: `await` it. Close ACP client and stop the environment.
- `.connect() -> 'None'`: Async: `await` it. Open an ACP connection to the agent. Can be called multiple times.
- `.connect_as(role: 'Role') -> 'None'`: Async: `await` it. Open an ACP connection for a specific role.
- `.create(config: 'RolloutConfig') -> 'Rollout'`: Async: `await` it. Create a Rollout instance. Preferred over __init__ for consistency.
- `.discard_prestart() -> 'None'`: Async: `await` it. Remove a prestarted trial the job never ran: its sandbox and folder.
- `.disconnect() -> 'None'`: Async: `await` it. Close the ACP client and clean up agent process, keeping the environment alive.
- `.env`: The rollout's sandbox, once `setup` created it.
- `.execute(prompts: 'list[str] | None' = None, *, node: 'RolloutNode | None' = None) -> 'tuple[list[dict], int]'`: Async: `await` it. Run prompts through the ACP session. Returns (new trajectory, new tool calls).
- `.finalize() -> 'RolloutResult'`: Async: `await` it. Finish a manually driven rollout, including required rubric review.
- `.install_agent() -> 'None'`: Async: `await` it. Install the primary agent binary, set up credentials, sandbox user, skills, lockdown.
- `.model_requests_in_flight() -> 'int | None'` (0.10): Model requests this rollout's proxy has in flight, when the proxy runs on this machine (the utilization watcher); else None.
- `.on_ask_user(handler: 'Any') -> 'None'`: Register the agent-initiated ``session/request_permission`` handler.
- `.on_sandbox_ready(callback: 'Callable[[], None]') -> 'None'`: Call *callback* once the trial's sandbox is started and its agent installed (the job's warm pool then prepares the next trial).
- `.prestarted`: Whether begin_prestart() ran for this rollout.
- `.record_external_tool_call(*, tool_name: 'str', event: 'dict') -> 'None'`: Record a tool call driven outside the ACP prompt loop.
- `.result`: The final result once the rollout is verified or finished, else None.
- `.run() -> 'RolloutResult'`: Async: `await` it. Run the complete trial lifecycle under a host-side hard deadline.
- `.setup(*, verifier_only: 'bool' = False) -> 'None'`: Async: `await` it. Resolve config, create environment object (not yet started).
- `.soft_verify() -> 'tuple[dict | None, str | None, str | None]'`: Async: `await` it. Run the verifier without full hardening; for intermediate feedback.
- `.start() -> 'None'`: Async: `await` it. Start the environment and upload task files.
- `.timing`: Seconds per phase and span so far (see ``timing_spans``).
- `.trajectory`: The ACP events captured so far.
- `.tree`: The RolloutTree this rollout grows as it executes.
- `.use_prebuilt_env(inner: 'Any') -> 'None'`: Inject a caller-owned sandbox; skip creation and teardown.
- `.verify() -> 'dict | None'`: Async: `await` it. Run the verifier and return rewards.

### `RolloutConfig`

Defined in `benchflow.rollout._config`.

Declarative rollout configuration.

| Field | Type | Default |
|---|---|---|
| `task_path` | `Path` | `required` |
| `scenes` | `list[Scene]` | `(computed)` |
| `environment` | `str` | `'docker'` |
| `sandbox_user` | `str \| None` | `'agent'` |
| `sandbox_locked_paths` | `list[str] \| None` | `None` |
| `sandbox_setup_timeout` | `int` | `120` |
| `skip_agent_install` | `bool` | `False` |
| `fresh_install` | `bool` | `False` |
| `allow_cloud_metadata` | `bool` | `False` |
| `services` | `list[str] \| None` | `None` |
| `job_name` | `str \| None` | `None` |
| `rollout_name` | `str \| None` | `None` |
| `jobs_dir` | `str \| Path` | `'jobs'` |
| `concurrency` | `int` | `1` |
| `context_root` | `str \| Path \| None` | `None` |
| `base_image_override` | `str \| None` | `None` |
| `pre_agent_hooks` | `list \| None` | `None` |
| `environment_manifest` | `EnvironmentManifest \| None` | `None` |
| `config_override` | `dict \| None` | `None` |
| `agent_idle_timeout` | `int \| None` | `600` |
| `agent_idle_timeout_given` (0.10) | `bool \| None` | `None` |
| `timeout` | `int \| None` | `None` |
| `prompt_suffix` | `str \| None` | `None` |
| `trial_budget` (0.10) | `TrialBudget \| None` | `None` |
| `usage_tracking` | `UsageTrackingConfig` | `(computed)` |
| `harness` | `str` | `'acp'` |
| `user` | `BaseUser \| None` | `None` |
| `max_user_rounds` | `int` | `5` |
| `oracle_access` | `bool` | `False` |
| `loop_strategy` | `LoopStrategySpec \| str \| None` | `None` |
| `uploads` | `dict[str, str]` | `(computed)` |
| `allow_document_user` | `bool \| None` | `None` |
| `agent` | `str` | `'claude-agent-acp'` |
| `prompts` | `list[str \| None] \| None` | `None` |
| `model` | `str \| None` | `None` |
| `reasoning_effort` | `str \| None` | `None` |
| `agent_env` | `dict[str, str] \| None` | `None` |
| `skills_dir` | `str \| Path \| None` | `None` |
| `skills_agent_derived` | `bool` | `False` |
| `skill_mode` | `str` | `'no-skill'` |
| `artifact_skill_mode` | `str \| None` | `None` |
| `skill_creator_dir` | `str \| Path \| None` | `None` |
| `generated_skills_root` | `str` | `'/app/generated-skills'` |
| `self_gen_no_internet` | `bool` | `False` |
| `skip_verify` | `bool` | `False` |
| `freeze_workspace` | `bool` | `False` |
| `keep_sandbox` (0.10) | `str` | `'never'` |
| `integrity` | `str` | `'off'` |
| `verifier_sandbox` | `str` | `'task'` |
| `codex_apps_policy` | `Literal['disabled', 'inherit'] \| None` | `None` |
| `separate_verifier` | `bool` | `False` |
| `separate_verifier_include` | `tuple[str, ...]` | `()` |
| `reviewer` | `ReviewerConfig` | `(computed)` |
| `purpose` | `Literal['task', 'reviewer']` | `'task'` |
| `parent_rollout` | `str \| None` | `None` |
| `checkpoints` | `CheckpointPolicy \| None` | `None` |
| `export_generated_skills_to` | `str \| Path \| None` | `None` |
| `source_provenance` | `dict[str, Any] \| None` | `None` |
| `dataset` | `dict[str, Any] \| None` | `None` |
| `task_digest` | `str \| None` | `None` |
| `planes` | `RolloutPlanes \| None` | `None` |

- `.effective_scenes`: Scenes to execute; falls back to legacy fields if scenes is empty.
- `.from_dict(raw: 'dict[str, Any]') -> 'RolloutConfig'`: Build a config from a rollout YAML mapping (``to_dict``'s shape).
- `.from_legacy(*, task_path: 'Path', agent: 'str' = 'claude-agent-acp', model: 'str | None' = None, reasoning_effort: 'str | None' = None, prompts: 'list[str | None] | None' = None, skills_dir: 'str | Path | None' = None, skill_mode: 'str' = 'no-skill', skill_creator_dir: 'str | Path | None' = None, generated_skills_root: 'str' = '/app/generated-skills', self_gen_no_internet: 'bool' = False, **kwargs: 'Any') -> 'RolloutConfig'`: Construct from flat SDK.run()-style args.
- `.from_yaml(path: 'str | Path') -> 'RolloutConfig'`: Read a rollout YAML file (same as ``bf.rollout_config_from_yaml``).
- `.loop_strategy_spec`: The parsed loop strategy; spec strings are parsed in __post_init__.
- `.primary_agent`: Agent name for the first role of the first scene.
- `.primary_model`: Model for the first role of the first scene.
- `.primary_reasoning_effort`: Reasoning effort for the first role of the first scene.
- `.recorded_skill_mode`: The skill mode written to the rollout's records.
- `.to_dict(*, include_agent_env: 'bool' = False) -> 'dict[str, Any]'`: This config as the rollout YAML mapping ``from_dict``/``from_yaml`` read.
- `.to_yaml(path: 'str | Path', *, include_agent_env: 'bool' = False) -> 'Path'`: Write `to_dict` as YAML and return the path.

### `RolloutResult`

Defined in `benchflow.models`.

Outcome of a single rollout execution.

```python
bf.RolloutResult(task_name: 'str', rollout_name: 'str' = '', rewards: 'dict[str, Any] | None' = None, trajectory: 'list[dict[str, Any]] | None' = None, agent: 'str' = '', agent_name: 'str' = '', model: 'str | None' = None, n_tool_calls: 'int' = 0, n_skill_invocations: 'int' = 0, n_prompts: 'int' = 0, n_input_tokens: 'int | None' = None, n_output_tokens: 'int | None' = None, n_cache_read_tokens: 'int | None' = None, n_cache_creation_tokens: 'int | None' = None, total_tokens: 'int | None' = None, cost_usd: 'float | None' = None, usage_source: 'UsageSource' = 'unavailable', price_source: 'str | None' = None, usage_details: 'dict[str, Any] | None' = None, error: 'str | None' = None, error_category: 'str | None' = None, verifier_error: 'str | None' = None, verifier_error_category: 'str | None' = None, export_error: 'str | None' = None, partial_trajectory: 'bool' = False, trajectory_source: 'TrajectorySource | None' = None, reward_events: 'list[RewardEvent] | None' = None, evolved_skills: 'dict[str, str] | None' = None, source_provenance: 'dict[str, Any] | None' = None, started_at: 'datetime | None' = None, finished_at: 'datetime | None' = None, scoring: 'ScoringResult | None' = None, purpose: "Literal['task', 'reviewer']" = 'task', parent_rollout: 'str | None' = None, rollout_dir: 'Path | None' = None, retry: 'dict[str, Any] | None' = None, integrity: 'dict[str, Any] | None' = None, harness_versions: 'dict[str, Any] | None' = None, model_refusal: 'dict[str, Any] | None' = None, trial_budget: 'dict[str, Any] | None' = None)
```

New arguments in 0.10: `harness_versions`, `model_refusal`, `trial_budget`.

- `.from_dict(data: 'dict[str, Any]', *, rollout_dir: 'Path | None' = None) -> 'RolloutResult'`: Build a result from a persisted ``result.json`` payload.
- `.load(path: 'str | Path') -> 'RolloutResult'`: Read a finished rollout from its directory (or its ``result.json``).
- `.messages`: Deprecated and always empty (``RuntimeResult`` never populated it).
- `.passed`: True when the rollout was scored and passed (``score_outcome == "passed"``).
- `.reward`: The canonical scalar reward; None when scoring has not completed.
- `.score_outcome`: Canonical scoring classification shared by runtime and saved reports.
- `.snapshots`: Deprecated and always empty (``RuntimeResult`` never populated it).
- `.success`: True when the trial completed without agent, verifier, or export error.
- `.to_record() -> 'dict[str, Any]'`: One flat, JSON-safe dict of the headline fields (a CSV/JSONL row).
- `.verified`: Deprecated: use ``score_outcome in ("passed", "failed")``.

### `RoundResult`

Defined in `benchflow.contracts.user`.

Outcome of one agent round, passed to ``BaseUser.run()``.

| Field | Type | Default |
|---|---|---|
| `round` | `int` | `required` |
| `trajectory` | `list[dict]` | `(computed)` |
| `rewards` | `dict[str, Any] \| None` | `None` |
| `verifier_output` | `str \| None` | `None` |
| `verifier_error` | `str \| None` | `None` |
| `n_tool_calls` | `int` | `0` |
| `scene` | `str \| None` | `None` |
| `role` | `str \| None` | `None` |
| `handoff_from` | `str \| None` | `None` |
| `handoff_to` | `str \| None` | `None` |

### `Rubric`

Defined in `benchflow.rewards.rubric`.

Weighted collection of reward functions.

| Field | Type | Default |
|---|---|---|
| `reward_funcs` | `list[RewardFunc]` | `required` |
| `weights` | `list[float] \| None` | `None` |

- `.score(rollout_dir: 'Path') -> 'VerifyResult'`: Async: `await` it. Run all reward functions and return weighted result.

### `RubricConfig`

Defined in `benchflow.rewards.rubric_config`.

Fully parsed ``rubric.toml``.

| Field | Type | Default |
|---|---|---|
| `judge` | `JudgeConfig` | `(computed)` |
| `criteria` | `list[Criterion]` | `(computed)` |
| `scoring` | `ScoringConfig` | `(computed)` |

### `Runtime`

Defined in `benchflow.runtime`.

The 0.3 execution center.

```python
bf.Runtime(env: 'Environment', agent: 'Agent', config: 'RuntimeConfig | None' = None) -> 'None'
```

- `.execute() -> 'RolloutResult'`: Async: `await` it. Run the full execution loop via Trial.

### `RuntimeConfig`

Defined in `benchflow.runtime`.

Configuration for ``bf.run(agent, env, config)`` and ``bf.run("agent", …)``.

| Field | Type | Default |
|---|---|---|
| `codex_apps_policy` | `Literal['disabled', 'inherit'] \| None` | `None` |
| `sandbox_user` | `str \| None` | `'agent'` |
| `sandbox_setup_timeout` | `int` | `120` |
| `max_rounds` | `int` | `10` |
| `snapshot_policy` | `str` | `'none'` |
| `reward_stream` | `bool` | `True` |
| `timeout` | `int \| None` | `None` |
| `jobs_dir` | `str \| Path` | `'jobs'` |
| `rollout_name` | `str \| None` | `None` |
| `skills_dir` | `str \| Path \| None` | `None` |
| `skill_mode` | `str` | `'no-skill'` |
| `context_root` | `str \| Path \| None` | `None` |
| `base_image_override` | `str \| None` | `None` |
| `separate_verifier` | `bool` | `False` |
| `separate_verifier_include` | `tuple[str, ...]` | `()` |
| `pre_agent_hooks` | `list \| None` | `None` |
| `sandbox_locked_paths` | `list[str] \| None` | `None` |
| `usage_tracking` | `Any` | `None` |
| `reviewer` | `ReviewerConfig` | `(computed)` |
| `allow_cloud_metadata` | `bool` | `False` |

### `Sandbox`

Defined in `benchflow.sandbox.protocol`.

Run-only: isolated execution environment.

```python
bf.Sandbox(*args, **kwargs)
```

- `.download_dir(src: 'str', dst: 'Path', service: 'str' = 'main') -> 'None'`: Async: `await` it. Copy the sandbox directory ``src`` to the host path ``dst``.
- `.download_file(src: 'str', dst: 'Path') -> 'None'`: Async: `await` it. Copy the sandbox file ``src`` to the host path ``dst``.
- `.exec(cmd: 'str', *, user: 'str' = 'root', timeout_sec: 'int' = 30) -> 'ExecResult'`: Async: `await` it. Run a shell command as ``user``; return its exit code and output.
- `.expose_ports`: Ports the sandbox exposes for inter-agent communication.
- `.host`: The host where the sandbox's exposed ports are reachable.
- `.restore(image: 'SandboxImage') -> 'None'`: Async: `await` it. Restore the container to a previously captured snapshot.
- `.snapshot(name: 'str | None' = None) -> 'SandboxImage'`: Async: `await` it. Capture the current container state as a re-usable image.
- `.start() -> 'None'`: Async: `await` it. Create and start the sandbox.
- `.stop(*, delete: 'bool' = True) -> 'None'`: Async: `await` it. Stop the sandbox, deleting it unless ``delete=False``.
- `.supports_snapshot`: Whether this backend implements container-level snapshot/restore.
- `.upload_dir(src: 'Path', dst: 'str', service: 'str' = 'main', *, admission: 'Any') -> 'None'`: Async: `await` it. Copy a host directory into the sandbox at ``dst`` (``admission`` as for `upload_file`).
- `.upload_file(src: 'Path', dst: 'str', *, mode: 'str | None' = None, admission: 'Any') -> 'None'`: Async: `await` it. Upload one file; ``mode`` chmods it (e.g. ``"600"`` for secrets).

### `SandboxExecResult`

Defined in `benchflow.sandbox.protocol`.

ExecResult(return_code: 'int', stdout: 'str', stderr: 'str')

| Field | Type | Default |
|---|---|---|
| `return_code` | `int` | `required` |
| `stdout` | `str` | `required` |
| `stderr` | `str` | `required` |

### `SandboxImage`

Defined in `benchflow.sandbox.protocol`.

A provider-opaque handle to a container-level checkpoint.

| Field | Type | Default |
|---|---|---|
| `provider` | `str` | `required` |
| `ref` | `str` | `required` |
| `meta` | `dict[str, str]` | `(computed)` |

- `.from_record(record: 'Mapping[str, Any]') -> 'SandboxImage'`: Read a persisted handle; legacy records have no provider metadata.
- `.to_record() -> 'dict[str, Any]'`: The complete JSON-compatible handle, including provider state.

### `SandboxSnapshotNotSupported`

Defined in `benchflow.sandbox.protocol`.

Raised when a Sandbox backend cannot satisfy ``snapshot``/``restore``.

### `Scene`

Defined in `benchflow._types`.

Authoring sugar for role/skill attribution over a sequence of turns.

| Field | Type | Default |
|---|---|---|
| `name` | `str` | `'default'` |
| `roles` | `list[Role]` | `(computed)` |
| `turns` | `list[Turn]` | `(computed)` |
| `skills_dir` | `str \| Path \| None` | `None` |

- `.single(*, agent: 'str', model: 'str | None' = None, reasoning_effort: 'str | None' = None, prompts: 'list[str | None] | None' = None, role_name: 'str' = 'agent', skills_dir: 'str | Path | None' = None) -> 'Scene'`: Shortcut for single-agent, single-role scene.

### `ScoringConfig`

Defined in `benchflow.rewards.rubric_config`.

How criteria scores are aggregated.

| Field | Type | Default |
|---|---|---|
| `aggregation` | `Literal['weighted_mean', 'all_pass', 'any_pass', 'threshold']` | `'weighted_mean'` |
| `threshold` | `float` | `0.7` |

### `SDK`

Defined in `benchflow.sdk`.

Deprecated backward-compat shim; delegates to `benchflow.rollout`.

```python
bf.SDK() -> 'None'
```

- `.run(task_path: 'str | Path', agent: 'str' = 'claude-agent-acp', prompts: 'list[str | None] | None' = None, *, model: 'str | None' = None, reasoning_effort: 'str | None' = None, agent_env: 'dict[str, str] | None' = None, job_name: 'str | None' = None, rollout_name: 'str | None' = None, trial_name: 'str | None' = None, jobs_dir: 'str | Path' = 'jobs', concurrency: 'int' = 1, agent_idle_timeout: 'int | None' = 600, environment: 'str' = 'docker', environment_manifest: 'EnvironmentManifest | None' = None, skills_dir: 'str | Path | None' = None, codex_apps_policy: "Literal['disabled', 'inherit'] | None" = None, sandbox_user: 'str | None' = 'agent', sandbox_locked_paths: 'list[str] | None' = None, sandbox_setup_timeout: 'int' = 120, pre_agent_hooks: 'list | None' = None, context_root: 'str | Path | None' = None, base_image_override: 'str | None' = None, separate_verifier: 'bool' = False, separate_verifier_include: 'Sequence[str]' = (), skill_mode: 'str' = 'no-skill', skill_creator_dir: 'str | Path | None' = None, self_gen_no_internet: 'bool' = False, source_provenance: 'dict[str, Any] | None' = None, usage_tracking: 'Any' = None, reviewer: 'ReviewerConfig | None' = None) -> 'RolloutResult'`: Async: `await` it. Run a task; delegates to `benchflow.run`.

### `SettingCheck`

Defined in `benchflow.jobs`.

One setting compared across a paired task's trials.

| Field | Type | Default |
|---|---|---|
| `setting` | `str` | `required` |
| `match` | `bool \| None` | `required` |
| `a` | `list[Any] \| None` | `None` |
| `b` | `list[Any] \| None` | `None` |

### `SettingMismatch`

Defined in `benchflow.jobs`.

A setting that differs between the sides on one paired task.

| Field | Type | Default |
|---|---|---|
| `task` | `str` | `required` |
| `setting` | `str` | `required` |
| `a` | `list[Any]` | `required` |
| `b` | `list[Any]` | `required` |

### `SkillInfo`

Defined in `benchflow.skills`.

Parsed skill metadata from SKILL.md frontmatter.

| Field | Type | Default |
|---|---|---|
| `name` | `<class 'str'>` | `required` |
| `description` | `<class 'str'>` | `''` |
| `version` | `<class 'str'>` | `''` |
| `path` | `<class 'pathlib.Path'>` | `(computed)` |
| `compatibility` | `<class 'str'>` | `''` |
| `metadata` | `<class 'dict'>` | `(computed)` |

### `StreamedRollout`

Defined in `benchflow.trajectories.rollout_stream`.

One finished rollout, as a trainer reads it (``benchflow.rollout-stream.v1``).

| Field | Type | Default |
|---|---|---|
| `job` | `str` | `required` |
| `rollout` | `str` | `required` |
| `rollout_path` | `str` | `required` |
| `task` | `str \| None` | `required` |
| `agent` | `str \| None` | `required` |
| `model` | `str \| None` | `required` |
| `group_id` | `str` | `required` |
| `group` | `dict[str, Any]` | `required` |
| `reward` | `float \| None` | `required` |
| `outcome` | `str` | `required` |
| `error` | `str \| None` | `required` |
| `finished_at` | `str \| None` | `required` |
| `token_capture` | `dict[str, Any]` | `required` |
| `calls` | `list[dict[str, Any]]` | `required` |
| `sequences` | `list[dict[str, Any]]` | `required` |
| `advantage` | `float \| None` | `None` |
| `group_complete` | `bool \| None` | `None` |
| `withheld` (0.10) | `str \| None` | `None` |
| `rollout_dir` | `Path` | `PosixPath('.')` |
| `replaces` | `str \| None` | `None` |
| `retried` | `bool` | `False` |

- `.scored`: Whether the rollout has a reward.
- `.to_json() -> 'str'`: The record as one compact JSON line.
- `.to_json_dict() -> 'dict[str, Any]'`: The record as a ``benchflow.rollout-stream.v1`` mapping.
- `.training_grade`: Whether every model call was captured token-in/token-out.

### `StringMatchRewardFunc`

Defined in `benchflow.rewards.builtins`.

Exact or fuzzy string matching against ``answer.txt``.

```python
bf.StringMatchRewardFunc(expected: 'str', fuzzy: 'bool' = False) -> 'None'
```

- `.score(rollout_dir: 'Path') -> 'float'`: Async: `await` it. 1.0 when ``answer.txt`` matches ``expected`` (substring when fuzzy), else 0.0.

### `Task`

Defined in `benchflow.task.task`.

Represents a task with the standard directory structure.

```python
bf.Task(task_dir: 'Path | str') -> 'None'
```

- `.task_dir`: The task's folder.

### `TaskConfig`

Defined in `benchflow.task.config`.

Full task configuration; the task specification ($T$).

| Field | Type | Default |
|---|---|---|
| `schema_version` | `str` | `'1.3'` |
| `task` | `benchflow.task.config.PackageInfo \| None` | `None` |
| `metadata` | `dict` | `(computed)` |
| `verifier` | `VerifierConfig` | `(computed)` |
| `agent` | `AgentConfig` | `(computed)` |
| `sandbox` | `SandboxConfig` | `(computed)` |
| `solution` | `SolutionConfig` | `(computed)` |
| `source` | `str \| None` | `None` |
| `reward` | `dict` | `(computed)` |
| `multi_step_reward_strategy` | `benchflow.task.config.MultiStepRewardStrategy \| None` | `None` |
| `steps` | `list[benchflow.task.config.StepConfig] \| None` | `None` |
| `artifacts` | `list` | `(computed)` |
| `preference` (0.10) | `dict` | `(computed)` |

- `.allows_cloud_metadata`: Whether the task opts out of refusing the cloud metadata endpoints (``allow_cloud_metadata = true`` in its ``[environment]`` table).
- `.cloud_metadata_opt_out`: How the task opts out of the cloud metadata block, for the record; empty when it does not. ``metadata.x-allow-cloud-metadata = true`` is the older, unreleased spelling of the same opt-out.
- `.expected_skills`: Hidden Memory-space fixture from ``[verifier.memory]``.
- `.handle_version_rename(data: 'Any') -> 'Any'`: Refuse the removed ``environment`` spelling and conflicting solution keys.
- `.ignored_keys`: task.toml keys this config does not model and a lenient load ignored (empty for strict loads). Keys BenchFlow knows it cannot honour are refused by the runtime capability check instead of being run.
- `.model_dump_toml() -> 'str'`: Write the config as ``task.toml`` text.
- `.model_post_init(context: 'Any', /) -> 'None'`: This function is meant to behave like a BaseModel method to initialize private attributes.
- `.model_validate_toml(toml_data: 'str') -> 'TaskConfig'`: Read a legacy ``task.toml`` configuration.
- `.validate_cloud_metadata_opt_out(value: 'dict[str, Any]') -> 'dict[str, Any]'`: Refuse an opt-out that is not a boolean, so a typo cannot leave the author believing the agent can (or cannot) reach the endpoints.
- `.validate_keep_services(value: 'dict[str, Any]') -> 'dict[str, Any]'`: Refuse a ``metadata.x-keep-services`` that is not a list of at most 8 ``&#123; port, name &#125;`` tables, naming every problem at its place, so a typo cannot leave the author believing a service is kept.
- `.validate_preference(value: 'dict[str, Any]') -> 'dict[str, Any]'` (0.10): The keys `bench preferences` reads must have their types; the rest is recorded as given (as in a task.md package's [preference]).
- `.validate_schema_version(value: 'str') -> 'str'`: Reject unknown/unparseable schema majors; stay permissive on minor.

### `TaskDocument`

Defined in `benchflow.task._document_parse`.

Parsed ``task.md`` document.

| Field | Type | Default |
|---|---|---|
| `frontmatter` | `dict[str, Any]` | `required` |
| `body` | `str` | `required` |
| `instruction` | `str` | `required` |
| `config` | `TaskConfig` | `required` |
| `roles` | `dict[str, Role]` | `required` |
| `scenes` | `list[Scene]` | `required` |
| `role_prompts` | `dict[str, str]` | `required` |
| `scene_prompts` | `dict[str, str]` | `required` |
| `user` | `dict[str, Any]` | `required` |
| `user_persona` | `str \| None` | `required` |
| `benchflow` | `dict[str, Any]` | `required` |
| `path` | `Path \| None` | `None` |

- `.from_path(path: 'str | Path') -> 'TaskDocument'`: Read a ``task.md`` file.
- `.from_text(text: 'str', *, path: 'str | Path | None' = None) -> 'TaskDocument'`: Parse ``task.md`` text (``path`` resolves relative references).

### `TaskDocumentParseError`

Defined in `benchflow.task._document_normalize`.

Raised when a ``task.md`` document cannot be parsed.

### `TaskRuntime`

Defined in `benchflow.rollout.task_runtime`.

Run one BenchFlow-compatible task as a reusable sandbox primitive.

```python
bf.TaskRuntime(config: 'TaskRuntimeConfig') -> 'None'
```

- `.bash(command: 'str', *, timeout_sec: 'int' = 30, user: 'str | None' = None) -> 'BashToolResult'`: Async: `await` it. Execute a bash-like command in the task workspace.
- `.close() -> 'None'`: Async: `await` it. Clean up the sandbox lifecycle owned by this runtime.
- `.create(config: 'TaskRuntimeConfig') -> 'TaskRuntime'`: Async: `await` it. Create and start a task runtime session.
- `.env`: The task's sandbox.
- `.rollout`: The underlying rollout; raises until `start` ran.
- `.rollout_dir`: The rollout's artifact folder on the host.
- `.start() -> 'None'`: Async: `await` it. Set up and start the sandbox without launching an ACP agent.
- `.verify() -> 'TaskRuntimeResult'`: Async: `await` it. Finish task scoring, including automatic review when a rubric exists.
- `.workspace`: The agent's working directory in the sandbox.

### `TaskRuntimeConfig`

Defined in `benchflow.rollout.task_runtime`.

Configuration for a BenchFlow task runtime session.

| Field | Type | Default |
|---|---|---|
| `task_path` | `str \| Path` | `required` |
| `environment` | `str` | `'docker'` |
| `codex_apps_policy` | `Literal['disabled', 'inherit'] \| None` | `None` |
| `sandbox_user` | `str \| None` | `'agent'` |
| `sandbox_locked_paths` | `list[str] \| None` | `None` |
| `sandbox_setup_timeout` | `int` | `120` |
| `job_name` | `str \| None` | `None` |
| `rollout_name` | `str \| None` | `None` |
| `jobs_dir` | `str \| Path` | `'jobs'` |
| `context_root` | `str \| Path \| None` | `None` |
| `base_image_override` | `str \| None` | `None` |
| `separate_verifier` | `bool` | `False` |
| `separate_verifier_include` | `tuple[str, ...]` | `()` |
| `pre_agent_hooks` | `list \| None` | `None` |
| `environment_manifest` | `EnvironmentManifest \| None` | `None` |
| `config_override` | `dict \| None` | `None` |
| `skills_dir` | `str \| Path \| None` | `None` |
| `skill_mode` | `str` | `'no-skill'` |
| `runtime_label` | `str` | `'task-runtime'` |
| `reviewer` | `ReviewerConfig` | `(computed)` |
| `planes` | `Any \| None` | `None` |
| `integrity` | `str` | `'off'` |
| `verifier_sandbox` | `str` | `'task'` |

- `.to_rollout_config() -> 'RolloutConfig'`: Lower to the Rollout configuration used to own lifecycle/artifacts.

### `TaskRuntimeResult`

Defined in `benchflow.rollout.task_runtime`.

Verifier result plus the normal BenchFlow artifact location.

| Field | Type | Default |
|---|---|---|
| `task_name` | `str` | `required` |
| `rollout_name` | `str` | `required` |
| `reward` | `float \| None` | `required` |
| `rewards` | `dict \| None` | `required` |
| `verifier_error` | `str \| None` | `required` |
| `error` | `str \| None` | `required` |
| `rollout_dir` | `Path` | `required` |
| `result` | `RolloutResult` | `required` |
| `integrity` | `Any` | `None` |

### `TestRewardFunc`

Defined in `benchflow.rewards.builtins`.

Wraps existing test.sh -&gt; reward.txt flow. Backward compatible.

```python
bf.TestRewardFunc()
```

- `.score(rollout_dir: 'Path') -> 'float'`: Async: `await` it. The reward the verifier wrote to ``reward.txt`` (0.0 when absent or empty).

### `Trajectory`

Defined in `benchflow.trajectories.types`.

Raw trajectory: ordered list of captured LLM API exchanges.

| Field | Type | Default |
|---|---|---|
| `session_id` | `str` | `required` |
| `agent_name` | `str` | `''` |
| `started_at` | `datetime` | `(computed)` |
| `finished_at` | `datetime.datetime \| None` | `None` |
| `exchanges` | `list` | `(computed)` |
| `metadata` | `dict` | `(computed)` |

- `.has_provider_usage`: Whether any exchange contains provider-supplied usage fields.
- `.messages`: Extract all messages from all exchanges (the conversation history).
- `.model_post_init(context: 'Any', /) -> 'None'` (0.10): This function is meant to behave like a BaseModel method to initialize private attributes.
- `.to_jsonl(*, redact_keys: bool = True) -> str`: Export as JSONL (one exchange per line).
- `.total_cache_creation_tokens`: Prompt-cache write tokens over the captured exchanges.
- `.total_cache_read_tokens`: Prompt-cache read tokens over the captured exchanges.
- `.total_cost_usd`: Extract cost if the API returns it (Anthropic does not, OpenAI does not).
- `.total_input_tokens`: Input tokens over the captured exchanges.
- `.total_output_tokens`: Output tokens over the captured exchanges.
- `.total_provider_tokens`: All tokens over the captured exchanges.

### `Trial`

Defined in `benchflow.jobs`.

One finished trial (rollout) directory.

| Field | Type | Default |
|---|---|---|
| `path` | `Path` | `required` |
| `result` | `RolloutResult` | `required` |
| `raw` | `dict[str, Any]` | `required` |
| `config` | `dict[str, Any]` | `(computed)` |
| `timing` | `dict[str, Any]` | `(computed)` |
| `verifier` | `VerifierOutput` | `(computed)` |
| `forks` | `list[Fork]` | `(computed)` |
| `checkpoints` | `list[dict[str, Any]]` | `(computed)` |
| `source` | `Literal['result.json', 'results.jsonl']` | `'result.json'` |
| `row` | `int \| None` | `None` |

- `.agent`: The harness that ran (``result.agent``).
- `.assessment`: Whether it got a score: scored, error (the verifier failed) or unscored.
- `.attempts`: List this trial's rollouts in the order they ran; the list includes this one.
- `.control`: ``oracle`` or ``empty`` for a control run, None for an agent run.
- `.cost_usd`: USD of this rollout, or None (see ``result.price_source``).
- `.duration_sec`: Wall-clock seconds from start to finish, when both were recorded.
- `.execution`: How the run ended: completed, errored, timed_out or integration_failed.
- `.model`: The model the agent used, or None.
- `.passed`: Whether the trial passed (and its agent integration did not break).
- `.reward`: The reward when the trial is scored; None when unscored (never 0 for a failure to score).
- `.scoring`: The automatic reviewer's gate verdict (``ScoringResult``), or None.
- `.settings() -> 'dict[str, Any]'`: The run settings ``bf.compare`` checks between two sides.
- `.solve_sample() -> 'Sample'`: This trial as one pass@k sample: the reward when scored (else None, unscored), and the review gate's verdict when it has one.
- `.task_name`: The task this trial ran.
- `.to_json(path: 'str | Path | None' = None, *, indent: 'int' = 2) -> 'Any'`: `to_json_dict` as a JSON string, or written to ``path`` (returned).
- `.to_json_dict(*, include_trajectory: 'bool' = True, include_verifier: 'bool' = True) -> 'dict[str, Any]'`: The trial as a ``benchflow.trial`` document (see ``benchflow.job_export``).
- `.to_record() -> 'dict[str, Any]'`: ``RolloutResult.to_record()`` plus control, execution, assessment and the settings ``bf.compare`` checks (flattened; ``model`` and ``agent`` already appear once).
- `.total_tokens`: Tokens of this rollout, or None when unknown.

### `TrialBudget` (0.10)

Defined in `benchflow.trial_budget`.

A per-trial budget: ``tokens``, ``turns`` or both (see the module).

| Field | Type | Default |
|---|---|---|
| `tokens` | `int \| None` | `None` |
| `turns` | `int \| None` | `None` |

- `.coerce(value: 'TrialBudget | Mapping[str, Any] | None') -> 'TrialBudget | None'`: A budget from itself, a mapping (`to_dict`'s shape) or None; a mapping with neither measure set is no budget.
- `.proxy_env(used: 'Mapping[str, int] | None' = None) -> 'dict[str, str]'`: The variables the trial's model proxy reads its budget from, and what earlier proxies of the trial counted (``used``).
- `.text() -> 'str'`: The budget in words: ``200,000 tokens and 50 turns``.
- `.to_dict() -> 'dict[str, int | None]'`

### `Turn`

Defined in `benchflow._types`.

One prompt in a scene. *role* selects which Role acts.

| Field | Type | Default |
|---|---|---|
| `role` | `str` | `required` |
| `prompt` | `str \| None` | `None` |

### `UsageLimitError`

Defined in `benchflow.agents.errors`.

The login the agent runs on has used up its subscription usage.

- `.describe() -> 'str'`: ``usage limit reached on login L: 7-day window, resets 2026-10-02 16:00 UTC (&lt;detail&gt;)``.
- `.from_headers(headers: 'Mapping[str, Any] | None', *, detail: 'str | None' = None) -> 'UsageLimitError | None'`: The error a Messages API answer's unified rate-limit headers describe.
- `.from_rate_limit_info(info: 'Mapping[str, Any] | None', *, detail: 'str') -> 'UsageLimitError | None'`: The error a rejected rate-limit record describes, or None.
- `.from_result(result: 'Any') -> 'UsageLimitError | None'`: The usage limit a finished trial ended on, or None.
- `.from_text(text: 'str | None', *, now: 'datetime | None' = None) -> 'UsageLimitError | None'`: The error a usage-limit message describes, or None when it is not one.
- `.to_dict() -> 'dict[str, Any]'`: The fields a result records (no secret: the login is a label).
- `.with_login(login: 'str | None') -> 'UsageLimitError'`: A copy naming ``login`` (the rollout knows it; the ACP client does not).

### `Verifier`

Defined in `benchflow.task.verifier_core`.

Runs the task's verifier and parses rewards.

```python
bf.Verifier(task: 'Any', rollout_paths: 'RolloutPaths', sandbox: 'Any', _logger: 'logging.Logger | None' = None, *, execution_receipt: 'bool' = False) -> 'None'
```

- `.finish_test_command(return_code: 'int') -> 'VerifierResult'`: Async: `await` it. Score a test command that ended although its exec never returned.
- `.save_timed_out_outputs() -> 'None'` (0.10): Async: `await` it. Keep what a test command still running at its deadline wrote.
- `.test_command_finishable`: Whether `finish_test_command` can score the issued test command: a script verifier whose command records its end.
- `.test_command_savable` (0.10): Whether `save_timed_out_outputs` has an issued test command whose outputs it can keep.
- `.verify() -> 'VerifierResult'`: Async: `await` it. Run the configured verifier and return the reward result.

### `VerifierOutput`

Defined in `benchflow.jobs`.

What the verifier left in ``verifier/`` (None for a missing file).

| Field | Type | Default |
|---|---|---|
| `reward_text` | `str \| None` | `None` |
| `reward_json` | `Any` | `None` |
| `stdout` | `str \| None` | `None` |
| `stderr` | `str \| None` | `None` |
| `ctrf` | `Any` | `None` |

- `.read(verifier_dir: 'Path | ReachablePath') -> 'VerifierOutput'`: Read ``verifier/`` (reward, stdout, stderr, test report); missing files are None.

### `VerifierResult`

Defined in `benchflow.task.verifier_errors`.

Result from the verifier; reward dict.

| Field | Type | Default |
|---|---|---|
| `rewards` | `dict[str, typing.Any] \| None` | `None` |

### `VerifyResult`

Defined in `benchflow.rewards.protocol`.

Aggregated result from a Rubric or node evaluation.

| Field | Type | Default |
|---|---|---|
| `reward` | `float` | `required` |
| `items` | `dict[str, float]` | `(computed)` |
| `events` | `list[RewardEvent]` | `(computed)` |
| `error` | `str \| None` | `None` |
| `space` | `Space` | `'output'` |
| `granularity` | `Granularity` | `'terminal'` |

## Values

### `__version__`

`bf.__version__`: `'0.10.0rc3'`

### `AGENTS`

`bf.AGENTS`: `dict`

### `SERVICES`

`bf.SERVICES`: `dict`

### `TASK_DOCUMENT_FILENAME`

`bf.TASK_DOCUMENT_FILENAME`: `'task.md'`
