# Python API
The Rollout/Scene API is the primary way to run agent benchmarks programmatically.

## Install

For a Python program, add BenchFlow to that program's environment (Python 3.12 or newer), with the Daytona extra if you run on Daytona:

```bash
uv add benchflow                     # or: uv add 'benchflow[sandbox-daytona]'
```

The CLI install (`uv tool install --python 3.12 --upgrade benchflow`) is isolated and cannot be imported from your code.

## Quick Start

Run one task with the oracle agent (the task's own solution, no model credentials needed) and read the result. `bf.run_sync` blocks until the rollout finishes; it also works inside a running event loop such as a Jupyter cell:

```python
import benchflow as bf

result = bf.run_sync(bf.RolloutConfig(task_path="tasks/my-task", agent="oracle", environment="docker"))
print(result.reward, result.passed, result.rollout_dir)
```

In async code, `await bf.arun(...)` takes the same arguments (`bf.run` is the same async function under its older name, so existing `await bf.run(...)` code keeps working).

Then a real agent. Credentials come from the environment, as for the CLI (`bench doctor` shows which one each agent uses); for Claude, `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY`:

```python
result = await bf.arun(
    bf.RolloutConfig(
        task_path="tasks/my-task",
        agent="claude-agent-acp",
        model="claude-haiku-4-5",
        environment="daytona",       # or "docker"
        jobs_dir="jobs/my-run",      # default "jobs"
    )
)
```

`bf.arun("claude-agent-acp", task_path=..., model=..., env="daytona")` is a shorter form of the same call. Every form uses the task's own agent timeout (`[agent] timeout_sec`) unless you pass one (`RolloutConfig(timeout=...)` or `RuntimeConfig(timeout=...)`). Before anything starts, the entry points check the request: a missing task directory raises `FileNotFoundError`; an unknown sandbox or a misspelt agent (`"claud-agent-acp"`) raises `ValueError` with a suggestion; Docker not running for `environment="docker"` raises `RuntimeError` with `bench doctor`'s fix; and a Claude agent that would fall back to an expired `~/.claude/.credentials.json` gets a `UserWarning` with the fix. `BENCHFLOW_SKIP_PREFLIGHT=1` skips the host checks, as for `bench eval run`. `Evaluation.run()` (and `stream()`, `run_sync()`) makes the same checks before the job directory exists; `bf.Evaluation(..., preflight=False)` skips them. BenchFlow reports progress through the standard `logging` module, so call `logging.basicConfig(level=logging.INFO)` to see it.

Runnable versions of every snippet on this page are in [`docs/examples/python-sdk/`](../examples/python-sdk/README.md), whose README is the gallery index; `quickstart.py` there is a tour in `# %%` cells that editors run like a notebook.

## Results

Every form (`bf.arun`/`bf.run`/`bf.run_sync` with a `RolloutConfig`, an agent name, or `bf.Agent` + `bf.Environment`, and `Rollout.run()`) returns a `RolloutResult`. The artifacts are also on disk under `result.rollout_dir` (`result.json`, `trajectory/acp_trajectory.jsonl`, `verifier/`, `timing.json`, `config.json`).

| Attribute | Meaning |
|---|---|
| `reward` | `rewards["reward"]` as a float, or `None` when the rollout was not scored. |
| `rewards` | The verifier's full reward dict, e.g. `{"reward": 1.0}`, or `None`. |
| `passed` | `True` when the scoring outcome is a pass. |
| `score_outcome` | `"passed"`, `"failed"`, `"errored"` or `"verifier_errored"`. |
| `success` | `True` when there was no agent, verifier or export error (a scored 0 is still a success). |
| `error`, `error_category` | Agent-side error text and a stable category (`provider_auth`, `timeout`, `sandbox_setup`, ...), or `None`. |
| `verifier_error`, `verifier_error_category` | Verifier-side error, kept separate from agent errors. |
| `trajectory` | The ACP events as a list of dicts with a `type` key: `user_message`, `agent_message`, `agent_thought`, `tool_call` (with `title`, `kind`, `status`, `raw_input`, `raw_output`). |
| `trajectory_source` | `"acp"` (captured over ACP), `"partial_acp"`, `"scraped"` (agent-writable, untrusted) or `"hosted_env"`. |
| `n_tool_calls`, `n_prompts` | Counts from the run. |
| `n_input_tokens`, `n_output_tokens`, `total_tokens`, `cost_usd`, `usage_source` | Provider usage; `None` when the provider reported none (`usage_source == "unavailable"`). |
| `price_source` | Who priced `cost_usd`: `"litellm"` (BenchFlow's model gateway), or `"agent_session_log"`, an estimate: a Claude subscription run bypasses the gateway, so its USD is Claude Code's own figure from its session log (copied to `agent/claude-sessions/`), or the logged usage at list prices; `usage_details["cost_estimate"]` says which. `None` when nothing priced it. |
| `task_name`, `rollout_name`, `agent`, `model` | Identity of the run. |
| `started_at`, `finished_at` | Local wall-clock times (naive `datetime`). |
| `rollout_dir` | The rollout's artifact directory, or `None` if the run failed before it was created. |

Read a finished rollout back later, for example from a notebook:

```python
result = bf.RolloutResult.load("jobs/my-run/2026-01-01__12-00-00/my-task__1a2b3c4d")
```

`result.to_record()` gives one flat, JSON-safe dict of the headline fields (the row the CSV and JSONL exports below write).

## Many rollouts

Run a list of configs (several agents or models on one task, say) with bounded concurrency. `bf.run_batch` / `await bf.arun_batch` return a `Results` list in input order; `bf.as_completed` yields each rollout as it finishes:

```python
import contextlib
import benchflow as bf

configs = [
    bf.RolloutConfig(task_path="tasks/my-task", agent=agent, model=model, environment="daytona")
    for agent, model in [("claude-agent-acp", "claude-haiku-4-5"), ("oracle", None)]
]
results = bf.run_batch(configs, concurrency=4, on_result=lambda done: print(done.index, done.result))
print(results.n_passed, results.mean_reward)
results.to_csv("results.csv")          # also to_jsonl(), to_records()

# in async code:
async with contextlib.aclosing(bf.as_completed(configs, concurrency=4)) as stream:
    async for done in stream:          # Completed(index, config, result)
        print(done.config.primary_agent, done.result.reward)
```

A rollout that raises comes back with `error` set instead of stopping the batch. Leaving `as_completed` early cancels the rollouts still running (`contextlib.aclosing` makes that immediate).

## Batches over a task directory

`Evaluation` runs every task under a directory with concurrency, retries and resume, and writes `summary.json`:

```python
evaluation = bf.Evaluation(
    tasks_dir="tasks",
    jobs_dir="jobs/my-batch",
    config=bf.EvaluationConfig(agent="oracle", environment="daytona", concurrency=4),
)
job = evaluation.run_sync()             # or: job = await evaluation.run()
print(job.passed, job.total, job.score, job.mean_reward, job.job_dir)
for name, result in job.results.items():   # task name -> RolloutResult
    print(name, result.reward, result.rollout_dir)
job.to_csv("job.csv")                   # also to_jsonl(), to_records()
```

Watch results arrive with `stream()`; the `EvaluationResult` is left in `evaluation.result`:

```python
async for name, result in evaluation.stream():
    print(name, result.reward)
print(evaluation.result.score)
```

To run Python in every trial's sandbox before its agent starts, give the job hooks, `async def hook(sandbox)`, as for a single rollout (`RolloutConfig.pre_agent_hooks`); to keep more files from every trial, add artifacts through the config overlay, collected besides each task's own:

```python
async def lock_inputs(sandbox):
    await sandbox.exec("chmod -R a-w /data", user="root", timeout_sec=60)

config = bf.EvaluationConfig(
    agent="claude-agent-acp",
    pre_agent_hooks=[lock_inputs],
    config_override={"artifacts": [{"source": "/home/agent/.claude/projects",
                                    "destination": "claude-sessions"}]},
)
```

Hooks are Python objects: a config file cannot hold them (`to_dict`/`to_yaml` refuse), `evaluation.json` records their names, and `Evaluation.resume(job_dir, pre_agent_hooks=[...])` takes them again.

A job records its tasks directory and config in `<job_dir>/evaluation.json` when it starts (the names of `agent_env` keys, never their values). To finish an interrupted job from its directory alone:

```python
job = bf.Evaluation.resume("jobs/my-batch/2026-01-01__12-00-00", agent_env={...}).run_sync()
```

A job holds `<job_dir>/.evaluation.lock` while it runs, so resuming a job that is still running is refused with the holder's process id. Finished tasks are read back from disk (they appear in `job.results`) and only the rest run; keyword overrides such as `concurrency=8` replace config fields. `on_result=lambda name, result: ...` on the constructor is still called as each task finishes.

## Branching

`bf.branch` (blocking) and `await bf.abranch` run what `bench eval branch` runs, with the same driver and job folder: the task runs up to a checkpoint, the sandbox is snapshotted, each child starts from the snapshot with its own prompt and is scored by the task's verifier, then the parent is restored, finished and verified.

```python
import benchflow as bf

result = bf.branch(
    "tests/examples/hello-world-task",
    agent="claude-agent-acp",
    model="claude-haiku-4-5",
    sandbox="daytona",
    prompts=["Create draft.txt containing: Hello world", "@instruction"],
    checkpoint_after=1,                  # fork after the first prompt
    children={
        "baseline": None,                # the parent's remaining prompts
        "hint": "Rename draft.txt to hello.txt and stop.",
    },
)
print(result.value)                      # V(checkpoint): mean child reward
for child in result.children:            # BranchChildResult
    print(child.label, child.status, child.reward, child.reward_source)
print(result.parent_reward, result.parent.rollout_dir, result.job_dir)
result.to_csv("children.csv")
```

Children start with a fresh agent session and only the files (and, with `snapshot_layers={"sandbox", "environment"}`, declared environment state) restored, so each child prompt must stand on its own; `resume_session=True` resumes the parent's conversation instead. Other keywords mirror the CLI flags: `parent="discard"`, `concurrency=K` / `isolate_children=True` (each child in its own sandbox), nested forks through `children=[bf.ChildSpec(label=..., prompt=..., parent="baseline"), ...]` or `"label=...,parent=...,prompt=..."` strings, `retain_snapshots=True` and `from_checkpoint=<trial dir>` (branch again from a kept snapshot; `checkpoint="prompt:1"` picks an automatic checkpoint or a fork id, and `task_path`, `agent` and `model` default to that trial's), `checkpoints=`. Each child has `advantage` (its reward minus its fork's V). `on_event=callback` receives progress dicts while the run goes (`event` is `trial_started`, `checkpoint_reached`, `child_finished` with `label` and `reward`, `fork_finished`, `parent_finished` or `trial_finished`), e.g. `on_event=print`. An invalid request raises `bf.BranchPlanError` (a `ValueError`, naming the Python keywords) before anything starts, and the host checks of `bf.run` apply (Docker not running for `sandbox="docker"` raises `RuntimeError`); a trial that fails later comes back with `result.error` set and `result.ok` false. A runnable version is [`docs/examples/python-sdk/run-branch.py`](../examples/python-sdk/run-branch.py); the manual `Rollout.branch()` lifecycle is below.

## Reading finished jobs

A walk-through with a worked example is in [Analysing runs](../analysing-runs.md).

`bf.load_job(path)` reads every trial under a job folder (a folder of jobs, one trial, or a list of folders also work) into typed objects, and `bf.load_trial(path)` reads one trial. They read the layouts BenchFlow writes today and older ones (a top-level `reward`, `trial_name`, `agent/acp_trajectory.jsonl`).

```python
import benchflow as bf

job = bf.load_job("jobs/my-batch/2026-01-01__12-00-00")
for trial in job.trials:                 # Trial
    print(trial.task_name, trial.reward, trial.execution, trial.assessment,
          trial.control, trial.cost_usd, trial.verifier.stdout)
    for fork in trial.forks:             # branch lineage from tree.json
        print("  V =", fork.value, [(c.label, c.reward) for c in fork.children])
d = job.denominators()
print(d.attempted, d.scored, d.assessment_errors, d.unscored, d.passed,
      d.pass_rate_scored, d.pass_rate_attempted, d.controls_excluded)
job.to_csv("trials.csv")

report = bf.compare("jobs/run-a", "jobs/run-b", labels=("a", "b"))
print(report.to_markdown())              # per-task rewards and deltas, with caveats
report.to_json("comparison.json")        # the versioned JSON document
```

`compare` also checks that both sides ran with comparable settings (task digest, model, harness, dataset, reasoning effort, sandbox, sandbox user, timeout, agent variables, prompts). Name the settings the comparison is about with `vary=("model",)`; any other difference warns (`on_mismatch="raise"` refuses, `"ignore"` stays quiet) and is listed in `report.mismatches`. `trial.review` is the `bench review` rubric verdict when one exists, and a folder with only `results.jsonl` rows loads too (`trial.source == "results.jsonl"`). `to_json_dict()` / `to_json()` on a trial, job or comparison give the versioned documents in [JSON export](./json-export.md); `bench eval inspect` and `bench eval compare` are the same functions on the command line.

Counting follows the viewer. `execution` is `completed`, `errored` or `timed_out`; `assessment` is `scored`, `error` (the verifier failed) or `unscored`; a timed-out run can still be scored. Control runs (the oracle, an empty run with `agent="nop"`, which runs nothing so the verifier scores the untouched workspace, and task copies suffixed `__o` (oracle) or `__e` (empty solution)) check the task, not an agent, so `denominators()` and `compare()` leave them out unless `include_controls=True`. A task that an Evaluation job retried keeps its best attempt (scored first, then newest) unless `attempts="all"`, and `trial.attempts` lists every attempt, oldest first (so `sum(len(t.attempts) for t in job.trials)` counts the rollouts a job ran); repeated rollouts of a task in a `bf.run_batch` folder are separate trials, never collapsed. Branch children are part of their parent trial's `forks`, not extra trials. `compare` pairs tasks by name, gives means over each side's scored runs, and adds an n = 1 caveat when each side has one run per task; it computes no significance.

`print(job)` (or `job.to_markdown()`, which a notebook shows for a bare `job`) summarises it: trials, tasks and agents, the solve rate with its 95% interval, unscored trials by reason, control runs left out, and what the rollouts cost (retried attempts included; estimates from an agent's session log marked). `repr(job)` stays one line.

`job.solve_rates(ks=None, solve_threshold=None)` gives pass@k, pass^k and the solve rate (with its 95% `interval`) over repeated trials (a task's trials in separate job folders are separate samples; unscored trials and controls are left out; a task with fewer than k scored trials is left out of that k and named in the caveats); `compare(..., ks=, solve_threshold=)` computes them for both sides (`report.solve_rates_a`, `report.solve_rates_b`). See [pass@k, pass^k and solve rates](./pass-at-k.md).

## Saving configs

A job or rollout built in Python can be saved for the CLI and read back:

```python
evaluation.to_yaml("job.yaml")                  # bench eval run --config job.yaml
same = bf.Evaluation.from_yaml("job.yaml")      # or from_dict(evaluation.to_dict())
config.to_yaml("rollout.yaml")                  # a RolloutConfig
config = bf.RolloutConfig.from_yaml("rollout.yaml")
```

`agent_env` values (the agent's and the reviewer's) are written only with `include_agent_env=True`; otherwise their names are listed under `agent_env_keys`. A `RolloutConfig` whose `user`, `pre_agent_hooks` or `planes` hold Python objects cannot be written. [CLI and Python equivalents](./cli-python-parity.md) lists every `bench eval run` and `bench eval branch` flag with its Python equivalent (a test keeps the table in step with the CLI).

## Environment manifests

An [environment manifest](../environment-plane.md) declares the world a task runs in: the image, services BenchFlow starts, readiness probes and restorable state. Pass it on the rollout or the batch config:

```python
manifest = bf.load_manifest("environment.toml")   # or bf.EnvironmentManifest.model_validate_toml(text)
result = await bf.run(
    bf.RolloutConfig(task_path="tasks/my-task", agent="oracle", environment="daytona",
                     environment_manifest=manifest)
)
```

`bf.Environment` is a different thing: a handle on one task's sandbox, used with `bf.Agent` and `bf.Runtime`.

## Deprecated names

| Deprecated | Use instead |
|---|---|
| `benchflow.SDK().run(task_path=..., agent=..., ...)` | `bf.run(bf.RolloutConfig(task_path=..., agent=..., ...))` (same keyword arguments) |
| `bf.snapshot`, `bf.restore`, `bf.list_snapshots` | `bf.workspace_snapshot`, `bf.workspace_restore`, `bf.list_workspace_snapshots` |
| `RuntimeConfig(max_rounds=..., snapshot_policy=..., reward_stream=...)` | Nothing reads these fields; use `RolloutConfig.max_user_rounds` for user loops. |
| `SDK.run(trial_name=...)` | `rollout_name=...` |
| `RuntimeConfig()`'s 900 s agent timeout | Removed: every form now uses the task's own timeout; pass `RuntimeConfig(timeout=900)` to keep it (the `Agent + Environment` form warns with `FutureWarning` when the task's timeout differs) |
| `RuntimeResult` (and `result.verified`, `result.messages`, `result.snapshots`) | `RolloutResult`; `result.score_outcome in ("passed", "failed")`; `messages` and `snapshots` were always empty (read `agent_message` events in `result.trajectory`) |

Each emits a `DeprecationWarning` and keeps working.

## Core Types

### RolloutConfig

Declarative configuration for a rollout — a sequence of Scenes in a shared sandbox.

```python
from pathlib import Path
from benchflow import RolloutConfig, Scene, Role, Turn

# Single-agent (simplest)
config = RolloutConfig(
    task_path=Path("tasks/my-task"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")],
    environment="daytona",
    sandbox_setup_timeout=120,
)

# Multi-scene BYOS (skill-gen → solve)
config = RolloutConfig(
    task_path=Path("tasks/my-task"),
    scenes=[
        Scene(name="prep", roles=[Role("gen", "gemini", "gemini-3.1-flash-lite-preview")],
              turns=[Turn("gen", "Generate a skill for this task...")]),
        Scene(name="solve", roles=[Role("solver", "gemini", "gemini-3.1-flash-lite-preview")],
              turns=[Turn("solver")]),
    ],
    environment="daytona",
    sandbox_setup_timeout=120,
)
```

Set `sandbox_setup_timeout` when sandbox user setup needs more than the default 120 seconds.
The same field is also available on `EvaluationConfig` and `RuntimeConfig`.

### Scene

Authoring sugar for role, prompt, and skill attribution. Scenes compile to
explicit rollout Steps before execution; there is no runtime Scene object or
message scheduler.

```python
# Single-role shortcut
scene = Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")

# Multi-role with explicit turn order
scene = Scene(
    name="coder-reviewer",
    roles=[
        Role("coder", "gemini", "gemini-3.1-flash-lite-preview"),
        Role("reviewer", "gemini", "gemini-3.1-flash-lite-preview"),
    ],
    turns=[
        Turn("coder"),                    # None prompt = native task goal
        Turn("reviewer", "Review the current workspace."),
        Turn("coder", "Fix the issues."),
    ],
)
```

### Rollout

The execution engine — decomposed into independently-callable phases.

```python
from benchflow import Rollout

rollout = await Rollout.create(config)

# Full lifecycle (most common)
result = await rollout.run()

# Manual composition (for custom flows)
await rollout.setup()
await rollout.start()
await rollout.install_agent()
await rollout.connect()
await rollout.execute(prompts=["custom prompt"])
# Optional fork here; branch() leaves the agent disconnected.
value = await rollout.branch(2, run_child, snapshot_layers={"sandbox"})
await rollout.connect()
await rollout.execute(prompts=["continue the parent"])
await rollout.disconnect()
await rollout.verify()
result = await rollout.finalize()  # cleanup() plus result.json
```

`cleanup()` stops the sandbox but writes no `result.json`; call `finalize()` to finish a manually driven rollout. For `branch()` and its child runner, see [Composed checkpoints](../composed-checkpoints.md) and `docs/examples/branch-agent-run.py`. `branch()` also takes `restore_parent=False` (skip the parent restore after the last child; the rollout can then only be finalized), `child_requests` (one description per child for `tree.json`), `isolate_children=True` with `concurrency=K` (each child in its own sandbox, K at once; the runner drives `child.rollout`, which can branch again) and `resume_session=True` (children resume the parent's agent conversation). `RolloutConfig(checkpoints=benchflow.checkpoints.parse_checkpoint_policy(...))` keeps automatic checkpoints. The same lifecycle is available as `bench eval branch`, whose driver is `benchflow.branch_run.run_branch_trial`.

### RuntimeConfig

Runtime-level configuration for the `Agent + Environment` execution path.

```python
from benchflow.runtime import Agent, Environment, Runtime, RuntimeConfig

config = RuntimeConfig(sandbox_setup_timeout=300)
agent = Agent("gemini", model="gemini-3.1-flash-lite-preview")
env = Environment.from_task("tasks/X", sandbox="daytona")
runtime = Runtime(env, agent, config=config)
result = await runtime.execute()
```

### bf.run()

Convenience function — multiple calling conventions:

```python
import benchflow as bf

# 1. RolloutConfig (full control); bf.arun is the same function
result = await bf.run(config)

# 2. Agent + Environment (0.3 style)
agent = bf.Agent("gemini", model="gemini-3.1-flash-lite-preview")
env = bf.Environment.from_task("tasks/X", sandbox="daytona")
runtime_config = bf.RuntimeConfig(sandbox_setup_timeout=300)
result = await bf.run(agent, env, runtime_config)

# 3. String shortcut (simplest)
result = await bf.run(
    "gemini",
    task_path="tasks/X",
    model="gemini-3.1-flash-lite-preview",
    config=bf.RuntimeConfig(sandbox_setup_timeout=300),
)
```

## Rollout Lifecycle

```
Rollout.run()
  │
  ├─ setup()          — resolve config, create env object
  ├─ start()          — spin up sandbox, upload task files, start services
  ├─ install_agent()  — install agent binary, credentials, sandbox user
  │                    (sandbox user setup: create non-root user, prepare
  │                     small config/auth dirs, chown the workspace — no
  │                     recursive copy of /root tool trees; agent binaries
  │                     must live on shared prefixes like /usr/local/bin)
  ├─ compile scenes → Steps
  ├─ for step in steps:
  │    ├─ connect_as(role) — open/reuse ACP session for this role
  │    └─ execute(prompt)  — send prompt, collect trajectory, grow tree
  ├─ verify()         — run verifier, collect rewards
  └─ cleanup()        — stop sandbox
```

Key: scene boundaries are gone by execution time; role changes are represented
as Step metadata and handled by the rollout executor.

## Multi-Turn vs Multi-Round

| Pattern | Roles | Turns | Communication | Example |
|---------|-------|-------|---------------|---------|
| **Single-turn** | 1 | 1 | — | Baseline benchmark |
| **Multi-turn** | 1 | 2+ | Same session, sequential prompts | Self-review |
| **Multi-role** | 2+ | 2+ | Explicit prompt sequence | Coder + Reviewer |

**Multi-turn** = same agent gets multiple prompts. Use when a second pass catches errors (self-review, iterative refinement). The agent keeps its context across turns.

**Multi-role** = different agents receive explicit turns. Use when tasks need multiple perspectives (code review, client-advisor). Any handoff text must be part of the declared prompt or agent-native communication, not a BenchFlow Scene scheduler.

Both use the same API — `RolloutConfig` with different `Scene` configurations.

## Multi-Agent Patterns

### Coder + Reviewer (followup-bench)

```python
config = RolloutConfig(
    task_path=task_path,
    scenes=[Scene(
        roles=[Role("coder", "gemini", "flash"), Role("reviewer", "gemini", "flash")],
        turns=[
            Turn("coder"),
            Turn("reviewer", "Review /app/. Summarize any issues."),
            Turn("coder", "Read feedback and fix."),
        ],
    )],
    environment="daytona",
)
```

### Skill Generation + Solve (BYOS)

```python
config = RolloutConfig(
    task_path=task_path,
    scenes=[
        Scene(name="skill-gen",
              roles=[Role("gen", "gemini", "flash")],
              turns=[Turn("gen", "Generate a skill document to /app/generated-skill.md")]),
        Scene(name="solve",
              roles=[Role("solver", "gemini", "flash")],
              turns=[Turn("solver")]),
    ],
    environment="daytona",
)
```

## User-Driven Loops

Use `BaseUser` or `FunctionUser` when one agent should run multiple rounds and
Python should decide the next prompt from verifier feedback. This is the
progressive-disclosure path: the user callback can stop early, read
`RoundResult` after each `soft_verify()`, and optionally receive the oracle
solution during `setup()` when `oracle_access=True`.

```python
from pathlib import Path

from benchflow import FunctionUser, RolloutConfig, RoundResult, Scene


def user(round: int, instruction: str, rr: RoundResult | None) -> str | None:
    if round == 0:
        return instruction.splitlines()[0]
    if rr and (rr.rewards or {}).get("reward") == 1.0:
        return None
    return f"Tests failed:\n{rr.verifier_output}\n\nUse the full spec:\n{instruction}"


config = RolloutConfig(
    task_path=Path("tasks/my-task"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")],
    user=FunctionUser(user),
    max_user_rounds=3,
    environment="daytona",
)
result = await bf.run(config)
```

Use multi-role Scenes when another LLM should act as the reviewer or simulated
user. Use `BaseUser` when the loop is deterministic or verifier-driven. See
[`progressive-disclosure.md`](../progressive-disclosure.md) and
[`docs/examples/scene-patterns.ipynb`](../examples/scene-patterns.ipynb).

## YAML Rollout Configs

```python
from benchflow._utils.yaml_loader import rollout_config_from_yaml

config = rollout_config_from_yaml("rollout.yaml")
result = await bf.run(config)
```

## Built-in Agents

| Agent | Protocol | Auth | Aliases |
|-------|----------|------|---------|
| `gemini` | ACP | GEMINI_API_KEY | — |
| `claude-agent-acp` | ACP | ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN | `claude` |
| `codex-acp` | ACP | OPENAI_API_KEY, CODEX_API_KEY, CODEX_ACCESS_TOKEN, or host login | `codex` |
| `opencode` | ACP | inferred from model/provider | — |
| `openhands` | ACP | LLM_API_KEY | `oh` |
| `pi-acp` | ACP | ANTHROPIC_API_KEY | `pi` |

The Auth column shows each agent's native/default credentials. Provider-prefixed
models can use provider-specific credentials instead; for example, Azure
Foundry models use `AZURE_API_KEY` plus `AZURE_API_ENDPOINT` with prefixes such
as `azure-foundry-openai/gpt-5.5` or
`azure-foundry-anthropic/claude-opus-4-5`. BenchFlow routes these providers
through LiteLLM on both Docker and Daytona.

Additional agents load lazily from the external agents catalog. See
[External agents](../external-agents.md).

Any agent can be prefixed with `acpx/` to run via [ACPX](https://acpx.sh/) (e.g. `acpx/gemini`, `acpx/claude`). ACPX is a headless ACP client with persistent sessions and crash recovery. The underlying agent's install, env, credentials, and skill paths are preserved.

## Retry and Error Handling

Rollout.run() catches common errors:
- `TimeoutError` — agent exceeded timeout
- `ConnectionError` — SSH/ACP pipe closed (retried 3x with exponential backoff)
- `ACPError` — agent protocol error

Evaluation-level retry with `RetryConfig`:
```python
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig

config = EvaluationConfig(
    retry=RetryConfig(
        max_retries=2,
        wait_multiplier=2.0,
        min_wait_sec=1.0,
        max_wait_sec=30.0,
    ),
)
```

---

## Sandbox and Reward Types

### Sandbox Protocol

The `Sandbox` protocol defines the interface any sandbox backend must implement.
Docker and Daytona are built-in; you can bring your own (Modal, Firecracker, E2B, etc.).

```python
from benchflow import Sandbox, ImageBuilder, ImageConfig, ImageRef

# Sandbox is a runtime-checkable Protocol
class MySandbox:
    async def exec(self, cmd: str, *, user: str = "root", timeout_sec: int = 30) -> ExecResult: ...
    async def upload_file(self, src: Path, dst: str) -> None: ...
    async def download_file(self, src: str, dst: Path) -> None: ...
    async def start(self) -> None: ...
    async def stop(self, *, delete: bool = True) -> None: ...
    # ... plus snapshot/restore + host/expose_ports; see sandbox/protocol.py

assert isinstance(my_sandbox, Sandbox)  # works at runtime
```

### Rubric + RewardFunc (Composable Rewards)

Declarative scoring via composable reward functions.

```python
from benchflow import Rubric, RewardFunc, RewardEvent, VerifyResult
from benchflow import TestRewardFunc, StringMatchRewardFunc, LLMJudgeRewardFunc

# Built-in reward functions
test_reward = TestRewardFunc()          # runs pytest, binary pass/fail
match_reward = StringMatchRewardFunc(expected="hello world")

# Compose into a weighted Rubric
rubric = Rubric(
    reward_funcs=[test_reward, match_reward],
    weights=[0.7, 0.3],
)

# Score a workspace
result: VerifyResult = await rubric.score(rollout_dir=my_rollout_dir)
print(result.reward)      # weighted float [0.0, 1.0]
print(result.events)      # list[RewardEvent] — per-function breakdown
```

### Adapters (Inspect AI + ORS)

Convert between BenchFlow types and external frameworks.

```python
from benchflow import InspectAdapter, ORSAdapter, to_inspect_task, to_ors_reward

# BenchFlow Scene → Inspect AI task format
inspect_task = to_inspect_task(scene, rubric=rubric)

# BenchFlow VerifyResult → ORS reward format
ors_payload = to_ors_reward(verify_result)
```

### Evaluation

Batch orchestration with concurrency and retries.

```python
from benchflow import Evaluation, EvaluationConfig, EvaluationResult, RetryConfig

# EvaluationConfig holds the per-job settings (agent/model/environment/...)
# applied to every task discovered under tasks_dir.
config = EvaluationConfig(
    agent="gemini",
    model="gemini-3.1-flash-lite-preview",
    environment="daytona",
    concurrency=8,
    retry=RetryConfig(max_retries=2),
)
evaluation = Evaluation(tasks_dir="tasks", jobs_dir="jobs/my-run", config=config)
eval_result: EvaluationResult = await evaluation.run()
```

`Evaluation(..., budget=bf.Budget(max_cost_usd=..., max_sandbox_seconds=..., max_tokens=..., max_rollouts=...))` caps the job: at a cap no new rollout starts (retries included), running ones are cancelled for the spend caps, and `eval_result.budget` / `summary.json` `budget` list them (never as failures). USD and tokens are known when a rollout finishes, so they are enforced between starts. See [Job budget caps](./budget.md).
