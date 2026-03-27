# benchflow SDK Reference

## SDK.run()

```python
await SDK().run(
    task_path,                          # Path to task dir (must have task.toml + instruction.md)
    agent="claude-agent-acp",           # Agent from registry or raw command
    prompts=None,                       # List[str|None]. None entries = instruction.md. Default: [instruction.md]
    model=None,                         # Model ID (e.g. "claude-haiku-4-5-20251001"). Set via ACP.
    agent_env=None,                     # Extra env vars. API keys auto-inherited (see below)
    job_name=None,                      # Auto-generated from timestamp
    trial_name=None,                    # Auto-generated from task name + uuid
    jobs_dir="jobs",                    # Output directory
    environment="docker",              # "docker" or "daytona"
    skills_dir=None,                    # Host path to skills dir (see Skills section)
    sandbox_user=None,                  # Non-root user (e.g. "agent"). See Sandbox section
    pre_agent_hooks=None,              # List of async callables(env). Run before agent launch
    context_root=None,                  # Repo root for Dockerfile COPY path resolution
) -> RunResult
```

### Auto-inherited env vars
These are forwarded from `os.environ` to the agent automatically:
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`

Also auto-set: `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`

### Agent timeout
Timeout comes from `task.toml` → `[agent].timeout_sec` (default 900s). NOT a SDK.run() parameter.

### RunResult

```python
result.task_name        # str
result.trial_name       # str
result.rewards          # dict | None — e.g. {"reward": 1.0}
result.trajectory       # list[dict] — tool calls, messages, thoughts
result.agent_name       # str — ACP-reported agent name
result.n_tool_calls     # int
result.n_prompts        # int
result.error            # str | None
result.started_at       # datetime
result.finished_at      # datetime
result.success          # bool — True if error is None
```

## Job

```python
job = Job(
    tasks_dir="path/to/tasks",
    jobs_dir="jobs/my-run",
    config=JobConfig(
        agent="claude-agent-acp",
        model="claude-haiku-4-5-20251001",
        environment="daytona",
        concurrency=64,
        prompts=None,                   # Multi-turn prompts (same as SDK.run)
        agent_env={},                   # Extra env vars
        skills_dir=None,
        sandbox_user=None,
        context_root=None,
        retry=RetryConfig(
            max_retries=2,              # Default: 2 (CLI default: 0)
            retry_on_install=True,
            retry_on_pipe=True,
            retry_on_acp=True,
        ),
    ),
    job_name=None,                      # Auto-generated
    on_result=None,                     # Callback: fn(task_name, RunResult)
)
result = await job.run()
```

### Resume behavior
Job automatically skips tasks that already have `result.json` with rewards in `jobs_dir`. Warns if resuming with a different agent config than previous runs.

### YAML config
```python
job = Job.from_yaml("config.yaml")
```

### JobResult

```python
result.job_name         # str
result.config           # JobConfig
result.passed           # int
result.failed           # int
result.errored          # int
result.total            # int
result.score            # float — passed / total
result.score_excl_errors # float — passed / (passed + failed)
result.elapsed_sec      # float
```

## Agents

| Agent | Launch | Requires | skill_paths |
|-------|--------|----------|-------------|
| `claude-agent-acp` | `claude-agent-acp` | ANTHROPIC_API_KEY | `$HOME/.claude/skills` |
| `pi-acp` | `pi-acp` | ANTHROPIC_API_KEY | `$HOME/.pi/agent/skills`, `$HOME/.agents/skills` |
| `openclaw` | ACP shim | ANTHROPIC_API_KEY | `$HOME/.claude/skills`, `$WORKSPACE/skills` |
| `openclaw-gemini` | ACP shim | GEMINI_API_KEY | `$HOME/.claude/skills`, `$WORKSPACE/skills` |
| `codex-acp` | `codex-acp` | OPENAI_API_KEY | `$HOME/.agents/skills` |
| `gemini` | `gemini --acp` | GOOGLE_API_KEY | `$HOME/.gemini/skills` |

All agents must speak ACP (JSON-RPC 2.0 over stdio).

### Register custom agent

```python
from benchflow import register_agent

register_agent(
    name="my-agent",
    install_cmd="npm install -g my-agent",  # Runs inside sandbox as root
    launch_cmd="my-agent --acp",            # Must speak ACP
    requires_env=["MY_API_KEY"],
    skill_paths=["$HOME/.my-agent/skills"],
    install_timeout=900,
)
```

Note: `register_agent()` must be called before creating `Job` or calling `SDK.run()`. If the agent name is not in the registry, `install_cmd` is skipped and `launch_cmd` is used as-is.

## Skills

### Precedence
CLI `--skills-dir` > task.toml `[environment].skills_dir` > Dockerfile COPY

### task.toml skills_dir
```toml
[environment]
skills_dir = "/skills"
```
The SDK reads this after env.start() and copies from this container path to the running agent's `skill_paths`.

### Distribution
Skills are copied to the running agent's `skill_paths` only. `$HOME` expands to `/root` or `/home/{sandbox_user}`. `$WORKSPACE` expands to agent's cwd.

### CLI
```bash
benchflow skills                                    # List discovered skills
benchflow skills --install owner/repo@skill-name    # Install from skills.sh
benchflow eval -t tasks/ --skills-dir skills/ -a claude-agent-acp -e daytona
```

## Sandbox User

```python
await sdk.run(..., sandbox_user="agent")
```

- **Requires gosu** in the container (`apt-get install -y gosu`)
- **Auto-creates user** via `useradd -m -s /bin/bash {sandbox_user}` if not exists
- **Name must match** `^[a-z_][a-z0-9_-]*$` (lowercase, underscores, hyphens)
- **Agent cwd** changes to `/home/{sandbox_user}` (not the container's WORKDIR)
- Agent config dirs (`.claude`, `.gemini`, `.openclaw`, `.pi`, `.agents`, `.codex`) are copied from root
- Install runs as root, agent runs as sandbox_user via gosu
- Custom agents: must support ACP — see [ACP spec](https://agentclientprotocol.com/)

## Environments

| Environment | Concurrency | Setup |
|-------------|-------------|-------|
| `docker` (default) | ~4 | Docker must be running locally |
| `daytona` | 64+ | Set `DAYTONA_API_KEY` from [daytona.io](https://app.daytona.io). No other config needed. |

## Pre-agent Hooks

Async callables that run after env setup but before agent launch. Use for starting background services.

```python
# Built-in: auto-detect and start claw-* services from Dockerfile
from benchflow import detect_services_from_dockerfile, build_service_hooks

services = detect_services_from_dockerfile("task-dir")  # reads Dockerfile for claw-*
hooks = build_service_hooks(services)                    # builds async start + health check

await sdk.run(task_path="task-dir", pre_agent_hooks=hooks)
```

```python
# Custom hook
async def my_hook(env):
    await env.exec("my-service start &", timeout_sec=10)
    await env.exec("curl -sf http://localhost:8080/health", timeout_sec=30)

await sdk.run(task_path="task-dir", pre_agent_hooks=[my_hook])
```

Note: `pre_agent_hooks` is SDK-only. Not available via Job or CLI.

See `examples/smolclaws_eval.py` for a complete example.

## context_root

Set this when your task's Dockerfile uses COPY instructions that reference files outside the `environment/` directory (relative to the repo root). Without it, the Docker build context won't find those files.

```python
# Dockerfile has: COPY packages/my-lib /app
# packages/my-lib lives at repo root, not inside environment/
await sdk.run(task_path="tasks/my-task", context_root="/path/to/repo")
```

## Errors

```python
from benchflow import AgentInstallError, AgentTimeoutError

# AgentInstallError — raised when install_cmd fails
#   .agent, .return_code, .stdout, .diagnostics, .log_path

# AgentTimeoutError — for timeout errors
#   .agent, .timeout_sec
```

## Trial Output

```
jobs/{job_name}/{trial_name}/
├── config.json              # SDK.run() parameters
├── result.json              # rewards, n_tool_calls, timing
├── timing.json              # {environment_setup, agent_setup, agent_execution, verifier, total}
├── prompts.json
├── agent/
│   ├── install-stdout.txt   # agent install output
│   └── {agent_name}.txt     # agent stderr/debug (non-JSON lines)
├── trajectory/
│   └── acp_trajectory.jsonl # tool calls, messages, thoughts
└── verifier/                # Written by Harbor verifier
    ├── reward.txt
    ├── test-stdout.txt
    └── ctrf.json            # pytest results (if verifier uses pytest)
```

## Caveats

1. **Dockerfile mutation**: `skills_dir` and `context_root` modify `environment/Dockerfile` in-place. Use separate task copies for parallel runs. Git will show dirty files.
2. **No env var pre-flight check**: Missing API keys only surface as errors deep in ACP connection.
3. **Oracle mode**: `agent="oracle"` skips ACP, runs `solution/solve.sh` directly. No multi-turn.
4. **codex-acp auth**: SDK auto-writes `~/.codex/auth.json` from OPENAI_API_KEY.
5. **Gemini trajectory**: Gemini sends `tool_call_update` without initial `tool_call`. SDK auto-creates records.
6. **API keys visible**: Docker exec `-e K=V` shows keys in `ps aux`.
7. **benchflow view side-effect**: Writes `trajectory.html` to the trial directory.
8. **collect_metrics deduplication**: Picks best result per task when multiple trials exist.
