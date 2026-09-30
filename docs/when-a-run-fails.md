# When a run fails

A failed run should tell you three things at once: what failed, whose fault it was (the task, the agent, the infrastructure, or your setup), and what to do next. This page maps each message BenchFlow prints to its cause and its fix.

## How `bench` ends on an error

- **An error you can fix** (a task file that does not parse, a bad flag, a missing file or login, a spent usage limit) prints one red message and its next step, and exits non-zero (1, or 2 for a flag Click rejects). No job folder is created when the problem is found before the run, as a missing login is.
- **Anything else is a bug in BenchFlow.** `bench` prints the Python traceback, then `bench: unexpected error (<type>). This is a bug in BenchFlow, not in your task or setup.` and the path of a crash log, and exits 1. The log is in `~/.cache/benchflow/logs/` (or `$BENCHFLOW_LOG_DIR`) and holds the command, the versions, the traceback and the run's last 2000 log lines. Credential values from the environment, `.env`, `KEY=VALUE` arguments (`--agent-env KEY=V` and `--agent-env=KEY=V`) and credential-named options (`--hf-token V`, `--hf-token=V`) are replaced with `***` in both. Attach the log to an issue at https://github.com/benchflow-ai/benchflow/issues.

| Message | Cause | Fix |
|---|---|---|
| `ANTHROPIC_API_KEY required for model '<model>' but not set.` then `next: for a Claude subscription run claude setup-token ...` and `No job was created.` | Setup: no credential for the agent's model. | `claude setup-token` and export `CLAUDE_CODE_OAUTH_TOKEN`, or `claude login`, or export the API key. For Codex: `codex login` or `OPENAI_API_KEY`. `bench doctor` lists what it finds. |
| `No such file or directory: <path>` | Setup or task: a file the command or the task names does not exist (for example `benchflow.environment.manifest` in task.md). | Create the file or fix the path. |
| A task.md or task.toml parse error | Task: the task file is invalid. | `bench tasks check <task>` names the line. |
| `Unknown sandbox ...`, `Invalid value for '--...'` | Setup: a bad flag. | The message lists the accepted values; `bench <command> --help`. |
| `usage limit reached ... (You've hit your ... limit ...)` and no job report | Setup: the login was spent before the job started. | See "Usage limits" below. |

## The end-of-run summary

After the score and one line per failed task, `bench eval run` and `bench eval resume` print:

```
Outcomes: 1 scored (1 passed, 0 failed), 1 unscored, 2 errored
  1 unscored: verifier plugin trust: test.sh installs a pytest plugin where the agent could write (task problem): powerlifting-coef-calc
      next: `bench tasks check tasks/powerlifting-coef-calc`
  1 errored: usage limit on login CLAUDE_CODE_OAUTH_TOKEN (environment), 7-day window, resets 2026-10-03 19:00 UTC (setup problem): hello
      next: switch to another login or wait for the reset, then `bench eval resume jobs/run`; `bench doctor` shows each login's headroom
  2 not started: the login's usage limit stopped the job; `bench eval resume jobs/run` runs them on another login or after the reset
Cost:      $0.75, 1,000 tokens (2 of 4 trials reported no cost)
Time:      2m 10s
Artifacts: jobs/run
Summary:   jobs/run/summary.json
View:      bench eval view jobs/run
```

- **Scored** trials got a reward from the verifier (passed or failed). A failure the agent causes, such as a wrong answer or running out of time, is scored 0.
- **Unscored** trials finished, but the verifier could not score them. They are left out of the score's denominator.
- **Errored** trials stopped before a score. They are left out of the denominator too, and most are retried within the run (see each row below).
- **Cost** sums the USD each trial reported; subscription logins report tokens only. **Time** is this invocation's wall-clock.
- `bench eval view <job>` opens the trajectories in the viewer.

## Errored trials

| Message in the trial's `error` | Category | Whose fault | Retried | What to do |
|---|---|---|---|---|
| `usage limit reached on login <label>: 7-day window, resets <time> (You've hit your weekly limit · resets ...)` | `usage_limit` | setup | never; the job starts no more trials | See "Usage limits" below. |
| `... provider auth failed (HTTP 401)`, `... was rejected as invalid` | `provider_auth` | setup | no | `bench doctor` checks the key or login. |
| `... rate limit ...`, `... HTTP 429` | `provider_rate_limit` | infrastructure | no | Lower `--concurrency`, then `bench eval resume <job>`. |
| `... HTTP 400` (often a context too long) | `provider_rejected` | agent | no | A model with a larger context, or a shorter prompt. |
| `provider api error [<kind>/<transient or permanent>] HTTP <status> ...` | `api_error` | infrastructure | transient ones | `api_error_info` in result.json has the status. |
| `suspected provider api error ...` | `suspected_api_error` | setup | no | Check the model id and login with `bench doctor`. |
| `agent integration failure [agent_model]: <agent> does not offer model '<model>' ...; it offers: ...` | `agent_integration` | setup | no | Pick a model from the list. |
| `agent integration failure [agent_auth]: ...` | `agent_integration` | setup | no | `bench doctor`. |
| `agent integration failure [agent_install]: ...` | `agent_integration` | infrastructure | no | `bench doctor --agent-start <agent>`. |
| `... install failed ...` | `install_failure` | infrastructure | yes | `bench doctor` checks the network to npm and Node.js; `agent/install-stdout.txt` has the log. |
| `PTY closed by the peer: agent process exited with code <n>` | `pipe_closed` | agent | yes | The agent process in the Daytona sandbox ended; its log is `agent/<agent>.txt`. |
| `PTY closed by the peer: websocket closed (close_code=<n>, ...)` | `pipe_closed` | infrastructure | yes | The connection to the Daytona sandbox dropped; `bench eval resume <job>` reruns it. |
| `PTY readline timeout (<n>s)` | `pipe_closed` | infrastructure | yes | No output for that long: a dead channel, or an agent silent past the read guard. |
| `Process closed stdout (rc=<n>): ...`, `Agent process closed stdout` | `pipe_closed` | infrastructure | yes | The local or SSH transport ended; `transport_error_info` in result.json says why. |
| `Agent idle for <n>s with no new tool call, message, or thought (...)` | `idle_timeout` | agent | yes | The idle watchdog stopped a session that went silent. Raise `--agent-idle-timeout` for an agent that thinks long. The verifier still scores the workspace when it can. |
| `Agent prompt exceeded wall-clock budget <n>s`, `Agent timed out: still running at the host hard deadline ...` | `timeout` | agent | no | The agent ran out of time; the verifier scores the workspace. Raise the task's `agent.timeout_sec` or `--timeout`. |
| `Rollout exceeded host hard deadline (<n>s) — transport or teardown wedged ...; sandbox abandoned` | `infra_failure` | infrastructure | yes | `bench eval resume <job>`. |
| `Sandbox startup failed: ...` | `sandbox_setup` | infrastructure (task when the Dockerfile names a missing path) | yes, unless it would fail the same way | `sandbox_startup_info` in result.json; `bench doctor`. |
| `ACP error -32603: Internal error: The connection to Claude was lost: the agent's own process ended or lost its stream ...` | `acp_error` | agent | yes | The Claude Code process in the sandbox died or its API stream failed; `agent/claude_agent_acp.txt` says which. This is not BenchFlow's connection to the sandbox, which reads `PTY closed ...` or `... closed stdout`. |
| `ACP error <code>: ...` | `acp_error` | agent | yes | The agent's log (`agent/<agent>.txt`) has its side. |

The Daytona PTY and the watchdogs never share a message: a `PTY ...` or `... closed stdout` error means the connection to the agent ended, and an `Agent idle ...`, `Agent prompt exceeded ...` or `Agent timed out ...` error means a watchdog stopped a session whose connection was still up. One case reads as the watchdog although the connection died: a connection that dies without closing (no heartbeat) looks, from the outside, like a silent agent until the idle watchdog fires.

## Unscored trials

| Message in the trial's `verifier_error` | Whose fault | What to do |
|---|---|---|
| `verifier crashed: PluginGuardLoadError: pytest plugin <name> <version> was installed after the agent stopped, by the verifier into <path>, where the agent could write; ... This is a task problem: bench tasks check <task> flags it. ...` | task | The task's test.sh installs a pytest plugin (usually `pytest-json-ctrf`) into a Python environment the agent could have written: the workspace, the agent's home, `/tmp`, `/var/tmp`, `/logs` or `/testbed`. The pytest plugin guard refuses it, so every run of the task is unscored, the reference solution's too. Install verifier plugins with `uvx` (`uvx --with pytest-json-ctrf==0.3.5 pytest ...`), whose cache BenchFlow moves to a directory the guard trusts, or bake them into the image. `bench tasks check <task>` warns about such installs. |
| `verifier crashed: ...` | task | `bench tasks check <task>`; the trial's `verifier/test-stdout.txt` has the output. |
| `verifier timed out ...` | task | Raise the task's `verifier.timeout_sec`. |
| a dependency install failure (`failed to download ...`, `no solution found ...`) | task | Pin or bake the verifier's dependencies into the image. |
| `separate verifier ...`, `verifier recovery ...` | infrastructure | `bench eval resume <job>` scores it again. |

## Usage limits

A Claude subscription (and a ChatGPT plan) allows a fixed amount of use in a 5-hour and a 7-day window. When a login's window is spent, every request on it fails until the window resets, so BenchFlow does not retry the trial, and an `Evaluation` starts no more trials: the running ones finish, and `summary.json`'s `usage_limit` block records the login, the window, the reset and the trials not started.

- **On the command line**, the summary names the login (`CLAUDE_CODE_OAUTH_TOKEN (environment)`, `host login (~/.claude/.credentials.json)`, or the name in `BENCHFLOW_LOGIN_LABEL`; never the token), the window and the reset. Switch to another login or wait, then `bench eval resume <job>`: it reruns the usage-limit trials and the ones never started, and keeps the rest.
- **From Python**, `Evaluation.run()` raises `bf.UsageLimitError` with `.login`, `.window`, `.resets_at` and the job's result as `.result`; catch it to switch logins (see [the Python API](reference/python-api.md#batches-over-a-task-directory)).
- **Before a run**, `bench doctor` shows each window's use and reset for the Claude login it finds, from one 8-token request.

## `bench doctor`

`bench doctor` checks what a first run needs and prints a fix under every problem:

- Python and uv;
- Docker: the daemon answers, its version, and the free disk in its data root (a warning below 10 GiB); or the Daytona SDK and a read-only API call with `DAYTONA_API_KEY`;
- each agent's credentials, by name and source (never the value), and for a Claude subscription login its 5-hour and 7-day windows from one 8-token `claude-haiku-4-5-20251001` request. That request is the only one doctor makes, and it goes only to api.anthropic.com with a subscription's own OAuth token (from the environment or `~/.claude/.credentials.json`): `--offline`, an API key or gateway token, and an `ANTHROPIC_BASE_URL` pointing elsewhere each skip it with a reason, so no token reaches a host it was not issued for;
- Codex's login file, Gemini and Bedrock keys, other provider keys;
- the agent pins the sandbox installs;
- the model proxy (LiteLLM) that API-key runs go through, and a custom `BENCHFLOW_PROVIDER_BASE_URL`;
- the network: the Docker registry, Node.js and npm, the Daytona API, PyPI for Daytona's model proxy, and each credential's model API.

It exits 1 only when something stops every run on the chosen sandbox: Python older than 3.12, no Docker daemon, no Daytona SDK or key, or a required endpoint out of reach. With no credential at all it warns: the oracle and nop controls still run.
