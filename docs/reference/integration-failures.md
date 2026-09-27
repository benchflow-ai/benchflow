# Agent integration failures

An agent integration can break without an error: the agent's install is missing a runtime, its login was rejected, or it exits before doing anything. The verifier then scores the untouched workspace, usually 0, and the trial looks like a model that failed the task. BenchFlow treats such a trial as an execution failure of the harness: unscored (never 0), with a named cause, and counted separately.

## When a trial is judged

After the prompt, the agent made no tool call, sent no message with text, produced no genuine thought and reported no output tokens. A thought that is a harness shim's diagnostic (`[openclaw stderr] …`, `[openclaw-acp-shim] …`) is not genuine. Any activity leaves the trial alone: a model that thought for an hour and timed out, or answered with a refusal, keeps its 0. Control runs (`oracle`, `nop`), session-factory agents and runs that captured no events at all are never judged.

## Causes

| Cause | Evidence |
|---|---|
| `agent_auth` | an auth or billing failure in a diagnostic thought or an agent log (`agent/*.txt`): credit balance too low, invalid API key, `authentication_error`, 401, not logged in, expired OAuth token, insufficient quota |
| `agent_install` | an install or runtime failure: `Node.js vX+ is required`, command not found, Cannot find module, `ModuleNotFoundError`, exec format error, `npm ERR!` |
| `truncated_trajectory` | the trajectory file has unparseable lines |
| `empty_trajectory` | no event after the prompt |
| `immediate_exit` | the agent phase lasted under 10 s |
| `no_activity` | the agent ran and produced nothing |

## What is recorded

At run time the result gets `error_category: "agent_integration"`, an `error` such as `agent integration failure [agent_auth]: LLM request rejected: Your credit balance is too low …`, `rewards: null`, and `integration_failure_info` with the cause, evidence, where it was found, the activity counts, the agent's seconds and `reward_withheld` (what the verifier gave). This applies to subscription (OAuth) runs too, which the older zero-token check (`suspected_api_error`) skips.

Results written before this existed are checked when they are read: `bf.load_job` / `bf.load_trial` (`Trial.integration_failure`, `Trial.execution == "integration_failed"`, `Trial.reward is None`), `bench eval inspect` (a `Cause` column and the headline count) and `bench eval metrics` (an `Integration failures` row, `integration_failures` in `--json`). `summary.json` has `integration_failures: {total, by_cause}`, and the job summary logs one warning line.

A batch stops early when the same `agent_auth` or `agent_install` failure repeats (the API-error circuit breaker, `BENCHFLOW_API_ERROR_BREAKER_THRESHOLD`, default 5); the other causes can be transient and do not trip it. These trials are not retried automatically.

## Checking an agent before a batch

`bench doctor --sandbox daytona --agent-start claude-agent-acp` creates a sandbox for the bundled hello-world task, installs the agent, opens its ACP connection (`initialize`, `session/new`) without a prompt, and deletes the sandbox. No model call is made. A failure names its cause with the evidence line and the tail of the agent's logs.
