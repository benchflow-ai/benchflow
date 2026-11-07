# Daytona

Daytona runs each trial in a cloud sandbox, so the number of trials at once isn't limited by your machine. Use it for runs too large for Docker on one machine. It needs a Daytona account, the `sandbox-daytona` extra, and nothing else installed locally.

## Set it up

Install BenchFlow with the extra, export your API key, and check that BenchFlow can reach Daytona:

```bash
uv tool install --python 3.12 'benchflow[sandbox-daytona]==0.9.1' \
  --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.9.1/constraints.txt
export DAYTONA_API_KEY=...
bench doctor --sandbox daytona
```

```text
BenchFlow doctor - benchflow 0.9.1, Linux x86_64, sandbox daytona
  PASS  daytona             SDK 0.207.0, DAYTONA_API_KEY accepted by app.daytona.io
  PASS  app.daytona.io      reachable — Daytona API
...
```

Keep `DAYTONA_API_KEY` in your shell. Passing it to the agent with `--agent-env` would put it inside the sandbox.

## Run on Daytona

`tasks/half-life` is the task [Write a task](../tasks/write-a-task.md) builds and shows how to copy.

```bash
bench run tasks/half-life -a oracle --sandbox daytona
```

```text
[half-life] Running verifier...
[half-life] Rewards: {'reward': 1.0}
[PASS] half-life (reward=1.00, tools=0)
Job complete: 1/1 (100.0%), mean_reward=1.00, errors=0, idle_timeouts=0, time=0.9min

✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
Outcomes: 1 scored (1 passed, 0 failed), 0 unscored, 0 errored
Cost:      none: no model tokens were used
Time:      55s
```

Everything else works as on Docker: the same tasks, agents and job folders. BenchFlow runs the model proxy inside each sandbox, since a cloud sandbox can't reach your machine.

## Limits

- **Disk:** Daytona caps each sandbox at 10 GB. A task with a large image or large downloads can fail with `No space left on device`, or stall at "Sandbox user agent ready" with no trajectory; run such tasks on Docker.
- **Concurrency:** your Daytona plan sets how many sandboxes can run at once; start with `-j` well under it.
- **Services:** tasks with compose services run, inside the sandbox.
- **Services the agent leaves running:** ten Terminal-Bench 2.0 tasks keep their services after the agent's turn, which a `--sandbox daytona` sandbox can't hold for an agent that isn't root, so those trials are refused: `configure-git-webserver`, `git-multibranch`, `hf-model-inference`, `install-windows-3.11`, `kv-store-grpc`, `mailman`, `nginx-request-logging`, `pypi-server`, `qemu-alpine-ssh` and `qemu-startup`. The oracle and nop aren't refused. Run those tasks on Docker or remote Docker ([Bring what you have](../evaluate/bring-what-you-have.md)).

## Clean up

A finished run deletes its sandboxes. `bench sandbox list` shows what's running on the account, and `bench sandbox cleanup` removes stale ones. On a shared account the list includes other people's sandboxes, so look first with `--dry-run`:

```bash
bench sandbox cleanup --dry-run
```

```text
Warning: a sandbox with a live run lease is kept (name it with --id to delete it). One Daytona stopped whose lease lapsed is deleted. Any other (no lease, or STARTED with a lapsed lease) is selected by age (1440m) and activity (30m idle) alone, which cannot tell a quiet live run from an abandoned one: use --dry-run first.
  d4d5492a-2869-4cd2-8259-b9e6b2bdafc2 state=SandboxState.STARTED age=36m (skip)
  f6ceece6-18fd-4df0-8775-fad6286f4821 state=SandboxState.STARTED age=172m (skip)
...
```

Every sandbox listed is younger than a day, so all are skipped; on this shared account they were other people's live runs. `--id <sandbox>` deletes one by name; `--max-age` changes the age, in minutes, past which an idle sandbox counts as stale.
