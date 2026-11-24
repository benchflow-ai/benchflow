# Daytona

Daytona runs each trial in a cloud sandbox, so the number of trials at once isn't limited by your machine. Use it for runs too large for Docker on one machine. It needs a Daytona account, the `sandbox-daytona` extra, and nothing else installed locally.

## Set it up

Install BenchFlow with the extra, export your API key, and check that BenchFlow can reach Daytona:

```bash
uv tool install --python 3.12 'benchflow[sandbox-daytona]==0.10.1' \
  --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.10.1/constraints.txt
export DAYTONA_API_KEY=...
bench doctor --sandbox daytona
```

This output is from 0.10.1, shortened:

```text
BenchFlow doctor - benchflow 0.10.1, Linux x86_64, sandbox daytona
...
  PASS  daytona             SDK 0.207.0, DAYTONA_API_KEY accepted by app.daytona.io
...
  PASS  app.daytona.io      reachable — Daytona API
  PASS  pypi.org            reachable — model proxy install inside the sandbox
...
14 passed, 1 warning, 0 failed, 5 skipped
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
- **Checkpoints:** a Daytona snapshot takes minutes: 110 to 500 seconds measured, for sandboxes with and without an agent's install (claude-agent-acp's is about 3 GiB). BenchFlow gives it `BENCHFLOW_DAYTONA_SNAPSHOT_TIMEOUT_SEC` seconds, 600 by default. When a create fails anyway, BenchFlow looks for the snapshot until Daytona shows it finished, for up to 300 s (and up to 240 s more at the end of the trial): an active one is the checkpoint after all, a failed one is deleted, and one still being made after that look is deleted at the end of the trial, before the sandbox (Daytona can't delete the sandbox while the snapshot is being made). The trial's `checkpoints.json` says which, under `leftover` (for `bench branch`, the fork's record in `tree.json`), and `daytona-snapshot-leftovers.json` names each one the end of the trial settled. A snapshot that still can't be deleted stays named there. With `BENCHFLOW_DAYTONA_OWNER` set, its name starts with that owner's prefix, `bf-snap2-<first 32 hex digits of the owner's SHA-256>-`, and `bench sandbox cleanup` deletes it once it's stale; without an owner nothing proves a snapshot is yours, so set one on a shared account. The time checkpoints take doesn't count against the trial's hard deadline. A snapshot also sets the ctime of the files the sandbox wrote to the time it starts. The trial's integrity record lists such a file as a `snapshot-ctime` change, not a flag, when only root can write it and its mode, owner, group, size, mtime and link are as before; any other file is compared as it would be without a snapshot.
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
