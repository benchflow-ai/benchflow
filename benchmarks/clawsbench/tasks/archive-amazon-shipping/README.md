# archive-amazon-shipping

A minimal, single-service ClawsBench task on the mock **Gmail** environment.

## Goal

The agent must archive one specific email — the Amazon shipping confirmation
from `shipment-tracking@amazon.com` — by removing its `INBOX` label, without
trashing, deleting, or touching any other message.

## How it works

| Piece | What it does |
|---|---|
| `environment/Dockerfile` | builds `FROM kywch/smolclaws-base:latest` and seeds `gmail.db` with the deterministic `default` scenario (`claw-gmail seed --scenario default --seed 42`). The base image already bundles `claw-gmail`. |
| Environment manifest | `../../environment.toml` declares the `claw-*` services; the BenchFlow Environment plane (`ManifestEnvironment`) starts `claw-gmail` inside the rollout sandbox on `localhost:9001`. |
| `instruction.md` | the prompt — points the agent at the Gmail REST API. |
| `tests/test.sh` + `tests/evaluate.py` | read `/_admin/state` and emit a binary reward to `/logs/verifier/reward.txt`: `1.0` iff the Amazon email still exists, is not trashed/spam, and no longer carries `INBOX`. |
| `solution/solve.sh` | oracle — archives the email via the Gmail API; gets reward `1.0`. |

The `default` scenario contains exactly one email from
`shipment-tracking@amazon.com`, so the verifier targets it by sender — no
build-time message id is baked in.

## Run it

```bash
bench eval run --tasks-dir benchmarks/clawsbench/tasks/archive-amazon-shipping \
  --environment-manifest benchmarks/clawsbench/environment.toml \
  --agent gemini --model gemini-3.1-flash-lite-preview \
  --sandbox daytona
```

Confirm the task is solvable with the oracle:

```bash
bench eval run --tasks-dir benchmarks/clawsbench/tasks/archive-amazon-shipping \
  --agent oracle --sandbox docker
```
