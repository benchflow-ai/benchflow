# Task format

A task is a folder: an instruction, an environment for the sandbox, a verifier that scores the result, and usually a reference solution. Here are its files, what each is for, and which one BenchFlow reads when two could apply. Every setting with its type and default is in [Files BenchFlow reads and writes](../reference/files.md).

## The files

The [half-life task](./write-a-task.md) has every part a task usually needs:

```text
half-life/
├── task.md                 settings on top, the instruction below
├── environment/
│   ├── Dockerfile          builds the sandbox
│   └── data/counts.csv     files the Dockerfile copies in
├── verifier/
│   ├── test.sh             scores the trial: writes /logs/verifier/reward.txt
│   └── rubric.json         optional: a rubric a reviewer agent grades against
└── oracle/
    └── solve.sh            the reference solution the oracle agent runs
```

| File | Seen by the agent? | Notes |
|---|---|---|
| `task.md` | the instruction only | The settings never reach the agent. |
| `environment/` | yes: it's the sandbox | Don't put answers or tests here. |
| `verifier/` | no | Copied into the sandbox at `/verifier` only after the agent has stopped. |
| `oracle/` | no | Uploaded only for oracle runs. |

## The task.md header

The header is YAML between two `---` lines:

```yaml
schema_version: "1.3"
metadata:
  author_name: BenchFlow docs
  difficulty: medium
  category: capability
  tags: [physics, data-analysis]
agent:
  timeout_sec: 900       # the agent's time limit, in seconds
verifier:
  timeout_sec: 120       # the verifier's time limit
sandbox:
  cpus: 1
  memory_mb: 2048
```

Below the header is the instruction. The body can be split at reserved headings: `## prompt` holds the instruction, and `## role:<name>`, `## scene:<name>` and `## user-persona` hold the parts of a [multi-turn task](./multi-turn.md). A body with none of them is the instruction as a whole.

The header's top-level keys are `schema_version`, `task`, `metadata`, `agent`, `verifier`, `sandbox`, `oracle`, `source`, `artifacts` and `reward`, and for multi-turn tasks `agents`, `scenes` and `user`. An unknown key is refused, so a typo can't silently change a run. `steps` describes a [multi-step task](./multi-step.md), which only 0.10.0rc1 runs, and 0.10.0rc1 adds `preference` for [human preferences](./human-preferences.md).

## The environment

With an `environment/Dockerfile`, BenchFlow builds the sandbox from it, with `environment/` as the build context. A task that needs no build can name an image instead:

```yaml
sandbox:
  docker_image: python:3.12-slim
  workdir: /app
```

The agent works as a user named `agent`, not root, in the image's working directory. Files it must not change, such as the data, can stay owned by root.

## Services the agent works with

A task whose agent needs other services, such as a database or a mock API, puts a `docker-compose.yaml` (or `compose.yaml`) next to the Dockerfile. Docker, remote Docker and Daytona run its services; Modal, AgentCore and Apple Container run one container, so they refuse such a task before it starts rather than run it without its services.

## Other task formats

BenchFlow also reads the `task.toml` layout that Terminal-Bench and many other benchmarks use. The files map one to one:

| task.md layout | task.toml layout | Holds |
|---|---|---|
| `task.md` | `task.toml` + `instruction.md` | settings and instruction |
| `verifier/` | `tests/` | tests and reward code |
| `oracle/` | `solution/` | the reference solution |

When both forms of a part exist in one task, they must match: `bench check` and `bench run` refuse a task whose `verifier/` and `tests/` differ, or whose `task.md` disagrees with its `task.toml`. `bench import` converts a `task.toml` task to `task.md`; [Bring what you have](../evaluate/bring-what-you-have.md) covers running such benchmarks as they are.
