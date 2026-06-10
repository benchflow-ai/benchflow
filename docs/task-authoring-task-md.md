# Authoring native task.md tasks

A native BenchFlow task is one `task.md` document plus sidecar directories.
The YAML frontmatter carries the task configuration; the markdown body **is**
the prompt. This page teaches the native format hands-on. For the normative
standard see [the task standard](./task-standard.md); for the legacy split
layout (`task.toml` + `instruction.md` + `tests/` + `solution/`) see
[Authoring tasks](./task-authoring.md).

When a directory contains both layouts, `task.md` is the authoritative task
definition — the runtime selects it and ignores the split pair.

---

## Minimal task — three files

```text
my-task/
├── task.md                # config frontmatter + prompt body
├── environment/
│   └── Dockerfile         # sandbox image
└── verifier/
    └── test.sh            # verifier entry point
```

That is the complete runnable surface: structural validation requires a task
definition (`task.md`, or the legacy `task.toml` + `instruction.md` pair), an
`environment/` directory with a `Dockerfile`, and a verifier directory with a
runnable entrypoint. An `oracle/` directory is optional.

```markdown
---
version: "1.0"
agent:
  timeout_sec: 300         # REQUIRED — fail-closed: omitting it fails validation
verifier:
  timeout_sec: 120
environment:
  cpus: 1
  memory_mb: 2048
---

Create a file `/app/hello.txt` containing exactly `Hello, world!`.
```

```bash
#!/bin/bash
# verifier/test.sh
REWARD=0
if [ "$(cat /app/hello.txt 2>/dev/null | tr -d '\n')" = "Hello, world!" ]; then
    REWARD=1
fi
echo "$REWARD" > /logs/verifier/reward.txt
```

Scaffold this shape with the CLI (task.md is the default format):

```bash
bench tasks init my-task                    # task.md, environment/, verifier/, oracle/
bench tasks check tasks/my-task             # structural validation
bench tasks check tasks/my-task --level schema   # frontmatter + prompt parse only
```

---

## Frontmatter

`task.md` must start with a `---`-delimited YAML frontmatter block, and the
frontmatter must be a mapping — a document without it fails to parse. The keys
fall into three classes.

**Task config keys** are the Harbor-compatible config surface, validated as
`TaskConfig`. Unknown keys are **rejected** (the schema is `extra="forbid"`),
so typos fail at parse time instead of becoming silently-ignored config:

| Key | Meaning |
|---|---|
| `schema_version` (alias `version`) | Config schema version, currently `"1.3"` |
| `task` | Package identity: `name` (`org/name` format), `description`, `authors`, `keywords` |
| `metadata` | Freeform mapping — difficulty, category, tags, anything descriptive |
| `agent` | Agent run policy: `timeout_sec`, `user`, `network_mode`, `allowed_hosts` |
| `verifier` | Verifier run policy: `timeout_sec` (default 600), `env`, `user`, `service`, … |
| `environment` | Sandbox: `docker_image`, `cpus`, `memory_mb`, `storage_mb`, `network_mode`, `env`, `workdir`, … |
| `oracle` | Oracle run policy: `env`, `timeout_sec` (import alias: `solution`) |
| `source`, `artifacts`, `steps`, `multi_step_reward_strategy`, `reward` | Provenance and Harbor-compatible extras |

`agent.timeout_sec` is **strictly required**: a task that omits it fails
validation closed rather than running with no wall-clock cap. Every other
config key has a default.

Declaring both `oracle` and the legacy `solution` alias in one config is
invalid and rejected; native tasks use `oracle`.

**Document orchestration keys** are parsed by `TaskDocument`, not `TaskConfig`:
`agents` (named roles with `agent`, `model`, `reasoning_effort`,
`capabilities`, …), `scenes` (ordered turns referencing declared roles — a
turn that names an undeclared role is a parse error), and `user` (simulated
user). `benchflow` is the reserved extension namespace.

**Authoring shorthands** are expanded during parsing and never reach the
canonical config under their short names:

| Shorthand | Expands to |
|---|---|
| `name: hello-world` | `task.name: benchflow/hello-world` (a `/` in the value keeps your org) |
| `image: ubuntu:24.04` | `environment.docker_image: ubuntu:24.04` |
| `verifier: verifier/` (string form) | `benchflow.verifier.path` / `.spec` / `.entrypoint` defaults |
| `oracle: oracle/` (string form) | `benchflow.oracle.path` |
| `profile: code-change` | Merges a named defaults bundle (see below) |

Profiles (`profile:` / `profiles:`) merge predefined default bundles —
`code-change`, `harbor-compatible`, `reward-kit`, `acceptance-live`,
`multi-agent`, `leaderboard-local` — under your explicit keys; an unknown
profile name is a parse error. `bench tasks normalize <task-dir>` prints the
fully expanded canonical document (`--write` replaces `task.md` in place), so
a minimal authored file and its canonical form never drift apart.

---

## Prompt body and prompts/ sidecars

The body below the frontmatter is the base prompt — free-form markdown, no
heading ceremony required. If the body contains no reserved section headings,
the entire body is the instruction the agent receives.

Four reserved headings are recognized for compatibility imports: `## prompt`,
`## role:<name>`, `## scene:<name>`, and `## user-persona`. Repeating the same
section heading is a parse error. New tasks should not use them;
multi-prompt material belongs in sidecar files under `prompts/`:

| File | Meaning |
|---|---|
| `prompts/role.<name>.md` | Role prompt — the whole file body is the prompt text |
| `prompts/scene.<name>.md` | Scene prompt |
| `prompts/user-persona.md` | Simulated-user persona |

Sidecar files take precedence over a reserved heading of the same name, so a
compat-imported task can be cleaned up incrementally. Runtime prompt
precedence for a turn is: inline turn prompt, then scene prompt, then role
prompt, then base prompt.

A multi-role task wires the pieces together in frontmatter:

```yaml
agents:
  roles:
    solver:
      agent: claude-agent-acp
scenes:
  - name: solve
    turns:
      - role: solver
```

with the solver guidance, if any, in `prompts/role.solver.md`. See
[docs/examples/task-md/](./examples/task-md/README.md) for runnable examples,
including real converted SkillsBench packages.

---

## Verifier package and strategy declaration

The native verifier directory is `verifier/` (`tests/` remains the legacy
alias; when both exist, `verifier/` wins and there is no fallback to
`tests/`). At verify time the directory is uploaded into the sandbox at
`/verifier` (legacy `tests/` uploads to `/tests`), and the verifier must write
its reward to `/logs/verifier/reward.txt` (and optionally
`/logs/verifier/reward.json`).

A plain `verifier/test.sh` is a complete verifier: with no other declaration,
the runtime executes it directly. The same contract as the legacy layout
applies — write a float `0.0`–`1.0` to `/logs/verifier/reward.txt`, then exit
`0`; a nonzero exit means verifier infrastructure failure, not a scored task
failure.

To declare *how* the task is scored, add `verifier/verifier.md`. Its
frontmatter must contain a `verifier:` mapping with at least one entry under
`strategies`; `default_strategy` selects which one runs (it defaults to the
first declared strategy and must name a declared one):

```markdown
---
document_version: "0.3"
verifier:
  name: my-task-verifier
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
---

## verifier intent

What the verifier measures and which task outputs it reads.
```

Five strategy types are recognized, each with fail-closed required fields:

| `type` | Required config | Notes |
|---|---|---|
| `script` | `command` | Runs as `cd /verifier && <command>`; local script files named in the command must exist in `verifier/` |
| `llm-judge` | `rubric` | Optional `model`, `input_dir`, and `context` *or* `context_file` (not both) |
| `reward-kit` | `root` | Optional `entrypoint` (default `reward.py`) and `criteria`; paths must be safe-relative |
| `agent-judge` | `role`, `isolation: verifier-only`, `inputs` | `role` must match a `## role:<name>` section in the verifier.md body |
| `ors-episode` | `inputs` | Optional `format`: `json`, `jsonl`, or `auto` |

An unknown `type` is a parse error. `bench tasks check` also verifies the
selected strategy is actually runnable — e.g. a `script` strategy whose
referenced files are missing, or an `llm-judge` strategy whose rubric file
does not exist, fails validation.

`outputs` declares the reward artifact contract (defaults shown above;
`details_json` and `aggregate_policy` are optional). `bench tasks check
--level publication-grade` additionally requires the native package shape:
`task.md`, native `oracle/`, `verifier/verifier.md` with rubric files, and an
explicit `reward_json` output contract.

---

## Oracle

`oracle/solve.sh` is the held-out reference solution (`solution/` is the
legacy alias; `oracle/` wins when both exist). Native oracles are uploaded to
`/oracle` in the sandbox (legacy `solution/` to `/solution`) and run instead
of an agent with `--agent oracle`:

```bash
bench eval create --tasks-dir tasks/my-task --agent oracle --sandbox docker
```

A correct task scores `1.0` on its oracle run before any model sees it.

---

## Migrating a legacy task

`bench tasks migrate` converts a `task.toml` + `instruction.md` pair into
`task.md`:

```bash
bench tasks migrate tasks/my-task                  # writes task.md, keeps legacy files
bench tasks migrate tasks/my-task --overwrite      # replace an existing task.md
bench tasks migrate tasks/my-task --remove-legacy  # delete the split pair and
                                                   # promote tests/ -> verifier/,
                                                   # solution/ -> oracle/
```

The migration is non-destructive by default and refuses to write anything
lossy: the generated document is re-parsed and must reproduce the original
config semantics and instruction text exactly, or the command fails. Unknown
`task.toml` keys that the schema does not model are preserved under
`benchflow.compat` in the generated frontmatter rather than dropped. After
migrating, validate the result:

```bash
bench tasks check tasks/my-task
bench eval create --tasks-dir tasks/my-task --agent oracle --sandbox docker
```
