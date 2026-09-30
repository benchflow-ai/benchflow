# task.md draft 2 packages

BenchFlow runs task.md draft 2 packages natively. Point `bench eval run --tasks-dir`, `bench tasks check`, `bf.Evaluation`, or `RolloutConfig` at a draft 2 folder, and BenchFlow runs it like any task: same sandboxes, agents, trial layout, and job tooling. Draft 2 is a design draft, so this page says what BenchFlow honors today, field by field, and what it refuses. It refuses a field it cannot honor by name, before anything runs, and never skips one silently.

A draft 2 package is a folder: one `task.md` (the instruction, then typed fenced blocks, with the config in a `toml task` block last), `sandbox/` (the agent's Dockerfile and its files), `verifier/` (`test.sh`, `rubric.json`, `judge.md`), `oracle/solve.sh`, and optionally `controls/`, `evidence/`, and `family/`. It is not BenchFlow's own `task.md` ([task-authoring-task-md.md](./task-authoring-task-md.md)), which opens with YAML frontmatter.

## Write a task

The smallest package, task-md's `examples/hello-world`:

````markdown
Create a file `/app/hello.txt` that contains exactly `Hello, world!` with no trailing newline.

```toml task
name = "examples/hello-world"
title = "Hello, world"
version = "1.3.0"

[agent]
timeout = "2m"
```
````

```text
hello-world/
├── task.md
├── sandbox/Dockerfile          FROM ubuntu:24.04, WORKDIR /app
├── verifier/rubric.json        one gate, decided by the test test_hello
├── verifier/test.sh            writes /logs/verifier/ctrf.json
└── oracle/solve.sh             printf 'Hello, world!' > /app/hello.txt
```

`verifier/test.sh` reports what happened in a CTRF test report, `/logs/verifier/ctrf.json`; it does not score. The rubric says what counts, and the runtime scores it.

## Check it

```bash
bench tasks check path/to/hello-world
```

A draft 2 folder gets three reports, in order:

1. **The reference checker's**, exactly what task-md's `python3 tools/taskmd.py check` prints: the parse, what grading needs, each judge's settings with defaults marked, and the worst-case grading time. BenchFlow runs the reference tool itself (below).
2. **BenchFlow's**: each field it refuses (`refused: ...`, which fails the check), each field it refuses only when an agent runs (`refused when an agent runs ...`), and how it honors or records every other field.
3. **The native package's**: when nothing is refused, BenchFlow materializes the task (a family at the reference checker's sample seed) and runs its own structural checks on the result.

## Run it

```bash
bench eval run --tasks-dir path/to/hello-world --agent oracle --sandbox docker   # the reference solution scores 1
bench eval run --tasks-dir path/to/hello-world --agent nop --sandbox docker      # doing nothing scores 0
bench eval run --tasks-dir path/to/suite --agent claude-agent-acp --model claude-sonnet-4-6 --sandbox docker
```

A `--tasks-dir` of draft 2 packages runs each one; a package BenchFlow refuses is skipped with a warning that names its fields, and the rest run. The Python API is the same: `bf.Evaluation(tasks_dir=...)`, or `RolloutConfig(task_path=...)` for one trial.

- **Families** (`[family]`, `family@1`) run per seed: `--seeds 0-4` or `--seeds 3,9`. BenchFlow runs the generator for each seed in a fresh container of the task's own image (offline, as the host's unprivileged uid, the package read-only at `/package`), fills the instruction's `{{placeholders}}`, bakes the instance's `agent/` files into the image, puts its `verifier/` files at `verifier/instance/`, and gives the oracle and the verifier `TASKMD_SEED` and `TASKMD_PARAMS`. Each seed is its own trial, `<task>--seed-<n>`. This needs a Docker daemon on the machine that loads the task. A family without `--seeds` is refused.
- **Controls** (`[integrity.controls]`): the do-nothing run is `--agent nop`. A control script runs as the oracle of a control variant: `benchflow.taskmd.TaskMdFormat().materialize_variant(task, out_root, control="known-bad-line-fit")` writes the package (`benchflow.taskmd.controls(task)` lists the ids), and `--agent oracle` runs it.
- **Model judges** need credentials in BenchFlow's own environment: `ANTHROPIC_API_KEY`, or a Claude Code OAuth token in `CLAUDE_CODE_OAUTH_TOKEN`. `BENCHFLOW_TASKMD_JUDGE_MODEL` substitutes a model, for every role (`claude-haiku-4-5-20251001`) or per role (`agent=...,llm=...`); every verdict records the model that gave it. `BENCHFLOW_TASKMD_MAX_JUDGE_SESSIONS` caps the judge sessions one process starts, for a budgeted run. A run whose judges cannot be served is refused before the solver starts.
- **Shared rubrics** (`extends`) are never fetched: set `TASKMD_SHARED_RUBRICS` to a folder holding `<name>.json` (for `https://task.md/rubrics/code-change@1`, `code-change@1.json`), as the reference tools do.

A trial's `verifier/` holds what the spec asks the verifier to write: `ctrf.json` (the script's report), `review.json` (one verdict per criterion, in task.md's review-1 shape), and `reward.json` with `reward` (the rubric's headline), `strict`, `partial`, and `timed_out`. A reward file the script wrote is kept as `script-reward.txt` or `.json` and ignored, since a rubric's reward is computed. With model judges, `verifier/taskmd-judge/` holds each session (its prompt, tool calls, usage, and accepted review), the solver's `trajectory-1.json`, and `fs/`, what the judges could read.

## How BenchFlow runs a package

**Detection.** The built-in format `taskmd` claims a folder whose `task.md` does not open with `---`. Every native, v0.6, and robouse `task.md` opens with YAML frontmatter, so no native folder is ever claimed. An empty `task.md`, a folder without one, and a legacy `task.toml` folder are left to the native loader.

**Parsing.** BenchFlow parses with task-md's own reference parser, vendored verbatim in `src/benchflow/taskmd/_vendor/` with the commit `VENDOR.json` pins, so a package parses exactly as `tools/taskmd.py json` parses it, diagnostics included. Any error the reference parser reports refuses the package. `tests/test_taskmd_differential.py` compares BenchFlow's parse with the tool's recorded output on the fixtures, and with a task-md checkout at hand (`TASKMD_REPO`), with the live tool on every example.

**Materializing.** The package becomes a native one under `~/.cache/benchflow/task-formats/taskmd/tasks/<key>/<name>/` (`BENCHFLOW_TASK_FORMAT_CACHE` moves it): `task.md` with native frontmatter and the agent's instruction, `environment/` from `sandbox/` (or a Dockerfile `FROM [sandbox] image`), `verifier/` or `tests/` where `[verifier] mount` places it (`/verifier` or `/tests`), `oracle/` or `solution/` (`/oracle` or `/solution`), and `metadata.taskmd` recording the source, its tree hash, the reference commit, and the draft 2 config. `<name>` is the source folder's name, so trial names and `--include` match it. `<key>` hashes the source tree and the writer, so a changed package gets a new folder.

**Grading.** The native package's `verifier/verifier.md` selects the `taskmd` strategy, which follows the spec's order: the saved outputs are copied for the judges before any script runs; `test.sh` runs in the task's working folder, bounded by `[verifier] timeout`; each test-judged criterion is decided from `ctrf.json` by the spec's matching rule; the model judges run; and the rubric is scored (gates, points, levels, continuous scores, `method`, `headline`, `pass_threshold`). A package with no rubric keeps Harbor's contract: `test.sh` writes the reward. When the verifier's network is `"none"`, a shared verifier's container is taken offline before `test.sh` runs if the agent's run left it online, and the trial is not scored if it cannot be.

**Judges** (`llm` and `agent` roles). BenchFlow compiles each session's prompt with the vendored `judge-prompt@1` compiler (`tests/test_taskmd_judge_vectors.py` holds it to the spec's four vectors, byte for byte) and runs `judge-loop@1` over the Anthropic Messages API: the `llm` role gets its evidence in the prompt and `submit_review`; the `agent` role gets `read`, `run`, and `submit_review`, each result in the spec's format, with the tool-call, token, and time budgets and the runtime's budget line. `submit_review` is checked against the assignment (three rejections end a session), each citation is checked against what the judge was shown and labeled `solver`, `solver-executed`, `judge`, or `environment`, `cite = "independent"` turns a pass without an independent citation into a fail, samples combine by `aggregate`, and a failed sample is judged once more. Lazy grading skips model-judged criteria once a test gate fails, except those that carry penalties, and a criterion whose every file the solver never saved fails without a session. An agent judge's `run` executes in the separate verifier sandbox, a fresh container of the task's image that holds only the saved outputs: BenchFlow removes `/verifier`, `/tests`, and any oracle from it first, hides `/logs/verifier` and `/logs/agent` from the runner's uid (they are mounted from the trial's folders, so they are never deleted), restores the saved outputs and a fresh working folder before every call, and runs the command as an unprivileged uid with no network, `bash --noprofile --norc -c`, stdin at `/dev/null`, under the call's timeout.

**Stages.** Stages that unlock one after another from the instruction (`at_start`, then `on_submit` or `after:<the stage before>`) become later turns of the same agent session: the stage's prompt arrives after the agent ends its previous turn. The oracle and `nop` run their scripts instead.

**Launch checks.** Before any sandbox starts, a trial is refused when an agent would run and the package holds a field BenchFlow honors only for scripted seats (an agent's `budget`, `system_prompt_append`), when `[agent] user` differs from the run's `--sandbox-user`, or when its judges cannot be served.

## What BenchFlow supports, field by field

"Honored" maps the field onto BenchFlow, "Partly" honors the values the column names and refuses the rest, "Refused for agents" honors the field for the oracle, controls, and `nop` and refuses an agent run, "Refused" refuses the package, and "Recorded" keeps metadata and scoring rules. Scoring rules (`[runs]`, `[integrity.controls]`, a rubric's validation) are checked only for a result that claims to be scored, and no BenchFlow run claims that yet, so they are recorded as not checked. `benchflow.taskmd.plan.SUPPORT` is this table; `tests/test_taskmd_support.py` checks that every key the reference tools document gets a decision.

| Field | Status | How, or why not |
|---|---|---|
| name, version, description, authors, keywords | Recorded | native task.name, version, description, authors, keywords |
| title | Recorded | metadata.title |
| [about] <key> | Recorded | native metadata |
| [agent] timeout | Honored | agent.timeout_sec |
| [agent] on_timeout | Partly | "grade" (BenchFlow grades what the agent left and records timed_out); "grade-flagged" and "fail" are refused |
| [agent] budget, [agent.budget] tool_calls, tokens | Refused for agents | BenchFlow enforces no tool-call or token budget on an agent; a scripted seat never approaches one |
| [agent] user | Partly | the oracle runs as it; an agent runs as the run's --sandbox-user, and a run whose agent user differs is refused |
| [agent] network | Partly | equal to [sandbox] network, or narrower than it: "none" runs the agent's phase offline in an online sandbox, a host list is an agent allowlist. A network wider than the sandbox's is refused, since the agent works inside the sandbox's container |
| [agent] network_reason | Honored | reviewer documentation: nothing to do at run time |
| [agent] system_prompt_append | Refused for agents | BenchFlow's harnesses take no system prompt addition |
| [agent] timeout_basis | Partly | "wall" (BenchFlow counts wall time); "environment" is refused |
| [sandbox] image | Honored | sandbox.docker_image, or FROM in environment/Dockerfile |
| [sandbox] os | Partly | "linux"; "windows" is refused |
| [sandbox] cpus, memory, disk | Honored | sandbox.cpus, memory_mb, storage_mb (a whole number of CPUs) |
| [sandbox] build_timeout | Honored | sandbox.build_timeout_sec |
| [sandbox] gpus | Honored | sandbox.gpus |
| [sandbox] gpu_types, tpu | Partly | gpu_types to sandbox.gpu_types; a TPU is refused |
| [sandbox] network | Partly | "none" (no-network), "open" (public), a host list (allowlist); { block = [...] } is refused |
| [sandbox] workdir | Honored | sandbox.workdir |
| [sandbox] env | Partly | literal values to sandbox.env; ${VAR} values are refused |
| [sandbox] skills | Refused for agents | BenchFlow installs task skills only in its with-skill mode; the oracle and nop run without them |
| [sandbox] mcp | Honored | sandbox.mcp_servers (Harbor's server tables, which an import keeps) |
| [sandbox] mounts | Partly | empty; task files are not mounted at start yet |
| [sandbox] services, [[sandbox.services]] name, image, build, command, env, ready | Partly | Compose services beside main; refused with network = "none", or a build folder outside sandbox/ |
| [sandbox] compose | Partly | only sandbox/docker-compose.yaml |
| [sandbox] ready, [sandbox.ready] run, interval, timeout, start_period, start_interval, retries | Honored | sandbox.healthcheck |
| [sandbox] outputs, [[sandbox.outputs]] path, save_as, exclude | Honored | native artifacts; the judges read the saved outputs |
| [[sandbox.outputs]] max_bytes, service | Refused | per-output caps and service outputs are not implemented |
| [sandbox] boundary | Partly | "container"; gvisor, microvm, and vm are refused |
| [sandbox] clock, [sandbox.clock] start, advance, enforce | Refused | BenchFlow sets no clock |
| [sandbox] timezone | Refused | BenchFlow cannot check that the image has the zone's data |
| [verifier] timeout | Honored | bounds test.sh; each judge session keeps its own timeout |
| [verifier] user | Honored | verifier.user |
| [verifier] env | Partly | literal values to verifier.env; ${VAR} values are refused |
| [verifier] network | Partly | a shared verifier: "none" (taken offline with iptables) or "open" (over an open sandbox or an agent allowlist); a separate verifier: "none" or "open" for its own sandbox. A host list written for the verifier is refused, since BenchFlow holds only the agent's uid to one; a host list the verifier only *inherited* from [sandbox] network is honored as binding the agent, with the shared verifier running as root outside it |
| [verifier] isolation | Honored | "shared", or "separate" (verifier.sandbox_mode: separate) |
| [verifier] sandbox, [verifier.sandbox] <key> | Partly | image, os, cpus, gpus, gpu_types, memory, disk, workdir, env, build_timeout, and empty mcp; the rest is refused. The image is found in the spec's order: [verifier.sandbox] image, verifier/Dockerfile, the task's image |
| [verifier] snapshot, [[verifier.snapshot]] run, reads, service, timeout, user | Refused | snapshot commands are not run |
| [verifier] combine_stages, unreached_stages | Honored | no stage is graded on its own, so the task's verifier alone decides the reward |
| [verifier] mount | Partly | "/verifier" or "/tests" |
| [verifier] feedback | Honored | BenchFlow shows an agent none of its review |
| [verifier] models | Partly | false; true (a script that calls a model) is refused |
| [verifier] judges and its role tables | Partly | llm and agent roles run judge-loop@1 (docs/task-authoring-taskmd-v2.md); vlm, panel, effort, a hosted harness, resources, services are refused |
| [verifier] human, [verifier.human] <key> | Refused | human assessment is not supported |
| [oracle] env | Partly | literal values to oracle.env |
| [oracle] mount | Partly | "/oracle" or "/solution" |
| [world], [tiers], [conventions] (every key) | Refused | worlds, tiers, and conventions are not provided |
| [stages.<name>] unlock | Partly | a chain from the instruction (at_start, on_submit, after:<previous>) becomes turns of one session; on_request and at_turn are refused |
| [stages.<name>] submit, mounts, agent, verifier, gate, ready, outputs | Refused | stages graded or set up on their own are not implemented |
| [roles.<name>] <key>, [interaction] <key> | Refused | multi-agent roles are not mapped (the spec leaves each role's prompt and order open) |
| [user] <key> | Refused | simulated users are not mapped |
| [variants.<name>] <key> | Recorded | BenchFlow runs the base task |
| [matrix] <key> | Refused | condition grids are not run |
| [family] generator, seed_param, params, splits, database | Honored | family@1 per seed (--seeds): the generator runs in a fresh container of the task's image |
| [training] reward, learnable_band, difficulty, admit | Recorded | metadata |
| [training] fork | Honored | a bench eval run sets every episode up from scratch |
| [training] max_concurrent | Refused | not enforced across a run's trials yet |
| [trajectory] require | Partly | "tool-calls"; the rest is refused |
| [preference] <key> | Refused | the human-preference protocol is not run |
| [integrity] profile | Partly | "shared-hardened", or "separated-verifier" with isolation = "separate"; "runner-separated" is refused |
| [integrity] intended_use | Honored | a BenchFlow eval run is not training |
| [integrity] canary, threat_model, residual_risks | Recorded | metadata |
| [integrity] forbidden_sources | Refused | not blocked yet |
| [integrity] resources, [integrity.resources] <key> | Refused | per-path resource classes are not enforced |
| [integrity] answers | Honored | judges are never served files the verifier writes |
| [integrity] controls and its keys | Recorded | scoring rule, not checked; controls run as control variants (Python API) and --agent nop |
| [runs] <key> | Recorded | scoring rule, not checked |
| [[credits]], [provenance], [import.<format>] | Recorded | metadata |
| x- keys and [x-<name>] | Partly | metadata; [x-benchflow] is refused |

The rest of the package:

| Part | Status |
|---|---|
| The instruction, and a canary comment before it | Honored: the canary is stripped from what the agent sees, and an instruction holding a reserved native heading is kept intact |
| ` ```stage <name> ` blocks | Partly, as the stage's unlock row says |
| ` ```role <name> ` and ` ```user ` blocks | Refused, with `[roles]`, `[interaction]`, and `[user]` |
| ` ```notes ` blocks | Recorded: never shown to an agent |
| `sandbox/` | Honored: the build context of `environment/` |
| `verifier/test.sh` | Honored: run in the task's working folder |
| `verifier/rubric.json` | Honored: test, llm, and agent criteria; rule, world, replay, human, vlm, and panel criteria are refused; `extends` needs `TASKMD_SHARED_RUBRICS` |
| `verifier/behaviors.json` | Refused when it watches or pairs any behavior: monitors, tags, and consequences are not run |
| `verifier/judge.md`, a role's `brief` | Honored: the brief of judge-prompt@1 |
| `verifier/verifier.md` | Refused: a BenchFlow verifier document would replace task.md's verifier contract |
| Evidence items | Honored: files, `trajectory`, `trajectory:reasoning`, `tests`; refused: `diff:`, `screenshots`, `video`, `request-record`, world streams |
| `oracle/` | Honored: `--agent oracle` runs `solve.sh` |
| `controls/`, `evidence/` | Recorded: never placed in the agent's image; controls run as control variants |
| `family/` | Honored with `--seeds` |
| `stages/<name>/` | Refused: stages graded on their own |
| `world/`, `variants/` | Refused with `[world]`; unused files of the base task otherwise |

## Harbor parity

An imported Harbor task should score what the original scores. To check that, 56 Harbor tasks were each run twice on one host, on Docker, with the oracle and with a do-nothing agent: once on BenchFlow's native path over the Harbor `task.toml` folder, and once on the `taskmd` format over the draft-2 package `task-md/tools/convert.py import` writes from it. The tasks were Terminal-Bench 2's terminal-mini set and BenchFlow's own Harbor-format examples and network, verifier-mode and sidecar matrices.

Twenty-six of them ran and scored on both paths, and every score is identical; twenty-two of those separate the oracle from doing nothing (1 and 0 on both paths), and four score 0 for both agents on both paths. Eleven more reach the same partial or failed outcome on both paths, such as a task that ships no oracle. Sixteen are declined by both paths — BenchFlow's native path raises `UnsupportedTaskFeatureError` for Harbor multi-step `steps` and per-step networks, and the `taskmd` format refuses the same task naming the field.

Three disagree, all in the network-policy matrix, and all because the `taskmd` format is deliberately stricter than the native path:

- `dynamic/e-a-diff` and `dynamic/e-a-diff-v-match` give the agent's phase *more* network than the sandbox. The agent works inside the sandbox's container, so this cannot be honored without rebuilding the container mid-run; the native path ignores the override and scores 0, and the `taskmd` format refuses naming `[agent] network`.
- `dynamic/e-v-diff` asks for an offline verifier in an image with no `iptables`. The native path ignores the setting and scores 0; the `taskmd` format reports that it could not take the verifier offline and does not score the trial.

Refusing by name rather than scoring a task whose settings were not honored is the point of the format hook, so these three stand as they are.

## Where BenchFlow's judges depart from the spec

Each verdict records the setup that produced it (`judge.model`, `judge.harness`, `judge.prompt_sha256`, `judge.judge_setup_sha256`, `judge.evidence_sha256`), and each session's record under `verifier/taskmd-judge/` shows what it sent and received. These rules of docs/runtime/judging.md are not yet honored:

- **The proxy.** Judges call the Anthropic Messages API directly with BenchFlow's own credentials, not through `proxy@1` with a seat token, so the proxy's request policy and its token records do not apply.
- **The system message.** With a Claude Code OAuth token, the system message is Claude Code's identity followed by `SYSTEM_PROMPT_1`, since Anthropic accepts such tokens only so; with an API key it is `SYSTEM_PROMPT_1` alone. Each session records which (`system_note`).
- **Wire APIs and roles.** Only the Anthropic Messages API is spoken, so a judge model must be Anthropic's. The `vlm` and `panel` roles, `effort`, a hosted harness, an agent judge's own `resources` and `services`, and images in an `llm` prompt are refused.
- **Seeds.** The Messages API takes no seed, so each session's `judge-seed@1` seed is computed and recorded, not sent.
- **Runners.** An agent judge's `run` uses one fresh container per session, reset before every call (the saved outputs restored, a fresh working folder and HOME, the uid's processes killed afterwards), not a fresh container per call: state a command leaves outside the saved outputs, such as in `/tmp`, can survive to the next call. Runners get a minimal environment (the image's `PATH`, `HOME`, `LANG`, `PYTHONSAFEPATH`, `PYTHONNOUSERSITE`), not the image's whole `Env`. No access watch runs, so every `run` result is labeled `solver-executed`, which the spec allows. The runner needs `[verifier] isolation = "separate"` and a verifier with no network; otherwise the agent role is refused.
- **Attribution.** A criterion left with fewer than `min_samples` valid samples, or a judge that cannot be reached, leaves the trial unscored (a verifier error) instead of being attributed by judging the reference with the same setup. A missing or malformed `ctrf.json` fails the test criteria without checking whether the script also fails on the reference.
- **Records.** `review.json` is the verifier-only record; the published form (`record = "published"`, rationales and quotes of protected verdicts withheld, HMAC'd prompt hashes) and `behavior-tags.json` are not written. PDF text extraction, notebook cells, and citations of the task's own inputs (outside the saved outputs) are not resolved; such citations are recorded as unverified with the reason. Files the verifier writes for the judges (`/logs/verifier/judge/`) are not served: a verifier that writes them leaves the trial unscored. Injection controls are not built or run.
- **Kept copy.** The saved outputs are copied through the sandbox's own download, after a probe that runs inside the container, rather than read from outside it with no symbolic link followed; links and special files are refused after the copy, and the size caps are checked on the copy.

## Interaction: stages, roles, and users

BenchFlow's scene lifecycle offers turns of one session, multiple agents taking turns, and a simulated user between rounds. What maps cleanly is implemented: a chain of stages becomes turns of the run's agent. The rest is refused, for these reasons:

- **Stages** that unlock `on_request` or `at_turn:<n>`, that name a `submit` file or `mounts`, or that are graded, timed, or checked on their own (`agent`, `verifier`, `gate`, `ready`, `outputs`, `stages/<name>/`): a BenchFlow turn ends only when the agent stops, and nothing mounts files or grades between turns. BenchFlow refuses Harbor's multi-step tasks, which import as graded stages, on its native path too (`steps`).
- **Roles** (`role` blocks, `[roles]`, `[interaction]`): BenchFlow can run roles as turns of different agents in one sandbox, but task.md does not say which role receives the instruction, whether a role also sees it beside its role prompt, or in which order roles act, and `tools`, `workspace = "read-only"`, and `policy` have no BenchFlow counterpart.
- **Simulated users** (` ```user `, `[user]`): BenchFlow's user loop starts a fresh agent session every round, so the agent would lose the conversation it is having with the user, and it runs the verifier between rounds, putting `verifier/` in the agent's sandbox mid-run and showing the user model the verifier's output. Neither fits a persona that answers the agent's questions.

## The reference tools in BenchFlow

`src/benchflow/taskmd/_vendor/` holds `tools/taskmd.py` and `tools/judgeprompt.py`, byte-identical to task-md at the commit `VENDOR.json` pins; lint and type checks skip them. `tools/rubrics.py` is not vendored, since importing it adds its folder to `sys.path`; its `score()` and `matching_tests()` are ported in `benchflow.taskmd.grading` and compared with the upstream file on a random battery when `TASKMD_REPO` is set. To follow a new task-md commit, copy the two files again, update `VENDOR.json`'s commit and hashes, refresh `tests/fixtures/taskmd/` (its `golden/` from the new `tools/taskmd.py json`), and run the tests with `TASKMD_REPO` pointing at a checkout of that commit.
