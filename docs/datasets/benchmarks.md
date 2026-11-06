# Benchmarks

BenchFlow ships converted copies of public benchmarks, and runs task folders in the task.toml format as they are.

```bash
bench run benchflow-ai/skillsbench/tasks/edit-pdf --agent gemini --model gemini-3.1-flash-lite-preview
GEMINI_API_KEY=... bench run --config benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml
bench run -d terminal-bench@2.0 --agent claude-agent-acp --model claude-haiku-4-5-20251001
```

## Available benchmarks

| Benchmark | Tasks | Verification | Run it |
|---|---|---|---|
| Terminal-Bench 2 | 89 | Task-specific tests | `bench run -d terminal-bench@2.0`; list tasks with `bench hub show terminal-bench@2.0` |
| [Harvey LAB](https://github.com/harveyai/harvey-labs) | 1,251 | LLM judge, per criterion | `bench run --config benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml` |
| [ProgramBench](https://programbench.com) | 201 | Unit tests | [Generate the tasks first](#programbench) |
| [SkillsBench](https://github.com/benchflow-ai/skillsbench) | 94+ | Unit tests | `bench run --source-repo benchflow-ai/skillsbench --source-path tasks` |
| ClawsBench | per task | Unit tests | Mock Gmail, Slack, Calendar, Docs and Drive. See [Environment manifests](../environment-plane.md) |
| chi-bench | per task | Unit tests | A hospital simulator. See [Environment manifests](../environment-plane.md) |

BenchFlow also runs PrimeIntellect Verifiers environments with `--source-env`, and exports results in Inspect AI and OpenReward formats. See [Any benchmark](../running-any-benchmark.md).

## Pinned dataset versions

For results others must reproduce, such as leaderboards and papers, run a dataset version instead of a folder:

```bash
bench run -d terminal-bench@2.0 --agent claude-agent-acp --model claude-haiku-4-5-20251001
bench run -d terminal-bench@2.0 --include fix-git --agent claude-agent-acp --model claude-haiku-4-5-20251001
```

Versions never float: there is no `latest`. `--include` and `--exclude` filter the version's task list.

A registry (`registry.json` at a dataset repo's root) pins each version to a commit and a sha256 per task. See skillsbench's [dataset versioning](https://github.com/benchflow-ai/skillsbench/blob/main/docs/dataset-versioning.md). Before the run, BenchFlow:

- clones that commit into `.cache/datasets-v2` and recomputes every task's digest. Any mismatch stops the run;
- leaves out folders the registry entry does not list;
- refuses to run on a BenchFlow outside the entry's `bench_version` range, since the results may not compare. `--ignore-bench-version` overrides this and records a warning.

`-d` reads BenchFlow's registry and nothing else: its published list (`registry.json` in the skillsbench repo) and the entries BenchFlow ships in the package for datasets that list does not hold yet. `terminal-bench@2.0` is one of those, every task pinned to a commit and a digest. `bench hub list` lists them all, and `bench hub show terminal-bench@2.0` lists one dataset's tasks.

A dataset BenchFlow's registry does not have stops the run with `Dataset '<name>@<version>' not found in BenchFlow's registry`, and the next step says how to read another registry. `--registry <url or file>` reads only that registry. Its entries may pin no digests:

- their tasks are not digest-checked, and the run line says how many were not;
- an entry at `HEAD` runs the repo's default branch as it is when the run starts;
- a task folder with local edits is refused, and your edits are kept.

An entry must point at `github.com/<org>/<repo>` with a full commit id or `HEAD`. If the registry has the name but not the version, the error lists the versions it has.

Each trial's `result.json` and `config.json` record `dataset_name`, `dataset_version` and `task_digest`, so results group by `dataset@version`. A run of a plain folder records `task_digest` only. `bench tasks digest <task-dir>` prints a folder's digest.

## SkillsBench

Use `bench run --source-repo benchflow-ai/skillsbench --source-path tasks` until a pinned version whose `bench_version` range accepts 0.9.0 is published; versions 1.0 and 1.1 do not.

```bash
bench run benchflow-ai/skillsbench/tasks/edit-pdf --agent gemini --model gemini-3.1-flash-lite-preview
bench run benchflow-ai/skillsbench/tasks --agent claude-agent-acp --model claude-sonnet-5 --sandbox daytona --concurrency 32
```

A path inside the repo picks a subset: `tasks/edit-pdf` is one task. `--task-skills` installs each task's own skills for the agent.

Daytona caps each sandbox at 10 GB, so run heavy tasks such as `latex-formula-extraction` with `--sandbox docker`. See [Daytona](../sandboxes/daytona.md).

For the four-cell skill-toggle matrix (two models, with and without skills), see [SkillsBench skill matrix](../datasets/skillsbench-skill-matrix.md).

## Harvey LAB

```bash
GEMINI_API_KEY=... bench run --config benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml

bench run benchflow-ai/benchmarks/datasets/harvey-lab/tasks/corporate-ma-analyze-cim-deal-teaser-scenario-01 \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox docker
```

The config clones the converted tasks from `benchflow-ai/benchmarks`. Task names are flattened: `corporate-ma/analyze-cim-deal-teaser/scenario-01` becomes `corporate-ma-analyze-cim-deal-teaser-scenario-01`.

The verifier is an LLM judge, so it needs `ANTHROPIC_API_KEY` as well as the agent's key.

**Parity runs.** `harvey-lab-harness` runs Harvey LAB's own agent loop (6 tools, its system prompt) inside BenchFlow's sandbox. Use it to run the same agent on the original and the converted tasks:

```bash
bench run benchflow-ai/benchmarks/datasets/harvey-lab/tasks/corporate-ma-analyze-cim-deal-teaser-scenario-01 \
  --agent harvey-lab-harness --model gemini-3.1-flash-lite-preview \
  --sandbox docker --concurrency 1 --jobs-dir jobs/smoke-test/harvey-harness
```

## ProgramBench

201 program-reconstruction tasks in 7 languages: C, Rust, Go, C++, Java, Haskell and Bash. The tasks are not in the repo; generate them from a ProgramBench checkout first.

You need:

- Docker on a Linux x86_64 machine (the images are `linux/amd64` only);
- about 20 GB of disk for the images;
- internet access during verification, for the Hugging Face test files;
- a ProgramBench checkout for `--programbench-dir`. Without it, the generator tries the installed `programbench` package.

Generate the tasks into `.cache/programbench-benchflow`, the folder the shipped config reads:

```bash
# All 201 tasks
python -m benchmarks.programbench.main --programbench-dir ~/programbench \
    --output-dir .cache/programbench-benchflow

# One task
python -m benchmarks.programbench.main --programbench-dir ~/programbench \
    --output-dir .cache/programbench-benchflow --task-ids abishekvashok__cmatrix.5c082c6
```

Then run all of them, or one:

```bash
bench run --config benchmarks/programbench/programbench-gemini-flash-lite.yaml
bench run .cache/programbench-benchflow/abishekvashok__cmatrix.5c082c6 \
  --agent gemini --model gemini-3.1-flash-lite-preview --sandbox docker
```

Check the generated task's structure before running an explicitly selected agent:

```bash
bench check .cache/programbench-benchflow/abishekvashok__cmatrix.5c082c6
```

The generated reference oracle is incomplete for some tasks. For the cmatrix example, it tries to clone GitHub inside a sandbox with internet disabled, and the upstream checkout supplies no `compile.sh`. Running without `--agent` selects that oracle and does not establish that the converted task is solvable. `bench check` validates structure; it does not validate a reference solution. Keep the agent sandbox's network policy intact when preparing a task-specific reference solution.

## Task folders

`bench run` runs a task.toml task folder (`task.toml`, `instruction.md`, `environment/`, `solution/`, `tests/`) as it is. The task loader reads `task.toml` itself; nothing is converted.

```bash
bench run path/to/toml-task --agent claude-agent-acp --model claude-sonnet-5
bench check path/to/toml-task --sandbox daytona     # what this sandbox would refuse
```

The sandbox, agent adapter and verifier hardening determine how a task runs and is scored.

| `task.toml` part | What BenchFlow does |
|---|---|
| `[task]`, `[metadata]`, `schema_version` (or the task.toml format's `version`), `source`, `[solution.env]` | Read and honoured |
| `[agent]` | Timeout, user and network mode |
| `[verifier]` | Timeout, env, user and network mode |
| `[environment]` | Read as BenchFlow's `sandbox`: image, cpus, memory, storage, build timeout, env, skills dir, workdir, healthcheck, `allow_internet`, MCP servers |
| Root `artifacts = [...]` | Collected on Docker, remote Docker and Daytona (below) |
| An unknown key outside the tables below | Ignored with a warning, listed in `task.config.ignored_keys` |
| A key that misspells a known one, or any unknown key in `[environment]`, `[agent]`, `[verifier]` or `[steps]` | Refused with a did-you-mean (`TaskConfigKeyError`), so a typo never runs with a default |
| Features BenchFlow cannot run | Refused before launch (below) |

`bench check` and the run apply the same rules, so they agree.

**Unknown keys.** A newer task.toml key BenchFlow chooses to ignore is listed in `benchflow.task.imports.KNOWN_IGNORABLE_KEYS`. For a corpus you do not control, `BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS=1` loads unknown keys anyway for that run.

**Artifacts.** After the agent stops, and before verifier hardening, BenchFlow collects each declared artifact and everything in `/logs/artifacts` into the trial's `artifacts/` folder:

- a declared artifact lands at `artifacts/<destination>`, by default the source's file name. A relative source is read from the agent's workspace;
- `artifacts-manifest.json` records each collection's status (`collected`, `empty`, `missing`, `refused`, `error`, `over_limit`) and each file's size and sha256;
- a collection stops at 1 GiB or 10,000 files. It refuses symlinks that leave the tree, a source that is a symlink, a destination that exists, and any `..` or absolute destination;
- a failed collection is a manifest status, never a failed trial.

**Refused before launch.** The run stops with `Task uses parsed runtime features that BenchFlow cannot execute` for:

- `steps` (multi-step tasks) and step `artifacts`;
- root `artifacts` on Modal, Apple Container and AgentCore;
- a separate verifier (`verifier.environment_mode = "separate"`, or `[verifier.environment]`) off Docker, remote Docker and Daytona, or with no verifier image. See [Separate verifier](../tasks/separate-verifier.md);
- `[[verifier.collect]]` hooks. An empty `collect = []`, counts as absent;
- `network_mode = "allowlist"` on `[verifier]`, or on sandboxes other than Docker, remote Docker and Daytona;
- an `[agent]` allowlist over a `no-network` or `denylist` `[environment]`;
- `no-network` or `denylist` on sandboxes that cannot enforce them;
- a verifier service other than `main`, off Docker and remote Docker;
- Windows `os` and TPUs;
- GPUs on backends other than Modal, or invalid [Modal GPU requests](../sandboxes/modal.md#gpus).

`bench check <dir> --sandbox <name>` lists the same refusals.

From Python, the inbound adapter reads a task.toml folder into an in-memory `InboundTask` and keeps unknown keys in `compatibility.config_extra`. `bench run` does not use it. `benchflow.adapters.inbound.detect_adapter()` picks the adapter for a folder.

## Parity validation

Parity validation checks that a converted benchmark keeps the original's results. It is for maintainers. The scripts live in each benchmark's folder:

```bash
uv run python benchmarks/harvey-lab/parity_test.py --mode full --harvey-root .cache/datasets-v2/harveyai/harvey-labs
ANTHROPIC_API_KEY=... uv run python benchmarks/harvey-lab/parity_test.py --mode eval-parity
GEMINI_API_KEY=... uv run python benchmarks/harvey-lab/parity_test.py --mode side-by-side
```

Recorded results are in each benchmark's `parity_experiment.json` and `benchmark.yaml`.

## Adding a benchmark

Each converted benchmark lives in `benchmarks/<name>/`:

| File | What |
|---|---|
| `benchflow.py` | The converter for the raw benchmark |
| `benchmark.yaml` | Metadata: task count, categories, verification method, parity results |
| `<name>-*.yaml` | Run configs for different agents and models |
| `parity_test.py` | The parity suite |
| `parity_experiment.json` | Recorded parity results |

`bench eval adopt <source>` creates this folder and has an agent convert the benchmark. `bench eval adopt <name> --verify` then checks parity. Harvey LAB and ProgramBench are the reference packages. See [Any benchmark](../running-any-benchmark.md#layer-2-convert-then-check).

## Next steps

- Run one: [Run a job](../jobs/run.md).
- Put the run in a file: [Configs](../jobs/configs.md).
- Benchmarks with running services: [Environment manifests](../environment-plane.md).
