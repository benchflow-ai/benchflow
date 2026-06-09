# task.md adapter capabilities

This guide turns the task.md dogfood validation work into the operating model
for benchmark adapters. The core rule is:

```text
foreign benchmark format
  -> native task.md package
  -> optional compatibility export
  -> optional hosted execution backend
```

Harbor, Pier, and Terminal-Bench split layouts are compatibility and execution
targets. They are not BenchFlow's canonical intermediate representation.

## Native package contract

Adapters that publish or standardize a benchmark should materialize native task
packages first:

```text
task-id/
  task.md
  environment/
  oracle/
  verifier/
    verifier.md
    rubrics/
```

Use `task.md` for the authoring surface: prompt, Harbor-compatible config,
roles, scenes, simulated-user intent, source provenance, and benchmark
metadata. Use `oracle/` for reference behavior and `verifier/` for scoring
logic, rubrics, judge inputs, reward details, and calibration evidence.

`solution/solve.sh` and `tests/test.sh` remain valid compatibility paths. In a
native package they should be treated as script strategies or export artifacts,
not as the whole task model.

## Harbor hosted environments

Harbor hosted environments can run exported split-layout tasks. The intended
flow is:

```bash
bench tasks export my-task exported/my-task --target harbor
harbor run -p exported/my-task -a <agent> -m <model>
```

The exported Harbor package should contain:

```text
exported/my-task/
  task.toml
  instruction.md
  environment/
  solution/
  tests/
  compatibility/export-report.json
```

The compatibility report is part of the contract. If it says the export is
lossy or wrapper-only, the native `task.md` package remains the source of truth
for benchmark semantics.

## Capability classes

| Class | Converts to native task.md | Harbor export | Examples |
|---|---|---|---|
| Native conversion | Lossless or near-lossless | Usually lossless | SkillsBench prompt tasks, simple Terminal-Bench-style tasks, deterministic file tasks |
| Native verifier package | Lossless when verifier metadata is represented | Usually wrapper or partial | Open-ended writing, research, report, rubric, LLM-judge, Reward Kit, agent-judge tasks |
| Native protocol adapter | Native metadata preserves protocol semantics | Wrapper-only unless backend implements protocol | ORS/OpenReward environments, AgentBeats assessor-agent evaluations |
| Wrapper-only export | Native package can explain semantics; export hides them in scripts/services | Runnable but not semantically lossless | ORS servers inside `environment/`, assessor loops inside `tests/test.sh`, external SaaS sessions |
| Unsupported or fail-closed | Parsed but not executable by selected runtime | Blocking diagnostic | Unsupported network policy, missing GPU/TPU backend, unavailable hosted protocol driver |

## What should convert cleanly

- Prompt-only tasks and deterministic verifier tasks.
- SkillsBench-style tasks, including skill/no-skill provenance.
- Harbor and Pier split packages whose `solution/` and `tests/` map directly
  to native `oracle/` and `verifier/`.
- Terminal-Bench-style tasks with a single instruction, container environment,
  solution script, and verifier script.
- Open-ended tasks whose rubric, judge model, calibration cases, and reward
  aggregation can be represented as a native verifier package.

## What should not be claimed lossless to Harbor

- ORS/OpenReward episode semantics: tool/action schemas, task splits, dense
  rewards, terminal `finished` signals, and reward streams.
- AgentBeats/AAA assessor-agent lifecycles: assessor setup, A2A task
  management, MCP resources, participant agent endpoints, and assessment
  orchestration.
- Multi-role, multi-scene, simulated-user, team handoff, or branch execution
  flows where the interaction graph is part of the benchmark.
- Human-in-the-loop approval and long-lived external account state.
- Process rewards and training evidence that must preserve action, memory,
  reasoning, or tool-output granularity.

These can often be wrapped so a Harbor hosted environment can execute them.
That wrapper is useful, but it is not a lossless Harbor task. Keep the native
package and export loss report as the auditable source of truth.

## Current implementation status

The dogfood validation pass validates the task.md-first path on real examples:

- Native `task.md` packages load and run through BenchFlow task discovery.
- Native `verifier/` packages run in the hardened verifier path; pytest
  hardening now uses `/verifier` for native packages and `/tests` for split
  compatibility packages.
- `verifier/verifier.md` can select `script`, `llm-judge`, `reward-kit`,
  `agent-judge`, and `ors-episode` verifier slices.
- SkillsBench task.md migration and all-task Daytona artifact coverage were
  checked against real c100 worker-sharded output.
- Worker-sharded result audits understand aggregate concurrency versus
  worker-local concurrency.

Still target work:

- `oracle/oracle.md`, so `solve.sh` becomes one oracle strategy rather than
  the only oracle model.
- Full OpenReward hosted environment import/export and ORS session driving.
- Full AgentBeats assessor-agent lifecycle support over ACP/A2A/MCP.
- A typed environment manifest inside `TaskPackage`, not only metadata.
- A conformance matrix for Harbor, Pier, Terminal-Bench, SkillsBench,
  ORS/OpenReward, AgentBeats, and open-ended rubric benchmarks.

## Adapter checklist

When adding or reviewing a converter:

1. Materialize native `task.md` packages by default.
2. Preserve source benchmark identity and hashes in `source` or
   `benchflow.compat`.
3. Keep foreign-only fields under compatibility metadata, not root config keys.
4. Put scoring semantics in `verifier/verifier.md`, rubrics, judge inputs, and
   reward artifacts.
5. Generate Harbor split layout only as an explicit export target.
6. Emit or preserve an export loss report for every compatibility target.
7. Fail closed when the selected runtime cannot honor parsed fields.
