# Task package reference

This page is the full contract of a BenchFlow task folder: its files, which one wins when two exist, the verifier's outputs, the evidence the acceptance checks read, and what import and export keep. For a guided start, see [Tasks](../tasks/overview.md).

Everything here runs today. Ideas that don't run yet are in the [design notes](../contributing/task-standard-design.md).

## Layout

```text
task/
|-- task.md
|-- environment/
|   `-- Dockerfile
|-- verifier/
|   |-- verifier.md        # optional
|   `-- test.sh
|-- oracle/
|   `-- solve.sh
`-- evidence/              # optional: files the acceptance checks read
```

| BenchFlow | task.toml | Holds |
|---|---|---|
| `task.md` | `task.toml` + `instruction.md` | Settings and instruction |
| `oracle/` | `solution/` | The reference solution |
| `verifier/` | `tests/` | Tests, reward code, rubrics and judges |

Selection is fail-closed:

- `task.md` wins over `task.toml` and `instruction.md`.
- `verifier/` wins over `tests/`. An empty or broken `verifier/` does not fall back to `tests/`.
- `oracle/` wins over `solution/`, likewise.

When both copies exist, `bench check` and `bench run` refuse the task unless they match (settings and instruction compared as parsed). Publish `task.md` as the only entry point; `bench check --level publication-grade` requires it.

## Versions

```yaml
schema_version: "1.3"        # task configuration schema version
benchflow:
  document_version: "0.6"    # BenchFlow-only task.md syntax
```

## Frontmatter keys

| Kind | Keys | Unknown keys |
|---|---|---|
| Task settings | `schema_version`, `task`, `metadata`, `agent`, `verifier`, `sandbox`, `oracle`, `source`, `artifacts`, `reward`, `steps`, `multi_step_reward_strategy` | Refused |
| Multi-agent keys | `agents`, `scenes`, `user` | Refused |
| BenchFlow extensions | `benchflow:` | Parsed; only the keys these pages document act |

Every key is described in [Configuration](../tasks/configuration.md).

## Verifier

Without `verifier/verifier.md`, BenchFlow runs `verifier/test.sh` (or `tests/test.sh` in a task.toml task). With it, the selected strategy runs, and `test.sh` is optional.

Native verifier files are mounted at `/verifier`; a task.toml task's `tests/` at `/tests`.

### A verifier with several strategies

```md
---
document_version: "0.3"
verifier:
  name: hidden-patch-and-quality
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
    criteria_runner:
      type: criteria-runner
      root: rewards/
      criteria: rewards/quality/criteria.toml
    judge:
      type: agent-judge
      role: verifier_judge
      model: gpt-5.5
      inputs: [trajectory/acp_trajectory.jsonl, /logs/artifacts/patch.diff]
      isolation: verifier-only
    ors:
      type: ors-episode
      reward_aggregation: last_non_null
      terminal_policy: finished_true
  rubric:
    combine: weighted_sum
    dimensions:
      correctness: {weight: 0.7, source: deterministic}
      maintainability: {weight: 0.2, source: judge}
      evidence_quality: {weight: 0.1, source: criteria_runner}
    files:
      human: rubrics/verifier.md
      structured: rubrics/verifier.toml
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/reward-details.json
    aggregate_policy:
      field: reward
      fallback: weighted_mean
---

## role:verifier_judge

Grade only the submitted artifact and declared evidence. Do not infer intent
from private oracle files or hidden verifier fixtures.
```

`rubric.combine` sets how the dimensions add up, and each dimension names the strategy that scores it. The `## role:verifier_judge` section is the judge's brief.

### Reward contract

| Rule | Behaviour |
|---|---|
| Files | `/logs/verifier/reward.txt` (one number) or `/logs/verifier/reward.json`; both are cleared before the verifier starts |
| Both present | `reward.json` wins; a numeric `reward` in it must equal `reward.txt`, or the trial scores 0 with a warning |
| Metrics only | Without a numeric `reward`, every top-level key must be a number. `outputs.aggregate_policy` computes the reward, or `reward.txt` supplies it. |
| Extra keys | Beside a numeric `reward`, keys that are not numbers are kept and not scored |
| Range | `[0, 1]`, or `verifier.reward_range`. A value outside is never clamped: the trial scores 0 with a warning and keeps `reward_out_of_range_info`. |
| Exit code | A reward written with a nonzero exit is scored. No reward gives score 0 with a warning. |
| Invalid files | Empty, malformed, non-finite, unsafe or conflicting reward files give score 0 with a named warning |
| Details | `reward-details.json` is kept when present |
| Reserved keys | `rubric`, `items`, `evidence`, `artifacts`, `metadata`, `reason`, `reasons`, `errors`, and payloads such as `metrics`, `regressions`, `participants`, `winner`, `raw`, `debug` are kept as written |

Aggregate methods: `mean`, `weighted_mean`, `weighted_sum`, `all_pass`, `any_pass`, `threshold`. See [Verifier](../tasks/verifier.md#reward-rules).

### Strategies

| Strategy | What runs |
|---|---|
| `script` | The declared command, from the verifier folder |
| `llm-judge` | BenchFlow's deliverables judge, with the strategy's rubric, model, input folder and context |
| `criteria-runner` | A `reward.py` runner inside the verifier folder. It writes `criteria-runner-manifest.json`, and with declared criteria checks `reward.json`'s metrics and computes `reward`. |
| `agent-judge` | A judge role over the declared inputs, scoped to the verifier |
| `ors-episode` | Reads declared OpenReward-style reward evidence and writes `reward.json` and `reward-details.json` |

A strategy BenchFlow can't run fails before scoring starts.

## Oracle

Native oracle files are mounted at `/oracle`; a task.toml task's `solution/` at `/solution`. Either is mounted only for an oracle run (`--agent oracle`), never while another agent runs. `bench export` writes `oracle/` as `solution/`. See [Oracle](../tasks/oracle.md).

## Evidence

`bench check --level acceptance` reads `benchflow.evidence`. Every artifact it names must exist and match its SHA-256 pin:

```yaml
benchflow:
  evidence:
    oracle_runs:
      required_reward: 1.0
      last_job: jobs/task-standard/oracle/...
      artifact: evidence/calibration/oracle-run.json
    verifier:
      reruns: 5
      flake_rate: 0.0
      report: evidence/calibration/verifier-stability-report.json
    review:
      anti_cheat: passed
      instruction_alignment: passed
      reviewer: benchflow
      artifact: evidence/calibration/review.json
    calibration:
      no_op_reward_max: 0.0
      known_bad_reward_max: 0.2
      partial_solution_range: [0.3, 0.8]
      report: evidence/calibration/calibration-report.json
      human_or_reference_examples:
        - name: gold
          expected_reward: 1.0
          artifact: evidence/calibration/gold-result.json
      judge_agreement:
        required: true
        sample_count: 5
        min_pairwise_agreement: 0.8
    trajectories:
      - path: trajectory/acp_trajectory.jsonl
        kind: acp
        visibility: evidence_only
        sha256: "<hash>"
      - path: trajectory/critique.jsonl
        kind: critique
        visibility: evidence_only
        sha256: "<hash>"
    artifacts:
      - path: evidence/calibration/oracle-run.json
        kind: oracle_run
        visibility: evidence_only
        sha256: "<hash>"
      - path: evidence/calibration/verifier-stability-report.json
        kind: verifier_stability
        visibility: evidence_only
        sha256: "<hash>"
      - path: evidence/calibration/review.json
        kind: acceptance_review
        visibility: evidence_only
        sha256: "<hash>"
      - path: evidence/calibration/calibration-report.json
        kind: calibration_report
        visibility: evidence_only
        sha256: "<hash>"
      - path: evidence/calibration/gold-result.json
        kind: reference_result
        visibility: evidence_only
        sha256: "<hash>"
      - path: artifacts/session.har
        kind: har
        mime_type: application/json
        produced_by: browser
        visibility: evidence_only
        sha256: "<hash>"
```

| Field | Values |
|---|---|
| Evidence artifact `kind` | `screenshot`, `video`, `har`, `browser_trace`, `trajectory`, `critique`, `viewer_metadata`, `verifier_artifact`, and the calibration kinds above |
| `visibility` | `agent`, `hidden_verifier`, `hidden_oracle`, `runtime_secret`, `external_dataset`, `evidence_only` |

Oracle and calibration evidence is required for leaderboard tasks, and recommended for any task with hidden tests, a subjective rubric or a model judge. See [Check a task](../tasks/check.md#evidence-for-acceptance).

## Compatibility

- Native `task.md` refuses unknown keys.
- `bench import` keeps the `task.toml` keys it does not model under `benchflow.compat.extra`, with their paths in `extra_paths`:

   ```yaml
   benchflow:
     compat:
       source: legacy
       extra_paths:
         - sandbox.modal.image
         - steps[0].runner
         - verifier.criteria_runner.metric
       extra:
         sandbox:
           modal:
             image: registry.example.com/task:latest
         steps:
           - runner: custom-step-runner
         verifier:
           criteria_runner:
             metric: exact_match
   ```

- `bench export` writes them back into `task.toml` without overwriting BenchFlow's own keys, and lists them as `restored_extension_paths`.
- `compatibility/export-report.json` lists the chosen files, input and output hashes, name collisions, restored keys and anything lost.
- A task.toml task may hold only `tests/test.sh`; it needs no `verifier.md`.

A round trip keeps meaning, not bytes. Settings are compared as parsed, the instruction as normalized text, and `oracle/` and `verifier/` as SHA-256 maps of their regular files.

## What runs today

| Feature | Status |
|---|---|
| `task.md` instruction, `verifier/`, `oracle/` | Runs |
| Strategies `script`, `llm-judge` | Run |
| Strategies `criteria-runner`, `agent-judge`, `ors-episode` | Run in part; see Strategies above |
| `agents.roles`, `scenes` | Run in part: in order, with explicit turns and sequential hand-off |
| `user` | Runs in part: scripted and model-backed users, linear scenes |
| `benchflow.teams` | One `handoff` with `mode: sequential`, `workspace_visibility: shared`, `trajectory_visibility: none` or `metadata` |
| `steps` | Parsed, refused |
| Root `artifacts` | Docker, remote Docker and Daytona |
| `allowlist`, `denylist` networks | Docker, remote Docker and Daytona; `verifier.network_mode: allowlist` is refused everywhere |
| Separate verifier | Docker, remote Docker and Daytona; not for `llm-judge` verifiers |
| `sandbox.workdir` | An absolute path other than `/` |
| `sandbox.gpus`, `sandbox.gpu_types` | [Modal only](../sandboxes/modal.md#gpus): positive count and at most one requested type |
| Windows, TPU | Parsed, refused |

A feature a sandbox can't run fails with `UnsupportedTaskFeatureError` before any sandbox starts. `bench check --sandbox <name>` runs the same gate.

## Next steps

- [Tasks](../tasks/overview.md): the task chapter.
- [Check a task](../tasks/check.md): the checks that enforce this contract.
- [Design notes](../contributing/task-standard-design.md): what the standard aims at beyond this.
