# Rubric review

Rubric review is a detached, agentic quality review of finished rollouts. A
reviewer agent reads a rollout's records — trajectory, result, verifier
output, and the task definition — inside its own sandbox and grades the run
against a rubric. Legacy rubrics produce one `pass` / `fail` /
`not_applicable` verdict per criterion; weighted rubrics combine binary
blocker verdicts with 0–2 scores.

Review is **report-only**. It runs after a job is over, from the host-side
rollout directories, and writes `review_report.json`. It never modifies a
reviewed rollout's `rewards` or `result.json`, and there is no code path
through which it could: the deterministic verifier is the only owner of
`reward`.

This is distinct from the [`llm-judge` verifier strategy](./llm-judge.md):
an llm-judge is part of a task's verifier and *produces* the reward, while
rubric review is downstream quality assurance *about* finished runs — is the
task well specified, did the agent game the grader, was the method sound.

## The rubric (versionless contracts v0.1 and v0.2)

A rubric is a JSON file with one `criteria` list and no in-file version key.
BenchFlow identifies its contract from the fields present on every criterion.

### Legacy v0.1 rubric

The original shape remains fully supported:

```json
{
  "criteria": [
    {
      "name": "reward_hacking",
      "description": "Author-facing note about why this criterion exists.",
      "guidance": "Study the recorded run under trial/ for signs the agent manipulated grading instead of doing the work. PASS when ... FAIL when ... NOT_APPLICABLE when ..."
    }
  ]
}
```

Each v0.1 criterion is exactly three strings:

| Field | Purpose |
|---|---|
| `name` | Stable identifier. Becomes a field in the reviewer's structured-output schema, so it must be a valid Python identifier. |
| `description` | Documentation for humans reading the rubric. **Never included in the reviewer prompt** — grading must not depend on it. |
| `guidance` | The grading contract the reviewer follows. Put the full pass/fail/not-applicable conditions here. |

The reviewer answers each criterion with `pass`, `fail`, or `not_applicable`
plus a non-empty explanation. There are no weights, gates, thresholds, or
aggregate scores. Consumers read per-criterion outcomes from the report and
apply their own policy.

### Weighted v0.2 rubric

The weighted contract used by
[FrontierPhysics PR #109](https://github.com/benchflow-ai/FrontierPhysics/pull/109)
adds `blocker` and `weight` to **every** criterion:

```json
{
  "criteria": [
    {
      "name": "required_artifact",
      "blocker": 1,
      "weight": 10,
      "description": "The required artifact must be present and usable.",
      "guidance": "PASS only when the evidence contains the complete artifact. Otherwise FAIL."
    },
    {
      "name": "technical_quality",
      "blocker": 0,
      "weight": 8,
      "description": "Quality of the technical result.",
      "guidance": "Score 0 for incorrect, 1 for partially correct, or 2 for complete and correct."
    }
  ]
}
```

`blocker` is a strict integer `0` or `1`; `weight` is a strict integer from
`1` through `10`. Booleans, numeric strings, floats, omitted fields, and mixed
v0.1/v0.2 criteria are rejected. A v0.2 rubric must contain at least one
non-blocker criterion so its quality denominator is nonzero.

- A criterion with `blocker: 1` receives `pass` or `fail`. Any failure closes
  the blocker gate. Its `weight` is validated for a uniform authoring format
  but is excluded from weighted quality.
- A criterion with `blocker: 0` receives an integer score of `0`, `1`, or `2`.
  Its contribution is `score * weight` out of `2 * weight`.

BenchFlow aggregates a structurally valid v0.2 review on the host:

```text
raw_quality = sum(score * weight) / (2 * sum(non-blocker weights))
gates_pass = deterministic reward is 1.0 with no recorded error
             AND every blocker passes
gated_quality = raw_quality if gates_pass, otherwise 0
```

The raw quality remains visible when a gate fails, making the rubric judgment
auditable without mistaking it for a publishable result. A failed gate always
produces `not_publishable`. With both gates open, the publication bands are:

| Gated quality | Decision |
|---|---|
| `>= 0.80` | `publishable` |
| `>= 0.65` and `< 0.80` | `presentable_with_revisions` |
| `< 0.65` | `not_publishable` |

### Validation and discovery

A rubric must contain at least one criterion, names must be unique, and
unknown fields are rejected. (Validation is stricter than the shape alone
requires: rubrics that would produce vacuous or ambiguous reviews are
refused.) `rubric.json` is an overloaded filename —
llm-judge verifier rubrics use `{id, match_criteria}` entries. Discovery is
fail-closed: a `rubric.json` is treated as a review rubric — and validated
loudly — **unless** every entry carries the full judge shape (both `id`
and `match_criteria`). Unreadable files, invalid JSON, empty or missing
`criteria`, and misspelled keys are all claimed and rejected with an
explicit error rather than silently replaced by the default rubric.

Rubric resolution order, per reviewed rollout:

1. an explicit `--rubric/-r` file,
2. the reviewed task's own `verifier/rubric.json` (or `tests/rubric.json`)
   when it is shaped like a review rubric,
3. the built-in default rubric (`reward_hacking`, `task_specification`).

## Running a review

```bash
# review one rollout
bench review jobs/<job>/<rollout> --sandbox docker \
  --model gemini/gemini-2.5-flash --tasks-root ./tasks

# review every rollout in a job, eight at a time, on Daytona
bench review jobs/<job> --sandbox daytona -n 8 \
  --model gemini/gemini-2.5-flash --tasks-root ./tasks

# audit the winners for grader manipulation
bench review jobs/<job> --passing --model gemini/gemini-2.5-flash

# analyze the losers for specification gaps
bench review jobs/<job> --failing -r spec-rubric.json \
  --model gemini/gemini-2.5-flash
```

`--passing` selects rollouts with reward 1.0 and no recorded error;
`--failing` selects everything else, including rollouts whose `result.json`
is unreadable. The reviewer agent (`--agent`, default `opencode`) and model
(`--model`; agents without a registry default require one — pass a gateway
model id such as `gemini/gemini-2.5-flash`) are independent of whatever ran
the original job.

## How a review executes

Each review is an ordinary rollout of a throwaway wrapper task assembled on
the host, which is why every sandbox backend (`docker`, `daytona`,
`agentcore`, ...) works unchanged:

- **Prebuilt image, pinned by digest.** The wrapper declares a
  digest-pinned `python` image and ships no Dockerfile, so Docker and
  Daytona never build one. AgentCore is the exception: it must wrap any
  image with its runtime-contract shim, so it still builds and pushes a
  derived ECR image once per distinct image, then reuses it.
- **Evidence by upload, outside the workdir.** A copy of the rollout
  directory is uploaded to `/evidence/trial`. A task copy is uploaded to
  `/evidence/task` only when it is admitted through the trusted-root and
  digest checks below. `/evidence` sits outside the agent workdir; after all
  uploads, a pre-agent hook fails closed unless the whole tree can be made
  root-owned, readable, and non-writable by the reviewer. Prior review
  outputs are excluded from the copy, so a
  re-review can never read an earlier verdict; symlinks anywhere in the
  evidence are dropped, never dereferenced; task skills and any shipped
  `rubric.json` are excluded from the task copy. The canonical ACP trajectory
  is retained. When an ACP implementation drops a completed tool observation
  or reduces a command title to the generic tool name, BenchFlow reconciles
  the missing detail from the matching exact-ID event in its trusted provider
  capture before the canonical record is finalized.
  The cumulative provider-history `llm_trajectory.jsonl` remains omitted: it
  repeats the growing conversation on every request and can exhaust a reviewer
  model's context. The reviewed rollout itself is never touched.
- **Post-initialization egress restriction, fail closed.** The wrapper
  declares `allow_internet: false`, which disables web tools, forces the
  model proxy sandbox-local, and arms the agent-UID egress firewall scoped
  to that loopback gateway. Backends that cannot enforce isolation (for
  example `agentcore`, whose runtime only offers PUBLIC/VPC networking)
  refuse the review at launch; `--allow-open-network` is the explicit,
  report-recorded override for them.

  Be precise about what this guarantees: the container needs network during
  image setup and agent installation, so the firewall is armed *after* the
  reviewer harness starts and completes ACP initialization. The guarantee is
  **restricted egress for the graded portion of the run**, not
  network isolation for the container's whole lifetime. Evidence is uploaded during
  sandbox setup, before the firewall is enforced, so a reviewer harness
  that is itself malicious could egress during startup **after evidence is
  present**. Treat the reviewer harness as trusted code; the untrusted
  input is the evidence it reads, and this guarantee constrains the graded
  portion of the run, not a hostile harness.
- **Task evidence requires an explicitly trusted root.** A rollout's
  recorded `task_path` is rollout-authored data, so it is never read
  directly — pass `--tasks-root <dir>` and the task is looked up *by name*
  beneath that root. Without it, the review proceeds from run records alone
  and says so in the trial's `notes`. When the rollout recorded a
  `task_digest` in `result.json` or `config.json`, the values must be valid
  and mutually consistent. A missing digest, mismatch against the on-disk
  task, conflict, or any verification failure **excludes the task from
  evidence** and says so in `notes`; an old or unverifiable rollout is never
  reviewed against current task content.
- **The rubric never enters the sandbox.** It is decomposed host-side:
  `guidance` lines render into the instruction; names and the minimum contract
  metadata needed for structural validation become the output schema and
  `tests/criteria.json`. `description` goes nowhere.
- **Validity-only reward.** The wrapper's verifier is a stdlib-only
  structural check of the reviewer's `review-result.json` (every criterion
  answered, outcomes or scores in the contract's vocabulary, non-empty
  explanations). Its reward remains structural for both rubric contracts:
  reward 1.0 means "a well-formed review exists" — never "the reviewed run
  was good." Weighted quality and publication decisions are separate,
  deterministic host-side report fields.
- **Failure isolation.** A review that crashes or produces malformed output
  becomes an error entry for that rollout; the rest of the job continues.

## Output

The review job directory contains `review_report.json`. This v0.1 example has
no aggregate score:

```json
{
  "path": "…/jobs/2026-08-03__12-00-00",
  "rubric": {"path": "…", "criteria": ["…"], "contracts": ["v0.1"]},
  "reviewer": {"agent": "opencode", "model": "gemini/gemini-2.5-flash", "environment": "docker", "network": "no-internet"},
  "job_summary": "Deterministic aggregation over VALID reviews only.",
  "trials": [
    {
      "trial_name": "hello-world-task__829cddb8",
      "source_rollout": "…",
      "review_valid": true,
      "summary": "Three-to-five sentence account of the run.",
      "checks": {
        "reward_hacking": {"explanation": "…", "outcome": "pass"},
        "task_specification": {"explanation": "…", "outcome": "fail"}
      },
      "error": null,
      "reviewer_rollout": "…/runtime/hello-world-task__829cddb8/<run-id>/…",
      "rubric_path": "…/verifier/rubric.json",
      "rubric_contract": "v0.1",
      "criteria": ["reward_hacking", "task_specification"],
      "criterion_metadata": [
        {"name": "reward_hacking", "blocker": null, "weight": null},
        {"name": "task_specification", "blocker": null, "weight": null}
      ],
      "scoring": null,
      "notes": ["task evidence skipped: no --tasks-root was given"]
    }
  ]
}
```

Each reviewer rollout's own records (trajectory, verifier output, raw
`review-result.json`) sit under the report's `runtime/` directory for
audit; every invocation uses a fresh unique runtime leaf. When a leaf is
successfully identified, `reviewer_rollout` points at that exact leaf;
otherwise it is `null` rather than an ambiguous parent directory. Reusing
`--out-dir` can therefore never resurface a stale review. The job summary is a deterministic
aggregation, not a model call — a host-side LLM call would bypass the
sandbox backend, egress policy, and telemetry.

For v0.2, blocker checks carry `outcome`, scored checks carry `score`, and the
trial also records its rubric contract, criterion metadata, and deterministic
aggregation. The scoring object has this shape:

```json
{
  "deterministic_pass": true,
  "all_blockers_pass": true,
  "failed_blockers": [],
  "weighted_points": 24,
  "max_weighted_points": 30,
  "raw_quality": 0.8,
  "gated_quality": 0.8,
  "decision": "publishable"
}
```

Aggregation is emitted only for a structurally valid v0.2 review. The report
keeps each trial's resolved contract and criterion metadata authoritative,
because one job may review tasks with different task-local rubrics.

## Writing good criteria

- Put the entire decision rule in `guidance`. For v0.1, include when to answer
  `not_applicable` (for example: infrastructure failure before the agent ever
  attempted the task). V0.2 has no `not_applicable`: define exact `pass` /
  `fail` conditions for blockers and exact `0` / `1` / `2` anchors for scored
  criteria.
- One judgment per criterion. A criterion that bundles several claims makes
  a blocker verdict or numeric score ambiguous.
- The reviewer reads evidence produced by the solver. Guidance should direct
  it to concrete records (`trial/result.json`, `trial/trajectory/`,
  `trial/verifier/`) rather than to intent.
- `description` is the right place for authorship context you do not want
  influencing the judge — provenance, rationale, links.
