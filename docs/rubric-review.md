# Rubric review

Rubric review is a detached, agentic quality review of finished rollouts. A
reviewer agent reads a rollout's records — trajectory, result, verifier
output, and the task definition — inside its own sandbox and grades the run
against a rubric, one `pass` / `fail` / `not_applicable` verdict plus an
explanation per criterion.

Review is **report-only**. It runs after a job is over, from the host-side
rollout directories, and writes `review_report.json`. It never modifies a
reviewed rollout's `rewards` or `result.json`, and there is no code path
through which it could: the deterministic verifier is the only owner of
`reward`.

This is distinct from the [`llm-judge` verifier strategy](./llm-judge.md):
an llm-judge is part of a task's verifier and *produces* the reward, while
rubric review is downstream quality assurance *about* finished runs — is the
task well specified, did the agent game the grader, was the method sound.

## The rubric (contract v0.1)

A rubric is a JSON file with one list:

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

Each criterion is exactly three strings:

| Field | Purpose |
|---|---|
| `name` | Stable identifier. Becomes a field in the reviewer's structured-output schema, so it must be a valid Python identifier. |
| `description` | Documentation for humans reading the rubric. **Never included in the reviewer prompt** — grading must not depend on it. |
| `guidance` | The grading contract the reviewer follows. Put the full pass/fail/not-applicable conditions here. |

There are no weights, gates, thresholds, or aggregate scores. Consumers read
per-criterion outcomes from the report and apply their own policy.

The contract is named **v0.1**; the document itself carries no version key
— a rubric is exactly its `criteria` list.

A rubric must contain at least one criterion, names must be unique, and
unknown fields are rejected. (Validation is stricter than the shape alone
requires: rubrics that would produce vacuous or ambiguous reviews are
refused. Every rubric that passes is exactly the v0.1 shape.) The task-root
`verifier/rubric.json` and legacy `tests/rubric.json` paths are reserved for
this detached-review contract. Ownership is determined by the path, not by
guessing a JSON dialect. Unreadable files, invalid JSON, empty or missing
`criteria`, misspelled keys, and verifier-style `{id, match_criteria}` entries
in either reserved slot are rejected explicitly.

Rubric resolution order, per reviewed rollout:

1. an explicit `--rubric/-r` file,
2. the reviewed task's own `verifier/rubric.json` (or `tests/rubric.json`),
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
  is retained at
  `/evidence/trial/trajectory/acp_trajectory.jsonl`. The default reviewer
  prompt identifies it as chronologically ordered, redacted JSONL, summarizes
  the event types it may contain, directs behavioral and chronological checks
  to it, and warns that missing events are not evidence of unrecorded behavior.
  Evidence content is explicitly untrusted data, never reviewer instructions.
  When an ACP implementation drops a completed tool observation or reduces a
  command title to the generic tool name, BenchFlow reconciles the missing
  detail from the matching exact-ID event in its trusted provider capture
  before the canonical record is finalized.
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
  `guidance` lines render into the instruction, criterion names become the
  output schema and `tests/criteria.json`. `description` goes nowhere.
- **Validity-only reward.** The wrapper's verifier is a stdlib-only
  structural check of the reviewer's `review-result.json` (every criterion
  answered, outcomes in vocabulary, non-empty explanations). Reward 1.0
  means "a well-formed review exists" — never "the reviewed run was good".
- **Failure isolation.** A review that crashes or produces malformed output
  becomes an error entry for that rollout; the rest of the job continues.

## Output

The review job directory contains `review_report.json`:

```json
{
  "path": "…/jobs/2026-08-03__12-00-00",
  "rubric": {"path": "…", "criteria": ["…"]},
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
      "criteria": ["reward_hacking", "task_specification"],
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

## Writing good criteria

- Put the entire decision rule in `guidance`, including when to answer
  `not_applicable` (for example: infrastructure failure before the agent
  ever attempted the task).
- One judgment per criterion. A criterion that bundles several claims makes
  `fail` ambiguous.
- The reviewer reads evidence produced by the solver. Guidance should direct
  it to concrete records (`trial/result.json`, `trial/trajectory/`,
  `trial/verifier/`) rather than to intent.
- `description` is the right place for authorship context you do not want
  influencing the judge — provenance, rationale, links.
