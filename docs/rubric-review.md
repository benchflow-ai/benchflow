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

## The rubric

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

A rubric must contain at least one criterion, names must be unique, and
unknown fields are rejected. `rubric.json` is an overloaded filename —
llm-judge verifier rubrics use `{id, match_criteria}` entries — so a file
only counts as a review rubric when every criteria entry carries exactly
these three keys; judge rubrics are never claimed or misvalidated.

Rubric resolution order, per reviewed rollout:

1. an explicit `--rubric/-r` file,
2. the reviewed task's own `verifier/rubric.json` (or `tests/rubric.json`)
   when it is shaped like a review rubric,
3. the built-in default rubric (`reward_hacking`, `task_specification`).

## Running a review

```bash
# review one rollout
bench review jobs/<job>/<rollout> --sandbox docker

# review every rollout in a job, eight at a time, on Daytona
bench review jobs/<job> --sandbox daytona -n 8

# audit the winners for grader manipulation
bench review jobs/<job> --passing

# analyze the losers for specification gaps
bench review jobs/<job> --failing -r spec-rubric.json
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

- **Prebuilt image, no build.** The wrapper declares a pinned
  `docker_image` (`python:3.13-slim`) and ships no Dockerfile, so no backend
  ever builds an image for a review.
- **Evidence by upload, outside the workdir.** A copy of the rollout
  directory is uploaded to `/evidence/trial`, and a copy of the task
  directory (when the rollout's `config.json` still points at one) to
  `/evidence/task`. `/evidence` sits outside the agent workdir, so the
  sandbox-user chown never touches it: it stays root-owned and unwritable
  by the reviewer. Prior review outputs are excluded from the copy, so a
  re-review can never read an earlier verdict; symlinks anywhere in the
  evidence are dropped, never dereferenced; task skills and any shipped
  `rubric.json` are excluded from the task copy. The reviewed rollout
  itself is never touched.
- **Gateway-scoped egress.** Reviewer rollouts run with
  `BENCHFLOW_DISALLOW_WEB_TOOLS=1`, so the sandbox egress lockdown confines
  the reviewer's network to the model gateway on backends that enforce the
  owner-matched firewall.
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
  "reviewer": {"agent": "opencode", "model": "google/gemini-2.5-flash", "environment": "docker"},
  "job_summary": "Deterministic outcome aggregation (multi-rollout runs only).",
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
      "reviewer_rollout": "…/runtime/hello-world-task__829cddb8"
    }
  ]
}
```

Each reviewer rollout's own records (trajectory, verifier output, raw
`review-result.json`) sit under the report's `runtime/` directory for
audit; every invocation uses a fresh unique runtime leaf, and each trial's
`reviewer_rollout` points at that exact leaf, so reusing `--out-dir` can
never resurface a stale review. The job summary is a deterministic
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
