# Rubric review internals

How [rubric review](../judging/rubric-review.md) captures evidence, isolates the reviewer and writes its report. This page is for maintainers and security reviewers.

## Evidence capture

At the end of a run, BenchFlow does these steps in order:

1. freezes the solver's actual working directory, before verifier hardening;
2. captures durable evidence;
3. runs the tests;
4. finalizes telemetry;
5. releases the solver sandbox, before waiting for reviewer capacity.

The reviewer gets read-only evidence plus its own writable workspace. File manifests and archive hashes check every transfer. A required capture that fails is a scoring error, not a scientific zero. Intermediate test feedback does not start the reviewer.

Capture limits, and files that change during capture, fail the capture.

### Exclusions

Evidence never drops a path silently. Each excluded path is listed in the manifest's `exclusions` with its reason:

| Reason | What |
|---|---|
| `credential` | credential files |
| `sandbox_runtime` | sandbox-provider state, such as Daytona's `~/.daytona` session tree inside a `/root` workspace |
| `package_cache` | pip, uv and npm caches |
| `special_file` | sockets, FIFOs and device nodes |
| `symlink_escape` | a link that leaves the workspace, such as an agent CLI helper linked from outside, or a link loop. It records its `link_target`. |
| `task_exclude` | paths the task itself excludes |

A symlink is kept only when both its resolved path and its target text stay inside the workspace. No outside file is ever copied. The host check and the reviewer's admission check both reject a manifest that captures and excludes the same path.

Provider traces that a supported native OAuth login can't produce are marked unavailable, never fabricated.

## Deadlines

Each reviewer is an ordinary rollout under the host hard deadline, the backstop above all its phase budgets. That deadline includes the wrapper's setup commands. It also adds transfer time for the uploaded evidence, at 1 MiB/s, so a large workspace is not cut off mid-transfer.

## How a review executes

Each review is an ordinary rollout of a throwaway wrapper task built on the host. So every sandbox backend (`docker`, `daytona`, `agentcore`, ...) works unchanged.

### Prebuilt image, pinned by digest

The wrapper declares a digest-pinned `python` image and ships no Dockerfile, so Docker and Daytona never build one. AgentCore is the exception: it must wrap any image with its runtime-contract shim. So it builds and pushes a derived ECR image once per distinct image, then reuses it.

### Evidence by upload, outside the workdir

- A copy of the trial folder is uploaded to `/evidence/trial`.
- A copy of the task goes to `/evidence/task`, only when it passes the trusted-root and digest checks below.
- `/evidence` sits outside the agent's workdir. After all uploads, a pre-agent hook fails closed unless the whole tree can be made root-owned, readable and not writable by the reviewer.
- Earlier review outputs are left out of the copy, so a re-review can never read an earlier verdict.
- Symlinks anywhere in the evidence are dropped, never followed.
- Task skills and any shipped `rubric.json` are left out of the task copy.
- The canonical ACP trajectory is kept. When an ACP implementation drops a completed tool observation, or shortens a command title to the generic tool name, BenchFlow fills in the detail from the matching event in its trusted provider capture before the record is finalized.
- `llm_trajectory.jsonl` is left out. It repeats the growing conversation on every request and can exhaust the reviewer model's context.

The reviewed trial itself is never touched.

### Network restricted after initialization, fail closed

The wrapper declares `allow_internet: false`. That disables web tools, makes the model proxy sandbox-local, and arms the agent-uid egress firewall, scoped to that loopback gateway.

Backends that can't enforce this refuse the review at launch. An example is `agentcore`, whose runtime offers only PUBLIC or VPC networking. `--reviewer-open-network` (on `bench run`) or `--allow-open-network` (on `bench review`) is the explicit override, and the report records it.

What this guarantees, precisely:

- The container needs the network during image setup and agent installation. So the firewall is armed after the reviewer harness starts and finishes ACP initialization.
- The guarantee is restricted egress for the graded part of the run, not network isolation for the container's whole life.
- Evidence is uploaded during sandbox setup, before the firewall is on. So a reviewer harness that is itself malicious could send data out during startup, after the evidence is present.

Treat the reviewer harness as trusted code. The untrusted input is the evidence it reads. This guarantee constrains the graded part of the run, not a hostile harness.

### Task evidence needs a trusted root

A trial's recorded `task_path` is data the trial wrote, so it is never read directly. Pass `--tasks-root <dir>`, and the task is looked up by name under that root. Without it, the review uses the run's records alone and says so in the trial's `notes`.

When the trial recorded a `task_digest` in `result.json` or `config.json`, the values must be valid and agree. A missing digest, a mismatch with the task on disk, a conflict or any failed check leaves the task out of the evidence, and `notes` says so. An old or unverifiable trial is never reviewed against current task content.

### The rubric never enters the sandbox

The rubric is split on the host:

- `guidance` lines go into the instruction;
- names and the minimum contract metadata for structural checks become the output schema and `tests/criteria.json`;
- `description` goes nowhere.

### Validity-only reward

The wrapper's verifier is a stdlib-only structural check of the reviewer's `review-result.json`: every criterion answered, outcomes or scores in the contract's vocabulary, and non-empty explanations. Its reward is structural for both rubric shapes. Reward 1 means "a well-formed review exists", never "the reviewed run was good". Weighted quality and publication decisions are separate, deterministic fields computed on the host.

### Failure isolation

A review that crashes or writes malformed output becomes an error entry for that trial. The rest of the job continues.

## The report

The review job folder contains `review_report.json`. This example uses a plain rubric, so it has no aggregate score:

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

`v0.1` is the plain shape and `v0.2` the weighted one. For a weighted rubric, blocker checks carry `outcome`, scored checks carry `score`, and `scoring` holds the deterministic aggregation shown on the [rubric review](../judging/rubric-review.md#results) page. Aggregation is written only for a structurally valid weighted review. Each trial keeps its own contract and criterion metadata, because one job may review tasks with different rubrics.

Each reviewer run's own records (trajectory, verifier output, the raw `review-result.json`) sit under the report's `runtime/` folder. Every invocation uses a fresh, unique runtime leaf. When the leaf is identified, `reviewer_rollout` points at it; otherwise it is `null`, never an ambiguous parent folder. So reusing `--out-dir` can never bring back a stale review.

The job summary is a deterministic aggregation, not a model call. A model call on the host would bypass the sandbox backend, the egress policy and telemetry.

## Next steps

- [Rubric review](../judging/rubric-review.md): the user guide.
- [Integrity](../integrity.md): how BenchFlow protects grading.
