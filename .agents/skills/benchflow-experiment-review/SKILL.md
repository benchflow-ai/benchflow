---
name: benchflow-experiment-review
description: Audit completed BenchFlow or SkillsBench trials, benchmark measurements, usage/cost evidence, and training exports against their tested versions and declared contracts. Use for run reliability, reward integrity, timing comparisons (speed or A/B comparisons between builds or frameworks), changes to sandbox setup, hardening, grading or teardown code, or targeted integration validation.
---

# BenchFlow Experiment Review

Determine which claims the evidence supports. Separate valid evaluation results,
complete observability, comparable measurements, and training readiness. A
passing reward alone establishes none of the other three.

## Establish the contract

Record the review purpose, intended task matrix, source commit, installed package
or binary hash, agent/version, requested and observed model, model route, backend,
task/image identities, configuration, and authoritative output roots. Use the
tested version's schema and behavior. A current README, historical skill, or
successful test on another build is not proof of the measured build's behavior.

| Claim | Required evidence |
| --- | --- |
| Evaluation validity | Authentic task attempt, trustworthy verifier outcome or documented scored timeout, task/result identity, and integrity of grading inputs. |
| Observability | Native agent/tool events, supported provider request/response and usage capture, terminal metadata, and explicit missing or unsupported fields. |
| Timing or cost comparison | Comparable boundaries/routes, complete evidence for that metric, intended/admitted denominators, and retained exclusions. |
| Training readiness | Complete structured trajectories and a valid export under the exact exporter/trainer contract. |

An evaluation-only result may be usable with observability limitations; do not
label it a fully healthy trajectory or training-ready. Missing training export
does not itself prove a verifier result false. Missing evidence required by the
declared evaluation contract still blocks that evaluation claim.

For timing or usage/cost investigations, read
[measurement-evidence.md](references/measurement-evidence.md). Do not silently
turn a timing question into training-data certification or a new paid campaign.

## Preserve the intended matrix

Enumerate slots by task, harness, observed model, skill mode, attempt identity,
backend, run root, and source/configuration. Distinguish planned, launched,
completed, scored, failed, missing, duplicated, and excluded slots.

- Preserve initial failures and administrative interventions. A rerun gets a
  new identity and a relationship to its predecessor.
- A parser correction is a new derived artifact: retain the original, bind the
  corrected rows to raw evidence, and explain changes. It is not a new trial.
- Reused trial labels require independent run identity, time window and source
  binding. Do not merge events across runs by label alone.
- Separate oracle/nop controls, fixed-response replay and real-model trials.
  Replay does not establish live solve rates. A harness comparison needs
  comparable agent, model route, task semantics, resources and policies.
- Improvements on another build/configuration do not replace older release
  evidence. Report them as a separate comparison.

## Inspect the attempt

Locate native configuration, agent/provider traces, tool events, terminal status,
verifier output, reward, timing, usage and lifecycle records. Read their contents
and join identities; file existence and aggregate counts are insufficient.

Classify outcomes separately from evidence quality:

- **pass / fail:** a valid verifier graded the completed task state.
- **scored timeout:** the agent hit its declared limit and a valid verifier
  graded that state; retain the timeout and inspect terminal capture.
- **unscored / infrastructure failure:** no trustworthy grade, including failed
  startup, unavailable inputs, auth/provider/transport errors, disk exhaustion
  or verifier failure. Never substitute reward zero.

Before attributing a failure to capability, verify inputs, credentials, skills,
paths, network policy, image resources and actual execution. Security defaults
can conflict with tasks requiring privileges or persistent services. Record the
incompatibility; do not weaken isolation just to recover a score. Validate an
explicitly chosen compatibility mode separately.

For a changed subsystem, choose representative real-model canaries for relevant
supported routes/backends after focused checks. A fake model, oracle or API
probe does not exercise the complete agent lifecycle. Do not require paid runs
for every documentation or read-only review. Use existing authorized accounts,
budgets and builders; this skill grants no new permissions. Record effective
effort/model settings rather than trusting requested labels.

## Training-export gate

For versions using BenchFlow's structured export contract, inspect:

- `trajectory/acp_trajectory.jsonl`: meaningful agent/tool events;
- `trajectory/llm_trajectory.jsonl`: complete recoverable provider request and
  response evidence with usage;
- rollout `results.jsonl`: task/score/export identity, valid roles, non-empty
  prompt/completion/trajectory, tokens, and tool definitions when used.

Require the exact exporter/trainer schema. Errored, incomplete, unsupported-tool,
malformed or missing-provider-trace rollouts remain `training_ready=false` with
their reason. A valid evaluation can correctly be non-training-ready. Never
remove a reason to make a gate green. Export success does not prove trainer
consumption or a learning improvement.

The bundled `scripts/validate_run_artifacts.py` implements a **strict structured
training-artifact contract**, not every historical layout or evaluation purpose.
Use it for compatible artifacts on an approved executor:

```bash
python .agents/skills/benchflow-experiment-review/scripts/validate_run_artifacts.py /path/to/run --json
```

Record its revision, command, exit and per-artifact findings. A missing or
deliberately non-training-ready export can fail this validator while leaving
evaluation validity undecided. Keep that finding under the training claim;
never report a nonzero validator as a pass or dismiss missing evidence required
by the evaluation contract. For unsupported layouts, use the version's validator
or equivalent explicit checks and state coverage. Read-only reviews must state
when execution was not performed.

The explicit `--allow-native-subscription-without-llm` mode is evidence only. It
requires `usage_source=agent_native_acp`, healthy ACP/native usage, and a completed
non-training-ready export with `missing_healthy_structured_llm_trajectory`. It
never creates training data or accepts a malformed present provider trace.
Native usage and OTel cost events do not supply missing request/response content.

## Integrity and skills

Check that the agent received the intended task without verifier-only content;
grading follows the relevant agent phase and binds the correct workspace and
attempt. For suspected grader gaming, read
[reward-hacking-patterns.md](references/reward-hacking-patterns.md) and
[verifier-hardening-checklist.md](references/verifier-hardening-checklist.md).
Check their controls against the task's declared contract; the reference does
not prove a version implements them. A root-owned path, separate container or
signature alone does not establish grader trust.

For skill comparisons, read
[harness-skill-catalog-sop.md](references/harness-skill-catalog-sop.md) or use
`scripts/extract_harness_skills.py` on a compatible trace. Unknown or zero
extraction needs manual inspection before concluding no skills were available.
Distinguish catalog presence, injection/loading and invocation; separate total
catalog size from task-specific skill loading. A no-skill claim needs checks of
task-specific files, prompts and events. No serialized catalog is not proof of
absence; an incidental `SKILL.md` string is not automatic leakage.

Audit owned-resource cleanup separately from scoring. A CLI exit, DELETE request
or desired destruction state does not confirm release. Record observed
404/destroyed states or equivalent native absence evidence for owned IDs; retain
ambiguous creates and incomplete audits. Do not duplicate retained campaigns,
mutate frozen evidence or clean up unrelated resources.

## Report the bounded conclusion

Lead with findings affecting the requested claim, intended/admitted counts,
exact versions/configuration, evidence pointers and limits. Separate measured
facts, source-supported mechanisms, hypotheses and untested optimizations.

Give separate verdicts where relevant: evaluation valid/invalid/unknown;
observability complete/incomplete/unsupported; metric admitted/excluded;
training ready/not ready/not assessed; cleanup confirmed/unknown. Conclude
publishable, publishable with quarantines, or not publishable **for the stated
purpose**. A scoped audit cannot certify a release, establish human approval or
prove the absence of bugs.

Install the complete skill directory in the active harness's skill root when
needed, including scripts, references, evals and metadata. Record its source
revision so installed copies cannot silently drift from the reviewed one.

On shared build VMs, the full test suite fails on the base for environment
reasons, such as group-writable ancestor folders or missing tools and browsers.
Two ways to handle this:

- Run the suite in an isolated container with root-owned ancestors.
- Replay every failure on the base with the same command on the same VM.

A gate counts as passing only when every failure reproduces on the base or is
explained. Don't run a test suite on a VM that is measuring timing. Remove your
own finished-run pytest scratch, never evidence, before it fills the disk.

## Speed and A/B Comparisons

When a run compares the time of two BenchFlow builds, or of BenchFlow and
another framework, also load `references/speed-ab-experiments.md`. In short:

- Register a manifest before the first trial, with its timestamp taken from
  `date -u`.
- Warm the image cache, or run a warm-up round, for any arm whose recipe
  changed.
- Isolate the arms: no `cp` into uv venvs, a fresh trial id per round, nothing
  else running on the orchestrator.
- Pair by task, with a seeded bootstrap CI, and report outer-clock phase
  segments at a loaded and a light load.
- Check that outcomes and task digests are identical between BenchFlow arms.
- Pass the other framework's telemetry opt-out explicitly at every launch, and
  keep the comparison outside git.

## Changes to Sandbox and Security Code

When a change touches setup, hardening, grading, exec transport, lifecycle or
image recipes, also load `references/sandbox-change-review.md`. In short:

- Require differential reproductions that fail on the head and pass on the
  base.
- Assume a transport error or timeout does not mean the command stopped.
- Check the eight defect classes listed there, for example a security answer
  discarded and re-run, or reuse across a kill or freeze.
- Run the measurement alongside the review, not after it.
