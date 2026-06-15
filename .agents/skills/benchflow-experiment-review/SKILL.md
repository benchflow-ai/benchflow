---
name: benchflow-experiment-review
description: Review Benchflow or SkillsBench task-run trajectories and integration-test Benchflow code changes. Use this skill whenever the user asks to audit traj health, failed or timed-out runs, healthy pass/fail/timeout status, no-skill leakage, skill loading, reward hacking, verifier isolation, metadata completeness, token usage, timing, Daytona-vs-Docker parity, path/root handling, coverage gaps, Docker/Daytona failures, or release-readiness of benchmark data.
---

# Benchflow Experiment Review

Use this skill to decide whether a Benchflow run trial is clean enough to
publish or whether a Benchflow code change is safe to use for new experiments.
The standard is: a clean sandbox with only task-needed resources, every agent
behavior logged, no verifier leakage, and a final score or healthy failure.

## Portability

This skill is intentionally harness-portable. Install or copy the entire
`benchflow-experiment-review/` directory into the active harness's skill root,
keeping `SKILL.md`, `scripts/`, `references/`, `evals/`, and optional
`agents/` metadata together. Use the same review procedure regardless of
whether the harness loads skills from `.claude/skills`, `.codex/skills`,
OpenHands/Gemini/pi-agent skill roots, or another compatible `SKILL.md`
directory.

## Operating Rule

Do not accept aggregate counts alone. Enumerate the intended matrix by
`task_id`, harness, model, skill mode, trial id, sandbox type, run root, and
source/ref. Mark each slot as healthy, missing, duplicate, stale, or unhealthy.

Healthy run outcomes are:

- `pass`: agent completed and verifier produced a valid score.
- `fail`: agent completed incorrectly and verifier produced a valid score.
- `normal_timeout`: agent genuinely ran, timed out, and still produced a
  complete trajectory plus reward/scoring metadata.

Infrastructure failures are not healthy failures. A stalled Docker daemon,
Daytona transport failure, missing trajectory, missing reward, missing timing,
missing token usage for new data, or verifier crash is unhealthy until rerun or
explicitly quarantined.

## Completed Trial Review

For each trial, first locate the authoritative run artifacts. Prefer repo
validators or existing scripts when available, then inspect raw files. Required
evidence usually includes:

- Run config: task id, harness, model, skill mode, trial id, source commit/ref,
  sandbox backend, timeout, provider settings, and output paths.
- Trajectories: ACP/tool trajectory and LLM/model trajectory, parseable from
  first event through final answer, failure, or timeout.
- Metadata: start/end timestamps, elapsed duration, token usage, tool usage,
  provider/model response metadata, sandbox metadata, and error fields.
- Results: verifier output, reward/score, result status, and enough provenance
  to connect the score to the exact trajectory.

Review the trajectory for meaning, not just existence:

- The agent received the intended task prompt and no verifier-only information.
- Tool calls and observations form a plausible task attempt.
- Skill-enabled trials show native skill loading or injection evidence for that
  harness; no-skill trials do not expose those skill resources.
- For harness-specific skill catalog recovery, load
  `references/harness-skill-catalog-sop.md` or run
  `scripts/extract_harness_skills.py` against `trajectory/llm_trajectory.jsonl`.
  Treat the script as a fast path: if it returns `unknown`, `skill_count: 0`,
  or `manual_review_required: true`, fall back to the SOP's manual procedure
  before concluding that no skills were available.
- Check `task_skills_loading`: with-skill trials should report `1`, meaning
  every expected task-specific skill was recovered in the startup skill catalog;
  without-skill trials should report `0`, meaning task-specific skills were not
  injected into the agent's startup catalog.
- Timeout trials show real progress or attempts before timeout and still have
  complete timing, token, trajectory, and verifier result metadata.
- The final answer/result and verifier score refer to the same task workspace
  and trial.
- Failures and timeouts are attributable to agent capability, not missing
  critical inputs. Audit the task spec, environment variables, API keys,
  mounted assets, package/runtime availability, network access, compute/memory,
  and sandbox rendering before labeling a failed or timed-out run as healthy.

Reject or quarantine the trial if any of these appear:

- Missing, truncated, or unparsable trajectory files.
- Empty transcript, agent never launched, or only setup logs.
- Missing token usage/timing/tool usage metadata for newly generated data.
- The agent lacked required task information, required skills, API keys,
  credentials, assets, runtime dependencies, or enough compute/resources to
  make a fair attempt.
- Verifier ran before the agent finished, leaked solution hints, or wrote into
  agent-visible resources.
- Reward appears hacked, copied from hidden files, or inconsistent with the
  visible task state.
- Path/root mismatch between task config, sandbox cwd, trajectory paths, and
  result paths.
- Provider, Daytona, Docker, or filesystem errors that prevented a normal
  agent attempt.

## Integration Tests After Code Changes

Before using a changed Benchflow version for production experiments, run an
end-to-end integration check that proves the changed code preserves artifact
shape and sandbox behavior.

1. Run focused unit/static checks for changed modules first. Use the repo's
   normal commands when appropriate, such as targeted `pytest`, `ty`, and
   `ruff`.
2. Pick representative canary tasks covering with-skill and no-skill modes,
   success/failure/timeout if feasible, and any changed subsystem.
3. Run the same canaries through Daytona and VM Docker with the same task,
   harness, model, skill mode, trial id pattern, timeout, and provider settings.
4. Compare artifact schema and semantics, not exact model wording: trajectory
   files, token usage, timing, tool usage, result status, verifier output, and
   source provenance must be equivalent.
5. Exercise path/root behavior explicitly. Check task root, sandbox cwd,
   mounted resources, result paths, locked paths, and any previous root-path
   regression case.
6. Verify skill loading. With-skill runs must show the harness-specific native
   skill metadata or prompt injection pattern; no-skill runs must not expose
   skill files or descriptions.
7. Verify verifier isolation and environment hardening. The verifier starts
   only after agent exit or timeout, cannot leak solution data into the agent
   phase, and records score provenance against the completed trajectory. Confirm
   the anti-reward-hacking invariants in
   `references/verifier-hardening-checklist.md`: network off by default (proxied
   and allowlisted when needed), git history scrubbed past the base commit, no
   answer/hidden-test/scorer files in agent-readable paths, empty patch fails
   while golden patch passes, and agent test edits are reset before grading.
8. Scale only after the canary passes. Increase concurrency gradually and keep
   each VM below the Docker/container pressure that makes the daemon unstable.

The pass condition is schema and lifecycle parity between Daytona and Docker:
both sandboxes should render the same task-needed resources, log the same
classes of agent behavior, and emit complete metadata for the same logical
trial outcome.

## Capability Attribution

When a run fails or times out, classify the cause before accepting it as a
healthy model failure.

- Model-capability failure: required task resources were present, credentials
  and skills were available when needed, the sandbox had sufficient compute and
  runtime dependencies, the agent made a real attempt, and the verifier scored
  the completed or timed-out state.
- Experiment-fidelity failure: the agent was blocked by missing API keys,
  absent skill resources, hidden or mis-mounted task assets, dependency
  installation failures, network/permission errors, insufficient compute or
  memory, broken rendering, path mismatches, provider outages, or sandbox
  instability.

Only the model-capability case can be counted as healthy `fail` or
`normal_timeout`. Experiment-fidelity failures must be rerun after fixing the
environment or documented as quarantined infra/context failures.

## Failure Playbook

Docker stuck or daemon unstable:

- Treat this as infrastructure failure, not model fail.
- Count active/exited containers and inspect daemon health before launching
  more work.
- Avoid putting more than about 60 containers on one VM. Drain, restart, or
  replace the VM when the daemon is unreliable.
- Rerun affected trials only after confirming new trajectories are complete.

Daytona vs VM Docker mismatch:

- Reproduce the same canary in both backends with identical logical settings.
- Diff artifact schema and metadata fields. Missing token usage or timing in
  one backend is a blocker for new production data.
- Do not publish data from the backend until the discrepancy is explained or
  fixed.

Path mismatch:

- Compare task config paths, sandbox cwd, mounted root, trajectory-reported
  paths, verifier input paths, and result output paths.
- Confirm the agent only sees task-needed resources and cannot read hidden
  verifier or answer files.
- Add or run a regression canary that covers the previous root-path failure.

Verifier leakage or reward hacking:

- Confirm verifier invocation begins after agent completion or timeout.
- Search the agent prompt, observations, mounted files, and tool outputs for
  answer keys, verifier hints, hidden tests, or reward files.
- Audit suspicious perfect scores, direct hidden-file reads, or trajectory
  actions that bypass the intended task.
- When the trajectory looks suspicious, load
  `references/reward-hacking-patterns.md` and classify the evidence against the
  solution-contamination, grader-gaming, and alignment-risk patterns there.
- To remediate and prevent recurrence, apply
  `references/verifier-hardening-checklist.md`, which maps each pattern to the
  environment/grader fix that closes it at the source.

Skill loading mismatch:

- For each harness, use the harness-specific expected system prompt or metadata
  pattern from `references/harness-skill-catalog-sop.md`.
- Recover and record the startup catalog source field, line index, skill names,
  skill count, and SHA-256 of the exact recovered prompt/catalog text when
  possible.
- Use `scripts/extract_harness_skills.py` for a first pass; it scans the LLM
  trajectory and falls back to sibling ACP system-prompt traces, but manual SOP
  review remains authoritative when the output requests review.
- Interpret `task_skills_loading` separately from `skill_count`.
  `skill_count` is the full recovered startup catalog size, while
  `task_skills_loading` is only whether the task's own expected skills were
  fully loaded. For with-skills runs expect `task_skills_loading: 1`; for
  without-skills runs expect `task_skills_loading: 0`.
- Do not infer a catalog when the harness does not serialize one. For example,
  current `pi-acp` trajectories may expose no startup skill catalog; mark this
  as `catalog_not_serialized` and rely on tool/file trace evidence.
- For no-skill trials, scan the full trajectory for skill leakage markers such
  as `SKILL.md`, `.codex/skills`, `.agents/skills`, `invoke_skill`,
  `activate_skill`, Claude Code `Skill` calls, and `ToolSearch` selection.

## Publishing And Reporting

Only export, commit, or push healthy latest trials. Exclude partial runs,
infra-failed trials, stale duplicates, local caches, and secrets.

Report findings in this order:

- Blockers: unhealthy trials or integration failures that invalidate data.
- Coverage: expected slots, healthy slots, missing slots, duplicates, and stale
  artifacts, grouped by task/skill mode/trial.
- Evidence: exact run roots, files, commit refs, and validation commands.
- Residual risk: anything not covered by canaries or validators.

When reviewing an already completed run, end with a clear verdict: publishable,
publishable with quarantines, or not publishable.
