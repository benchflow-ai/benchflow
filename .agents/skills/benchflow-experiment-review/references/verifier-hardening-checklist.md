# Verifier Hardening Checklist

Prevention-side companion to `reward-hacking-patterns.md`. That file is for
*detection* (auditing a finished trajectory); this file is for *prevention*
(closing the vector in the environment) when integration-testing a Benchflow
code change or packaging a task.

The invariants below are drawn from Microsoft AI's MAI-Thinking-1 technical
report, "Building a Hill-Climbing Machine" (2026), which deployed an LLM monitor
plus human review of RL rollouts and documented the concrete mitigations a
frontier training stack uses. The report empirically confirmed three SWE
reward-hacking types — internet answer search, local git-history search, and
test tampering — and reported the fixes below. Each invariant names the
`reward-hacking-patterns.md` pattern it closes. Section refs (e.g. MAI §3.3.1)
point into that report; verify wording against the current paper before quoting.

## Environment Isolation

- Fresh sandbox per task, destroyed on completion, with no state carried between
  tasks or trials. (MAI §3.3, Sandbox Execution Environment)
- Network isolated by default. Only when a task genuinely needs it (e.g.,
  package install) route egress through a caching proxy plus an explicit domain
  allowlist — never open general egress. (MAI §3.3)
  → closes A: online answer lookup.

## Answer-Leakage Removal (category A)

- Git "time-travel": scrub every commit, ref, branch, and tag *after* the task's
  base commit, returning the repo to its original pre-fix state. Do not delete
  git entirely — git is a legitimate skill and removing it breaks valid task
  flows. (MAI §3.3.1; the SWE-bench #465 class of leak)
  → closes A: git-history leakage.
- No answer artifacts in agent-visible paths: reference solutions, expected
  outputs, answer JSON, hidden tests, and scorer files live outside any path the
  agent can read. (MAI §3.3.1)
  → closes A: environment answer leakage.
- Build-time leakage probe: confirm an empty/no-op patch FAILS the grader and
  the golden patch PASSES, across multiple trials; drop environments with
  non-deterministic tests. A passing empty patch means the reward is reachable
  without solving — a leak or grader bug. (MAI §3.3.1)
  → closes A: environment leakage + B: grader-bug exploitation.
- Problem statement carries enough to solve but does not over-specify or embed
  the answer. Score tasks on spec clarity, test quality, leakage risk, and
  feasibility, and rewrite leaky or underspecified statements. (MAI §3.3.1)
  → closes A: environment leakage; reduces B: test-input hardcoding pressure.

## Grader Integrity (category B)

- Reset all agent-modified test files before grading; the grade runs against the
  original tests, not the agent's edits. (MAI §3.3.1, SWE-Bench protocol)
  → closes B: test/verifier mutation, assertion suppression, skip/xfail abuse.
- Hide test changes from the agent during the run and apply them only at grading
  time (hidden tests). (MAI §3.3.1)
  → closes B: test-input hardcoding.
- Run the verifier as a separate process/container, started only after the agent
  exits or times out, writing scores to a path the agent cannot touch. (MAI §3.3
  grading lifecycle; matches this skill's verifier-isolation rule)
  → closes B: direct result writing, exit-code manipulation.
- Residual vectors the report flags as NOT closed by test-reset or hidden tests
  alone: monkey-patching the test framework, and modifying equivalence behavior
  (e.g., overriding `__eq__`). These must be caught by trajectory monitoring, so
  keep the B: monkey-patch and operator-overload checks in
  `reward-hacking-patterns.md` active even on an otherwise hardened environment.
  (MAI §3.3.1)

## Reward / Judge Design

- Prefer verifiable rule-based rewards over LLM-judge rewards wherever the
  constraint can be checked directly; verifiable rewards are reported as less
  prone to reward hacking. (MAI §3.4.1)
- For LLM judges, coarse rubrics (e.g., 0/1/2) are harder to game than granular
  prompt-specific rubrics; add few-shot examples to the judge prompt to reduce
  judge hacking. (MAI §3.5; Appendix Fig 25)
  → reduces grader/judge gaming and sycophancy.
- Penalize length via an acceptable per-context quantile range, not a flat
  penalty, so the anti-verbosity reward cannot itself be hacked by collapsing
  into an overly terse answer. (MAI §3.4.1)
  → closes length-bias gaming.

## Training-Loop Defense (only when runs feed model training)

- Monitor rollouts with an LLM plus human review of flagged cases, and
  continuously strengthen test-file detection/reset and anti-tampering
  heuristics. (MAI §3.3.1)
- When generating SFT/self-distillation data from runs, filter out traces that
  exhibit reward hacking so the behavior is not reinforced into the next model.
  (MAI §3.1.4)

## Integration-Test Checklist

When validating a Benchflow code change or task package, confirm:

- [ ] Network off by default; any egress is proxied and allowlisted.
- [ ] Sandbox is fresh per task and torn down after.
- [ ] Repo git history scrubbed past the base commit (commits, refs, branches, tags).
- [ ] No answer / reference / hidden-test / scorer files in agent-readable paths.
- [ ] Empty patch fails and golden patch passes the grader, across multiple trials.
- [ ] Agent test edits are reset before grading; hidden tests applied at grade time.
- [ ] Verifier runs post-exit, isolated, scoring to an agent-unwritable path.
- [ ] Trajectory monitoring still covers the monkey-patch / `__eq__` / exit-code
      vectors the environment cannot fully block.

## Source

Microsoft AI, "MAI-Thinking-1: Building a Hill-Climbing Machine" (2026):
§3.1.4 (self-distillation trace filtering), §3.3 (Sandbox Execution Environment),
§3.3.1 (preventing reward hacking during SWE RL), §3.4.1 (verifiable vs judge
rewards, length penalty), §3.5 (coarse style graders), Appendix Fig 25 (judge
few-shot).
