# Task Standard Learning Checklist

Status: in progress

Purpose: verify that the human deeply understands the BenchFlow `task.md`
standard work before the session is considered done. This is a running
checklist, not a final report.

## Stage 1: Problem And Motivation

- [ ] Explain why the previous `task.md` goal is not complete yet.
- [ ] Explain the difference between "good authoring format" and "completed
  executable standard".
- [ ] Explain why BenchFlow wants `task.md` instead of making
  `instruction.md` the entrypoint.
- [ ] Explain why `task.toml + instruction.md + solution/ + tests/` remains
  important as Harbor/Pier compatibility rather than the native ideal.
- [ ] Explain the three major branches of the work:
  - authoring: `task.md`, frontmatter, prompts, roles, scenes, user persona
  - verifier/oracle vocabulary: `verifier/` and `oracle/` as native names
  - runtime/compatibility: capability gates, reward semantics, export loss
    reports
- [ ] Explain why real dogfood tasks matter more than toy examples.

## Stage 2: Current State

- [ ] Identify what already landed in the current worktree:
  - `bench tasks init` defaults to `task.md`
  - legacy format remains available
  - `bench tasks migrate` exists
  - `oracle/` and `verifier/` are native, with `solution/` and `tests/` aliases
  - trace and skill-eval generators default to native packages
  - real BenchFlow wanted-feature tasks exist as dogfood examples
- [ ] Explain what the HTML mind map is for and why it is not the standard
  itself.
- [ ] Explain why parser support alone is weaker evidence than runtime support.

## Stage 3: Design Decisions

- [ ] Explain why BenchFlow keeps Harbor-compatible root config keys but puts
  BenchFlow-specific concepts under `benchflow:`.
- [ ] Explain why `solution -> oracle` and `tests -> verifier` are semantic
  renames, not just cosmetic directory changes.
- [ ] Explain why verifier should be a peer package, similar to `task.md`, with
  `verifier/verifier.md` as its own entrypoint.
- [ ] Explain why agent-as-judge belongs under verifier scope rather than as a
  solver scene.
- [ ] Explain why `reward.json` should become authoritative while `reward.txt`
  remains scalar compatibility.
- [ ] Explain how multi-metric reward maps differ from a single scalar reward.
- [ ] Explain why unsupported runtime semantics should fail closed instead of
  being silently ignored.

## Stage 4: Edge Cases And Failure Modes

- [ ] Explain what should happen when both `oracle/` and `solution/` exist.
- [ ] Explain what should happen when both `verifier/` and `tests/` exist.
- [ ] Explain what should happen when `task.md` parses a field the selected
  sandbox cannot execute.
- [ ] Explain what should happen when `reward.json` and `reward.txt` disagree.
- [ ] Explain what should happen when a verifier emits a multi-metric map but
  no aggregate policy.
- [ ] Explain why judge credentials and judge prompts must not be visible to
  the solver agent.
- [ ] Explain why export to Harbor/Pier needs a typed loss report.

## Stage 5: Broader Context And Impact

- [ ] Explain how this affects task authors.
- [ ] Explain how this affects benchmark adapters and fork users.
- [ ] Explain how this affects rollout/runtime code.
- [ ] Explain how this affects reward tracking, leaderboards, and trajectory
  review.
- [ ] Explain how this affects future NudgeBench, multi-agent, agent-team, and
  multi-scene work.

## Stage 6: Demonstrated Mastery

- [ ] Human can give a concise five-minute explanation of the whole standard.
- [ ] Human can describe at least three remaining gaps without prompting.
- [ ] Human can choose the next implementation slice and justify why it is next.
- [ ] Human can answer quiz questions about problem, solution, edge cases, and
  impact.

## Current Teaching Notes

- 2026-06-05: The previous standardization goal is paused, not complete. The
  next teaching step is to have the human restate her current understanding of
  why it is incomplete and what remains.
