# Speed and A/B Experiments

Use this when a run compares the time of two BenchFlow builds, or of BenchFlow
and another framework. Run health is checked as usual. What's added here is
whether the comparison measures what it claims to.

## Register before the first trial

Write a manifest before any trial of the experiment, warm-ups included. It
names:

- each arm, with its exact commit and how it was installed;
- the tasks, the mode and the load;
- the primary metric, and how its confidence interval is computed;
- the stop rules.

Rules for the manifest:

- **Timestamps.** Take the registration time from `date -u` at write time.
  Never type it by hand; a typed timestamp has been wrong before.
- **Scripts.** Hash every harness script into the manifest.
- **Validate the analysis.** Before registering, run the analysis script on a
  previous round and check that it reproduces that round's published numbers.
- **Freezing.** Once a round starts, the manifest is frozen. Record any later
  correction in a separate note beside it.
- **Results.** Keep every round, including invalid ones, with the reason. If
  the primary target is missed, report it as missed and say by how much.

## Remove the confounds that have bitten before

- **Image builds.**
  - A build cache on the provider means the first arm to request an image pays
    for its build.
  - An arm whose image recipe changed pays that build on its first trial of
    every task.
  - Run a warm-up round for those arms, keep it in the record, and leave it out
    of the analysis.
- **Shared caches between arms.**
  - Never `cp` a changed file into a uv-managed venv to make an arm. uv
    hardlinks files from its cache, so the copy rewrites every venv that
    shares them, and the arm compares the fix to itself.
  - Install each arm from `git archive` of its commit.
  - Don't run two arms' sandbox builds for the same task at the same time if
    they share a builder or cache.
- **Trial identity.** Each round uses fresh trial ids, or a fresh output root
  and proxy root. That keeps one round's approvals, telemetry and folders from
  mixing with another's.
- **The orchestrator machine.**
  - The machine that launches trials runs nothing else: no test suites, no
    other lanes' probes.
  - Host-side phases are CPU and disk bound under load, so a neighbour's
    pytest changes them.
  - Check the load before each round and record it.
- **Disk.**
  - Copying a large task folder per trial, plus pytest scratch, can fill a
    shared VM.
  - Check free space before and after each round. Remove only your own
    finished-run scratch, never evidence.

## Measure in a way that says which phase moved

- **Pair by task.** Report the paired median of per-task differences, with a
  seeded bootstrap confidence interval: 4,000 resamples, tasks in sorted order.
- **Report segments from an outer clock** that both arms share:
  - launch to framework start (CLI);
  - framework start to sandbox create (host);
  - create;
  - create end to agent launch (post-create);
  - agent;
  - verifier;
  - teardown.

  Alongside them, report the framework's own spans (for BenchFlow,
  `timing.json`) and its exec and request counters (`daytona-execs.json`).
- **Run at least two loads:** a loaded round (for example 40 trials at once)
  and a light round (for example 6). Queueing under load can hide a fixed
  per-trial cost, and contention under load can create one. In one experiment a
  13–15 s create cost showed only at light load, and a 45 s host-phase cost
  only at 40 at once.
- **Model-free rounds** (a fake upstream through the recording proxy) measure
  harness time, not capability. Outcomes and recorded task digests must still
  be identical between BenchFlow arms; check that as the run's own control.

## Comparing with another framework

- Pass that framework's telemetry opt-out explicitly at the launch boundary of
  every invocation: remote shells, subprocess environments, CI and containers.
  Don't rely on an inherited environment. Keep BenchFlow's own usage, cost and
  timing evidence enabled.
- Keep the comparison and its numbers outside git, in the data folder or
  evidence store. Code, tests, commits and PR text name no other framework.

## What a finished A/B report contains

- The manifest's path and hash.
- The rounds that ran, with start and end times.
- The primary result and its CI.
- The per-segment table for each round.
- Outcome and digest checks.
- Sandboxes left behind. The required number is zero.
- What was not in the tested build.
- Where the raw trial folders are kept.
