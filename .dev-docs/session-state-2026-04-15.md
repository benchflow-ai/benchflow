# Session state — 2026-04-15

**Local-only note. Not committed. Uncommitted under `.dev-docs/` per kywch's request.**

## What moved this session

Big stretch of work across the internalize-harbor-experiments branch plus one bug-fix on `main`-targeted code. Summary below so the next session can start cold.

### Oracle parity (headline)

- 172-task oracle parity run between benchflow and Harbor (same Daytona backend, same task set, same oracle agent).
- Morning pre-fix baseline: benchflow 143/173 (82.7%) vs Harbor 147/173 (85.0%) — benchflow ~2.3 pts behind.
- Found three bugs, fixed all three (see issue #137), rerun landed at **148/172 = 86.0%**, ahead of Harbor.
- The 15% gap that remains is upstream skillsbench/tb2 task issues (broken solve.sh/test.sh pairs, task-internal flakes) — not benchflow bugs. Neither runner passes them.

### Three bugs (0.2.3 scope)

All three are documented in detail in issue #137. Pointers:

1. **Oracle tee-stream slowdown** — `benchflow.sdk._run_oracle` piped solve.sh through `| tee`, which streamed every line back over the Daytona exec channel. For apt-heavy tasks (tens of MB of output) the per-line overhead caused ~20-30× slowdown and 900s/1200s timeouts. Fix: redirect to container-local file + `DEBIAN_FRONTEND=noninteractive`. Patched in `sdk.py:_run_oracle`. Unmerged on the working branch, not yet in its own commit.
2. **PYTHONSAFEPATH broke sibling imports** — cancel-async-tasks was failing with `ModuleNotFoundError: No module named 'run'` because `-P` dropped the script dir from sys.path. Already shipped in commit `9b0d484`.
3. **PATH scrub dropped Dockerfile compile-time entries** — fix-build-agentops + fix-build-google-auto failing with `uv: command not found` even though Dockerfile set `ENV PATH="/root/.local/bin:$PATH"`. Verifier._harden now merges compile-time PATH via `bash -lc 'printenv PATH'`. Already shipped in commit `6ec00f3`.

### Harbor internalization (larger thread)

Four tasks worth, all except the last one landed in commit `a2571c4`:

- Task vendored → `benchflow.schemas.task.{Task, TaskConfig, SkillRef}` with schema_version 1.2
- Verifier vendored → `benchflow.verifier.Verifier` with `_harden()` structural (runs as first line of `verify()`, can't be bypassed)
- Multi-agent schema → `[[agents]]` + `RoleAgentConfig` (for the clawsbench multi-agent mock scenarios)
- DockerEnvironment still NOT vendored — see issue #142 for cold-start plan

### managed-evals prototype

Full server/client/resources/runner/store/cli scaffolding shipped as part of the 4-primitive refactor (task #15). Lives under `src/benchflow/managed_evals/`. API mirrors Anthropic's Agent / Environment / Session / Events shape but simplified to {Agent, Environment, Trajectory, Job} per xdotli's redirect ("we don't need Session + Events, too Anthropic-y"). Event catalog adds `TrajectoryEnvSnapshotEvent` + `TrajectoryProcessRewardEvent` for process/dense rewards (OpenRewardStandard alignment).

Race fix during prototyping: `store.append_event` used MAX(seq)+1 which races under concurrent writers and hit UNIQUE constraint crashes. Fixed with a Python `threading.Lock` serialising the read-modify-write window. Not the prettiest but works.

SSE envelope drift: `/stream` wrapped events under `{payload:}` key but the prod SDK expects flattened event fields at root. Fixed.

### Harbor registry compat + benchflow registry

- `harbor://<dataset>` prefix → fetch from upstream Harbor registry.json using sparse git checkout
- `benchflow://<dataset>` prefix → fetch from benchflow's own registry (differentiation)
- Both prefixes work in `bf eval create`, `bf run`, `bf job`

### CLI shape (task #25, issue #139)

Scaffolded `bf eval create <TASK_REF>` as the main entry point, replacing `bf run`. Not yet finished — see issue #139 for what's left.

### skillsbench cleanup (committed on a different branch)

Branch: `cleanup/task-resources-2026-04-15` on the skillsbench repo. One commit `8d6716de` fixed 8 tasks to respect Daytona limits (cpu≤4, memory≤8GB, storage≤10GB). Also fixed `parallel-tfidf-search` name format (had a space, which Harbor's TaskConfig validator rejected).

Then today's regression rerun surfaced 2 more tasks I missed — `organize-messy-files` and `syzkaller-ppdev-syzlang` both had storage_mb=20480. Fixed in-place but NOT YET COMMITTED on the skillsbench branch. Either add those to the same cleanup commit (amend) or create a follow-up commit.

## Unmerged state across worktrees

- `benchflow/internalize-harbor-experiments` — ~4 clean commits ahead of main (`d09d336 refactor: extract _sandbox from sdk.py`, `a2571c4 feat: vendor Task + Verifier + structural hardening`, `6ec00f3 fix(verifier): auto-merge compile-time Dockerfile PATH`, `9b0d484 fix(verifier): remove PYTHONSAFEPATH`), plus today's tee-fix in `sdk.py:_run_oracle` still uncommitted in the working tree.
- `skillsbench/cleanup/task-resources-2026-04-15` — 1 commit ahead of main (`8d6716de`), plus 2 uncommitted task.toml fixes from today's regression rerun.
- benchflow `main` is untouched by this session.

## What the next session should start with

1. **Ship 0.2.3** (blocked by the commit prep):
   - Rebase/squash the internalize-harbor-experiments branch into a logical 0.2.3 PR
   - Add the tee fix as its own commit
   - Bump version in `pyproject.toml`
   - Cut the PR referencing issue #137
2. **Close the skillsbench cleanup** — amend or follow-up the `organize-messy-files` + `syzkaller-ppdev-syzlang` fixes
3. **Work through the future-plan issues**, in this order:
   - #139 `bf eval create` CLI finish (2-3 hrs)
   - #140 declarative skills in task.toml (1-2 days)
   - #141 Haiku agent-parity run (2-3 hrs, Daytona $$)
   - #142 vendor DockerEnvironment (1 day, deferred, lowest priority)

## Useful invariants discovered

- **No subprocess fanout**. 2-worker × 32-coroutine pool beats N subprocesses × 1-coroutine for OOM and import overhead. Already in memory.
- **Harbor's OracleAgent** is the reference for what a minimal-overhead runner looks like for oracle runs. `harbor.agents.oracle.OracleAgent.run`. Read this if you're tempted to stream oracle output through the exec channel.
- **Structural hardening** > procedural hardening. `Verifier._harden()` runs as the first line of `verify()` and cannot be bypassed — don't regress to calling a helper from `sdk.py`.

## Loose threads I didn't close

- `fix-build-agentops` + `fix-build-google-auto` still fail on benchflow's side due to task-internal flakiness. Would be worth filing upstream in skillsbench.
- 3 Daytona errors from the 0.2.3 regression (`multilingual-video-dubbing`, `shock-analysis-demand`, `speaker-diarization-subtitles`) — 2 are sandbox spin-up timeouts (transient), 1 is "path does not exist" which is weirder. Worth revisiting.
- `test_agent_spec.py` import error + `test_registry_invariants.py::test_agent_field_shapes[oracle]` — both pre-existing failures unrelated to today's work but visible in any test run. Someone should look at them.
