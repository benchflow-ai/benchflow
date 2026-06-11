# clawsbench ↔ benchflow compat — issue tracker (fork branch)

This branch (`clawsbench-compat`, off benchflow v0.6.0) is a **fork-style working branch** for
fixes that make benchflow work cleanly with clawsbench/worldcraft, aligned with benchflow's own
directions + abstractions. Issues are tracked here; each fix lands on a `fix/<id>` branch and
merges into `clawsbench-compat`. Nothing here is pushed to `benchflow-ai/benchflow` without an
explicit go — treat it like a fork.

Status legend: `OPEN` · `IN-PROGRESS` · `FIXED` (merged to clawsbench-compat) · `WONTFIX`.

---

## BF-1 — `daytona` optional extra: unclear failure when env selected without it
**Severity:** medium · **Status:** FIXED
Default `benchflow` install drops the Daytona SDK (now `[sandbox-daytona]`), but selecting the
Daytona environment still hard-`ImportError`s deep in the stack. Align with benchflow's
optional-deps pattern: a clear, early "install benchflow[sandbox-daytona]" error at env
selection, not a raw ModuleNotFoundError.
**Fix sketch:** guard the daytona env entrypoint with a friendly `OptionalDependencyError`.

## BF-2 — `SDK.run(trial_name=...)` removed without alias
**Severity:** low · **Status:** FIXED
The rename to `rollout_name` is a hard break for SDK callers. Add a deprecation alias
(`trial_name` → `rollout_name` with a `DeprecationWarning`) for one minor, per benchflow's
public-API stability norms.

## BF-3 — verifier rejects legacy rich `reward.json`
**Severity:** medium · **Status:** FIXED
`validate_reward_map` rejects any top-level key beyond `reward`/`metrics`/`rubric`/… and
non-numeric metrics, so the classic `test.sh → reward.json` flow that emitted `{"reward":1.0,
"done":true,"metrics":{...}}` now fails the run. Align with the "backward compatible" promise
for the test.sh→reward flow: either (a) a documented lenient mode that ignores unknown keys, or
(b) clearly document the canonical schema + a one-line migration. clawsbench currently
stages a canonicalizer to work around it.
**Fix sketch:** add `validate_reward_map(..., lenient=True)` that drops unknown keys with a
warning; default strict.

## BF-4 — provider resolution requires the `provider/` prefix; bare models default to anthropic
**Severity:** medium · **Status:** FIXED
`find_provider` only matches with the `provider/` prefix; the openclaw shim's
`_infer_provider_prefix` only knows gemini/gpt/o1/o3, so a stripped `deepseek-v4-flash` silently
runs as `anthropic/...`. Align with the provider-registry abstraction: have
`_infer_provider_prefix` (and the SDK's stripped-model path) consult `agents.providers` so any
registered provider (deepseek/zai/glm/…) routes correctly even without the prefix.
**Fix sketch:** `_infer_provider_prefix` → registry lookup by model substring; fall back to
anthropic.

## BF-5 — `daytona.list()` returns a generator (daytona ≥0.184) — `.items` breaks
**Severity:** low · **Status:** FIXED
The bumped daytona SDK changed `list()` to a generator; callers doing `result.items` raise
`AttributeError`. If benchflow exposes a sandbox-list helper, normalize its return shape;
otherwise document the daytona-SDK change.
**Fix sketch:** wrap sandbox listing in a helper returning a concrete list.

## BF-6 — `DaytonaSandbox.exec(timeout_sec=None)` wedges forever on a backgrounded command
**Severity:** medium · **Status:** FIXED
`DaytonaSandbox._sandbox_exec` (`src/benchflow/sandbox/daytona.py`, pinned rev `7d49b595`) runs
every command as a Daytona **session** command via `process.execute_session_command(run_async=
True)` (`daytona.py:1562`) and then blocks in `_poll_response` (`daytona.py:1479`) on
`while response.exit_code is None` (`daytona.py:1497`). When the caller passes no `timeout_sec`,
`deadline` is `None` (`daytona.py:1492`) and the loop has **no escape**. The Daytona toolbox only
reports a session command's `exit_code` once the command *completes*, and it treats the command as
still-running while any child holds the session's stdout/stderr stream open (daytona SDK
`process.get_session_command`: `exit_code` = "exit status (if completed)"). So a backgrounded
daemon (`mysvc &` with no `nohup`/redirection/`disown`) keeps that stream open → `exit_code` never
arrives → `exec` hangs indefinitely.

This is **Daytona-specific**: `DockerSandbox.exec` (`src/benchflow/sandbox/docker.py:717`) shells
out via `asyncio.create_subprocess_exec` + `process.communicate()`, which returns as soon as the
foreground `sh -c` exits regardless of orphaned background children — so a `mysvc &` setup hook
passes locally but wedges on Daytona. Any benchflow consumer that backgrounds a service before the
agent runs (the common "start server, poll /health" pre-agent hook) silently wedges at setup.
clawsbench/worldcraft hit this in its `_build_service_hooks` (clawbench `backend.py`); fixed
clawbench-side by fully detaching the daemon's std fds (`nohup … </dev/null >log 2>&1 &` —
redirection alone severs the inherited session stream; no `disown`, which is bash/zsh-only and
absent from the local `sh`/dash hook shell) **and** passing a bounded `timeout_sec`. The clawbench
fix is sufficient for clawbench, but benchflow's `exec` is still a foot-gun for every other
consumer.

**Proposed fix (benchflow-side, defense-in-depth):**
1. In `_poll_response` (`daytona.py:1479`), apply a non-`None` hard-cap deadline even when
   `timeout_sec is None` (e.g. a generous default ceiling) so a never-completing session command
   can never spin the loop forever; raise the existing `"Command timed out"` `RuntimeError` on hit.
2. Document on `SandboxProcess.exec` (`sandbox/protocol.py:106`) that, on Daytona, a command which
   leaves a child holding the session stdout/stderr stream open will block until `timeout_sec`
   (or forever if unset), and recommend `nohup … </dev/null >log 2>&1 &` (full fd redirection) for
   backgrounded processes — mirroring the Docker-vs-Daytona asymmetry.
3. (Optional) For `run_async=True`, detect "launched but foreground exited" and return on the
   foreground command's exit rather than waiting on the whole session stream.
**Fix sketch:** add a `_DAYTONA_DEFAULT_EXEC_CEILING_SEC` and use it as the fallback deadline in
`_poll_response`; add the protocol docstring note.

## BF-7 — `bench eval create` has no `--context-root`; YAML config has no key either
**Severity:** medium · **Status:** FIXED · **Upstream:** benchflow-ai/benchflow#674
Surfaced running skillsgym directly on `benchflow eval create` (smolclaws PR #93).
`context_root` / `stage_dockerfile_deps` exist end-to-end in benchflow — `EvaluationConfig.
context_root` (`evaluation.py:300`) threads through both single-task paths into
`RolloutConfig.context_root` → `stage_dockerfile_deps` (`sandbox/setup.py:310`), and even the
sharded eval worker payload carries it (`eval_sharding.py:120` → `eval_worker.py:58`) — but the
two operator entry points never set it: `eval_create` (`cli/main.py`) has no `--context-root`
flag and `Evaluation._from_native_yaml` has no `context_root` key. So task Dockerfiles that COPY
repo-root paths (e.g. `COPY packages/environments/stripe …`) cannot build from the CLI;
downstream consumers must pre-stage `_deps` with their own script.
**Fix sketch:** add `--context-root PATH` to `eval create` (full-name flag, no short form per
ENG-74) + the native-YAML key, threading to the existing `EvaluationConfig.context_root` — no
new logic.

## BF-8 — verifier rejects negative rewards even when the benchmark's contract floors at −1.0
**Severity:** high · **Status:** FIXED · **Upstream:** benchflow-ai/benchflow#675
Surfaced running skillsgym directly on `benchflow eval create` (smolclaws PR #93). skillsgym's
safety tasks floor reward to **−1.0** on safety violations — the floor sits BELOW
doing-nothing 0.0, which is core to the benchmark's thesis. benchflow's reward contract is
hard-coded [0,1] in `is_valid_reward_number` (`rewards/validation.py`) and enforced at every
chokepoint — `Verifier._parse_reward_text`, `validate_reward_map` (reward.json),
`apply_aggregate_policy`, and the final `_ensure_canonical_rewards` gate in `rollout.py` — and
negatives raise **even in lenient mode** (BF-3 lenient drops an out-of-range `reward` then
fails on "missing numeric 'reward'", verified empirically). Unsafe runs therefore become
verifier ERRORS instead of scored −1.0. The task schema also rejects any declaration: the
`[verifier]` table is `extra="forbid"`, so a task cannot even say its range is wider.
**Fix sketch:** declared reward range — `[verifier] reward_range = [-1.0, 1.0]` in
task.toml/task.md frontmatter (`VerifierConfig.reward_range`, default `[0.0, 1.0]`, validated
min < max + finite), threaded as an explicit `reward_range` parameter through the existing
validators. Default stays [0,1] strict — zero behavior change unless a task declares a range.
Avoid blanket lenient-mode acceptance of negatives (silent contract erosion).

## BF-9 — `ManifestEnvironment` service start lacks the W9 Daytona detachment
**Severity:** high · **Status:** OPEN · **Upstream:** benchflow-ai/benchflow#676
Surfaced running skillsgym directly on `benchflow eval create` (smolclaws PR #93).
`ManifestEnvironment.provision()`/`reset()` (`environment/manifest_env.py`) start each
`[[environment.services]]` command as `{cmd} > {log} 2>&1 &` — stdout/stderr are redirected but
**stdin is not, and there is no `nohup`**. That is the exact failure class as BF-6/W9: on
Daytona, a backgrounded service holding any of the session's std streams open keeps the session
command "running", so `exec` wedges until the timeout (now bounded by BF-6's 3600s cap when
unset — but here `timeout_sec=15` turns the wedge into a hard provision failure on Daytona
while passing on Docker). clawsbench fixed the same bug clawbench-side in
`_build_service_hooks` (`packages/clawbench/clawbench/backend.py`): full three-fd redirection
(`nohup … </dev/null >log 2>&1 &`) severs the inherited session stream; no `disown`, which is
bash/zsh-only and absent from the `sh`/dash hook shell.
**Fix sketch:** start manifest services with `nohup {cmd} </dev/null >{log} 2>&1 &` via one
shared start-command helper (provision and reset currently duplicate the shape), keep the
bounded `timeout_sec` and the per-service log path.

---

### Upstream recheck — 2026-06-11 (vs `origin/release/v0.6.0` head `f642afdc`)
Re-evaluated all 5 against the current 0.6 head, **+20 commits** over the pin `7d49b595`. The 20
commits are almost entirely docs + RC version rolls (the head self-reports `0.6.0rc3`); only 6
`src/` files changed, and of the BF-relevant files **only `sandbox/daytona.py`** moved — for an
unrelated orphan-label guard, not a list helper. `setup.py`, `sdk.py`, `rewards/validation.py`,
`task/verifier.py`, `rollout.py`, `agents/providers.py`, `agents/openclaw_acp_shim.py` are
**byte-identical to the pin**.

| BF | Upstream status | Verdict |
|----|-----------------|---------|
| BF-1 | setup.py unchanged; daytona import-safe (#358) so the `ModuleNotFoundError` guard still never trips → late leak | **STILL-NEEDED** (keep) |
| BF-2 | `trial_name` appears nowhere in upstream `src/`; `SDK.run` is `rollout_name`-only | **STILL-NEEDED** (keep) |
| BF-3 | `validate_reward_map` still strict; no `lenient`/`BENCHFLOW_REWARD_LENIENT` | **STILL-NEEDED** (keep) |
| BF-4 | `_infer_provider_prefix` still defaults non-gemini/gpt → anthropic; no `model_prefixes`/`find_provider_for_bare_model` | **STILL-NEEDED** (keep) |
| BF-5 | no `list_sandboxes` upstream, but reaper + CLI iterate the generator directly and no bare `.items` access exists (daytona pinned ≥0.184 = generator-only) | **NOMINALLY NEEDED** — defensive only; droppable unless clawbench imports `list_sandboxes` |

**Conclusion:** none of BF-1..5 are made redundant by the current 0.6 head — the fork remains the
source of these fixes and is the right thing to upstream. Bumping clawbench's benchflow pin to
`f642afdc` is safe but has **no functional upside for clawbench** and carries one watch-item: 0.6
now **locks `/testbed_verify` by default** (`lockdown.py`, commit `71d2f4fc`) — the `sandbox_user`
agent can no longer read the seeded verifier-side snapshot, which would break any task relying on
that. Recommendation: hold the pin at `7d49b595` until there's a concrete 0.6 fix/feature clawbench
needs; validate the `/testbed_verify` lockdown before any bump.

---

### Worklog
- _(loop appends one line per resolved issue: id, branch, what changed, tests run)_
- **BF-1** · `fix/bf-1-daytona-extra-error` → merged (`--no-ff`) into `clawsbench-compat` ·
  The daytona module is import-safe without the SDK (#358), so the factory's existing
  `try/except ModuleNotFoundError` import guard never tripped — selecting the Daytona env
  leaked a raw `ImportError` deep inside `DaytonaSandbox.__init__` (after CPU/memory clamping).
  Fix: in `_create_sandbox_environment` (`src/benchflow/sandbox/setup.py`) force
  `_load_daytona_sdk()` at env selection and route any `ImportError` through the existing
  `_raise_missing_optional_sandbox_dependency` helper (now typed `exc: ImportError`), matching
  the `modal` branch — a clear `uv sync --extra sandbox-daytona` / `pip install
  'benchflow[sandbox-daytona]'` error that fails fast before construction. Added unit test
  `test_daytona_without_sdk_fails_fast_with_install_hint` (simulates absent SDK via monkeypatch).
  Tests: `pytest tests/test_env_setup.py tests/test_base_install_imports.py` green; broader
  `-k "daytona or sandbox or environment"` → 457 passed / 45 skipped (SDK-gated). ruff check + format clean.
- **BF-2** · `fix/bf-2-sdk-run-trial-name-alias` → merged (`--no-ff`) into `clawsbench-compat` ·
  v0.6 renamed `SDK.run(trial_name=...)` to `rollout_name` with no alias — a hard break for
  pre-v0.6 callers (clawsbench hit it). Fix: re-add an optional `trial_name: str | None = None`
  keyword to `SDK.run` (`src/benchflow/sdk.py`). When only `trial_name` is given it maps to
  `rollout_name` and emits a `DeprecationWarning` via `warnings.warn(..., DeprecationWarning,
  stacklevel=2)` — matching the existing deprecation style in `task/config.py`
  (`memory`/`storage` → `*_mb`). Passing both `trial_name` and `rollout_name` is ambiguous and
  raises `TypeError`; docstring documents the alias. Added `tests/test_sdk_run_alias.py` (warn+map
  path via `pytest.warns`, no-warn `rollout_name` path, `TypeError` on both — `Rollout.create`
  stubbed so no sandbox/cloud). Tests: `pytest tests/test_sdk_run_alias.py` 3 passed; `-k "sdk or
  run or rollout"` → 596 passed / 7 skipped. SDK import smoke + ruff check + format clean.
- **BF-3** · `fix/bf-3-reward-lenient` → merged (`--no-ff`) into `clawsbench-compat` ·
  `validate_reward_map` (`src/benchflow/rewards/validation.py`) rejected any unrecognized
  non-numeric top-level key (the Harbor-era `{"reward":1.0,"done":true,...}`) and any
  non-numeric metric, failing the classic `test.sh → reward.json` run; clawsbench had to stage a
  canonicalizer. Fix: add `validate_reward_map(..., lenient=False)`. Lenient drops
  unrecognized/non-numeric top-level keys, non-numeric metric *entries* (pruned from `metrics`),
  and malformed recognized-structured keys (`rubric`/`space`/`granularity`/`aggregate`, via a
  shared `_apply_structured` helper), emitting ONE `warnings.warn` that lists everything dropped
  instead of raising. A usable scalar `reward` is still REQUIRED — taken from `reward`, else
  derived from a numeric `score`/`rewards` alias, else from numeric metrics + a declared
  aggregate policy (the existing structured path is untouched). Default stays strict (no
  behaviour change). Operator opt-in `reward_lenient_from_env()` reads `BENCHFLOW_REWARD_LENIENT`
  (truthy `1/true/yes/on`), matching the `BENCHFLOW_*` runtime-toggle convention (e.g.
  `BENCHFLOW_DAYTONA_AUTO_REAP`); threaded into `Verifier._parse_reward_json` (the reward.json
  parse path) and the final `_ensure_canonical_rewards` gate in `rollout.py` so both stay
  consistent. Added `tests/test_reward_lenient.py` (strict still raises on `{reward,done}` +
  non-numeric metric; lenient drops `done` + `metrics.label`, keeps `reward`, single warning;
  `score`/`rewards` alias derivation; unusable-`reward` drop-then-derive; still-requires-reward;
  env truthy/falsy parse). Tests: `pytest tests/test_reward_lenient.py` 26 passed; `-k "reward or
  verifier or validation"` → 477 passed / 3009 deselected. ruff check + format clean; import
  smoke clean.
- **BF-4** · `fix/bf-4-bare-model-provider` → merged (`--no-ff`) into `clawsbench-compat` ·
  `strip_provider_prefix` hands downstream a *bare* model id (e.g. `deepseek-v4-flash` from
  `deepseek/deepseek-v4-flash`) that `find_provider` (`src/benchflow/agents/providers.py`) can no
  longer match — it only resolves an explicit `provider/` prefix. So the openclaw ACP shim's
  `_infer_provider_prefix` (`src/benchflow/agents/openclaw_acp_shim.py`), which only knew
  gemini/gpt/o1/o3, defaulted EVERY other bare id to `anthropic` — silently running
  deepseek/zai/glm/minimax/qwen/… as `anthropic/…` (wrong endpoint + wrong key → fail). Fix: keep
  provider knowledge in the registry. Added `ProviderConfig.model_prefixes` — the bare-model
  family tokens a provider owns (`deepseek`→deepseek, `glm`→glm, `qwen`→qwen-dashscope,
  `kimi`/`moonshot`→kimi, `minimax`→minimax, `xiaomi`/`mimo`→xiaomi, `hunyuan`→hunyuan, and the
  full `doubao-seed-2-lite`/`-pro` tokens) — and a registry helper
  `find_provider_for_bare_model(model)` that does longest-token matching with a non-letter family
  boundary (so `glm` matches `glm-4.6`/`glm5`/`glm-5.1` and `qwen` matches the version-suffixed
  `qwen3.6-max`, but `glmnext` does NOT match), then falls back to an unambiguous declared
  `models[].id`; inputs still carrying a registered prefix return None (defer to `find_provider`).
  `_infer_provider_prefix` now consults that helper FIRST (guarded by `try/except ImportError` for
  the no-benchflow shim env), then its native gemini/gpt heuristics, then `anthropic` — so
  gemini/gpt/anthropic behaviour is byte-for-byte unchanged. Added
  `tests/test_bare_model_provider.py` (bare `deepseek-v4-flash`→deepseek, `glm-4.6`→glm,
  `qwen3.6-max-preview`→qwen-dashscope, doubao longest-token disambiguation, `glmnext` boundary
  reject, prefixed-input defers to `find_provider`, empty→None; `_infer_provider_prefix`:
  `gemini-3.1-flash-lite`→google, `gpt-4o`→openai, `whatever-7b`/`claude-*`→anthropic). Tests:
  `pytest tests/test_bare_model_provider.py` 29 passed; `-k "provider or openclaw or shim or
  agent"` → 699 passed / 2 skipped / 2814 deselected; `test_registry_invariants.py` 139 passed.
  ruff check + format clean.
- **BF-5** · `fix/bf-5-daytona-list` → merged (`--no-ff`) into `clawsbench-compat` ·
  daytona is pinned `>=0.184.0` (lock resolves `0.184.0`), whose `Daytona.list()` returns an
  auto-paginating `Iterator[Sandbox]` (a generator) — pre-0.18 returned a *page object* exposing
  sandboxes on a `.items` attribute. benchflow's two listing sites
  (`reap_stale_sandboxes` in `src/benchflow/sandbox/daytona.py` and `environment_list` /
  `bench env list` in `src/benchflow/cli/main.py`) already iterated `client.list()` *directly*, so
  the literal `result.items` `AttributeError` was NOT live inside benchflow — but the listing was
  scattered across both sites, single-pass (a bare generator has no `len()` and exhausts after one
  pass), and not robust to the legacy `.items` page-object shape. Fix (benchflow-aligned, a single
  helper not scattered shape-checks): added `list_sandboxes(client) -> list[Any]` in
  `sandbox/daytona.py` that calls `client.list()` once and materializes a concrete, re-iterable
  `list` from all three shapes — generator/iterator, a non-callable `.items` page object (a
  `callable` guard skips a `Mapping`-style `.items` *method*), and an already-concrete list — then
  routed both the reaper and the CLI through it. Added `tests/test_daytona_list.py`: fakes the
  client at the `.list()` boundary (no daytona SDK / creds — SDK absent in the fork venv) with
  generator-returning, `.items`-returning, and list-returning clients; asserts a concrete
  re-iterable `list` across shapes (incl. `.items` as its own generator, empty cases) and that the
  reaper stays correct on BOTH the generator and `.items` shapes. Tests: `pytest
  tests/test_daytona_list.py` 9 passed; `-k "daytona or sandbox or reap or gc"` → 363 passed / 43
  skipped (SDK-gated) / 3118 deselected. ruff check + format clean; import smoke clean.
- **BF-6** · `fix/bf-6-poll-deadline` → landed in `clawsbench-compat` via the rc4 integration
  (`b87f268c`, cherry-flowed through `compat/clawsbench-v06-fixes`) · `DaytonaSandbox.
  _poll_response` looped `while response.exit_code is None` with `deadline = None` when the
  caller passed no `timeout_sec`, so a backgrounded daemon holding the session's std streams
  open wedged `exec` forever. Fix: module-level safety-net ceiling
  `_DAYTONA_EXEC_HARD_CAP_SEC = 3600` applied ONLY when `timeout_sec is None` (explicit
  timeouts byte-identical to before); on hitting the cap, the SAME "Command timed out"
  `RuntimeError` via a shared `_poll_timeout_error` helper; `SandboxProcess.exec` protocol
  docstring documents the Daytona backgrounded-command semantics and recommends full three-fd
  detachment (`nohup … </dev/null >log 2>&1 &`). Tests: `tests/test_daytona_poll_deadline.py`.
- **2026-06-11 tracker restore** · `CLAWSBENCH-COMPAT.md` was orphaned on
  `fix/bf-6-poll-deadline` when `clawsbench-compat` was re-pointed at the rc4 merge
  (`ac1c2c18`, release/v0.6.0 rc4 + BF-1..6 + audit fixes); restored the latest version
  (`45f68888`) onto the branch, marked BF-6 FIXED, and filed BF-7/8/9 (below) from running
  skillsgym directly on `benchflow eval create` (smolclaws PR #93).
- **BF-7** · `fix/bf-7-eval-create-context-root` → merged (`--no-ff`) into `clawsbench-compat`
  (merge `73fc57b6`, change `887757b9`) · filed upstream as benchflow-ai/benchflow#674 ·
  `context_root`/`stage_dockerfile_deps` existed end-to-end (`EvaluationConfig.context_root` →
  `RolloutConfig` → `stage_dockerfile_deps`, sharded eval worker payload included) but neither
  operator entry point could set it. Fix threads the value to the existing plumbing only — no
  new logic: `eval_create` gains `--context-root PATH` (full-name flag, no short form per
  ENG-74), wired into `_make_eval_config` for the tasks-dir/source-repo paths and as a CLI-wins
  override on the `--config` path (same pattern as `--agent-idle-timeout`);
  `Evaluation._from_native_yaml` gains the `context_root` key (string, default None). Tests:
  `tests/test_context_root_cli.py` (mirrors `test_agent_idle_timeout_cli`) — flag reaches
  `EvaluationConfig`; YAML key parses; CLI overrides YAML; defaults unchanged (None) with and
  without a config file; `--help` documents the flag → 6 passed; broader
  `-k "context_root or eval_create or idle_timeout"` → 76 passed.
- **BF-8** · `fix/bf-8-reward-range` → merged (`--no-ff`) into `clawsbench-compat`
  (change `021a6935`) · upstream benchflow-ai/benchflow#675 · Task-config opt-in:
  `[verifier] reward_range = [lo, hi]` (`VerifierConfig.reward_range`, default None =
  canonical strict [0,1]), parsed from task.toml AND task.md frontmatter (both funnel
  through `TaskConfig.model_validate`). **Design: widen-only** — `lo <= 0.0`, `hi >= 1.0`,
  `lo < hi`, finite, validated once at config parse (`validate_declared_reward_range` in
  `rewards/validation.py`, delegated to by a pydantic field validator) so 0 ("did nothing")
  and 1 ("solved") stay scorable anchors for every task and a range can never narrow or
  shift the contract. Threaded as an explicit `reward_range` parameter through every
  scalar-reward chokepoint: `is_valid_reward_number`, `validate_reward_map`,
  `apply_aggregate_policy` (incl. strict consistency check), `Verifier._parse_reward_text`
  / `_parse_reward_json` / aggregate compat path (range resolved once in `Verifier.__init__`
  via the `declared_reward_range(task)` accessor, MagicMock/legacy-task safe), and rollout's
  final `_ensure_canonical_rewards(…, task=…)` gate. Defaults byte-identical; lenient (BF-3)
  still never accepts negatives on its own — range and lenient compose but neither implies
  the other. **Scoped out, documented:** LLM-judge/agent-judge/ORS-episode scores stay [0,1]
  (normalized judge criteria, not task rewards); string safety verdicts (e.g. `safety_gate`)
  still drop-with-warning in lenient mode per BF-3 — the strict-mode escape hatch is nesting
  them under a structured key (`details`/`metadata`/`reason`), covered by a test. Tests:
  `tests/test_reward_range.py` → 45 passed (declared range accepts −1.0; undeclared still
  raises; malformed/narrowed/shifted ranges rejected at parse; range+lenient and
  range+strict-aggregate compose; task.toml + task.md round-trips; rollout final gate);
  sweep `-k "reward or verifier or validation or config"` → 788 passed / 1 skipped.
  ruff check + format clean; import smoke OK.
