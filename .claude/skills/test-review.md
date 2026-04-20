---
name: test-review
description: Mutation-kill oriented review of benchflow test files — find bloat to delete/collapse and coverage gaps to add, without naive deletion
user_invocable: true
---

# Test Review Skill

Review test files for mutation-kill value. Most bloat hides as assertions
that restate the mock's `return_value` or the expected dict; most gaps
hide behind a wall of happy-path coverage. Sibling to `/code-cleanup` —
that skill reshapes production code; this one reshapes the tests that
guard it.

**Do not auto-edit.** Report ranked findings; user approves. Test
deletions have audit-trail implications — a deleted test may be the only
surviving record of a fixed bug. Human sign-off required.

## Commands

- `/test-review` — whole `tests/`, split by module group across 2–4 parallel agents
- `/test-review <path>` — single test file or subtree
- `/test-review --recent` — tests whose source changed in the last ~20 commits

## The core question

**"If I mutate the production code, does this test fail?"** A test that
passes against any plausible implementation of the function under test is
zero-value — it's asserting on the mock, the dict literal, or a guard
that can't fire. Every rule below is a specialization of this question.

Acceptance test for every finding: **name the mutation the finding would
kill** (for adds) or **name the mutation the test fails to kill** (for
deletes). If you can't, drop the finding.

## Delete / collapse rules

Each rule: (a) one-line rule, (b) pattern to look for, (c) when NOT to apply.

### 1. Mock-echo — delete

(a) Test asserts a value the mock itself produced, through a pass-through.
(b) `mock.return_value = X; ...; assert result == X` (or `AsyncMock(return_value=X)`) with no transform between.
(c) Keep if the forwarding applies a rename, merge, default-fill, or conditional — then mock value and assertion differ. ALSO keep if the assertion uses `!=` / `not in` / `assert not re.search(...)` (adversarial guard), or asserts a production-side call signature via `mock.assert_called_with(...)` / `mock.call_args == ...` — that's verifying what the code under test *invoked*, not echoing a return value.

### 2. Setter/getter parade — collapse to one smoke test

(a) One `def test_...()` per field on a dataclass / config builder / `TaskConfig` / `RunResult`.
(b) Three or more tests on a 5-line function, each checking one field read.
(c) Keep separate if a field has real branching (default fallback, validation, derived value, computed property).

### 3. Template-echo / dict-echo assertions — delete or replace with shape check

(a) Expected dict / string constructed in the test using the same dict-literal or f-string the production code uses.
(b) Test imports the same constant, duplicates a multi-line f-string verbatim, or builds the expected dict with a `**input_dict` spread on the same input fed to production.
(c) Keep if asserting a stable contract external consumers rely on (ACP envelope shape, trajectory JSON on-disk format, task YAML schema, CLI JSON output). Keep if adversarial — e.g. input that *looks like* it should render but must be stripped (empty list, placeholder stub); negative assertions are the whole point. Replace template-echo with targeted `in` / shape assertions (`assert "foo" in result["bar"]`).

### 4. Defensive-check tests for unreachable states — delete

(a) Exercises an early `if not x: return` / `raise` guard on input real callers can't produce.
(b) Guard exists because the function is exported for tests; no production path passes `None`.
(c) Keep if the guard is a public-API contract (a CLI-exposed command, a public SDK entry real users can hit with bad input). Keep if guarded by a "don't regress" source comment.

### 5. Parallel one-liners — collapse to `@pytest.mark.parametrize`, **carefully**

(a) 5+ tests with identical setup differing only by one input value.
(b) `def test_handles_x(...)`, `def test_handles_y(...)` with near-duplicate bodies.
(c) **Keep separate when cases have distinct failure narratives.** Before collapsing, ask: (i) do the cases exercise different regex branches, registry-lookup layers, filter-chain stages, or guard clauses? (ii) Do they target different constants / boundaries in the source? (iii) Would `case 3 failed` in a parametrized run be less useful than the named `def test_` when debugging? If any yes, keep separate. Default to keep-separate for tests on invariants (`test_registry_invariants`, `test_invariants`), permission / sandbox guards, and format-builders with multiple branches. Parametrize is right only when the differentiator is purely the input literal and every case fails at the same assertion for the same reason. Use `pytest.param(..., id="human-readable")` when collapsing so failure output remains legible.

### 6. Integration tests that mock the integration — delete or demote

(a) Test is marked `@pytest.mark.live` but patches the seam it claims to integrate.
(b) Count SEAM mocks, not total mocks: a mock target resolves to a *seam* if it lives in `src/benchflow/` (sdk, job, agents, acp, sandbox, verifier — the modules the integration claims to exercise). A mock target resolves to a *boundary* if it resolves to a third-party SDK (`anthropic`, `openai`), a system boundary (`subprocess.run`, `docker`, `asyncio.create_subprocess_exec`, `socket`, `httpx`, the OS filesystem), or the ACP transport wire. Flag if ≥2 mocks hit seams. Raw `patch(...)` count is not the signal.
(c) Keep if every mock target is a boundary (LLM provider SDK, Docker CLI, subprocess, fs). **But** if the `@pytest.mark.live` test is the only surviving coverage of a cross-module triangle, keep it even if heavily mocked — deleting kills the only seam record. **To invoke this escape, name the three seams and the call site where they interact** (e.g., "sdk + agent_env + sandbox meet in `SDK.run()` at sdk.py:412; no other test exercises all three"). Don't assert the triangle exists without naming it — an unfalsifiable escape clause always wins.

### 7. Vendor smoke tests — rewrite or delete

(a) Test exercises behavior of a dependency, not this codebase's usage of it.
(b) Test could live in the vendor's own repo unchanged (e.g. checks `typer` arg parsing rules, `pydantic` validation mechanics, `pyyaml` round-trip, `httpx` retries).
(c) **Production-usage gate before deleting:** run `git grep "<dependency>" src/benchflow/` (production only, not tests). If grep finds hits, the dependency is live in production — rewrite the test to pin the specific flags/wrapper contract we rely on, don't delete. If grep finds zero production hits, the dependency is dead — safe to delete. **Ordering-contract check:** before deleting a vendor-smoke test, scan the assertion sequence for an ordering invariant the vendor requires (e.g., `subprocess` open → write → close, ACP `session/new` → `session/prompt` → response, container start → exec → stop, `asyncio` cancel → await). If the test asserts both calls happened but ignores order, and reversal would silently break production (e.g., ACP request out of sequence, container exec before start), loosen to `mock_calls == [call(...), call(...)]` (ordered) or use `assert_has_calls(..., any_order=False)` — don't delete.

### 8. Over-specified forwarding — relax assertion shape

(a) `mock.assert_called_with(big_dict)` / `assert result == big_dict` when the real downstream contract is looser. Also applies to `assert rr == RunResult(...)` checks where only 1–2 fields carry the contract — the rest is snapshot noise that will fail on unrelated field additions.
(b) Expected object constructed via `**input_dict` of the same input the test fed in, or by copying every field from an existing fixture.
(c) Keep exact-match if the production code *assembles* the shape from multiple sources — the assembly is what's under test. Otherwise loosen to field-subset assertions (`assert rr.error is None`, `assert rr.success is True`) or use `mock.assert_called_once()` + targeted kwarg checks.

### 9. Thin forwarding — delete if mutation-insensitive

(a) Test exercises a pass-through whose only real work is calling one downstream SDK/function.
(b) `def wrap(x): return sdk.record(x)` tested as `sdk.record.assert_called_with(x)` — no transform, no guard, no branch between input and forwarded call.
(c) Keep if the wrapper applies a rename/merge/default-fill, enforces a cap / auth check / permission guard, or the test adversarially asserts the call was *skipped* under a condition (e.g., `--dry-run` short-circuit, oracle bypass in `sdk.py`). Also keep if it's the only seam record of a cross-module contract (name the seam, same discipline as 6(c)). Delete if the assertion only pins "the SDK was called" — that's a tautology against a one-line wrapper.

### 10. Stale mock target — delete or repoint

(a) `patch("<path>")` where `<path>` doesn't resolve to a real attribute (renamed file, split package, deleted export), or uses `create=True` against a symbol that no longer exists.
(b) Ordinary `patch` raises `AttributeError` at call time — but `patch("x.y", create=True)` or `patch.dict(...)` against a missing key silently create ghosts; `monkeypatch.setattr("x.y", v, raising=False)` does the same. Every assertion downstream of that patch is then asserting against an auto-mocked ghost — mutation-blind by construction. Also flag `patch(...)` where the *import path* is correct but the test patches the wrong module (patching `anthropic.Anthropic` when production imports via `from anthropic import Anthropic` in `agents/providers.py` — patch must target `benchflow.agents.providers.Anthropic`).
(c) Never keep as-is. Either repoint to the real attribute (and re-verify the test against the new seam) or delete the test if the original triangle no longer exists. Does not go through rule 6's seam-count threshold — one dead patch is enough.

## Keep-as-is signals

Do not touch a test that shows any of these:

1. **Paired with a "don't regress" comment in the source**, or lives in
   `test_invariants.py` / `test_registry_invariants.py` / `test_reexport.py`
   / `test_sdk_lockdown.py`. The source comment or filename points back
   at the test — they're load-bearing by construction. `lockdown` /
   `invariants` / `reexport` tests are specifically *designed* to fail
   loudly on refactors; that's the point.
2. **Uses real deps where cheap.** Real fs in `tmp_path` fixtures, real
   subprocesses against local scripts, real YAML parsing, real task
   loading. Low mock-tautology risk.
3. **Exercises a genuine branch readable in the source.** If you can
   point at the `if` / `match` / `except` the test covers, it earns its
   keep.
4. **Documents an ordered contract.** A filter chain, a layered guard,
   a fallback sequence, the SDK run phases (SETUP → START → AGENT →
   VERIFY). Named `def test_` blocks encode the order;
   `@pytest.mark.parametrize` hides it.
5. **Adversarial / false-positive guard.** Tests that something *doesn't*
   match are often the only defense against over-eager regex or
   fuzzy-match logic (sandbox escape detection, credential redaction,
   path traversal). Easy to mistake for bloat; they're the opposite.

## Add-coverage rules

New tests must tie to one of these observed gap patterns. Don't invent
coverage for code that already has adequate honest tests.

### A. Untested complex public functions — one happy path + one error path

Prioritize functions with ≥2 branches, ≥1 external dep, and ≥1 importer
in the main code path. Smell: file has heavy coverage of trivial
dataclass reads and zero coverage of the load-bearing public function in
the same module.

### B. Retry / fallback `except` blocks — one test per non-trivial handler

Source has a `try/except` where the `except` has real logic (not just
`logger.error(e)`) and no test ever raises to reach it. Common
benchflow surfaces: subprocess failures in `_sandbox.py` / `process.py`,
ACP transport errors in `_acp_run.py` and `acp/`, provider auth failures
in `agents/providers.py`, verifier crashes in `sdk.py`.

### C. Concurrency / locking branches — one contention test

Source has a module-level `dict[id, asyncio.Task]`, `asyncio.Lock`, or
`is_running` flag, and all tests call the function once. Add a test
that fires two overlapping calls and asserts serialization. **Skip if
the contention is architecturally closed upstream** — don't mock both
sides of a guard that real callers can't defeat.

### D. Cap / limit boundary values — one test at `cap - 1` and `cap + 1`

`git grep -n 'MAX_\|TIMEOUT\|_LIMIT' src/benchflow/` returns constants
whose exact gate value is never fed as a test input. **Skip timing-
dependent caps** (timeout seconds, `asyncio.wait_for`) — unit tests are
flaky against them; leave to `@pytest.mark.live` integration.

## Execution

### Pass 1 — discovery (parallel subagents)

Spawn 2–4 subagents in parallel, each covering a disjoint slice of test
files. Suggested slice for benchflow:

- `test_sandbox*.py` + `test_verify.py` (the two biggest)
- `test_sdk_*.py` + `test_job.py` + `test_acp.py`
- `test_providers.py` + `test_pi_acp_launcher.py` + `test_agent_*.py` + `test_registry_invariants.py`
- everything else (`test_tasks.py`, `test_metrics.py`, `test_scoring.py`, `test_skills.py`, `test_yaml_config.py`, …)

Each agent prompt includes:

- Exact test file list for that slice
- Instruction to **read the module under test first** — without the
  source, every test looks reasonable
- The Delete/collapse rules 1–10 verbatim, with `(c)` clauses
- The Keep-as-is signals verbatim
- The Add-coverage rules A–D
- **Required output format per finding** (not advice — required):
  `Test: <def test_... name>. File: <path:line>. Rule: <# or keep-signal #>. Mutation: <one-line concrete code change>.`
  For deletes/collapses: the mutation must be one the test currently *fails* to catch. For keeps/adds: the mutation is one the test *does* or *would* catch. If the only mutation you can name is "delete the whole function", drop the finding — that's removal, not a mutation.
- **Output discipline** (each sub-rule is binding — do not elide):
  - Emit only the required finding lines, grouped under the bucket headers (`delete | collapse | loosen | weak | missing | keep`). No preamble, no trailing summary table, no narrative.
  - **Each `def test_` appears in at most one bucket.** If you're torn between delete and keep, the tie goes to keep. Never emit the same test twice across buckets.
  - Silence means keep-as-is applies — do NOT list every keep.
  - **Hard cap: at most ~5 entries in `keep`** per slice. List a test under `keep` only if it was a plausible rule-1-through-10 candidate you *decided* to keep (cite the keep-signal #). If you have more than 5, you're over-reporting; cut to the most genuinely-ambiguous ones. This is the one place verbosity earns its keep, because it shows the skeptic where to challenge.
- **Mutation concreteness.** The mutation must be a single concrete code edit a skeptic could re-verify by reading the source. Not good enough: "change the assembly", "update the forwarding", "modify the guard". Required: name the exact identifier and the exact change — "rename `metadata.timestamp` → `metadata.date`", "flip `>=` to `>` at metrics.py:47", "return `None` instead of empty list from `parse_manifest`'s empty-input branch". If the only mutation you can name amounts to removing the function or deleting the loop body, that's not a mutation — drop the finding per the acceptance test.
- Line numbers point at the `def test_` line, not the enclosing `class Test...` or module.
- Before flagging rule 6 (`@pytest.mark.live` mock), classify each `patch(...)` / `monkeypatch.setattr(...)` target as *seam* (resolves to `src/benchflow/`) or *boundary* (third-party SDK, `subprocess`, `os`, `docker`, transport wire). Only flag if ≥2 are seams.
- Before flagging rule 7 (vendor smoke), run `git grep "<dependency>" src/benchflow/` to check production usage. **Quote the grep hit count in the finding rationale** (e.g. "grep typer src/benchflow/ → 6 hits in cli/*.py") so Pass 2 can audit the gate. If the dependency is invoked from `src/benchflow/`, rewrite — don't delete. Also scan the test's assertion sequence for vendor ordering contracts per rule 7(c).

Each agent returns: per file, `delete | collapse | loosen | weak |
missing | keep` buckets with specific test names and mutation rationale.

### Pass 2 — challenge

Spawn **two skeptic subagents in parallel**, split by input type.
Combining both jobs into one prompt produces bias-mixing: the observed
failure mode is rubber-stamping adds while engaging only deletes (or
vice versa). Splitting by input type honors the disjoint guardrail sets
(#1–7 generic + delete-focused; #8–13 add-focused) and the different
tally floors.

**Skeptic A — delete/collapse/loosen/weak.** Input: the union of Pass 1
findings in buckets `delete | collapse | loosen | weak`. Apply
guardrails #1–7 verbatim. Do NOT emit verdicts on `missing` entries.

**Skeptic B — adds.** Input: Pass 1 `missing` bucket only. Apply
guardrails #8–13 verbatim with the ≥50% reject floor (when ≥6 adds).
Do NOT emit verdicts on delete/collapse/loosen/weak. Each REJECT must
cite which of #8–13 it invoked.

Both skeptics share the same prompt preamble:

> "Push back on any recommendation that: (A) loses real signal when collapsed, (B) misapplies the tautological label to a regression-pinned test (e.g. anything in `test_invariants.py`, `test_lockdown*.py`, `test_reexport.py`), (C) recommends coverage that wouldn't earn its keep, (D) shrinks the suite at cost of readability, (E) misses bigger bloat the reviewers overlooked.
>
> **Calibration guardrails — these bind your verdicts:**
> 1. Pass 1 has already applied every `(c)` escape clause verbatim. Do not re-apply escape clauses — only flag a case where Pass 1 *misapplied* one (and say which).
> 2. Focus challenges on **mutation concreteness**: is the named mutation a real single-identifier edit, or hand-waved assembly?
> 3. If the only mutation a test catches is 'delete the whole function' or 'delete the loop body', the test MUST be dropped — do not rescue it as a 'weak keep' or 'init contract'. The acceptance test is non-negotiable.
> 4. Do not emit self-contradicting verdicts ('weak but keep ... confirm as bloat'). One verdict per finding: CONFIRM | REJECT | AMEND.
> 5. A 70%+ reject rate is a signal of defensive-keep bias, not rigor. If you're rejecting more than half, re-check each for rule 3 before finalizing.
> 6. **AMEND is downward-only** — you may re-bucket keep→weak→delete or loosen→collapse→delete, but never the reverse. Upward rescues (delete/weak/collapse → keep) must be emitted as REJECT, and the reject reason must quote the specific keep-signal # or rule-(c) clause that Pass 1 misapplied. This forces the skeptic to justify escape-clause invocations rather than silently re-applying them. Edge case: "collapse → keep-as-parametrized" is downward when starting from collapse-to-delete (tests survive densified) but upward when starting from keep-as-separate. Judge by the starting bucket, not the ending bucket.
> 7. **AMEND cap: at most ~10% of findings.** AMEND is the lazy verdict; prefer CONFIRM or REJECT. Each AMEND must state the new mutation the re-bucketed test catches, not just the new bucket name.
>
> **Add-bloat guardrails (apply to `missing` / Rule A–D findings):**
> 8. **Indirect-coverage check.** REJECT an Add if an existing sibling test's assertions would fail under the named mutation *at the same boundary comparison or branch*. Example: a proposed `cap - 1` test is redundant with an at-cap test only when both target the same `>=` vs `>` flip; if the sibling pins `cap` exactly and the Add targets the off-by-one mutation `i > MAX → i >= MAX - 1`, they target distinct operators — keep both. The test is whether the sibling's *assertion* fails under the Add's *named mutation*, not whether the sibling is "nearby."
> 9. **Harness cost gate.** REJECT if the add requires substantial new scaffolding (mocking a class not already mocked in the file, faking async streaming, spinning up a real container, wiring a new ACP fixture) *and* the named mutation is low-probability *and low blast-radius*. Carve-out: sandbox-escape, credential-redaction, trajectory-emit, ACP protocol-ordering catches (session/new → session/prompt, flush → shutdown) are high-blast-radius even when low-probability — the cost gate does not apply; write the test. Scaffolding that earns its keep covers multiple future adds — one-off scaffolding for one low-value low-impact add is bloat.
> 10. **Phantom-branch check.** Read the production source at the cited line and walk the actual call path to that branch *including the proposed test's own setup* — flag-toggles (`--oracle`, `--dry-run`), injected deps, and mocked preconditions count as reachable. REJECT only if the branch is unreachable even after the test's setup (e.g., an early-return upstream the test cannot bypass, or a `platform.system()` gate the test doesn't patch).
> 11. **Sibling-redundancy.** If two proposed adds target different branches inside the same function, REJECT the second unless the two branches require *contradictory* fixture shapes that one test cannot carry (e.g., cycle-break needs a cycle; author-mismatch needs a linear chain — different fixtures). When a single fixture *can* cover both, AMEND the first to exercise both branches; the merged test must carry **one distinct assertion per branch**, each keyed to its own named mutation. A merged test whose only assertion checks the outer outcome is mutation-blind against the inner branch.
> 12. **Rule A exclusion.** REJECT a Rule-A add whose only named mutation is 'delete the function' or 'remove the whole return statement' — those are removals. Operator flips (`>=` → `>`), constant changes, argument-order swaps, identifier renames, and boolean inversions (`return True` → `return False`) are NOT removals and do NOT invoke this guardrail (consistent with guardrail 3). This rule bites only when the Add targets a one-line wrapper whose production mutation collapses to deleting the line — cosmetic local-variable renames or no-op identifier changes fail the acceptance test independently via guardrail 2.
> 13. **Mock-depth gate.** REJECT any Add whose setup would mock both a wrapper and the wrapper's own downstream seam (i.e., the call chain is mocked to the leaf). Assertions against a fully-mocked chain are mutation-blind by construction — the test cannot fail unless the mocks themselves change. Either the Add belongs under `@pytest.mark.live` with real seams, or it doesn't earn its keep."

**Tally discipline:** each skeptic reports its reject rate separately.
Typical: Skeptic A runs **20–40% reject**; Skeptic B runs **≥50% reject
on adds when there are ≥6 adds** (Pass 1 is biased toward over-proposing
coverage; a well-calibrated Pass 1 with ≥6 adds will still lose half).
Below 6 adds, Skeptic B judges per-finding — a floor on a tiny sample is
noise. The ≥50% target is a *floor*, not a ceiling — do not stop
rejecting at 50% and rubber-stamp the rest. Each Skeptic-B REJECT must
cite the specific guardrail # invoked (one of #8–13); if Skeptic B
processed ≥5 adds and any of #8–13 was never invoked, the skeptic
skipped it — flag for a second pass. Outside Skeptic A's 10–50% band
warrants attention: >50% suggests defensive-keep bias; <10% is noise —
only re-run Skeptic A if **<10% reject AND ≥1 AMEND AND Pass 3 drops at
least one finding**. Do not skip Pass 2.

**Coherence reconciler (cheap bolt-on between Pass 2 and Pass 3).** After
both skeptics return, do a single mechanical check: for each surviving
Add, does it propose covering a branch that a surviving Delete just
orphaned? If yes, drop the Add — you can't simultaneously say "this
test is worthless" and "we need a new test for this branch." This
closes the cross-skeptic gap the split creates. One prompt, ~50 lines.
Skip if either bucket returned zero survivors.

### Pass 3 — acceptance-test audit

Spawn **one audit subagent** over the skeptic-adjusted list. Single
responsibility: for each surviving finding, read the test and the
production source, then decide whether the Pass-1 named mutation is (i)
a concrete single-identifier edit to **production code**, and (ii)
genuinely uncaught (for deletes/collapses/loosens/weak) or genuinely
caught (for adds) by the test as written. Verdicts:

- `VERIFIED` — the named production-code mutation is concrete and the catch/miss claim holds.
- `DROP (only-mutation-is-removal)` — apply ONLY when the Pass-1 mutation field literally reduces to "delete the function", "delete the loop body", or "remove the file". A rename, constant flip, operator swap, argument reorder, or identifier change is NEVER this verdict, **even for delete-bucket findings**. The skill's bucket label (delete/collapse/add) is irrelevant to this verdict; only the shape of the named *production* mutation matters. "Removal" here means the production mutation collapses to deleting code, not the test-deletion action the skill is recommending.
- `DROP (mutation unverifiable from source)` — the named identifier or line doesn't exist in the source, or the claimed branch isn't reachable (e.g., a test-only short-circuit like `os.environ.get("BENCHFLOW_TEST")` bypasses the mutation target).

No new findings. Do not re-apply rule-(c) escape clauses — that's Pass
2's job. Keep the prompt short since the job is mechanical. Expected
DROP rate: <15%; a run that DROPs most of a bucket indicates the
auditor is misreading the verdict labels (observed failure: treating
every delete-bucket entry as removal).

### Synthesize

Present the skeptic-adjusted list, grouped:

- **Collapse** — parametrize candidates that survived challenge
- **Delete** — clear bloat (mock-echo, template-echo, unreachable guards)
- **Loosen** — over-specified assertions to soften, not delete
- **Add** — coverage gaps tied to rule A–D
- **Rejected** — findings the skeptic killed, with one-line reason (builds trust; avoids re-surfacing)
- **Deferred** — deep bloat needing a shared fixture refactor (surface, don't tackle inline)

### Ask for approval

Present the list. Do NOT edit. Wait for "collapse 1-3", "delete all",
"add A+B skip C", or similar. Apply only what's approved.

## Anti-patterns

- **Don't delete without naming the mutation.** Bypassing the acceptance test turns this skill into vibes-based pruning.
- **Don't add coverage without pointing at a specific branch.** "More tests for X" without a named branch is cargo-cult.
- **Don't collapse parallel tests whose failure messages carry distinct debugging value.** See rule 5(c). Benchflow's `test_registry_invariants.py` and `test_invariants.py` are canonical keep-separate surfaces.
- **Don't promote a unit test to `@pytest.mark.live` by adding `patch(...)` calls.** `live` means real seams and real API calls; mocking the seam defeats the mark.
- **Don't batch test deletions with production-code changes.** Separate commit — a later bisect can then distinguish "behavior changed" from "guard test removed."
- **Don't skip Pass 2 or Pass 3.** Pass 2 catches Pass 1's trigger-happiness; Pass 3 catches Pass 2's defensive-keep bias. Both failure modes are observed; both passes earn their keep.
- **Don't grow the rule list.** 10 delete rules + 4 add rules is the observed surface. New patterns → new examples under existing rules, not new rules.
- **Don't touch CI-gate tests casually.** Anything under `test_invariants.py`, `test_registry_invariants.py`, `test_reexport.py`, `test_sdk_lockdown.py` is a refactor guard by design — default keep, only delete if a specific rule 1–10 trip survives Pass 2 skepticism.
