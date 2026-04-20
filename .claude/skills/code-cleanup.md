---
name: code-cleanup
description: Two-pass subagent sweep for trivial/small refactoring wins — find candidates, then verify each before recommending
user_invocable: true
---

# Code Cleanup Skill

Find quick refactoring wins (dead code, duplication, stale comments,
latent bugs hiding as style) via parallel subagents, then **verify each
finding in a second pass before presenting**. Most value comes from the
verification pass: single-pass sweeps routinely produce plausible-but-wrong
suggestions, and a polluted punch list is worse than none.

**Do not auto-fix.** Report verified findings; let the user approve edits.

**Every fix must not grow the file.** A cleanup that adds net lines has
failed the intent — dead code deletion, dedup via shared helper, trimming
verbose comments, and header additions should all come out flat or
negative in LOC. Header additions (Cat 7) are the only cleanup that adds
lines; they are capped at one docstring line + optional section markers.
If a proposed fix would net-increase a file, drop it or reframe as a
refactor request for the user, not a cleanup.

## Commands

- `/code-cleanup` — whole `src/benchflow/`, split across 3–4 parallel agents
- `/code-cleanup <path>` — single file or subtree (e.g. `/code-cleanup src/benchflow/agents/`)
- `/code-cleanup --recent` — limit to files changed in the last ~20 commits

## In-scope categories

Only these. If a candidate doesn't fit one of these, drop it.

1. **Dead code** — unused public symbols (no external or in-package importers), orphan modules, no-op functions, unreachable branches. Use `ty check src/` + `git grep` before flagging — `ty` catches some but not all.
2. **Duplicated logic** — same snippet in ≥2 sites with an obvious shared extraction. Benchflow has natural dedup surfaces around env var loading, subprocess shelling, provider config resolution.
3. **Stale comments / docstrings** — comment contradicts the code, docstring mentions renamed parameters or removed behavior, `TODO`/`FIXME` for work that shipped.
4. **Over-defensive checks** — validation for cases the type system or earlier guard already rules out. `if x is None` after `x: str` with no reassignment; `try/except` around code that can't raise; `isinstance(x, T)` in a body where `x: T`.
5. **Dependency hygiene** — imports with no matching `pyproject.toml` entry (transitive-by-accident), imports pulled in only for types that can move under `if TYPE_CHECKING:`, conditional imports for platforms we don't support.
6. **Latent bugs disguised as style** — un-awaited coroutines, swallowed exceptions (`except Exception: pass`), mutable default arguments, shadowed builtins (`list`, `id`, `type`), `is` / `is not` on strings or numbers, off-by-one in slicing, fire-and-forget `asyncio.create_task` without storing a reference, blocking calls inside `async def`, opening files without `with`.
7. **Orientation headers** — every non-test `src/benchflow/**/*.py` module gets a one-line top-of-file docstring stating its responsibility (≤ ~110 chars, concrete domain language, no restating the filename). Modules >400 LOC with ≥3 loosely-related symbol groups also get section markers (`# ── <section> ──`). Serves both agents cold-reading a file *and* humans scanning the flat `src/benchflow/` layout — the filename tells you the domain, the docstring tells you the boundary (e.g. `_sandbox.py` is 800 LOC mixing user-setup / path-lockdown / verifier-hardening). Co-located beats a central architecture doc because it drifts less. This is the one exception to CLAUDE.md's "write no comments" — module-level role is architectural, not line-level. Skip modules that already have a meaningful one-line docstring.
8. **Verbose comments / docstrings** — multi-paragraph blocks that restate the *what* of the next statement, or `Args:` / `Returns:` sections whose lines add nothing beyond the typed signature. Trim to one line or delete — never expand. **Strong bias to keep:** any comment encoding rationale, invariants, workarounds, cross-module intent, or non-obvious "why" is load-bearing even if long. When in doubt, leave it. Orientation headers (Cat 7) are exempt.

## Out of scope (drop without mentioning)

- Renames, reorganizations, new abstractions — not quick wins (that's `/arch-audit` territory)
- "Add error handling for X" suggestions — violates CLAUDE.md guidance
  ("Don't add error handling, fallbacks, or validation for scenarios
  that can't happen")
- Style nits without behavioral benefit (`ruff format` owns formatting; don't duplicate)
- Public API changes — anything in `src/benchflow/__init__.py` re-exports is a contract; changes are out of scope for this skill
- Anything estimated medium+ effort — only trivial/small qualify
- Test-file cleanups — use `/test-review` instead, different rules apply

## Execution

### Pass 1 — discovery (run twice, union)

Discovery is unstable run-to-run: two passes over the same code with the
same prompts routinely surface disjoint findings. A single sweep misses
roughly half the real wins. So run discovery **twice** and union the
results before verification — cheaper and more complete than over-slicing
one run.

Each discovery run spawns 2–4 `Explore` subagents in parallel, each
covering a disjoint slice of the scope. Suggested split for a whole-repo
`/code-cleanup`:

- `src/benchflow/sdk.py` + `src/benchflow/job.py` + `src/benchflow/_acp_run.py` + `src/benchflow/_scoring.py`
- `src/benchflow/_sandbox.py` + `src/benchflow/_env_setup.py` + `src/benchflow/_agent_setup.py` + `src/benchflow/_agent_env.py` + `src/benchflow/_credentials.py`
- `src/benchflow/agents/` + `src/benchflow/acp/` + `src/benchflow/cli/`
- everything else (`environments.py`, `metrics.py`, `models.py`, `process.py`, `tasks.py`, `task_download.py`, `_trajectory.py`, `skills.py`, `viewer.py`)

Each agent prompt includes:

- Exact file list for that slice
- The in-scope categories verbatim
- The out-of-scope list verbatim
- "Rank by value/effort; drop anything medium+; cite file:line; under 400 words"
- "Each finding must include a verbatim code excerpt (≤3 lines) — Pass 2 verifies against this, not your paraphrase"
- "Every proposed fix must keep the file's LOC flat or shrink it. Cat 7 headers are the one allowed exception (one docstring line + optional section markers)."
- "Don't re-flag `ruff format` / `ruff check` concerns — CI already owns those."

Each returns a ranked list with `file:line | excerpt | one-line change |
effort (trivial/small)`.

Union the two runs' findings (dedup by file:line + category), then hand
the combined list to Pass 2.

### Pass 2 — verification

Spawn one `Explore` subagent per discovery agent's output. Prompt:

> "Verify each claim against the actual code at <repo>. For each, quote
> the relevant code (file:line), run `Grep` / `ty check src/` where
> needed, and return a verdict: **real / false positive / nuanced**.
> Include one-sentence justification and an updated effort estimate.
> Explicitly check: (i) for Cat 1 (dead code), does the symbol have zero
> in-package importers AND zero external consumers via
> `src/benchflow/__init__.py` re-exports? (ii) for Cat 4 (over-defensive),
> is the 'unreachable' branch truly unreachable after considering
> optional kwargs, `None` defaults, and `Unpack`-style typed dicts?
> (iii) for Cat 6 (latent bug), can you name the concrete failure mode?"

This pass typically kills 30–50% of Pass 1 findings. That is the point —
do not skip it.

### Synthesize

Present only **verified-real** findings, grouped by bucket:

- **Latent bugs** — do now, separate commit (category 6)
- **Trivial cleanups** — batch into one "janitor" commit (categories 1, 3, 8)
- **Small cleanups** — second commit if user wants (categories 2, 4, 5, 7)
- **Needs judgment** — Pass 2 verdicts of "nuanced"; surface the caveat
  and let the user decide
- **Rejected** — list briefly with one-line reason, so user sees what was
  checked and excluded (builds trust; avoids re-surfacing next sweep)

### Ask for approval

Present the list. Do NOT edit. Wait for "fix 1-4", "all", "skip 3",
"just the latent bugs", or similar. Apply only what's approved. After
edits, run `ruff format`, `ruff check`, `ty check src/`, and
`.venv/bin/python -m pytest tests/` (fast unit subset) to re-verify only
what you touched.

## Anti-patterns

- **Don't skip Pass 2.** The verification pass is the skill's whole value.
  Without it this is just a prompt.
- **Don't suggest out-of-scope items.** A tempting rename or reorg is not
  a quick win — it's scope creep. Note it as a rejected finding or drop
  it, don't smuggle it in. Send renames/splits to `/arch-audit`.
- **Don't batch bug-fixes with cleanups.** Latent bugs go in their own
  commit so `git bisect` stays useful.
- **Don't auto-fix**, even for "obvious" trivia. User approves.
- **Don't grow the category list.** If a finding doesn't fit the
  in-scope categories, it's not a quick win by definition.
- **Don't expand comments or docstrings.** Trimming verbose comments
  (Cat 8) is in scope; *adding* explanatory comments to clarify
  confusing code is not — that's a refactor request, surface it to the
  user instead.
- **Don't touch `src/benchflow/__init__.py` re-exports.** Public API
  surface — changes go through a normal PR review, not a cleanup sweep.

## Example output

```
Code-cleanup sweep — 5 verified real (2 latent bugs, 3 trivial), 5 rejected

Latent bugs (separate commit):
- src/benchflow/_acp_run.py:87 — asyncio.create_task(...) result discarded; coroutine may GC mid-run
- src/benchflow/agents/pi_acp_launcher.py:142 — except Exception: pass swallows auth failures silently

Trivial cleanups (janitor commit):
- src/benchflow/metrics.py:220-232 — compute_pass_rate has zero importers outside the module; inline at sole caller
- src/benchflow/_env_setup.py:41 + src/benchflow/_agent_setup.py:58 — _resolve_home_path byte-identical; extract shared util
- src/benchflow/viewer.py:1 — no module docstring; add one-line header (Cat 7)
- src/benchflow/_sandbox.py:1 — 800 LOC, 3 symbol groups; add `# ── User setup ──` / `# ── Path lockdown ──` / `# ── Verifier hardening ──` markers (Cat 7)

Rejected (checked, not real):
- src/benchflow/sdk.py:512 — isinstance check is for dict-vs-TypedDict at the YAML boundary; necessary
- src/benchflow/job.py:74 — TODO is a planning marker matched by an active ticket; leave
- src/benchflow/process.py:198 — try/except around subprocess is required; CalledProcessError is real
- src/benchflow/models.py:22 — RunResult fields look redundant but one feeds the SDK, other feeds viewer
- src/benchflow/tasks.py:31 — Optional[...] | None is defensive but loader feeds untyped YAML; keep

Reply with which to fix, or "all".
```
