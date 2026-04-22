# Sandbox Setup Shared Tool Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate per-trial copying of heavyweight tool installations during `setup_sandbox_user()` while keeping sandboxed agents able to read configs, credentials, and installed binaries.

**Architecture:** Treat tool installations as shared container state, not per-user home state. Keep `setup_sandbox_user()` responsible only for creating the non-root user, preparing small home-scoped config/credential directories, and granting workspace ownership. Add a narrow compatibility path for root-home installs that BenchFlow cannot relocate itself.

**Tech Stack:** Python 3.12, pytest, Harbor sandbox env exec, BenchFlow agent registry / sandbox setup.

---

## Recommendation

Pick **alternative 2: shared install paths** as the primary direction, with a **small compatibility fallback from alternative 1** for existing images that already install tools under `/root`.

Why this is the best fit:

- `setup_sandbox_user()` currently copies `/root/.nvm`, `/root/.local/bin`, and every derived home dir. That is the direct cause of the timeout.
- BenchFlow already installs some helpers into shared locations like `/usr/local/bin`, which matches the right ownership boundary for binaries.
- Pre-creating the user in the Dockerfile (alternative 3) is useful for image authors, but it is not a sufficient product-level fix because BenchFlow must still support arbitrary user-provided images.
- Pure symlinking of `/root` home directories (alternative 1) would reduce I/O, but it keeps the wrong model: heavyweight tool installs still live in a root-only home and `setup_sandbox_user()` still has to know about them.

So the implementation target should be:

- **Primary path:** stop relying on root-home installs for agent binaries and shared tooling.
- **Compatibility path:** if an existing image already has a heavy tool dir in `/root` and BenchFlow needs it, link the minimum necessary path instead of copying it.

## File Map

- Modify: `src/benchflow/_sandbox.py`
  - Remove heavyweight recursive copies from `setup_sandbox_user()`.
  - Add a minimal compatibility helper for sharing root-owned tool paths without recursive copy.
- Modify: `src/benchflow/agents/registry.py`
  - Clarify the contract for `home_dirs` / `get_sandbox_home_dirs()` so it covers user config, not shared tool installs.
- Modify: `src/benchflow/_agent_setup.py`
  - Ensure the install path assumptions match the new sandbox contract.
- Modify: `tests/test_sandbox.py`
  - Add contract tests for which home dirs are copied versus treated as shared tooling.
- Create or modify: `tests/test_sandbox_setup.py`
  - Add targeted tests for the generated `setup_sandbox_user()` command.
- Modify: `docs/task-authoring.md`
  - Document preferred shared install locations for benchmark/task images.
- Modify: `docs/api-reference.md`
  - Update the sandbox setup description to match the new behavior.

### Task 1: Lock the new sandbox contract in tests

**Files:**

- Modify: `tests/test_sandbox.py`
- Create: `tests/test_sandbox_setup.py`
- Modify: `src/benchflow/_sandbox.py`

- [x] Add a focused async test file for `setup_sandbox_user()` that mocks `env.exec`, calls `setup_sandbox_user(env, "agent", "/app")`, and asserts the command no longer contains recursive copies of `/root/.nvm` or `/root/.local/bin`.
- [x] In the same test file, assert the command still creates the user, prepares the target home directory, and `chown`s the workspace.
- [x] Add a compatibility test that asserts the command uses a link-or-shared-path strategy for heavyweight root-owned tool dirs rather than `cp -a`.
- [x] Run: `.venv/bin/python -m pytest tests/test_sandbox.py tests/test_sandbox_setup.py -q`
- [x] Expected: new tests fail on current `main` because the command still contains `cp -a` / `cp -aL` for heavyweight tool dirs.

### Task 2: Narrow `setup_sandbox_user()` to user state only

**Files:**

- Modify: `src/benchflow/_sandbox.py:103-128`
- Test: `tests/test_sandbox_setup.py`

- [x] Refactor `setup_sandbox_user()` so it does only three things:
  - create the sandbox user if missing,
  - prepare only small home-scoped directories BenchFlow actually needs for config/auth,
  - grant ownership to the workspace.
- [x] Remove the unconditional copy of `/root/.nvm`.
- [x] Remove the special-case copy of `/root/.local/bin`.
- [x] Keep copying only the small dirs derived from `get_sandbox_home_dirs()` that represent config/auth state, not tool installations.
- [x] Add a minimal helper or inline shell snippet that, when a legacy image exposes required tool paths only under `/root`, creates a symlink to the required path instead of copying the directory tree.
- [x] Run: `.venv/bin/python -m pytest tests/test_sandbox_setup.py -q`
- [x] Expected: the new setup contract tests pass.

### Task 3: Align registry semantics with the new contract

**Files:**

- Modify: `src/benchflow/agents/registry.py:338-366`
- Modify: `tests/test_sandbox.py`

- [x] Update the `get_sandbox_home_dirs()` docstring to state that the returned dirs are user home config/auth dirs that BenchFlow may need to materialize for the sandbox user.
- [x] Decide whether `.local` should remain in `get_sandbox_home_dirs()`:
  - if BenchFlow still needs `.local/share` or similar user-scoped state, keep it but stop special-casing `.local/bin` in `_sandbox.py`;
  - if BenchFlow only needed `.local` for tool binaries, remove `.local` from the always-include set and update tests accordingly.
- [x] Prefer the more minimal option based on the actual credential/skill paths in the registry.
- [x] Run: `.venv/bin/python -m pytest tests/test_sandbox.py -q`
- [x] Expected: registry contract tests pass with updated semantics.

### Task 4: Verify agent install assumptions still hold

**Files:**

- Modify: `src/benchflow/_agent_setup.py`
- Modify: `src/benchflow/agents/registry.py`
- Test: `tests/test_sandbox_setup.py`

- [x] Review each registry `install_cmd` and confirm BenchFlow-installed binaries resolve from shared paths already on `PATH` for the sandbox user.
- [x] If any BenchFlow-managed install still lands in a root-home path, change that install command to a shared location such as `/usr/local/bin`, `/usr/local/lib`, or another non-home prefix.
- [x] Add or adjust a test that asserts the sandbox user setup no longer depends on home-copying to make BenchFlow-installed agents executable.
- [x] Run targeted tests covering sandbox setup and any registry invariant tests touched by the change.

### Task 5: Document the benchmark image guidance

**Files:**

- Modify: `docs/task-authoring.md`
- Modify: `docs/api-reference.md`

- [x] Add a short section to `docs/task-authoring.md` explaining that benchmark images should install shared tooling into shared prefixes like `/usr/local` or `/opt`, not `/root/.nvm` or `/root/.local/bin`, when the tools must be usable by a sandbox user.
- [x] Document that pre-creating the sandbox user in the Dockerfile is optional optimization, not the primary compatibility mechanism.
- [x] Update `docs/api-reference.md` to describe `setup_sandbox_user()` as lightweight user setup plus workspace ownership, not recursive home cloning.

### Task 6: End-to-end verification

**Files:**

- No code changes required unless verification exposes a gap.

- [x] Run: `.venv/bin/python -m pytest tests/test_sandbox.py tests/test_sandbox_setup.py tests/test_sandbox_hardening.py tests/test_sandbox_verifier_workspace.py -q`
- [x] Run: `.venv/bin/ty check src/`
- [x] If available, run one trial startup path that exercises `Trial.install_agent()` with `sandbox_user="agent"` and capture setup timing before/after.
- [x] Confirm the setup command no longer performs recursive copies of heavyweight tool trees per trial.

## Notes / Risks

- The collaborator comment on the issue asks to try shared install paths first. This plan follows that request as the architectural choice.
- Shared install paths alone cannot retrofit arbitrary existing benchmark images that already baked tools into `/root`. That is why the plan keeps a narrow symlink-based compatibility path instead of a full copy.
- The key review question before implementation is whether `.local` remains a true user-state directory in BenchFlow, or whether it only existed because of previous tool-install assumptions.
