# Fix: connect_as() ignores cfg.agent_env

Issue: https://github.com/EYH0602/benchflow/issues/2

## Problem

`Trial.connect_as()` at `src/benchflow/trial.py:641` calls:

```python
agent_env=resolve_agent_env(role.agent, role.model, role.env or None)
```

`role.env` is `{}` for scenes created via `Scene.single()` (line 115 creates `Role` with default empty dict). Because `{}` is truthy, `role.env or None` evaluates to `{}`, so `resolve_agent_env` receives an empty dict and loses all env vars from `cfg.agent_env`.

The legacy path at line 292 correctly uses `cfg.agent_env`:

```python
self._agent_env = resolve_agent_env(cfg.primary_agent, cfg.primary_model, cfg.agent_env)
```

But `connect_as()` never reads `self._agent_env` — it re-resolves from scratch with an empty dict.

### Impact

User-supplied providers (e.g. `vllm` with `base_url=""`) get `BENCHFLOW_PROVIDER_BASE_URL=""` instead of the value from the YAML config. Providers with hardcoded `base_url` (e.g. `zai`) are unaffected.

## Fix

### Change 1 — `src/benchflow/trial.py:641`

Merge `cfg.agent_env` as base, overlay `role.env` on top (role-specific wins):

```python
# Before
agent_env=resolve_agent_env(role.agent, role.model, role.env or None),

# After
agent_env=resolve_agent_env(
    role.agent, role.model,
    {**(cfg.agent_env or {}), **(role.env or {})},
),
```

One-line semantic change. Role-level vars override config-level vars.

### Change 2 — Regression test

Add a test that:
1. Creates a `TrialConfig` with `agent_env={"BENCHFLOW_PROVIDER_BASE_URL": "http://localhost:8080/v1"}` and a scene via `Scene.single()`
2. Mocks `connect_acp` and verifies `resolve_agent_env` receives the merged env
3. Verifies role-specific `env` overrides config-level `agent_env`
4. Verifies non-overlapping keys from both dicts are all present in the merge
5. Verifies `cfg.agent_env=None` + empty `role.env` does not crash

### No change to `Scene.single()`

Pushing `agent_env` into `Role` at construction time would conflate config-level and role-level env, breaking multi-agent scenes where different roles share config-level env but have different role-specific overrides.

## Tasks

- [ ] Fix `connect_as()` env merging in `trial.py:641`
- [ ] Add regression test for env propagation
- [ ] Run test suite (`pytest tests/` + `ty check src/`)

## Risk

Low. Standard dict unpacking, one-line change. Role-level > config-level precedence matches the specificity expectation. Legacy path (line 292) unchanged.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 0 issues, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **VERDICT:** ENG CLEARED — ready to implement.
