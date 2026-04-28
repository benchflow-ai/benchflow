"""benchflow.contracts — the default home for contract surfaces.

Every file in this subpackage is a semver boundary: declarative dataclasses
(trial_config, job_config, agent_config, task_config, paths),
invariant-enforcing pure functions (scoring.extract_reward,
scoring.classify_error), and shared grammar (retry-category constants).
Changes here may break downstream importers, so diffs to this directory
should get extra review.

This is the *default* home — not the exclusive one. Per the hybrid layout
rule (PLAN_V2_impl §4 + memory feedback_core_dir.md): contract surfaces
that anchor a plug axis live next to their adapters, not here. Known
examples today:

    - benchflow.sandbox._base       — BaseSandbox ABC (sandbox plug axis)
    - benchflow.agents.registry     — AGENTS dict (agents plug axis)

Both are semver-stable by the same rule as this directory. The full list
is the source of truth in code review, not in this docstring — when a new
contract surface is added outside contracts/, add the banner comment in
the file and update the project allowlist.

An import-linter rule forbids any module under ``benchflow.contracts``
from importing periphery (``trial.py``, ``job.py``, ``agents.registry``,
``sandbox`` adapters) — promises can't depend on implementations.
"""
