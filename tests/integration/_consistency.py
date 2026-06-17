"""Config <-> manifest consistency for the integration suite (ENG-265 slice 3).

One source of truth: the per-agent configs/*.yaml must agree with the release
manifest (suites/release.yaml) on agent membership, model pins, and task set.
``config_drifts`` is pure (manifest + configs -> list of human-readable drifts)
so it is unit-testable and reusable by load_suite + a CI consistency step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def config_drifts(
    manifest: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
    *,
    task_set: str = "skillsbench_release_subset",
) -> list[str]:
    """Return drift descriptions between per-agent configs and the manifest.

    Empty list == consistent. Checks, per config: the agent is in
    ``axes.agents.credentialed``; the model matches ``axes.models.default[agent]``
    when that agent is pinned; and ``include`` matches the referenced task set's
    ``include``.
    """
    axes = manifest.get("axes", {})
    credentialed = set(axes.get("agents", {}).get("credentialed", []))
    default_models = axes.get("models", {}).get("default", {})
    task_include = axes.get("task_sets", {}).get(task_set, {}).get("include")

    drifts: list[str] = []
    for name, cfg in configs.items():
        agent = cfg.get("agent", name)
        if agent not in credentialed:
            drifts.append(
                f"config {name!r}: agent {agent!r} is not in axes.agents.credentialed"
            )
        pinned = default_models.get(agent)
        actual_model = cfg.get("model")
        if pinned is not None and actual_model != pinned:
            drifts.append(
                f"config {name!r}: model {actual_model!r} != axes.models.default "
                f"pin {pinned!r}"
            )
        cfg_include = cfg.get("include")
        if (
            task_include is not None
            and cfg_include is not None
            and list(cfg_include) != list(task_include)
        ):
            drifts.append(
                f"config {name!r}: include {list(cfg_include)} != task set "
                f"{task_set!r} include {list(task_include)}"
            )
    return drifts
