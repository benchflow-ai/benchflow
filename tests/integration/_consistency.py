"""Config <-> manifest consistency for the integration suite (ENG-265 slice 3).

One source of truth: the per-agent configs/*.yaml must agree with the release
manifest (suites/release.yaml) on agent membership, model pins, and task set.
``config_drifts`` is pure (manifest + configs -> list of human-readable drifts)
so it is unit-testable and reusable by load_suite + a CI consistency step.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_agent_configs(configs_dir: Path) -> dict[str, dict[str, Any]]:
    """Load per-agent config YAMLs under *configs_dir*, keyed by file stem.

    Returns ``{}`` when the directory is absent so callers can treat "no configs
    to check" explicitly rather than silently. Non-mapping YAML files are
    skipped (a malformed config is not this function's concern).
    """
    if not configs_dir.is_dir():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(configs_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            out[path.stem] = data
    return out


def config_drifts(
    manifest: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
    *,
    task_set: str = "skillsbench_release_subset",
) -> list[str]:
    """Return drift descriptions between per-agent configs and the manifest.

    Empty list == consistent. The membership invariant is checked in BOTH
    directions so "single source of truth" actually holds:

    * every config's agent is in ``axes.agents.credentialed`` (or explicitly in
      ``axes.agents.experimental`` — e.g. ``mimo``, which runs the Xiaomi MiMo
      path and is deliberately outside the credentialed release set), and
    * every credentialed agent has a config (a credentialed agent with no
      config is the symmetric twin of the mimo drift and must not pass silently).

    Per config it also checks: the model matches ``axes.models.default[agent]``
    when that agent is pinned; ``include`` matches the task set's ``include`` (as
    a set — ordering is irrelevant); and the config ``source`` {repo, path, ref}
    matches the task set's ``source`` (same task names from a different repo/ref
    are different tasks). A reference to an undefined task set is itself a drift,
    so a typo never passes vacuously. Axis values default to empty (a manifest
    with ``experimental:`` left blank must not crash the loader).
    """
    axes = manifest.get("axes") or {}
    agents_axis = axes.get("agents") or {}
    credentialed = set(agents_axis.get("credentialed") or [])
    experimental = set(agents_axis.get("experimental") or [])
    default_models = (axes.get("models") or {}).get("default") or {}
    task_sets = axes.get("task_sets") or {}
    task_spec = task_sets.get(task_set) or {}
    task_include = task_spec.get("include")
    task_source = task_spec.get("source")

    drifts: list[str] = []
    if task_set not in task_sets:
        drifts.append(
            f"manifest defines no task set {task_set!r} to validate config "
            "includes against"
        )

    config_agents = {cfg.get("agent", name) for name, cfg in configs.items()}
    if configs:
        # Reverse direction: a credentialed agent must ship a config.
        for agent in sorted(credentialed - config_agents):
            drifts.append(
                f"credentialed agent {agent!r} has no config in configs/*.yaml"
            )

    for name, cfg in configs.items():
        agent = cfg.get("agent", name)
        if agent not in credentialed and agent not in experimental:
            drifts.append(
                f"config {name!r}: agent {agent!r} is neither in "
                "axes.agents.credentialed nor axes.agents.experimental"
            )
        pinned = default_models.get(agent)
        actual_model = cfg.get("model")
        if pinned is not None and actual_model != pinned:
            drifts.append(
                f"config {name!r}: model {actual_model!r} != axes.models.default "
                f"pin {pinned!r}"
            )
        cfg_include = cfg.get("include")
        if task_include is not None and cfg_include is not None:
            want = set(task_include)
            have = set(cfg_include)
            if want != have:
                drifts.append(
                    f"config {name!r}: include differs from task set "
                    f"{task_set!r}: missing={sorted(want - have)} "
                    f"extra={sorted(have - want)}"
                )
        cfg_source = cfg.get("source")
        if task_source is not None and cfg_source is not None:
            mismatched = [
                k
                for k in ("repo", "path", "ref")
                if cfg_source.get(k) != task_source.get(k)
            ]
            if mismatched:
                drifts.append(
                    f"config {name!r}: source differs from task set "
                    f"{task_set!r} on {mismatched} "
                    f"(config={ {k: cfg_source.get(k) for k in mismatched} })"
                )
    return drifts
