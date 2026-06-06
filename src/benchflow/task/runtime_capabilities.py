"""Fail-closed runtime capability validation for parsed task fields.

Parsed Harbor-compatible fields that the selected sandbox backend cannot honor
must be rejected before sandbox construction. See ``docs/task-standard.md``
Runtime Capability Matrix (P1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow.task.config import (
    NetworkMode,
    TaskConfig,
    TaskOS,
    VerifierEnvironmentMode,
)

if TYPE_CHECKING:
    from benchflow.task.task import Task

_GATED_SANDBOX_TYPES = frozenset({"docker", "daytona", "modal"})
_METADATA_ONLY_RUNTIME_VALUES = frozenset({"metadata-only", "metadata_only"})


@dataclass(frozen=True)
class UnsupportedTaskFeature:
    """One parsed task field the selected sandbox cannot execute."""

    path: str
    reason: str
    sandbox_type: str


class UnsupportedTaskRuntimeError(ValueError):
    """Raised when a task uses parsed fields unsupported on the selected sandbox."""

    def __init__(
        self,
        issues: list[UnsupportedTaskFeature],
        *,
        task_path: Path | str,
    ) -> None:
        self.issues = issues
        self.task_path = Path(task_path)
        lines = [
            f"{issue.path}: {issue.reason} (sandbox={issue.sandbox_type})"
            for issue in issues
        ]
        message = (
            f"Task {self.task_path} uses runtime fields unsupported on "
            f"{issues[0].sandbox_type!r}: " + "; ".join(lines)
        )
        super().__init__(message)


def _task_has_compose(task_path: Path) -> bool:
    return (task_path / "environment" / "docker-compose.yaml").exists()


def _benchflow_section_runtime_mode(
    benchflow: dict[str, Any],
    section: str,
) -> str | None:
    value = benchflow.get(section)
    if not isinstance(value, dict):
        return None
    runtime = value.get("runtime")
    return runtime if isinstance(runtime, str) else None


def _is_metadata_only_runtime(benchflow: dict[str, Any], section: str) -> bool:
    """Return whether *section* user/nudge semantics are explicitly metadata-only."""
    umbrella = benchflow.get("user_runtime")
    if isinstance(umbrella, str) and umbrella in _METADATA_ONLY_RUNTIME_VALUES:
        return True
    mode = _benchflow_section_runtime_mode(benchflow, section)
    return mode in _METADATA_ONLY_RUNTIME_VALUES if mode is not None else False


def _append_user_semantics_issues(
    issues: list[UnsupportedTaskFeature],
    *,
    task: Task,
    sandbox_type: str,
) -> None:
    document = getattr(task, "document", None)
    if document is None:
        return

    benchflow = document.benchflow
    from benchflow.task.user_loop import compile_document_user_loop

    compiled_user_loop = compile_document_user_loop(task)
    user_loop_executable = (
        compiled_user_loop is not None and compiled_user_loop.executable
    )

    if (
        document.user
        and not _is_metadata_only_runtime(benchflow, "user")
        and not user_loop_executable
    ):
        issues.append(
            UnsupportedTaskFeature(
                path="user",
                reason=(
                    "document-declared simulated-user frontmatter is parsed but "
                    "not compiled into a rollout user loop yet"
                ),
                sandbox_type=sandbox_type,
            )
        )

    nudges = benchflow.get("nudges")
    nudges_supported = (
        isinstance(nudges, dict)
        and nudges.get("mode") == "simulated-user"
        and user_loop_executable
    )
    if (
        isinstance(nudges, dict)
        and not _is_metadata_only_runtime(benchflow, "nudges")
        and not nudges_supported
    ):
        issues.append(
            UnsupportedTaskFeature(
                path="benchflow.nudges",
                reason=(
                    "document-declared nudge policy is parsed but not executed "
                    "by the selected sandbox yet"
                ),
                sandbox_type=sandbox_type,
            )
        )

    if (
        document.user_persona
        and not _is_metadata_only_runtime(benchflow, "user_persona")
        and not user_loop_executable
    ):
        issues.append(
            UnsupportedTaskFeature(
                path="prompt.user-persona",
                reason=(
                    "document-declared user persona is parsed but not compiled "
                    "into a rollout user loop yet"
                ),
                sandbox_type=sandbox_type,
            )
        )


def _append_allowlist_issue(
    issues: list[UnsupportedTaskFeature],
    *,
    sandbox_type: str,
    prefix: str,
    network_mode: NetworkMode | None,
    allowed_hosts: list[str] | None,
) -> None:
    if network_mode == NetworkMode.ALLOWLIST:
        issues.append(
            UnsupportedTaskFeature(
                path=f"{prefix}.network_mode",
                reason=(
                    "network_mode='allowlist' is parsed but egress allowlists "
                    "are not enforced by the selected sandbox yet"
                ),
                sandbox_type=sandbox_type,
            )
        )
    elif allowed_hosts:
        issues.append(
            UnsupportedTaskFeature(
                path=f"{prefix}.allowed_hosts",
                reason=(
                    "allowed_hosts is parsed but egress allowlists are not "
                    "enforced by the selected sandbox yet"
                ),
                sandbox_type=sandbox_type,
            )
        )


def validate_task_runtime_support(
    task: Task,
    sandbox_type: str,
    task_path: Path | str,
) -> list[UnsupportedTaskFeature]:
    """Return unsupported parsed fields for *sandbox_type* (empty when supported)."""
    if sandbox_type not in _GATED_SANDBOX_TYPES:
        return []

    config = getattr(task, "config", None)
    if not isinstance(config, TaskConfig):
        return []

    root = Path(task_path).resolve()
    env = config.environment
    issues: list[UnsupportedTaskFeature] = []

    if config.steps:
        issues.append(
            UnsupportedTaskFeature(
                path="steps",
                reason=(
                    "Harbor multi-step tasks are parsed but step execution is "
                    "not implemented for this sandbox yet"
                ),
                sandbox_type=sandbox_type,
            )
        )
        for step in config.steps:
            if step.artifacts:
                issues.append(
                    UnsupportedTaskFeature(
                        path=f"steps[{step.name!r}].artifacts",
                        reason=(
                            "step-level artifact collection is parsed but not "
                            "implemented for this sandbox yet"
                        ),
                        sandbox_type=sandbox_type,
                    )
                )
            if step.healthcheck is not None:
                issues.append(
                    UnsupportedTaskFeature(
                        path=f"steps[{step.name!r}].healthcheck",
                        reason=(
                            "step-level healthchecks are parsed but not "
                            "implemented for this sandbox yet"
                        ),
                        sandbox_type=sandbox_type,
                    )
                )

    if config.artifacts:
        issues.append(
            UnsupportedTaskFeature(
                path="artifacts",
                reason=(
                    "root-level artifact collection is parsed but not "
                    "implemented for this sandbox yet"
                ),
                sandbox_type=sandbox_type,
            )
        )

    _append_allowlist_issue(
        issues,
        sandbox_type=sandbox_type,
        prefix="environment",
        network_mode=env.network_mode,
        allowed_hosts=env.allowed_hosts,
    )
    _append_allowlist_issue(
        issues,
        sandbox_type=sandbox_type,
        prefix="agent",
        network_mode=config.agent.network_mode,
        allowed_hosts=config.agent.allowed_hosts,
    )
    _append_allowlist_issue(
        issues,
        sandbox_type=sandbox_type,
        prefix="verifier",
        network_mode=config.verifier.network_mode,
        allowed_hosts=config.verifier.allowed_hosts,
    )

    verifier = config.verifier
    if (
        verifier.environment_mode == VerifierEnvironmentMode.SEPARATE
        or verifier.environment is not None
    ):
        issues.append(
            UnsupportedTaskFeature(
                path="verifier.environment"
                if verifier.environment is not None
                else "verifier.environment_mode",
                reason=(
                    "separate verifier environments are parsed but not "
                    "materialized for this sandbox yet"
                ),
                sandbox_type=sandbox_type,
            )
        )

    if env.os == TaskOS.WINDOWS:
        issues.append(
            UnsupportedTaskFeature(
                path="environment.os",
                reason="Windows task containers are parsed but not runnable yet",
                sandbox_type=sandbox_type,
            )
        )

    if env.tpu is not None:
        issues.append(
            UnsupportedTaskFeature(
                path="environment.tpu",
                reason="TPU resource requests are parsed but not runnable yet",
                sandbox_type=sandbox_type,
            )
        )

    if env.healthcheck is not None:
        issues.append(
            UnsupportedTaskFeature(
                path="environment.healthcheck",
                reason=(
                    "environment healthchecks are parsed but not executed "
                    "during sandbox startup yet"
                ),
                sandbox_type=sandbox_type,
            )
        )

    if env.workdir is not None and sandbox_type not in ("docker", "modal"):
        issues.append(
            UnsupportedTaskFeature(
                path="environment.workdir",
                reason=(
                    "environment.workdir is parsed but default working "
                    "directories are not applied by the materializer yet"
                ),
                sandbox_type=sandbox_type,
            )
        )

    if verifier.service != "main" and not _task_has_compose(root):
        issues.append(
            UnsupportedTaskFeature(
                path="verifier.service",
                reason=(
                    f"verifier.service={verifier.service!r} requires a "
                    "multi-container docker-compose.yaml task, which is not "
                    "present under environment/"
                ),
                sandbox_type=sandbox_type,
            )
        )

    _append_user_semantics_issues(issues, task=task, sandbox_type=sandbox_type)

    return issues


def ensure_task_runtime_support(
    task: Task,
    sandbox_type: str,
    task_path: Path | str,
) -> None:
    """Raise :class:`UnsupportedTaskRuntimeError` when validation finds blockers."""
    issues = validate_task_runtime_support(task, sandbox_type, task_path)
    if issues:
        raise UnsupportedTaskRuntimeError(issues, task_path=task_path)


__all__ = [
    "UnsupportedTaskFeature",
    "UnsupportedTaskRuntimeError",
    "ensure_task_runtime_support",
    "validate_task_runtime_support",
]
