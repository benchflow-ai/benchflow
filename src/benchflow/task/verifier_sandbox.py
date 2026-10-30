"""Which sandbox a separate verifier runs in (Harbor ``environment_mode = "separate"``).

Pure config logic shared by the launch gate (``runtime_capabilities``) and the
rollout (``benchflow.rollout._separate_verifier``), which runs it.
"""

from __future__ import annotations

import posixpath
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from benchflow.task.config import (
    NetworkMode,
    SandboxConfig,
    TaskConfig,
    VerifierSandboxMode,
)
from benchflow.task.paths import TaskPaths

SEPARATE_VERIFIER_SANDBOXES = frozenset({"docker", "daytona"})

ImageSource = Literal[
    "verifier.sandbox.docker_image",
    "sandbox.docker_image",
    "tests/Dockerfile",
    "environment/Dockerfile",
]


class SeparateVerifierError(RuntimeError):
    """The verifier sandbox could not be given the agent's outputs."""


# What ``--separate-verifier-include`` may not name: pseudo filesystems, and
# roots so broad that carrying them would carry the agent's system state back
# into the verifier sandbox.
_INCLUDE_FORBIDDEN_TREES = ("/proc", "/sys", "/dev", "/run")
_INCLUDE_FORBIDDEN_ROOTS = frozenset(
    {
        "/",
        "/bin",
        "/boot",
        "/etc",
        "/home",
        "/lib",
        "/lib32",
        "/lib64",
        "/libx32",
        "/sbin",
        "/usr",
        "/var",
    }
)


def normalize_include_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Validated, de-duplicated ``--separate-verifier-include`` paths.

    Each names a directory or file outside the workspace that a separate
    verifier sandbox receives too, at the same path: a task whose agent writes
    graded files to ``/root`` while its workspace is ``/app`` needs ``/root``.
    """
    seen: dict[str, None] = {}
    for raw in paths:
        value = str(raw).strip()
        if not value.startswith("/"):
            raise ValueError(
                f"--separate-verifier-include needs an absolute path, got {raw!r}"
            )
        if ".." in PurePosixPath(value).parts:
            raise ValueError(
                f"--separate-verifier-include path {raw!r} must not contain '..'"
            )
        path = "/" + posixpath.normpath(value).lstrip("/")
        if path in _INCLUDE_FORBIDDEN_ROOTS or any(
            path == tree or path.startswith(tree + "/")
            for tree in _INCLUDE_FORBIDDEN_TREES
        ):
            raise ValueError(
                f"--separate-verifier-include cannot carry {path}: name the "
                "directory the agent writes its outputs to, such as /root"
            )
        seen.setdefault(path, None)
    return tuple(seen)


def separate_verifier_requested(config: TaskConfig | None) -> bool:
    """Whether ``config`` asks for a separate verifier sandbox.

    Harbor: an explicit ``environment_mode = "separate"``, or a
    ``[verifier.environment]`` table with the mode unset. Partial stand-ins
    (no verifier section) are shared-mode.
    """
    verifier = getattr(config, "verifier", None)
    mode = getattr(verifier, "sandbox_mode", None)
    if mode is not None:
        return mode == VerifierSandboxMode.SEPARATE
    return getattr(verifier, "sandbox", None) is not None


def force_separate_verifier(config: TaskConfig) -> TaskConfig:
    """A copy of ``config`` whose verifier runs in a separate sandbox.

    Used by ``--separate-verifier``: the task's files (and so its digest) are
    unchanged, only this run's resolved config is.
    """
    forced = config.model_copy(deep=True)
    forced.verifier.sandbox_mode = VerifierSandboxMode.SEPARATE
    return forced


@dataclass(frozen=True)
class VerifierImage:
    """Where the verifier sandbox comes from."""

    source: ImageSource
    sandbox: SandboxConfig
    # Directory whose contents become the image build context; None for a
    # prebuilt ``docker_image``.
    context_dir: Path | None


def _verifier_sandbox_config(config: TaskConfig) -> SandboxConfig:
    declared = config.verifier.sandbox
    base = (declared or config.sandbox).model_copy(deep=True)
    # Agent-side provisioning never runs in the verifier sandbox.
    base.setup_commands = []
    base.healthcheck = None
    base.skills_dir = None
    base.mcp_servers = []
    return base


def plan_verifier_image(config: TaskConfig, task_dir: Path) -> VerifierImage:
    """The verifier sandbox's image, Harbor's precedence first.

    1. ``[verifier.sandbox].docker_image``;
    2. with no ``[verifier.sandbox]``, a fresh copy of ``[sandbox]``
       (Harbor ``[environment]``): its ``docker_image`` when set;
    3. otherwise the image built from ``tests/Dockerfile`` (Harbor builds the
       separate verifier from ``tests/``);
    4. BenchFlow extension: with neither a ``[verifier.sandbox]`` nor a
       ``tests/Dockerfile``, a fresh sandbox from the task's own
       ``environment/Dockerfile``. Nothing the agent changed survives in it.

    Raises :class:`SeparateVerifierError` when none applies.
    """
    sandbox = _verifier_sandbox_config(config)
    paths = TaskPaths(task_dir)
    declared = config.verifier.sandbox is not None
    if sandbox.docker_image:
        source: ImageSource = (
            "verifier.sandbox.docker_image" if declared else "sandbox.docker_image"
        )
        return VerifierImage(source=source, sandbox=sandbox, context_dir=None)
    if (paths.tests_dir / "Dockerfile").is_file():
        return VerifierImage(
            source="tests/Dockerfile", sandbox=sandbox, context_dir=paths.tests_dir
        )
    if not declared and (paths.environment_dir / "Dockerfile").is_file():
        return VerifierImage(
            source="environment/Dockerfile",
            sandbox=sandbox,
            context_dir=paths.environment_dir,
        )
    raise SeparateVerifierError(
        "no verifier image: add tests/Dockerfile or a docker_image to "
        "[verifier.sandbox] (Harbor [verifier.environment])"
    )


def verifier_image_issue(config: TaskConfig, task_dir: Path) -> str | None:
    """Why no verifier image can be chosen, for the launch gate."""
    try:
        plan_verifier_image(config, task_dir)
    except SeparateVerifierError as exc:
        return str(exc)
    return None


def separate_verifier_issues(config: TaskConfig, *, sandbox: str) -> list[str]:
    """Why ``config``'s separate verifier cannot run on ``sandbox`` (config only)."""
    issues: list[str] = []
    if sandbox not in SEPARATE_VERIFIER_SANDBOXES:
        names = " and ".join(sorted(SEPARATE_VERIFIER_SANDBOXES))
        issues.append(f"separate verifier sandboxes are implemented for {names}")
    if config.verifier.type != "test-script":
        issues.append("separate verifier sandboxes run test-script verifiers only")
    if config.verifier.service != "main":
        issues.append(
            "a separate verifier sandbox has no compose services; "
            "verifier.service must be 'main'"
        )
    verifier_sandbox = config.verifier.sandbox
    if verifier_sandbox is not None and verifier_sandbox.network_mode in (
        NetworkMode.ALLOWLIST,
        NetworkMode.DENYLIST,
    ):
        # Egress filtering binds the agent's uid through the proxy the agent
        # connection starts; the verifier sandbox has neither.
        issues.append(
            f"verifier.sandbox.network_mode={verifier_sandbox.network_mode.value!r} "
            "is not enforced in a separate verifier sandbox"
        )
    return issues


def apply_separate_verifier_flag(
    config: TaskConfig, *, sandbox: str, task_dir: Path
) -> TaskConfig:
    """``--separate-verifier`` for one task: its config with a separate verifier.

    A task that already asks for one is returned as is. Raises ``ValueError``
    listing why the separate verifier cannot run on ``sandbox``, before any
    sandbox exists.
    """
    if separate_verifier_requested(config):
        return config
    issues = separate_verifier_launch_issues(
        config, sandbox=sandbox, task_dir=Path(task_dir)
    )
    if issues:
        raise ValueError(
            "--separate-verifier runs the verifier in a separate verifier "
            "sandbox, which this task cannot use here:\n- " + "\n- ".join(issues)
        )
    return force_separate_verifier(config)


def separate_verifier_launch_issues(
    config: TaskConfig, *, sandbox: str, task_dir: Path
) -> list[str]:
    """Every reason a separate verifier cannot run here, image included."""
    issues = separate_verifier_issues(config, sandbox=sandbox)
    problem = verifier_image_issue(config, task_dir)
    if problem is not None:
        issues.append(problem)
    return issues
