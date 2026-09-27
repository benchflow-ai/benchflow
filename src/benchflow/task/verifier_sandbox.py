"""Which sandbox a separate verifier runs in (Harbor ``environment_mode = "separate"``).

Pure config logic shared by the launch gate (``runtime_capabilities``) and the
rollout (``benchflow.rollout._separate_verifier``), which runs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from benchflow.task.config import SandboxConfig, TaskConfig, VerifierSandboxMode
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
