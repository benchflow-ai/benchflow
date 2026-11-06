"""Which sandbox a separate verifier runs in (task.toml ``environment_mode = "separate"``).

Pure config logic shared by the launch gate (``runtime_capabilities``) and the
rollout (``benchflow.rollout._separate_verifier``), which runs it.
"""

from __future__ import annotations

import fnmatch
import posixpath
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from benchflow._utils.sandbox_providers import SEPARATE_VERIFIER_PROVIDERS
from benchflow.task.config import (
    NetworkMode,
    SandboxConfig,
    TaskConfig,
    VerifierSandboxMode,
)
from benchflow.task.paths import TaskPaths

SEPARATE_VERIFIER_SANDBOXES = SEPARATE_VERIFIER_PROVIDERS
# How refusals name them (remote-docker is the docker backend on another host).
_SEPARATE_VERIFIER_BACKENDS = "daytona, modal and docker (local or remote-docker)"

#: The run-level switch (``--verifier-sandbox``, ``verifier_sandbox:``):
#: ``task`` leaves every task's verifier where the task declares it,
#: ``separate`` moves every task's verifier into the separate verifier
#: sandbox. There is no ``shared`` value: a run can never weaken a task that
#: declares a separate verifier (the run overlay refuses verifier keys for the
#: same reason, ``_utils/config_override.py``).
VERIFIER_SANDBOX_SETTINGS: tuple[str, ...] = ("task", "separate")


def normalize_verifier_sandbox(value: object) -> str:
    """``task`` or ``separate``; None means ``task``. Anything else is refused."""
    if value is None:
        return "task"
    text = str(value).strip().lower()
    if text in VERIFIER_SANDBOX_SETTINGS:
        return text
    if text == "shared":
        raise ValueError(
            "verifier_sandbox has no 'shared' value: a run can only move "
            "verifiers into the separate verifier sandbox, never out of it; "
            "use task (each task's own choice) or separate"
        )
    raise ValueError(f"verifier_sandbox must be task or separate, got {value!r}")


def effective_verifier_sandbox(setting: object, separate_verifier: object) -> str:
    """The run's one isolation setting: ``--separate-verifier`` is ``separate``.

    Every reader (plan validation, the rollout's resolver, the records) takes
    this, so the retained flag and ``--verifier-sandbox separate`` are one
    setting wherever it is set (b127 PY8-MAIN-27).
    """
    return (
        "separate" if separate_verifier is True else normalize_verifier_sandbox(setting)
    )


#: Paths in the verifier sandbox that belong to the verifier, never to what
#: the agent's run transfers there. ``tests/`` is uploaded over ``/tests``
#: (or ``/verifier``) without clearing it and the plugin-guard sweep skips
#: both, ``/logs/verifier`` and ``/logs/agent`` hold the verifier's outputs
#: and the published trajectory, and the last two are hardening's own scratch
#: files (``sandbox/lockdown.py``: the verifier's ``/_benchflow_verifier_*``
#: and the plugin guard's ``/_benchflow_guard_*``). A workspace or declared
#: artifact that equals, lies under or contains one of them is refused: at
#: launch when the task or run declares it, at transfer time otherwise.
# PT's protected runtime contract reserves this tree before its route merges.
# Keep the leaf module independent of the not-yet-integrated runtime package.
VERIFIER_OWNED_PATHS: tuple[str, ...] = (
    "/tests",
    "/verifier",
    "/logs/verifier",
    "/logs/agent",
    "/_benchflow_verifier_*",
    "/_benchflow_guard_*",
    "/opt/benchflow",
)


def verifier_owned_overlap(path: str) -> str | None:
    """The verifier-owned path ``path`` equals, lies under or contains.

    ``path`` is an absolute sandbox path; ``..`` and repeated slashes are
    folded first, so ``/app/../tests`` is ``/tests``. Returns None for a
    relative path (it has no place in the sandbox until a workspace is known).
    """
    if not path.startswith("/"):
        return None
    # One leading slash: POSIX keeps "//x" apart, Linux does not.
    normal = PurePosixPath(posixpath.normpath("/" + path.lstrip("/")))
    for owned in VERIFIER_OWNED_PATHS:
        pattern = PurePosixPath(owned)
        if "*" in owned:
            # A pattern names top-level entries: "/" contains them, and a
            # path lies under one when its first component matches.
            if normal == PurePosixPath("/") or (
                len(normal.parts) > 1
                and fnmatch.fnmatchcase(normal.parts[1], pattern.parts[1])
            ):
                return owned
            continue
        if normal == pattern or pattern in normal.parents or normal in pattern.parents:
            return owned
    return None


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
# into the verifier sandbox. The trees the verifier sandbox gets from the
# harness (tests, logs, oracle and solution mounts, BenchFlow's own installs)
# are refused too: an agent copy laid over them would decide what the verifier
# runs or reports (PY8 main sync).
_INCLUDE_FORBIDDEN_TREES = (
    "/proc",
    "/sys",
    "/dev",
    "/run",
    "/tests",
    "/logs",
    "/oracle",
    "/solution",
    "/verifier",
    "/opt/benchflow",
    "/.benchflow",
)
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
        # A refused tree inside the path counts too: /opt carries /opt/benchflow.
        if path in _INCLUDE_FORBIDDEN_ROOTS or any(
            path == tree or path.startswith(tree + "/") or tree.startswith(path + "/")
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

    An explicit ``environment_mode = "separate"``, or a
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
    """The verifier sandbox's image, in this order.

    1. ``[verifier.sandbox].docker_image``;
    2. with no ``[verifier.sandbox]``, a fresh copy of ``[sandbox]``
       (task.toml ``[environment]``): its ``docker_image`` when set;
    3. otherwise the image built from ``tests/Dockerfile`` (the task.toml
       layout builds the separate verifier from ``tests/``);
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
        "[verifier.sandbox] ([verifier.environment] in task.toml)"
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
        issues.append(
            f"separate verifier sandboxes are implemented for {_SEPARATE_VERIFIER_BACKENDS}"
        )
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


def verifier_context_services(plan: VerifierImage) -> list[str]:
    """Compose services besides ``main`` that the verifier image would start.

    The verifier sandbox is built from ``plan.context_dir`` copied as its
    ``environment/``; a compose file there makes Docker start every service it
    names. With the ``environment/Dockerfile`` fallback that is the agent's
    own topology, so its side services would run beside the verifier, which
    the separate mode says has none. A file that does not parse, or pulls in
    other files (``include``), counts as unknown services.
    """
    if plan.context_dir is None:
        return []
    from benchflow.sandbox._compose import compose_definition_path

    compose = compose_definition_path(plan.context_dir)
    if compose is None:
        return []
    import yaml

    try:
        data = yaml.safe_load(compose.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return [f"<{compose.name} does not parse>"]
    if not isinstance(data, dict):
        return [f"<{compose.name} does not parse>"]
    services = data.get("services") or {}
    if not isinstance(services, dict):
        return [f"<{compose.name} does not parse>"]
    others = sorted(str(name) for name in services if str(name) != "main")
    if data.get("include"):
        others.append(f"<{compose.name} include>")
    return others


def verifier_services_issue(config: TaskConfig, task_dir: Path) -> str | None:
    """Why the planned verifier image would start compose services, if it would."""
    try:
        plan = plan_verifier_image(config, task_dir)
    except SeparateVerifierError:
        return None
    services = verifier_context_services(plan)
    if not services:
        return None
    return (
        f"the verifier image comes from {plan.source}, whose compose file also "
        f"starts {', '.join(services)}; a separate verifier sandbox has no "
        "compose services: add tests/Dockerfile or a docker_image to "
        "[verifier.sandbox]"
    )
