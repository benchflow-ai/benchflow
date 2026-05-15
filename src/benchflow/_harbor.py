"""Internal Harbor compatibility adapter.

BenchFlow v0.4 is moving to native task, path, sandbox, and verifier
interfaces. Until each callsite is migrated, all remaining Harbor runtime
touchpoints should pass through this module instead of importing Harbor
directly from orchestration code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harbor.models.task.task import Task as HarborTask
from harbor.models.trial.paths import TrialPaths as HarborTrialPaths
from harbor.utils.env import resolve_env_vars as harbor_resolve_env_vars
from harbor.verifier.verifier import Verifier as HarborVerifier


def make_task(task_path: str | Path) -> HarborTask:
    """Create the current Harbor-backed task object."""

    return HarborTask(Path(task_path))


def make_trial_paths(trial_dir: str | Path) -> HarborTrialPaths:
    """Create the current Harbor-backed trial path object."""

    return HarborTrialPaths(Path(trial_dir))


def resolve_env_vars(env_config: Any) -> dict[str, str]:
    """Resolve Harbor-style env var declarations."""

    return harbor_resolve_env_vars(env_config)


def make_verifier(task: Any, trial_paths: Any, environment: Any) -> HarborVerifier:
    """Create the current Harbor verifier."""

    return HarborVerifier(task=task, trial_paths=trial_paths, environment=environment)


def docker_environment_env_vars_class() -> Any:
    """Return Harbor's Docker env-var helper class."""

    from harbor.environments.docker.docker import DockerEnvironmentEnvVars

    return DockerEnvironmentEnvVars


def docker_environment_class() -> Any:
    """Return Harbor's Docker environment class."""

    from harbor.environments.docker.docker import DockerEnvironment

    return DockerEnvironment


def daytona_environment_class() -> Any:
    """Return Harbor's Daytona environment class."""

    from harbor.environments.daytona import DaytonaEnvironment

    return DaytonaEnvironment


def modal_environment_class() -> Any:
    """Return Harbor's Modal environment class."""

    from harbor.environments.modal import ModalEnvironment

    return ModalEnvironment


def modal_environment_paths_class() -> Any:
    """Return Harbor's environment-path constants used by Modal setup."""

    from harbor.models.trial.paths import EnvironmentPaths

    return EnvironmentPaths
