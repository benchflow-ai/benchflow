"""Native BenchFlow task config models."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow.sandboxes import SandboxSpec


@dataclass(frozen=True)
class AgentTaskConfig:
    timeout_sec: int = 300


@dataclass(frozen=True)
class VerifierTaskConfig:
    timeout_sec: int = 120
    env: dict[str, str] = field(default_factory=dict)
    user: str = "root"
    pytest_plugins: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EnvironmentTaskConfig:
    docker_image: str | None = None
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int | None = None
    allow_internet: bool = True
    env: dict[str, str] = field(default_factory=dict)

    def to_sandbox_spec(self, provider: str = "docker") -> SandboxSpec:
        """Convert task environment requirements into a sandbox request."""

        return SandboxSpec(
            provider=provider,
            cpus=self.cpus,
            memory_mb=self.memory_mb,
            storage_mb=self.storage_mb,
            gpus=self.gpus,
            allow_internet=self.allow_internet,
            env=dict(self.env),
        )


@dataclass(frozen=True)
class TaskConfig:
    raw: dict[str, Any]
    agent: AgentTaskConfig = field(default_factory=AgentTaskConfig)
    verifier: VerifierTaskConfig = field(default_factory=VerifierTaskConfig)
    environment: EnvironmentTaskConfig = field(default_factory=EnvironmentTaskConfig)


@dataclass(frozen=True)
class TaskPaths:
    task_dir: Path

    @property
    def instruction(self) -> Path:
        return self.task_dir / "instruction.md"

    @property
    def environment_dir(self) -> Path:
        return self.task_dir / "environment"

    @property
    def dockerfile(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / "tests"

    @property
    def solution_dir(self) -> Path:
        return self.task_dir / "solution"


@dataclass(frozen=True)
class Task:
    task_dir: Path
    config: TaskConfig
    paths: TaskPaths

    @classmethod
    def from_dir(cls, task_dir: str | Path) -> Task:
        path = Path(task_dir)
        return cls(
            task_dir=path,
            config=load_task_config(path / "task.toml"),
            paths=TaskPaths(path),
        )


def load_task_config(path: Path) -> TaskConfig:
    """Load a BenchFlow/Harbor-compatible task.toml into native dataclasses."""

    with path.open("rb") as f:
        raw = tomllib.load(f)

    agent_raw = raw.get("agent", {})
    verifier_raw = raw.get("verifier", {})
    env_raw = raw.get("environment", {})

    return TaskConfig(
        raw=raw,
        agent=AgentTaskConfig(timeout_sec=int(agent_raw.get("timeout_sec", 300))),
        verifier=VerifierTaskConfig(
            timeout_sec=int(verifier_raw.get("timeout_sec", 120)),
            env=dict(verifier_raw.get("env", {})),
            user=str(verifier_raw.get("user", "root")),
            pytest_plugins=list(verifier_raw.get("pytest_plugins", [])),
        ),
        environment=EnvironmentTaskConfig(
            docker_image=env_raw.get("docker_image"),
            cpus=_optional_int(env_raw.get("cpus")),
            memory_mb=_optional_int(env_raw.get("memory_mb", env_raw.get("memory"))),
            storage_mb=_optional_int(env_raw.get("storage_mb", env_raw.get("storage"))),
            gpus=_optional_int(env_raw.get("gpus")),
            allow_internet=bool(env_raw.get("allow_internet", True)),
            env=dict(env_raw.get("env", {})),
        ),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
