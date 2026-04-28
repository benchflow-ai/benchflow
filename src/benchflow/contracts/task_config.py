"""Task config data shapes.

CONTRACT SURFACE — semver-stable. Changes here break every downstream task
author (task.toml schema) and every importer. Prefer extending in periphery
(``benchflow.tasks``, ``benchflow.task``) unless the shape itself must change.

Declarative pydantic models + enums that describe a task.toml on disk.
The loader (``Task`` / ``TaskPaths``) lives in periphery (``benchflow.task``)
because it touches the filesystem; the shapes here are pure.
"""

from __future__ import annotations

import re
import tomllib
import warnings
from typing import Any

import toml
from pydantic import BaseModel, Field, field_validator, model_validator

ORG_NAME_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$"


class Author(BaseModel):
    """Author information for a package or dataset."""

    name: str = Field(..., description="Author name")
    email: str | None = Field(default=None, description="Author email address")


class PackageInfo(BaseModel):
    """Package metadata for the ``[task]`` section of task.toml."""

    name: str = Field(
        ..., description="Package name in org/name format (e.g., 'harbor/hello-world')"
    )
    description: str = Field(default="", description="Human-readable description")
    authors: list[Author] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        if not re.match(ORG_NAME_PATTERN, v) or ".." in v:
            raise ValueError(
                f"Package name must be in 'org/name' format with alphanumeric "
                f"characters, hyphens, underscores, and dots. Cannot start with a "
                f"dot or contain '..'. Got: {v}"
            )
        return v

    @property
    def org(self) -> str:
        return self.name.split("/")[0]

    @property
    def short_name(self) -> str:
        return self.name.split("/")[1]


class VerifierConfig(BaseModel):
    timeout_sec: float = 600.0
    env: dict[str, str] = Field(default_factory=dict)
    user: str | int | None = Field(
        default=None,
        description="Username or UID to run the verifier as. None uses the environment's default USER.",
    )
    harden: bool = Field(
        default=True,
        description=(
            "When True (default), sandbox.verifier_harden.harden_before_verify "
            "runs its full sequence (kill sandbox-user procs, wipe /logs/verifier, "
            "purge symlinks/__pycache__, chown workspace to root, scrub injected "
            "conftest/.pth files, assemble trusted env). "
            "Set to False when the task verifier legitimately needs paths, envs, "
            "or workspace files that hardening would strip — verified tasks only, "
            "since this reduces isolation between the agent and verifier."
        ),
    )


class SolutionConfig(BaseModel):
    env: dict[str, str] = Field(default_factory=dict)


class TaskAgentSection(BaseModel):
    """The ``[agent]`` section of task.toml."""

    timeout_sec: float | None = None
    user: str | int | None = Field(
        default=None,
        description="Username or UID to run the agent as. None uses the environment's default USER.",
    )


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server available to the agent."""

    name: str
    transport: str = "sse"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_transport_fields(self) -> MCPServerConfig:
        if self.transport in ("sse", "streamable-http") and not self.url:
            raise ValueError(f"'url' is required for transport '{self.transport}'")
        if self.transport == "stdio" and not self.command:
            raise ValueError("'command' is required for transport 'stdio'")
        return self


class EnvironmentConfig(BaseModel):
    build_timeout_sec: float = 600.0
    docker_image: str | None = None
    cpus: int = 1
    memory_mb: int = 2048
    storage_mb: int = 10240
    gpus: int = 0
    gpu_types: list[str] | None = Field(
        default=None,
        description="List of acceptable GPU types. None means any GPU type is acceptable.",
    )
    allow_internet: bool = Field(
        default=True, description="Whether to allow internet access in the environment."
    )
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables required for the task and resolved from the host at runtime.",
    )
    skills_dir: str | None = Field(
        default=None,
        description="Path to skills directory in the environment.",
    )
    memory: str | None = Field(
        default=None,
        deprecated="Use 'memory_mb' instead. This field will be removed in a future version.",
        exclude=True,
    )
    storage: str | None = Field(
        default=None,
        deprecated="Use 'storage_mb' instead. This field will be removed in a future version.",
        exclude=True,
    )

    @staticmethod
    def _parse_size_to_mb(size_str: str) -> int:
        size_str = size_str.strip().upper()
        if size_str.endswith("G"):
            return int(float(size_str[:-1]) * 1024)
        if size_str.endswith("M"):
            return int(float(size_str[:-1]))
        if size_str.endswith("K"):
            return int(float(size_str[:-1]) / 1024)
        raise ValueError(
            f"Invalid size format: {size_str}. Expected format like '1G', '512M'."
        )

    @model_validator(mode="after")
    def handle_deprecated_fields(self) -> EnvironmentConfig:
        if self.memory is not None:
            warnings.warn(
                "The 'memory' field is deprecated. Use 'memory_mb' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.memory_mb = self._parse_size_to_mb(self.memory)
            self.memory = None

        if self.storage is not None:
            warnings.warn(
                "The 'storage' field is deprecated. Use 'storage_mb' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.storage_mb = self._parse_size_to_mb(self.storage)
            self.storage = None

        return self


class TaskConfig(BaseModel):
    schema_version: str = "1.1"
    task: PackageInfo | None = Field(
        default=None,
        description="Package information for the task, parsed from the [task] section of task.toml.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    agent: TaskAgentSection = Field(default_factory=TaskAgentSection)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    solution: SolutionConfig = Field(default_factory=SolutionConfig)
    source: str | None = None

    @model_validator(mode="before")
    @classmethod
    def handle_version_rename(cls, data: Any) -> Any:
        if isinstance(data, dict) and "version" in data:
            data.setdefault("schema_version", data.pop("version"))
        return data

    @classmethod
    def model_validate_toml(cls, toml_data: str) -> TaskConfig:
        toml_dict = tomllib.loads(toml_data)
        return cls.model_validate(toml_dict)

    def model_dump_toml(self) -> str:
        return toml.dumps(self.model_dump(mode="json"))


__all__ = [
    "ORG_NAME_PATTERN",
    "Author",
    "EnvironmentConfig",
    "MCPServerConfig",
    "PackageInfo",
    "SolutionConfig",
    "TaskAgentSection",
    "TaskConfig",
    "VerifierConfig",
]
