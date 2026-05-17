"""Task configuration models — internalized from Harbor, aligned to RL terminology.

Terminology (per https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/):
    - Task ($T$): problem specification the agent solves
    - Sandbox: isolated execution environment (replaces Harbor "environment")
    - Rollout: single episode of agent interaction (replaces Harbor "trial")
    - Verifier ($V$): maps completion → reward signal
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
    """Author information for a task package."""

    name: str = Field(..., description="Author name")
    email: str | None = Field(default=None, description="Author email address")


class PackageInfo(BaseModel):
    """Package metadata from the [task] section of task.toml."""

    name: str = Field(
        ...,
        description="Package name in org/name format (e.g., 'benchflow/hello-world')",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the task",
    )
    authors: list[Author] = Field(
        default_factory=list,
        description="List of package authors",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords for search and categorization",
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        if not re.match(ORG_NAME_PATTERN, v) or ".." in v:
            raise ValueError(
                f"Package name must be in 'org/name' format with alphanumeric characters, "
                f"hyphens, underscores, and dots. Cannot start with a dot or contain '..'. Got: {v}"
            )
        return v

    @property
    def org(self) -> str:
        return self.name.split("/")[0]

    @property
    def short_name(self) -> str:
        return self.name.split("/")[1]


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


class VerifierConfig(BaseModel):
    """Verifier ($V$) configuration — maps completion → reward."""

    timeout_sec: float = 600.0
    env: dict[str, str] = Field(default_factory=dict)
    user: str | int | None = Field(
        default=None,
        description="Username or UID to run the verifier as.",
    )


class AgentConfig(BaseModel):
    """Agent harness ($H$) configuration."""

    timeout_sec: float | None = None
    user: str | int | None = Field(
        default=None,
        description="Username or UID to run the agent as.",
    )


class SandboxConfig(BaseModel):
    """Sandbox configuration — the isolated execution environment.

    Replaces Harbor's EnvironmentConfig with RL-aligned naming.
    """

    build_timeout_sec: float = 600.0
    docker_image: str | None = None
    cpus: int = 1
    memory_mb: int = 2048
    storage_mb: int = 10240
    gpus: int = 0
    gpu_types: list[str] | None = Field(
        default=None,
        description="List of acceptable GPU types (e.g., ['H100', 'A100', 'T4']).",
    )
    allow_internet: bool = Field(
        default=True,
        description="Whether to allow internet access in the sandbox.",
    )
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables resolved from host at runtime. "
        "Supports ${VAR} and ${VAR:-default} template syntax.",
    )
    skills_dir: str | None = Field(
        default=None,
        description="Path to skills directory in the sandbox.",
    )

    # Deprecated fields
    memory: str | None = Field(
        default=None,
        deprecated="Use 'memory_mb' instead.",
        exclude=True,
    )
    storage: str | None = Field(
        default=None,
        deprecated="Use 'storage_mb' instead.",
        exclude=True,
    )

    @staticmethod
    def _parse_size_to_mb(size_str: str) -> int:
        size_str = size_str.strip().upper()
        if size_str.endswith("G"):
            return int(float(size_str[:-1]) * 1024)
        elif size_str.endswith("M"):
            return int(float(size_str[:-1]))
        elif size_str.endswith("K"):
            return int(float(size_str[:-1]) / 1024)
        else:
            raise ValueError(
                f"Invalid size format: {size_str}. Expected format like '1G', '512M', etc."
            )

    @model_validator(mode="after")
    def handle_deprecated_fields(self) -> SandboxConfig:
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


class SolutionConfig(BaseModel):
    env: dict[str, str] = Field(default_factory=dict)


class TaskConfig(BaseModel):
    """Full task.toml configuration — the task specification ($T$).

    Maps task.toml sections to BenchFlow's RL-aligned models.
    The ``environment`` key in task.toml is loaded into ``sandbox``
    for internal use, maintaining file-level backward compatibility.
    """

    schema_version: str = "1.1"
    task: PackageInfo | None = Field(
        default=None,
        description="Package information from the [task] section.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    # Stored as 'sandbox' internally, but loaded from 'environment' key in TOML
    sandbox: SandboxConfig = Field(
        default_factory=SandboxConfig,
        alias="environment",
    )
    solution: SolutionConfig = Field(default_factory=SolutionConfig)
    source: str | None = None

    model_config = {"populate_by_name": True}

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
        return toml.dumps(self.model_dump(mode="json", by_alias=True))

    @property
    def environment(self) -> SandboxConfig:
        """Backward-compat alias: task.config.environment → task.config.sandbox."""
        return self.sandbox
