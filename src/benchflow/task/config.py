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
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

ORG_NAME_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$"
_NETWORK_HOST_LABEL_PATTERN = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class TaskConfigModel(BaseModel):
    """Base model for task schema sections.

    Harbor-compatible config keys should be modeled explicitly. Unknown keys
    are rejected so ``task.md`` and ``task.toml`` do not silently become a
    lossy subset of the upstream task schema.
    """

    model_config = ConfigDict(extra="forbid")


class NetworkMode(StrEnum):
    """Network access policy for task execution."""

    NO_NETWORK = "no-network"
    PUBLIC = "public"
    ALLOWLIST = "allowlist"


class TaskOS(StrEnum):
    """Target operating system for a task container."""

    LINUX = "linux"
    WINDOWS = "windows"


class VerifierEnvironmentMode(StrEnum):
    """Whether the verifier runs in the agent environment or a separate one."""

    SHARED = "shared"
    SEPARATE = "separate"


class MultiStepRewardStrategy(StrEnum):
    """Strategy for deriving one rollout reward from step-level rewards."""

    MEAN = "mean"
    FINAL = "final"


def _validate_allowed_hosts(hosts: list[str] | None) -> list[str] | None:
    if hosts is None:
        return None
    normalized: list[str] = []
    for raw_host in hosts:
        host = raw_host.strip().lower().rstrip(".")
        if not host:
            raise ValueError("allowed_hosts entries must be non-empty hostnames")
        if "://" in host or "/" in host or ":" in host:
            raise ValueError(
                "allowed_hosts entries must be hostnames, not URLs, ports, or paths"
            )
        labels = host.split(".")
        if not all(_NETWORK_HOST_LABEL_PATTERN.match(label) for label in labels):
            raise ValueError(
                "allowed_hosts entries must be valid hostnames containing only "
                "letters, digits, hyphens, and dots"
            )
        normalized.append(host)
    return normalized


def _validate_network_policy_fields(
    network_mode: NetworkMode | None,
    allowed_hosts: list[str] | None,
) -> None:
    if network_mode == NetworkMode.ALLOWLIST and not allowed_hosts:
        raise ValueError("allowed_hosts must be non-empty for network_mode='allowlist'")
    if network_mode != NetworkMode.ALLOWLIST and allowed_hosts:
        raise ValueError("allowed_hosts is only valid for network_mode='allowlist'")


class Author(TaskConfigModel):
    """Author information for a task package."""

    name: str = Field(..., description="Author name")
    email: str | None = Field(default=None, description="Author email address")


class PackageInfo(TaskConfigModel):
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


MCPTransport = Literal["stdio", "sse", "streamable-http"]


class MCPServerConfig(TaskConfigModel):
    """Configuration for an MCP server available to the agent."""

    name: str
    transport: MCPTransport = "sse"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)

    @field_validator("transport", mode="before")
    @classmethod
    def normalize_transport(cls, value: Any) -> Any:
        return "streamable-http" if value == "http" else value

    @model_validator(mode="after")
    def validate_transport_fields(self) -> MCPServerConfig:
        if self.transport in ("sse", "streamable-http") and not self.url:
            raise ValueError(f"'url' is required for transport '{self.transport}'")
        if self.transport == "stdio" and not self.command:
            raise ValueError("'command' is required for transport 'stdio'")
        return self


class JudgeVerifierConfig(TaskConfigModel):
    """The ``[verifier.judge]`` section — config for the LLM-as-judge verifier.

    Used only when ``[verifier].type == "llm-judge"``.
    """

    model: str = Field(
        default="claude-sonnet-4-6",
        description="Judge model identifier. Provider is routed from the prefix "
        "(claude-* / gpt-* / gemini-*).",
    )
    rubric_path: str = Field(
        default="tests/rubric.toml",
        description="Path to the rubric file, relative to the task directory. "
        "Both rubric.toml (native) and rubric.json (Harvey LAB style) are "
        "supported.",
    )
    input_dir: str = Field(
        default="/app",
        description="Directory inside the sandbox holding the agent's "
        "deliverables. Its contents are downloaded and shown to the judge.",
    )
    input_type: Literal["deliverables"] = Field(
        default="deliverables",
        description="What the judge evaluates. Only 'deliverables' (agent "
        "output files) is supported — trajectory judging is not available at "
        "verify time.",
    )
    context: str = Field(
        default="",
        description="Optional extra context passed to the judge prompt "
        "(defaults to the task instruction when empty).",
    )


class MemoryVerifierConfig(TaskConfigModel):
    """The ``[verifier.memory]`` section — hidden Memory-space fixtures."""

    expected_skills: list[str] | None = Field(
        default=None,
        description=(
            "Skill pack names the task expects the agent to add/update/remove. "
            "None means no answer key was supplied; [] means no skill should change."
        ),
    )


class VerifierHardeningConfig(TaskConfigModel):
    """The ``[verifier.hardening]`` section for per-task opt-outs."""

    cleanup_conftests: bool = True


class VerifierConfig(TaskConfigModel):
    """Verifier ($V$) configuration — maps completion → reward.

    ``type`` selects the verification method:

    - ``"test-script"`` (default): run ``tests/test.sh`` in the sandbox and
      parse ``reward.txt`` / ``reward.json``.
    - ``"llm-judge"``: score the agent's deliverables against a human-authored
      rubric using an LLM judge (see :class:`JudgeVerifierConfig`).
    """

    type: Literal["test-script", "llm-judge"] = Field(
        default="test-script",
        description="Verification method.",
    )
    timeout_sec: float = 600.0
    env: dict[str, str] = Field(default_factory=dict)
    user: str | int | None = Field(
        default=None,
        description="Username or UID to run the verifier as.",
    )
    network_mode: NetworkMode | None = Field(
        default=None,
        description="Optional verifier-specific network policy override.",
    )
    allowed_hosts: list[str] | None = Field(
        default=None,
        description="Hostnames reachable when network_mode='allowlist'.",
    )
    environment_mode: VerifierEnvironmentMode | None = Field(
        default=None,
        description=(
            "Whether the verifier runs in the agent environment ('shared') "
            "or a dedicated verifier environment ('separate')."
        ),
    )
    environment: SandboxConfig | None = Field(
        default=None,
        description="Optional separate verifier environment configuration.",
    )
    service: str = Field(
        default="main",
        description=(
            "Compose service the test-script verifier runs in. Defaults to "
            "'main' (the agent container). Multi-container (vulhub-style) "
            "tasks set this to a target/database service so test.sh can "
            "inspect target-side state — RCE markers, DB modifications — "
            "rather than only the agent's workspace. The agent's "
            "anti-tamper hardening only applies to 'main'; deliberately "
            "vulnerable target containers are intentionally not hardened. "
            "See #248."
        ),
    )
    judge: JudgeVerifierConfig = Field(
        default_factory=JudgeVerifierConfig,
        description="LLM-judge configuration (used when type == 'llm-judge').",
    )
    memory: MemoryVerifierConfig = Field(
        default_factory=MemoryVerifierConfig,
        description="Memory-space scoring fixtures.",
    )
    hardening: VerifierHardeningConfig = Field(
        default_factory=VerifierHardeningConfig,
        description="Per-task verifier hardening opt-outs.",
    )
    pytest_plugins: list[str] = Field(
        default_factory=list,
        description=(
            "pytest11 plugin names to allow under PYTEST_DISABLE_PLUGIN_AUTOLOAD=1. "
            "Container-side auto-discovery handles most cases; this is the "
            "explicit fallback for plugins that discovery cannot see "
            "(e.g. ctrf, playwright)."
        ),
    )

    @field_validator("allowed_hosts")
    @classmethod
    def validate_allowed_hosts(
        cls, hosts: list[str] | None
    ) -> list[str] | None:
        return _validate_allowed_hosts(hosts)

    @model_validator(mode="after")
    def validate_verifier_environment(self) -> VerifierConfig:
        _validate_network_policy_fields(self.network_mode, self.allowed_hosts)
        if (
            self.environment_mode == VerifierEnvironmentMode.SHARED
            and self.environment is not None
        ):
            raise ValueError(
                "[verifier].environment_mode='shared' is incompatible with "
                "[verifier.environment]"
            )
        return self


class AgentConfig(TaskConfigModel):
    """Agent harness ($H$) configuration."""

    timeout_sec: float | None = None
    user: str | int | None = Field(
        default=None,
        description="Username or UID to run the agent as.",
    )
    network_mode: NetworkMode | None = Field(
        default=None,
        description="Optional agent-specific network policy override.",
    )
    allowed_hosts: list[str] | None = Field(
        default=None,
        description="Hostnames reachable when network_mode='allowlist'.",
    )

    @field_validator("allowed_hosts")
    @classmethod
    def validate_allowed_hosts(
        cls, hosts: list[str] | None
    ) -> list[str] | None:
        return _validate_allowed_hosts(hosts)

    @model_validator(mode="after")
    def validate_network_policy(self) -> AgentConfig:
        _validate_network_policy_fields(self.network_mode, self.allowed_hosts)
        return self


class HealthcheckConfig(TaskConfigModel):
    """Healthcheck configuration mirroring Docker HEALTHCHECK options."""

    command: str
    interval_sec: float = 5.0
    timeout_sec: float = 30.0
    start_period_sec: float = 0.0
    start_interval_sec: float = 5.0
    retries: int = 3


class TpuSpec(TaskConfigModel):
    """Specification for a TPU slice attached to an environment."""

    type: str = Field(min_length=1)
    topology: str

    @field_validator("topology")
    @classmethod
    def validate_topology(cls, value: str) -> str:
        topology = value.strip()
        if not re.match(r"^[1-9]\d*(x[1-9]\d*)+$", topology):
            raise ValueError(
                f"Invalid TPU topology {value!r}; expected e.g. '2x4' or '2x2x1'"
            )
        return topology

    @property
    def chip_count(self) -> int:
        product = 1
        for axis in self.topology.split("x"):
            product *= int(axis)
        return product


class SandboxConfig(TaskConfigModel):
    """Sandbox configuration — the isolated execution environment.

    Replaces Harbor's EnvironmentConfig with RL-aligned naming.
    """

    network_mode: NetworkMode = Field(
        default=NetworkMode.PUBLIC,
        description="Network access policy for this environment.",
    )
    allowed_hosts: list[str] | None = Field(
        default=None,
        description="Hostnames reachable when network_mode='allowlist'.",
    )
    build_timeout_sec: float = 600.0
    docker_image: str | None = Field(
        default=None,
        validation_alias=AliasChoices("docker_image", "image"),
        description="Prebuilt image name; legacy SkillsBench tasks may use image.",
    )
    os: TaskOS = Field(
        default=TaskOS.LINUX,
        description="Target operating system for the task container.",
    )
    cpus: int = 1
    memory_mb: int = 2048
    storage_mb: int = 10240
    gpus: int = 0
    gpu_types: list[str] | None = Field(
        default=None,
        description="List of acceptable GPU types (e.g., ['H100', 'A100', 'T4']).",
    )
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables resolved from host at runtime. "
        "Supports ${VAR} and ${VAR:-default} template syntax.",
    )
    tpu: TpuSpec | None = Field(
        default=None,
        description="Optional TPU accelerator request.",
    )
    skills_dir: str | None = Field(
        default=None,
        description="Path to skills directory in the sandbox.",
    )
    healthcheck: HealthcheckConfig | None = Field(
        default=None,
        description="Healthcheck to run after environment start.",
    )
    workdir: str | None = Field(
        default=None,
        description="Default working directory for command execution.",
    )
    bugswarm_image_tag: str | None = Field(
        default=None,
        description="BugSwarm image tag used by build-repair benchmark tasks.",
    )

    # Deprecated fields
    allow_internet: bool = Field(
        default=True,
        description="Deprecated compatibility field; use network_mode instead.",
    )
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

    @field_validator("allowed_hosts")
    @classmethod
    def validate_allowed_hosts(
        cls, hosts: list[str] | None
    ) -> list[str] | None:
        return _validate_allowed_hosts(hosts)

    @field_validator("os", mode="before")
    @classmethod
    def normalize_os(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @model_validator(mode="after")
    def handle_deprecated_fields_and_network_policy(self) -> SandboxConfig:
        _validate_network_policy_fields(self.network_mode, self.allowed_hosts)
        memory = self.__dict__.get("memory")
        storage = self.__dict__.get("storage")
        if memory is not None:
            warnings.warn(
                "The 'memory' field is deprecated. Use 'memory_mb' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.memory_mb = self._parse_size_to_mb(memory)
            self.memory = None
        if storage is not None:
            warnings.warn(
                "The 'storage' field is deprecated. Use 'storage_mb' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.storage_mb = self._parse_size_to_mb(storage)
            self.storage = None
        # Reconcile the deprecated allow_internet flag with network_mode.
        # The new field is authoritative: when network_mode was explicitly
        # provided, allow_internet must not silently override it. An explicit
        # contradiction (e.g. network_mode='allowlist' + allow_internet=False)
        # is a hard error rather than a silent downgrade to no-network.
        network_mode_explicit = "network_mode" in self.model_fields_set
        if self.allow_internet is False:
            if network_mode_explicit:
                if self.network_mode != NetworkMode.NO_NETWORK:
                    raise ValueError(
                        "allow_internet=False contradicts the explicit "
                        f"network_mode={self.network_mode.value!r}; "
                        "drop the deprecated allow_internet field and rely on "
                        "network_mode (use network_mode='no-network' to "
                        "disable networking)."
                    )
            else:
                self.network_mode = NetworkMode.NO_NETWORK
        if self.network_mode == NetworkMode.NO_NETWORK:
            self.allow_internet = False
        # Reconciliation must never leave the object in a state that
        # _validate_network_policy_fields itself rejects.
        _validate_network_policy_fields(self.network_mode, self.allowed_hosts)
        return self


class SolutionConfig(TaskConfigModel):
    env: dict[str, str] = Field(default_factory=dict)
    timeout_sec: float | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_inline_env(cls, data: Any) -> Any:
        """Accept legacy ``[solution] FOO = "..."`` env shorthands.

        Some SkillsBench task.toml files predate the stricter ``[solution.env]``
        shape and place provider tokens directly under ``[solution]``. Treat
        env-var-shaped string keys as env entries while keeping arbitrary
        unknown solution keys forbidden.
        """
        if not isinstance(data, dict):
            return data

        env_raw = data.get("env")
        if env_raw is not None and not isinstance(env_raw, dict):
            return data

        known_keys = {"env", "timeout_sec"}
        env = dict(env_raw or {})
        normalized: dict[str, Any] = {}
        for key, value in data.items():
            if key in known_keys:
                normalized[key] = value
            elif isinstance(value, str) and _ENV_VAR_NAME_PATTERN.match(key):
                existing = env.get(key)
                if existing is not None and existing != value:
                    raise ValueError(
                        f"Conflicting values for solution env var {key!r}"
                    )
                env[key] = value
            else:
                normalized[key] = value

        normalized["env"] = env
        return normalized


class ArtifactConfig(TaskConfigModel):
    """Artifact path copied out of a task environment after verification."""

    source: str
    destination: str | None = None
    exclude: list[str] = Field(default_factory=list)


class StepConfig(TaskConfigModel):
    """Harbor-style multi-step task configuration."""

    name: str
    agent: AgentConfig = Field(default_factory=AgentConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    min_reward: float | dict[str, float] | None = Field(default=None)
    healthcheck: HealthcheckConfig | None = None
    artifacts: list[str | ArtifactConfig] = Field(default_factory=list)


class TaskConfig(TaskConfigModel):
    """Full task.toml configuration — the task specification ($T$).

    Maps task.toml sections to BenchFlow's RL-aligned models.
    The ``environment`` key in task.toml is loaded into ``sandbox``
    for internal use, maintaining file-level backward compatibility.
    """

    schema_version: str = "1.3"
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
    solution: SolutionConfig = Field(
        default_factory=SolutionConfig,
        serialization_alias="oracle",
        validation_alias=AliasChoices("oracle", "solution"),
    )
    source: str | None = None
    reward: dict[str, Any] = Field(
        default_factory=dict,
        description="Legacy task-level reward expression metadata.",
    )
    multi_step_reward_strategy: MultiStepRewardStrategy | None = Field(
        default=None,
        description="How to derive one rollout reward from per-step verifier results.",
    )
    steps: list[StepConfig] | None = None
    artifacts: list[str | ArtifactConfig] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    _SUPPORTED_SCHEMA_MAJORS: ClassVar[frozenset[int]] = frozenset({1})

    @model_validator(mode="before")
    @classmethod
    def handle_version_rename(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "oracle" in data and "solution" in data:
                raise ValueError(
                    "Task config cannot contain both 'oracle' and legacy "
                    "'solution'; use 'oracle' for native tasks or 'solution' "
                    "only when importing legacy Harbor/Pier tasks."
                )
            if "version" in data:
                data.setdefault("schema_version", data.pop("version"))
        return data

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        """Reject unknown/unparseable schema majors; stay permissive on minor.

        The schema version is ``MAJOR.MINOR``; only majors the loader knows how
        to interpret are accepted. Unknown majors or non-numeric versions are a
        hard error rather than a value silently carried through.
        """
        major_part = value.split(".", 1)[0]
        try:
            major = int(major_part)
        except ValueError:
            raise ValueError(
                f"schema_version {value!r} is not a valid MAJOR.MINOR version"
            ) from None
        if major not in cls._SUPPORTED_SCHEMA_MAJORS:
            supported = ", ".join(
                str(m) for m in sorted(cls._SUPPORTED_SCHEMA_MAJORS)
            )
            raise ValueError(
                f"Unsupported schema_version major {major} (from {value!r}); "
                f"supported major version(s): {supported}"
            )
        return value

    @classmethod
    def model_validate_toml(cls, toml_data: str) -> TaskConfig:
        toml_dict = tomllib.loads(toml_data)
        return cls.model_validate(toml_dict)

    @property
    def expected_skills(self) -> list[str] | None:
        """Hidden Memory-space fixture from ``[verifier.memory]``.

        ``None`` means the task did not supply an answer key, so the
        Memory-space scorer can only grade activity. An empty list is a real
        fixture: the task expects no skill changes.
        """
        expected = self.verifier.memory.expected_skills
        return None if expected is None else list(expected)

    def model_dump_toml(self) -> str:
        import tomli_w

        public = self.model_dump(
            mode="json",
            by_alias=True,
            exclude={"verifier": {"memory": {"expected_skills"}}},
            exclude_none=True,
        )
        memory = (public.get("verifier") or {}).get("memory") or {}
        if isinstance(memory, dict) and not memory:
            public["verifier"].pop("memory", None)
        return tomli_w.dumps(public)

    @property
    def environment(self) -> SandboxConfig:
        """Backward-compat alias: task.config.environment → task.config.sandbox."""
        return self.sandbox
