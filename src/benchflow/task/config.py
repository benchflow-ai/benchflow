"""Task configuration models, in RL terminology.

Terminology (per https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/):
    - Task ($T$): problem specification the agent solves
    - Sandbox: isolated execution environment (task.toml spells it "environment")
    - Rollout: single episode of agent interaction
    - Verifier ($V$): maps completion → reward signal
"""

from __future__ import annotations

import ipaddress
import re
import tomllib
import warnings
from enum import StrEnum
from typing import Any, ClassVar, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from benchflow.rewards.validation import validate_declared_reward_range
from benchflow.sandbox._recovery_submission import validate_contract
from benchflow.task import keep_services

#: The task metadata key that lets the agent reach the cloud metadata endpoints
#: under an open network (BenchFlow refuses them by default; see
#: benchflow.sandbox.lockdown.CLOUD_METADATA_V4). Read from ``metadata`` in
#: task.md front-matter and task.toml alike, where the Rust ``bench`` reads it;
#: ``allow_cloud_metadata = true`` in ``[environment]`` is the same opt-out.
CLOUD_METADATA_OPT_OUT_KEY = "x-allow-cloud-metadata"
# A task with no agent.timeout_sec (and a caller with no timeout) gives the
# agent no time limit of its own: the rollout's whole deadline bounds it
# (rollout/_deadline.uncapped_agent_budget_sec), as in 0.7.11.
# The deadline counts the agent's share of it as this many seconds, as 0.7.11's
# did. Before, the rollout read the missing timeout as 0, which the sandbox
# exec takes as "time out at once" (DF-p1-first-run-2).
AGENT_DEADLINE_SHARE_SEC = 900.0


def effective_agent_timeout_sec(agent: Any) -> float | None:
    """*agent*'s ``timeout_sec``, or None: no time limit of its own.

    The one reader of a task's agent timeout. Takes any object with
    ``timeout_sec`` (an AgentConfig, or a test's stand-in for one); a 0 from
    such a stand-in reads as unset, as 0.7.11 read it.
    """
    value = getattr(agent, "timeout_sec", None)
    return value or None


#: How the record names the task's ``allow_cloud_metadata = true``
#: (``[environment]`` in task.toml, ``sandbox:`` in task.md front-matter).
CLOUD_METADATA_TASK_OPT_OUT = "allow_cloud_metadata = true in the task's environment"
# pydantic's lax bool: true/false, 1/0 and "yes"/"no"-style strings.
_LAX_BOOL: TypeAdapter[bool] = TypeAdapter(bool)

ORG_NAME_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$"
_NETWORK_HOST_LABEL_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")

_ENVIRONMENT_RENAMED_ERROR = (
    "the 'environment' key was renamed to 'sandbox' — rename 'environment:' "
    "to 'sandbox:' (legacy task.toml imports convert '[environment]' "
    "automatically)"
)
_VERIFIER_ENVIRONMENT_RENAMED_ERROR = (
    "the 'verifier.environment' key was renamed to 'verifier.sandbox' — "
    "rename the key (legacy task.toml imports convert "
    "'[verifier.environment]' automatically)"
)
_VERIFIER_ENVIRONMENT_MODE_RENAMED_ERROR = (
    "the 'verifier.environment_mode' key was renamed to "
    "'verifier.sandbox_mode' — rename the key (legacy task.toml imports "
    "convert 'environment_mode' automatically)"
)


def _make_key_rename(old: str, new: str, reason: str):
    """Build a rename closure with the key pair and error phrasing baked in.

    ``reason`` explains, in the both-declared error, why one spelling must
    win (e.g. which one is the legacy task.toml spelling).
    """

    def rename(table: Any, context: str) -> None:
        if not isinstance(table, dict) or old not in table:
            return
        if new in table:
            raise ValueError(
                f"task config declares both '{context}{old}' and "
                f"'{context}{new}' — they are the same section "
                f"({reason}); keep only '{context}{new}'"
            )
        table[new] = table.pop(old)

    return rename


def _convert_sandbox_spelling(
    data: dict[str, Any], *, old: str, new: str, old_mode: str, new_mode: str
) -> dict[str, Any]:
    # True in both directions; the "keep only '{new}'" tail names the
    # direction-appropriate resolution.
    rename_sandbox = _make_key_rename(
        old, new, "'environment' is the legacy task.toml spelling"
    )
    rename_mode = _make_key_rename(
        old_mode, new_mode, "'environment_mode' is the legacy task.toml spelling"
    )

    rename_sandbox(data, "")
    verifier = data.get("verifier")
    rename_sandbox(verifier, "verifier.")
    rename_mode(verifier, "verifier.")
    steps = data.get("steps")
    if isinstance(steps, list):
        for index in range(len(steps)):
            step = steps[index]
            if isinstance(step, dict):
                step_verifier = step.get("verifier")
                rename_sandbox(step_verifier, f"steps[{index}].verifier.")
                rename_mode(step_verifier, f"steps[{index}].verifier.")
    return data


def convert_legacy_environment_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Rename legacy task.toml ``environment`` keys to native ``sandbox``.

    ``task.toml`` is a foreign/legacy format, so its
    ``environment`` spelling is translated here at the import boundary
    instead of being modeled as an alias on the native schema. That keeps
    exactly one spelling — ``sandbox`` — in the native surface (task.md
    frontmatter, Python constructors) while existing and SkillsBench task.toml
    corpora keep loading unchanged. Declaring both spellings in one file is
    a hard error rather than a silent merge.

    Converts the top-level ``environment`` table, ``verifier.environment``,
    ``verifier.environment_mode``, and their ``steps[*].verifier``
    equivalents. Mutates ``data`` in place and returns it; callers pass
    freshly parsed or copied mappings.
    """

    return _convert_sandbox_spelling(
        data,
        old="environment",
        new="sandbox",
        old_mode="environment_mode",
        new_mode="sandbox_mode",
    )


def convert_native_keys_to_legacy_environment(data: dict[str, Any]) -> dict[str, Any]:
    """Rename native ``sandbox`` keys to the task.toml ``environment`` spelling.

    The exact inverse of :func:`convert_legacy_environment_keys`, for
    task.toml emitters (``bench tasks export``): a split layout written
    for a task.toml consumer must spell the sandbox spec
    ``[environment]`` / ``environment_mode``, which such a consumer reads and
    ``[sandbox]`` is invisible to. Mutates ``data`` in place and returns it.
    """

    return _convert_sandbox_spelling(
        data,
        old="sandbox",
        new="environment",
        old_mode="sandbox_mode",
        new_mode="environment_mode",
    )


class TaskConfigModel(BaseModel):
    """Base model for task schema sections.

    task.toml config keys should be modeled explicitly. Unknown keys
    are rejected so ``task.md`` and ``task.toml`` do not silently become a
    lossy subset of the upstream task schema.
    """

    model_config = ConfigDict(extra="forbid")


class NetworkMode(StrEnum):
    """Network access policy for task execution."""

    NO_NETWORK = "no-network"
    PUBLIC = "public"
    ALLOWLIST = "allowlist"
    DENYLIST = "denylist"


class TaskOS(StrEnum):
    """Target operating system for a task container."""

    LINUX = "linux"
    WINDOWS = "windows"


class VerifierSandboxMode(StrEnum):
    """Whether the verifier runs in the agent sandbox or a separate one."""

    SHARED = "shared"
    SEPARATE = "separate"


class MultiStepRewardStrategy(StrEnum):
    """Strategy for deriving one rollout reward from step-level rewards."""

    MEAN = "mean"
    FINAL = "final"


def _validate_hostnames(hosts: list[str] | None, field_name: str) -> list[str] | None:
    """Normalize a hostname list, rejecting URLs, ports, paths, and bad labels."""
    if hosts is None:
        return None
    normalized: list[str] = []
    for raw_host in hosts:
        host = raw_host.strip().lower().rstrip(".")
        if not host:
            raise ValueError(f"{field_name} entries must be non-empty hostnames")
        if "://" in host or "/" in host or ":" in host:
            raise ValueError(
                f"{field_name} entries must be hostnames, not URLs, ports, or paths"
            )
        labels = host.split(".")
        if not all(_NETWORK_HOST_LABEL_PATTERN.match(label) for label in labels):
            raise ValueError(
                f"{field_name} entries must be valid hostnames containing only "
                "letters, digits, hyphens, and dots"
            )
        normalized.append(host)
    return normalized


def _allowed_address_entry(entry: str) -> str | None:
    """Canonical form of an IP or strict CIDR allowlist entry, else None."""
    if "%" in entry:
        return None
    try:
        if "/" in entry:
            return ipaddress.ip_network(entry, strict=True).compressed
        return ipaddress.ip_address(entry).compressed
    except ValueError:
        return None


def _validate_allowed_hosts(hosts: list[str] | None) -> list[str] | None:
    """Accepted allowlist entry kinds: hostname, ``*.`` wildcard, IP, CIDR."""
    if hosts is None:
        return None
    normalized: list[str] = []
    for raw_host in hosts:
        entry = raw_host.strip().lower()
        address = _allowed_address_entry(entry)
        if address is not None:
            normalized.append(address)
            continue
        if entry.startswith("*."):
            suffix = _validate_hostnames([entry[2:]], "allowed_hosts")
            normalized.append("*." + (suffix or [""])[0])
            continue
        if "*" in entry:
            raise ValueError(
                "allowed_hosts wildcards must be a single leading '*.' label"
            )
        if "/" in entry and "://" not in entry:
            raise ValueError(
                f"allowed_hosts entry {raw_host!r} is not a valid CIDR range"
            )
        normalized.extend(_validate_hostnames([raw_host], "allowed_hosts") or ())
    return normalized


def _validate_blocked_hosts(hosts: list[str] | None) -> list[str] | None:
    normalized = _validate_hostnames(hosts, "blocked_hosts")
    for host in normalized or ():
        try:
            ipaddress.ip_address(host)
        except ValueError:
            continue
        raise ValueError("blocked_hosts entries must name a hostname, not an IP")
    return normalized


def _validate_blocked_urls(urls: list[str] | None) -> list[str] | None:
    """Normalize URL prefixes to ``scheme://host/path`` with query and fragment dropped."""
    if urls is None:
        return None
    normalized: dict[str, None] = {}
    for raw_url in urls:
        url = raw_url.strip()
        if not url:
            raise ValueError("blocked_urls entries must be non-empty URLs")
        if "://" not in url:
            url = f"https://{url}"
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise ValueError("blocked_urls entries must use the http or https scheme")
        if "@" in parts.netloc:
            raise ValueError("blocked_urls entries must not contain userinfo")
        if ":" in parts.netloc:
            raise ValueError("blocked_urls entries must not contain a port")
        hosts = _validate_hostnames([parts.netloc], "blocked_urls")
        host = hosts[0] if hosts else ""
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise ValueError("blocked_urls entries must name a hostname, not an IP")
        normalized.setdefault(f"{parts.scheme}://{host}{parts.path.rstrip('/')}")
    return list(normalized)


def _reject_role_denylist(network_mode: NetworkMode | None, role: str) -> None:
    if network_mode == NetworkMode.DENYLIST:
        raise ValueError(
            f"{role}.network_mode='denylist' is not supported; "
            "declare the denylist on sandbox.network_mode"
        )


def _validate_network_policy_fields(
    network_mode: NetworkMode | None,
    allowed_hosts: list[str] | None,
    blocked_urls: list[str] | None = None,
    blocked_hosts: list[str] | None = None,
) -> None:
    # An empty allowlist is a policy, not a mistake: the agent then reaches
    # only what the runner admits for it (its model endpoint), nothing else.
    if network_mode != NetworkMode.ALLOWLIST and allowed_hosts:
        raise ValueError("allowed_hosts is only valid for network_mode='allowlist'")
    if network_mode == NetworkMode.DENYLIST and not (blocked_urls or blocked_hosts):
        raise ValueError(
            "network_mode='denylist' requires blocked_urls or blocked_hosts"
        )
    if network_mode != NetworkMode.DENYLIST and (blocked_urls or blocked_hosts):
        raise ValueError(
            "blocked_urls and blocked_hosts are only valid for network_mode='denylist'"
        )


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
    version: str = Field(
        default="",
        description=(
            "Package version (task.toml schema 1.3 [task] field). Informational "
            "only — recorded, never interpreted. Absent from this model it was "
            "extra_forbidden, which made schema-1.3 tasks that set it unloadable."
        ),
    )
    description: str = Field(
        default="",
        description="What the task asks, in a sentence or two, for people browsing tasks",
    )
    authors: list[Author] = Field(
        default_factory=list,
        description="List of package authors",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Tags that help people find and group the task",
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
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    tools: list[str] | None = None
    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None

    @field_validator("transport", mode="before")
    @classmethod
    def normalize_transport(cls, value: Any) -> Any:
        return "streamable-http" if value == "http" else value

    @field_validator("tools", "include_tags", "exclude_tags")
    @classmethod
    def validate_non_empty_filter_entries(
        cls, value: list[str] | None
    ) -> list[str] | None:
        if value is None:
            return None
        normalized = [item.strip() for item in value if item.strip()]
        if not normalized:
            raise ValueError("MCP tool filters must contain at least one entry")
        return normalized

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
        # The current Sonnet; Sonnet 4.6 is a legacy model (Anthropic's models
        # page, read 2026-11-04). A task that names its judge keeps it.
        default="claude-sonnet-5-5",
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
    # False keeps the workspace repositories' hooks and the config entries
    # that run commands, for a task whose answer is one (test.sh then names
    # a hook with git -c core.hooksPath=...).
    cleanup_git: bool = True


# A pinned PyPI release: a project name, then == and a version.
_PINNED_RELEASE_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?==[0-9][0-9A-Za-z.!+_-]*"
)


MAIN_SERVICE = "main"
_SERVICE_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")


def _compose_service(value: str | None, what: str) -> str | None:
    if value is not None and not _SERVICE_NAME.fullmatch(value):
        raise ValueError(f"{what} {value!r} is not a compose service name")
    return value


class VerifierCollectConfig(TaskConfigModel):
    """A command the task runs in a compose service once the agent's turn is
    over and before its artifacts are collected (``[[verifier.collect]]``),
    to snapshot runtime state (a database, a patch of the workspace) into
    files its artifacts then carry to the verifier. Best effort: a hook that
    fails is logged and the trial goes on."""

    command: str
    service: str = MAIN_SERVICE
    timeout_sec: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    user: str | int | None = None

    @field_validator("service")
    @classmethod
    def _service_name(cls, value: str) -> str:
        return _compose_service(value, "collect service") or MAIN_SERVICE


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
    # The BASH_ENV hardening validated (its trusted-file check) for this
    # config, set by lockdown._build_verifier_env; None until it runs. The
    # verifier's shell reads only a validated value, never [verifier.env]'s
    # raw one (b143 PY8-MAIN-28).
    _trusted_bash_env: str | None = PrivateAttr(default=None)
    submission_files: list[str] = Field(
        default_factory=list,
        description="Explicit verifier-recovery contract (submission-files-v1): the "
        "absolute paths of the task's submission files, at most 8, literal (no globs), "
        "outside /dev, /logs, /proc, /solution, /sys and /tests. Only these regular "
        "files are captured and restored; never a whole directory.",
    )

    @field_validator("submission_files")
    @classmethod
    def validate_submission_files(cls, value: list[str]) -> list[str]:
        if value:
            return list(validate_contract(value))
        return value

    workspace_recovery: bool = Field(
        default=False,
        description="Allow verifier-only recovery from captured workspace after transport failure. "
        "The task author guarantees all solver changes needed for scoring are inside the workspace "
        "and a fresh task image/setup reproduces all other dependencies; no mutable external state.",
    )
    timeout_sec: float = Field(
        default=600.0,
        gt=0,
        allow_inf_nan=False,
        description="Wall-clock budget in seconds for verifier execution. "
        "Omitting the field inherits this default; zero, negative, and "
        "non-finite budgets are rejected at validation time.",
    )
    env: dict[str, str] = Field(default_factory=dict)
    collect: list[VerifierCollectConfig] = Field(
        default_factory=list,
        description="Commands run in compose services after the agent's turn and "
        "before artifact collection (task.toml [[verifier.collect]]).",
    )
    required_tools: list[str] | None = Field(
        default=None,
        description="Complete list of PATH program names whose availability or lookup "
        "can affect grading, including subprocesses and later-created helpers. "
        "Required when any image PATH entry is omitted by hardening. Every "
        "name must already resolve in the trusted verifier PATH; use absolute paths "
        "for uploaded/generated helpers. [] declares no required PATH tools.",
    )

    @field_validator("required_tools")
    @classmethod
    def validate_required_tools(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if any(not name or "/" in name or "\x00" in name for name in value):
            raise ValueError("required_tools must contain nonempty program basenames")
        return list(dict.fromkeys(value))

    optional_tools: list[str] = Field(
        default_factory=list,
        description="PATH program names used only for optional probes. The task author "
        "guarantees that every use tolerates the program being unavailable without "
        "changing grading. Refusals of any other program invalidate every reward.",
    )

    @field_validator("optional_tools")
    @classmethod
    def validate_optional_tools(cls, value: list[str]) -> list[str]:
        if any(not name or "/" in name or "\x00" in name for name in value):
            raise ValueError("optional_tools must contain nonempty program basenames")
        return list(dict.fromkeys(value))

    reward_range: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Declared [lo, hi] reward contract for this task. Omitted (None) "
            "keeps BenchFlow's strict canonical [0.0, 1.0]. A declared range "
            "may only WIDEN the canonical range — lo <= 0.0, hi >= 1.0, "
            "lo < hi, both finite — so 0 ('did nothing') and 1 ('solved') "
            "stay meaningful for every task; e.g. safety-floor benchmarks "
            "declare reward_range = [-1.0, 1.0] to score deliberate "
            "violations below doing nothing. Applies to the test-script "
            "reward contract (reward.txt / reward.json scalar reward, "
            "numeric metrics, aggregate results) and the final reward "
            "canonicalization gate; LLM-judge rubric scores stay [0, 1]."
        ),
    )
    user: str | int | None = Field(
        default=None,
        description="Account (name or numeric id) the verifier's commands run under.",
    )
    network_mode: NetworkMode | None = Field(
        default=None,
        description="Optional verifier-specific network policy override.",
    )
    allowed_hosts: list[str] | None = Field(
        default=None,
        description="Hostnames reachable when network_mode='allowlist'.",
    )
    sandbox_mode: VerifierSandboxMode | None = Field(
        default=None,
        description=(
            "Whether the verifier runs in the agent sandbox ('shared') "
            "or a dedicated verifier sandbox ('separate')."
        ),
    )
    sandbox: SandboxConfig | None = Field(
        default=None,
        description="Optional separate verifier sandbox configuration.",
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
    pytest_plugin_releases: list[str] = Field(
        default_factory=list,
        description=(
            "Pinned PyPI releases (name==version, e.g. pytest-asyncio==0.24.0) "
            "of the pytest plugins test.sh installs itself. The plugin guard "
            "loads such a plugin from the agent-writable trees or uvx's cache "
            "only when its installed files are that release's wheel, byte for "
            "byte, as PyPI publishes it; pytest-json-ctrf needs no entry."
        ),
    )

    @field_validator("pytest_plugin_releases")
    @classmethod
    def validate_pytest_plugin_releases(cls, releases: list[str]) -> list[str]:
        for release in releases:
            if not _PINNED_RELEASE_RE.fullmatch(release):
                raise ValueError(
                    f"pytest_plugin_releases entry {release!r} is not a pinned "
                    "PyPI release (name==version, e.g. pytest-asyncio==0.24.0)"
                )
        return releases

    @field_validator("allowed_hosts")
    @classmethod
    def validate_allowed_hosts(cls, hosts: list[str] | None) -> list[str] | None:
        return _validate_allowed_hosts(hosts)

    @field_validator("reward_range")
    @classmethod
    def validate_reward_range(
        cls, value: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        # Widen-only rule (lo <= 0.0 <= 1.0 <= hi, lo < hi, finite) lives with
        # the reward contract in benchflow.rewards.validation.
        return None if value is None else validate_declared_reward_range(value)

    @model_validator(mode="before")
    @classmethod
    def reject_renamed_environment_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "environment" in data:
                raise ValueError(_VERIFIER_ENVIRONMENT_RENAMED_ERROR)
            if "environment_mode" in data:
                raise ValueError(_VERIFIER_ENVIRONMENT_MODE_RENAMED_ERROR)
        return data

    @model_validator(mode="after")
    def validate_verifier_sandbox(self) -> VerifierConfig:
        if self.workspace_recovery and self.submission_files:
            raise ValueError(
                "Choose either workspace_recovery or explicit submission_files"
            )
        _reject_role_denylist(self.network_mode, "verifier")
        _validate_network_policy_fields(self.network_mode, self.allowed_hosts)
        if self.sandbox_mode == VerifierSandboxMode.SHARED and self.sandbox is not None:
            raise ValueError(
                "[verifier].sandbox_mode='shared' is incompatible with "
                "[verifier.sandbox]"
            )
        return self


class AgentConfig(TaskConfigModel):
    """Agent harness ($H$) configuration."""

    prompt_prefix: str | None = Field(
        default=None,
        description=(
            "Optional run-specific policy text prepended to each resolved task "
            "prompt. Intended for generic harness constraints such as benchmark "
            "integrity rules; must not contain task-specific solution content."
        ),
    )
    timeout_sec: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Wall-clock budget in seconds for the agent harness. "
        "Omitting the field gives the agent no limit of its own: the "
        "rollout's whole deadline bounds it (as in 0.7.11) unless the caller "
        "passes a timeout; zero, negative, and non-finite budgets are "
        "rejected at validation time, matching verifier.timeout_sec.",
    )
    token_budget: int | None = Field(
        default=None,
        gt=0,
        strict=True,
        description="The most tokens a trial's model calls may use, input "
        "(cached included) plus output, subagents' calls included. BenchFlow's "
        "model proxy refuses the next call once they are spent and the agent "
        "is stopped (benchflow.trial_budget). A run's --token-budget replaces it.",
    )
    turn_budget: int | None = Field(
        default=None,
        gt=0,
        strict=True,
        description="The most model calls (turns) a trial may make through "
        "BenchFlow's model proxy, subagents' and side calls included; the proxy "
        "refuses the next one and the agent is stopped. A run's --turn-budget "
        "replaces it.",
    )
    user: str | int | None = Field(
        default=None,
        description="Account (name or numeric id) the agent's commands run under.",
    )
    network_mode: NetworkMode | None = Field(
        default=None,
        description="Optional agent-specific network policy override.",
    )
    allowed_hosts: list[str] | None = Field(
        default=None,
        description="Hostnames reachable when network_mode='allowlist'.",
    )

    @property
    def effective_timeout_sec(self) -> float | None:
        """The task's own agent limit; None when the rollout deadline bounds it."""
        return effective_agent_timeout_sec(self)

    @field_validator("prompt_prefix")
    @classmethod
    def validate_prompt_prefix(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("prompt_prefix must contain non-whitespace text")
        return value

    @field_validator("allowed_hosts")
    @classmethod
    def validate_allowed_hosts(cls, hosts: list[str] | None) -> list[str] | None:
        return _validate_allowed_hosts(hosts)

    @model_validator(mode="after")
    def validate_network_policy(self) -> AgentConfig:
        _reject_role_denylist(self.network_mode, "agent")
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


class SetupCommandConfig(TaskConfigModel):
    """Command to run after sandbox startup and before agent installation."""

    command: str = Field(min_length=1)
    timeout_sec: float = Field(default=600.0, gt=0, allow_inf_nan=False)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    user: str | int | None = None
    service: str = "main"
    host_lock: str | None = None
    capture_dir: str | None = Field(
        default=None,
        description=(
            "Optional absolute sandbox directory to archive after the command "
            "succeeds. The archive is written to capture_dir_b64_env."
        ),
    )
    capture_dir_b64_env: str | None = Field(
        default=None,
        description=(
            "Host environment variable to update with a base64 tar.gz archive "
            "of capture_dir after the command succeeds."
        ),
    )
    capture_dir_b64_env_file_var: str | None = Field(
        default=None,
        description=(
            "Optional host environment variable whose value is a dotenv file "
            "path. When set, capture_dir_b64_env is also upserted there."
        ),
    )

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: str) -> str:
        command = value.strip()
        if not command:
            raise ValueError("setup command must be non-empty")
        return command

    @field_validator("cwd")
    @classmethod
    def validate_cwd(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cwd = value.strip()
        if not cwd:
            raise ValueError("setup command cwd must be non-empty when provided")
        if not cwd.startswith("/"):
            raise ValueError("setup command cwd must be an absolute path")
        return cwd

    @field_validator("service")
    @classmethod
    def validate_service(cls, value: str) -> str:
        service = value.strip()
        if not service:
            raise ValueError("setup command service must be non-empty")
        return service

    @field_validator("host_lock")
    @classmethod
    def validate_host_lock(cls, value: str | None) -> str | None:
        if value is None:
            return None
        host_lock = value.strip()
        if not host_lock:
            raise ValueError("setup command host_lock must be non-empty when provided")
        return host_lock

    @field_validator("capture_dir")
    @classmethod
    def validate_capture_dir(cls, value: str | None) -> str | None:
        if value is None:
            return None
        capture_dir = value.strip()
        if not capture_dir:
            raise ValueError(
                "setup command capture_dir must be non-empty when provided"
            )
        if not capture_dir.startswith("/"):
            raise ValueError("setup command capture_dir must be an absolute path")
        return capture_dir

    @field_validator("capture_dir_b64_env")
    @classmethod
    def validate_capture_dir_b64_env(cls, value: str | None) -> str | None:
        if value is None:
            return None
        env_var = value.strip()
        if not _ENV_VAR_NAME_PATTERN.fullmatch(env_var):
            raise ValueError(
                "setup command capture_dir_b64_env must be a valid environment "
                "variable name"
            )
        return env_var

    @field_validator("capture_dir_b64_env_file_var")
    @classmethod
    def validate_capture_dir_b64_env_file_var(cls, value: str | None) -> str | None:
        if value is None:
            return None
        env_var = value.strip()
        if not _ENV_VAR_NAME_PATTERN.fullmatch(env_var):
            raise ValueError(
                "setup command capture_dir_b64_env_file_var must be a valid "
                "environment variable name"
            )
        return env_var

    @model_validator(mode="after")
    def validate_capture_pair(self) -> SetupCommandConfig:
        if bool(self.capture_dir) != bool(self.capture_dir_b64_env):
            raise ValueError(
                "setup command capture_dir and capture_dir_b64_env must be "
                "provided together"
            )
        if self.capture_dir_b64_env_file_var and not self.capture_dir_b64_env:
            raise ValueError(
                "setup command capture_dir_b64_env_file_var requires "
                "capture_dir_b64_env"
            )
        return self


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

    Named in RL terms; task.toml calls this section ``[environment]``.
    """

    network_mode: NetworkMode = Field(
        default=NetworkMode.PUBLIC,
        description="Network access policy for this environment.",
    )
    allowed_hosts: list[str] | None = Field(
        default=None,
        description="Hostnames reachable when network_mode='allowlist'.",
    )
    blocked_urls: list[str] | None = Field(
        default=None,
        description=(
            "URL prefixes the agent cannot reach when network_mode='denylist'; "
            "scheme optional, query ignored"
        ),
    )
    blocked_hosts: list[str] | None = Field(
        default=None,
        description=(
            "Hostnames (and their subdomains) the agent cannot reach when "
            "network_mode='denylist'"
        ),
    )
    build_timeout_sec: float = 600.0
    docker_image: str | None = Field(
        default=None,
        validation_alias=AliasChoices("docker_image", "image"),
        description="Prebuilt image name; legacy SkillsBench tasks may use image.",
    )
    os: TaskOS = Field(
        default=TaskOS.LINUX,
        description="Operating system the task's container runs.",
    )
    cpus: int = 1
    memory_mb: int = 2048
    storage_mb: int = 10240
    gpus: int = Field(default=0, ge=0, strict=True)
    gpu_types: list[str] | None = Field(
        default=None,
        description=(
            "Acceptable GPU types. Modal accepts one entry, e.g. ['H100']; "
            "omit for any type when gpus is positive."
        ),
    )
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Variables set in the sandbox, read from the host when the "
        "trial starts. A value can name a host variable as ${VAR}, or as "
        "${VAR:-default} to fall back to a default.",
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
    setup_commands: list[SetupCommandConfig] = Field(
        default_factory=list,
        description=(
            "Commands run after sandbox services start and before the agent is "
            "installed. Useful for benchmark-native workspace/database seeding."
        ),
    )
    bugswarm_image_tag: str | None = Field(
        default=None,
        description="BugSwarm image tag used by build-repair benchmark tasks.",
    )
    allow_cloud_metadata: bool = Field(
        default=False,
        description=(
            "Let the agent reach the cloud metadata endpoints under an open "
            "network: no block, and the trial records the opt-out. Without it "
            "the agent starts only once the block is proven on."
        ),
    )
    require_cloud_metadata_block: bool = Field(
        default=False,
        description=(
            "Refuse an agent that runs as root on a sandbox provider without "
            "BenchFlow's metadata helper (Daytona, Modal, AgentCore, Apple), "
            "where it otherwise runs under the provider's own metadata policy."
        ),
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
    def validate_allowed_hosts(cls, hosts: list[str] | None) -> list[str] | None:
        return _validate_allowed_hosts(hosts)

    @field_validator("blocked_urls")
    @classmethod
    def validate_blocked_urls(cls, urls: list[str] | None) -> list[str] | None:
        return _validate_blocked_urls(urls)

    @field_validator("blocked_hosts")
    @classmethod
    def validate_blocked_hosts(cls, hosts: list[str] | None) -> list[str] | None:
        return _validate_blocked_hosts(hosts)

    @field_validator("os", mode="before")
    @classmethod
    def normalize_os(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @model_validator(mode="after")
    def handle_deprecated_fields_and_network_policy(self) -> SandboxConfig:
        _validate_network_policy_fields(
            self.network_mode,
            self.allowed_hosts,
            self.blocked_urls,
            self.blocked_hosts,
        )
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
        _validate_network_policy_fields(
            self.network_mode,
            self.allowed_hosts,
            self.blocked_urls,
            self.blocked_hosts,
        )
        return self

    # Set by a run's config overlay (``timeout_multipliers.build``, which a
    # YAML job config's build multiplier gives): the
    # whole environment start, not only the image build, must finish in this
    # many seconds. Never serialized,
    # never read from a task.
    _start_deadline_sec: float | None = PrivateAttr(default=None)

    @property
    def start_deadline_sec(self) -> float | None:
        """The whole environment start's budget, or None for no bound of its own."""
        return self._start_deadline_sec

    # Set by the sandbox factory on its own copy when the run refuses the
    # cloud metadata endpoints inside ``main`` (lockdown.cloud_metadata_plan);
    # the compose backends then give ``main`` NET_ADMIN for the rules. Never
    # serialized, never read from a task.
    _cloud_metadata_block: bool = PrivateAttr(default=False)
    # Set with it when the agent runs as root: the rules then
    # refuse root too, and main must start without NET_ADMIN and NET_RAW.
    _cloud_metadata_root_agent: bool = PrivateAttr(default=False)

    @property
    def cloud_metadata_block(self) -> bool:
        """Whether this run installs the cloud metadata block in ``main``."""
        return self._cloud_metadata_block

    @property
    def cloud_metadata_root_agent(self) -> bool:
        """Whether the block holds an agent that runs as root."""
        return self._cloud_metadata_root_agent

    def arm_cloud_metadata_block(self, *, root_agent: bool = False) -> None:
        """Mark this (copied) config as one whose run installs the block,
        for an agent that runs as root when ``root_agent``."""
        self._cloud_metadata_block = True
        self._cloud_metadata_root_agent = root_agent

    # Set by the Docker sandbox, before the freeze that starts main, when its
    # metadata helper adds the block's rules from outside main
    # (sandbox._metadata_helper): main then gets no NET_ADMIN for them.
    _cloud_metadata_by_helper: bool = PrivateAttr(default=False)
    # Set by the sandbox factory when the run enforces no-web at the agent
    # layer: root's uid firewall in main needs main's NET_ADMIN then.
    _agent_uid_firewall: bool = PrivateAttr(default=False)

    @property
    def cloud_metadata_by_helper(self) -> bool:
        """Whether the block's rules come from the helper, with no NET_ADMIN
        in ``main``."""
        return self._cloud_metadata_by_helper

    def serve_cloud_metadata_from_helper(self) -> None:
        """Mark this (copied) config's block as one the helper installs."""
        self._cloud_metadata_by_helper = True

    @property
    def agent_uid_firewall(self) -> bool:
        """Whether the run's agent-layer no-web firewall needs root's
        NET_ADMIN in ``main``."""
        return self._agent_uid_firewall

    def need_agent_uid_firewall(self) -> None:
        """Mark this (copied) config as one whose run enforces no-web in main."""
        self._agent_uid_firewall = True


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
                    raise ValueError(f"Conflicting values for solution env var {key!r}")
                env[key] = value
            else:
                normalized[key] = value

        normalized["env"] = env
        return normalized


#: The compose service the agent runs in (the task.toml format's ``main``).
class ArtifactConfig(TaskConfigModel):
    """Artifact path copied out of a task environment after verification."""

    source: str
    destination: str | None = None
    exclude: list[str] = Field(default_factory=list)
    service: str | None = Field(
        default=None,
        description="The compose service to collect this artifact from (the task.toml "
        "format's sidecar artifacts): None or 'main' is the agent's own container; any "
        "other names a service of the task's compose file, needs a sandbox that runs "
        "compose services and an absolute source, and is read from that service's own "
        "filesystem.",
    )

    @field_validator("service")
    @classmethod
    def _service_name(cls, value: str | None) -> str | None:
        return _compose_service(value, "artifact service")

    @model_validator(mode="after")
    def _sidecar_source_is_absolute(self) -> Self:
        if self.service not in (None, MAIN_SERVICE) and not self.source.startswith("/"):
            raise ValueError(
                f"artifact {self.source!r} from service {self.service!r} must be an "
                "absolute path"
            )
        return self


class StepConfig(TaskConfigModel):
    """Multi-step task configuration."""

    name: str
    agent: AgentConfig = Field(default_factory=AgentConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    min_reward: float | dict[str, float] | None = Field(default=None)
    healthcheck: HealthcheckConfig | None = None
    artifacts: list[str | ArtifactConfig] = Field(default_factory=list)


class TaskConfig(TaskConfigModel):
    """Full task configuration — the task specification ($T$).

    Maps task.md frontmatter / task.toml sections to BenchFlow's RL-aligned
    models. The native key for the sandbox spec is ``sandbox``; the legacy
    ``environment`` spelling is only accepted on the task.toml import
    path, where :func:`convert_legacy_environment_keys` translates it.
    """

    schema_version: str = "1.3"
    task: PackageInfo | None = Field(
        default=None,
        description="Package information from the [task] section.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
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
    preference: dict[str, Any] = Field(
        default_factory=dict,
        description="How `bench preferences` pairs and ranks this task's outputs "
        "(protocol, randomize_sides, raters_per_pair, min_agreement; other keys "
        "are recorded). The run itself is unchanged. "
        "See docs/tasks/human-preferences.md.",
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("preference")
    @classmethod
    def validate_preference(cls, value: dict[str, Any]) -> dict[str, Any]:
        """The keys `bench preferences` reads must have their types; the rest
        is recorded as given (as in a task.md package's [preference])."""
        protocol = value.get("protocol", "pairwise")
        if protocol not in ("pairwise", "rubric", "best-of-n"):
            raise ValueError(
                "preference.protocol must be pairwise or rubric "
                f"(best-of-n is not supported yet), not {protocol!r}"
            )
        if "randomize_sides" in value and not isinstance(
            value["randomize_sides"], bool
        ):
            raise ValueError("preference.randomize_sides must be true or false")
        raters = value.get("raters_per_pair")
        if raters is not None and (
            isinstance(raters, bool) or not isinstance(raters, int) or raters < 1
        ):
            raise ValueError("preference.raters_per_pair must be a whole number from 1")
        agreement = value.get("min_agreement")
        # A number, or task.md draft 2's {metric, value} table.
        if isinstance(agreement, dict):
            agreement = agreement.get("value")
        if agreement is not None and (
            isinstance(agreement, bool)
            or not isinstance(agreement, int | float)
            or not 0 <= agreement <= 1
        ):
            raise ValueError(
                "preference.min_agreement must be a number from 0 to 1 "
                "(or a table whose value is one)"
            )
        return value

    @field_validator("metadata")
    @classmethod
    def validate_cloud_metadata_opt_out(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Refuse an opt-out that is not a boolean, so a typo cannot leave the
        author believing the agent can (or cannot) reach the endpoints."""
        if CLOUD_METADATA_OPT_OUT_KEY in value:
            raw = value[CLOUD_METADATA_OPT_OUT_KEY]
            try:
                _LAX_BOOL.validate_python(raw)
            except ValidationError:
                raise ValueError(
                    f"metadata.{CLOUD_METADATA_OPT_OUT_KEY} must be true or false, "
                    f"not {raw!r}"
                ) from None
        return value

    @property
    def allows_cloud_metadata(self) -> bool:
        """Whether the task opts out of refusing the cloud metadata endpoints
        (``allow_cloud_metadata = true`` in its ``[environment]`` table)."""
        return bool(self.cloud_metadata_opt_out)

    @property
    def cloud_metadata_opt_out(self) -> str:
        """How the task opts out of the cloud metadata block, for the record;
        empty when it does not. ``metadata.x-allow-cloud-metadata = true`` is
        the older, unreleased spelling of the same opt-out."""
        if self.sandbox.allow_cloud_metadata:
            return CLOUD_METADATA_TASK_OPT_OUT
        raw = self.metadata.get(CLOUD_METADATA_OPT_OUT_KEY)
        if raw is not None and _LAX_BOOL.validate_python(raw):
            return f"metadata.{CLOUD_METADATA_OPT_OUT_KEY} = true"
        return ""

    @field_validator("metadata")
    @classmethod
    def validate_keep_services(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Refuse a ``metadata.x-keep-services`` that is not a list of at most
        8 ``{ port, name }`` tables, naming every problem at its place, so a
        typo cannot leave the author believing a service is kept."""
        if keep_services.KEY in value:
            keep_services.validate(value[keep_services.KEY])
        return value

    # Terminal-Bench 4.1 labels some tasks schema_version "2.0" with the 1.x
    # keys: the runner that writes them reads the label without gating on it.
    # A 2.x task is read as 1.x; a key 1.x does not know is still refused.
    _SUPPORTED_SCHEMA_MAJORS: ClassVar[frozenset[int]] = frozenset({1, 2})

    @model_validator(mode="before")
    @classmethod
    def handle_version_rename(cls, data: Any) -> Any:
        """Refuse the removed ``environment`` spelling and conflicting solution keys."""
        if isinstance(data, dict):
            if "environment" in data:
                raise ValueError(_ENVIRONMENT_RENAMED_ERROR)
            if "oracle" in data and "solution" in data:
                raise ValueError(
                    "Task config cannot contain both 'oracle' and legacy "
                    "'solution'; use 'oracle' for native tasks or 'solution' "
                    "only when importing legacy task.toml tasks."
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
            supported = ", ".join(str(m) for m in sorted(cls._SUPPORTED_SCHEMA_MAJORS))
            raise ValueError(
                f"Unsupported schema_version major {major} (from {value!r}); "
                f"supported major version(s): {supported}"
            )
        return value

    @classmethod
    def model_validate_toml(cls, toml_data: str) -> TaskConfig:
        # task.toml is the legacy surface, so the loader translates
        # the foreign 'environment' spelling to native 'sandbox' before
        # validation. This is deliberate format translation, not an alias:
        # the native schema (task.md frontmatter, direct model_validate)
        # accepts only 'sandbox'.
        """Read a legacy ``task.toml`` configuration."""
        toml_dict = convert_legacy_environment_keys(tomllib.loads(toml_data))
        return cls.model_validate(toml_dict)

    # Keys a lenient task.toml load ignored (dotted paths, sorted); set by
    # benchflow.task.imports.load_task_config_toml. Never serialized.
    _ignored_keys: tuple[str, ...] = PrivateAttr(default=())

    @property
    def ignored_keys(self) -> tuple[str, ...]:
        """task.toml keys this config does not model and a lenient load
        ignored (empty for strict loads). Keys BenchFlow knows it cannot honour
        are refused by the runtime capability check instead of being run."""
        return self._ignored_keys

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
        """Write the config as ``task.toml`` text."""
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
