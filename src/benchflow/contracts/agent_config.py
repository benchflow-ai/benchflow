"""Agent config dataclasses — CredentialFile, HostAuthFile, SubscriptionAuth, AgentConfig.

CONTRACT SURFACE — semver-stable. Changes here break downstream importers.
Prefer extending in periphery (``benchflow.agents.registry`` and friends)
unless the shape itself must change.

Declarative shape of a supported agent. Lives in ``contracts/`` so the registry
machinery in ``benchflow.agents.registry`` (AGENTS dict, install shell
snippets, runtime helpers) can evolve without dragging semver guarantees
into every adapter refactor. ``agents.registry`` re-exports these names
for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CredentialFile:
    """A file to write inside the container before agent launch."""

    path: str  # Target path in container (may use {home} placeholder)
    env_source: str  # Env var to read value from
    template: str = ""  # Template with {value} placeholder. Empty = raw value.
    mkdir: bool = True  # Create parent directory


@dataclass
class HostAuthFile:
    """A single file to copy from the host into the container."""

    host_path: str  # Path on host, e.g. "~/.claude/.credentials.json"
    container_path: str  # Destination in container (may use {home} placeholder)


@dataclass
class SubscriptionAuth:
    """Host CLI login credentials that can substitute for an API key.

    When the user has logged in via the agent CLI (e.g. ``claude login``),
    BenchFlow detects the auth files on the host, copies them into the
    container, and skips the API key requirement.

    ``detect_file`` is checked to determine if the user is logged in.
    All ``files`` are copied into the container when subscription auth is used.
    """

    replaces_env: str  # The env var this substitutes, e.g. "ANTHROPIC_API_KEY"
    detect_file: str  # Host path to check for login, e.g. "~/.claude/.credentials.json"
    files: list[HostAuthFile] = field(default_factory=list)  # All files to copy


@dataclass
class AgentConfig:
    """Configuration for a supported agent."""

    name: str
    install_cmd: str
    launch_cmd: str
    protocol: str = "acp"  # "acp" or "cli"
    requires_env: list[str] = field(default_factory=list)
    description: str = ""
    skill_paths: list[str] = field(default_factory=list)
    install_timeout: int = 900  # seconds
    default_model: str = ""  # default model ID when --model is omitted
    api_protocol: str = ""
    # The LLM API protocol the agent natively speaks:
    # "anthropic-messages" | "openai-completions" | "" (runtime/native).
    # Used to pick the correct provider endpoint when a provider exposes
    # multiple (e.g. zai has both anthropic-messages and openai-completions).
    env_mapping: dict[str, str] = field(default_factory=dict)
    # Maps BENCHFLOW_PROVIDER_* → agent-native env var names.
    # Applied by SDK after provider resolution.
    credential_files: list[CredentialFile] = field(default_factory=list)
    # Files to write into container before agent launch (e.g. auth.json).
    home_dirs: list[str] = field(default_factory=list)
    # Extra dot-dirs under $HOME to copy to sandbox user (for dirs not
    # derivable from skill_paths or credential_files, e.g. ".openclaw").
    subscription_auth: SubscriptionAuth | None = None
    # Host CLI login that can substitute for an API key (e.g. OAuth tokens
    # from `claude login`). Detected automatically; API keys take precedence.
    error_taxonomy: list[str] = field(default_factory=list)
    # Canonical error substrings this agent is known to emit, to be matched
    # against the failure text in classify_error. Declaring ``[]`` explicitly
    # means "this agent accepts the silent-non-retryable sink" — useful when
    # the agent has no distinctive failure modes worth pattern-matching.
    #
    # The invariant in tests/test_registry_invariants.py asserts that no two
    # agents claim overlapping substrings. Wiring this field into
    # classify_error (so these substrings are matched before falling through
    # to "other") is deferred — schema and uniqueness are v2; live retry
    # behavior change is a v3 decision (see PLAN_V2_impl.md §5.1).
