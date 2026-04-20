"""Agent registry — supported agents and their configurations.

Adding a new agent is a registry-only change: append one entry to ``AGENTS``
below. The SDK reads everything it needs about the agent from this dict, so no
``sdk.py`` edits are required. ``tests/test_registry_invariants.py`` runs
contract checks against every entry — read it for the executable schema.

Required fields
---------------
- ``name``               Must equal the dict key.
- ``install_cmd``        Bash command run inside the sandbox to install the
                         agent. Idempotent (use ``command -v ... ||`` guards).
- ``launch_cmd``         Command that starts the agent process. The SDK pipes
                         ACP messages to its stdin / reads from its stdout.

Common optional fields
----------------------
- ``protocol``           "acp" (default) or "cli". Almost always "acp".
- ``requires_env``       List of env var names the SDK must propagate into the
                         sandbox (e.g. ``["ANTHROPIC_API_KEY"]``). Validated at
                         run start; missing keys raise before the container
                         spins up.
- ``api_protocol``       "anthropic-messages" | "openai-completions" | "" — the
                         LLM API the agent natively speaks. Used to pick the
                         right endpoint when a provider exposes multiple
                         (e.g. zai). Empty means "agent infers from model name".
- ``env_mapping``        ``BENCHFLOW_PROVIDER_*`` → agent-native env var.
                         Applied by the SDK after provider resolution. Keys
                         **must** start with ``BENCHFLOW_PROVIDER_``.
- ``skill_paths``        Sandbox paths where benchflow should mount skill
                         content. Must start with ``$HOME/`` or ``$WORKSPACE/``.
- ``credential_files``   ``CredentialFile`` entries for files written into the
                         container before launch (e.g. ``~/.codex/auth.json``).
- ``home_dirs``          Extra dot-dirs under ``$HOME`` to copy to the sandbox
                         user (for dirs not derivable from ``skill_paths`` /
                         ``credential_files``, e.g. ``.openclaw``).
- ``subscription_auth``  ``SubscriptionAuth`` describing host CLI login files
                         (e.g. ``claude login`` credentials) that can stand in
                         for an API key. API keys still take precedence.

Look at the existing entries below for worked examples:
``claude-agent-acp`` (subscription auth + env_mapping), ``codex-acp``
(credential_files), ``openclaw`` (home_dirs + custom shim), ``gemini``
(multi-file subscription auth).
"""

import base64
from dataclasses import dataclass, field
from pathlib import Path


def _install_python_script(container_path: str, source: str) -> str:
    """Shell snippet that ensures python3 and writes `source` to container_path.

    Base64 transport — makes the install shell line content-agnostic so a line
    like `SHIMEOF` or `LAUNCHEREOF` inside the Python source can't collide with
    a heredoc terminator.

    Used by pi-acp and openclaw — both ship a Python launcher/shim baked into
    install_cmd. Rule of three: if you're adding a THIRD agent that needs this
    pattern, read both pi_acp_launcher.py and openclaw_acp_shim.py first and
    reconcile their semantics (env bridging, provider-name derivation, model
    metadata) before writing a new one. Only consider extracting a shared base
    after the third data point — divergence is cheap, premature abstraction isn't.
    """
    encoded = base64.b64encode(source.encode()).decode()
    return (
        "( command -v python3 >/dev/null 2>&1 || "
        "(apt-get update -qq && apt-get install -y -qq python3 >/dev/null 2>&1) ) && "
        f"echo {encoded} | base64 -d > {container_path} && "
        f"chmod +x {container_path}"
    )


# Node.js bootstrap — handles missing node, old node (<22), Ubuntu + Debian slim
_NODE_INSTALL = (
    "set -o pipefail; "
    "export DEBIAN_FRONTEND=noninteractive; "
    "NODE_OK=0; "
    "if command -v node >/dev/null 2>&1; then "
    "  NODE_VER=$(node -e 'console.log(process.versions.node.split(\".\")[0])' 2>/dev/null || echo 0); "
    '  [ "$NODE_VER" -ge 22 ] 2>/dev/null && NODE_OK=1; '
    "fi; "
    'if [ "$NODE_OK" = 0 ]; then '
    "  apt-get update -qq && "
    "  apt-get install -y -qq curl ca-certificates && "
    "  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && "
    "  apt-get install -y -qq nodejs; "
    "fi >/dev/null 2>&1"
)

# Path to the openclaw ACP shim script
_OPENCLAW_SHIM = (Path(__file__).parent / "openclaw_acp_shim.py").read_text()

# Path to the Pi launch wrapper (bridges BENCHFLOW_PROVIDER_* → Pi config)
_PI_LAUNCHER = (Path(__file__).parent / "pi_acp_launcher.py").read_text()


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


# Agent registry — all supported agents
AGENTS: dict[str, AgentConfig] = {
    "claude-agent-acp": AgentConfig(
        name="claude-agent-acp",
        description="Claude Code via ACP (Anthropic's Agent Client Protocol)",
        skill_paths=["$HOME/.claude/skills"],
        install_cmd=(
            f"{_NODE_INSTALL} && "
            "( command -v claude-agent-acp >/dev/null 2>&1 || "
            "npm install -g @zed-industries/claude-agent-acp@latest >/dev/null 2>&1 ) && "
            "command -v claude-agent-acp >/dev/null 2>&1"
        ),
        launch_cmd="claude-agent-acp",
        protocol="acp",
        requires_env=["ANTHROPIC_API_KEY"],
        api_protocol="anthropic-messages",
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "ANTHROPIC_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "ANTHROPIC_AUTH_TOKEN",
            "BENCHFLOW_PROVIDER_MODEL": "ANTHROPIC_MODEL",
        },
        subscription_auth=SubscriptionAuth(
            replaces_env="ANTHROPIC_API_KEY",
            detect_file="~/.claude/.credentials.json",
            files=[
                HostAuthFile(
                    "~/.claude/.credentials.json", "{home}/.claude/.credentials.json"
                ),
            ],
        ),
    ),
    "pi-acp": AgentConfig(
        name="pi-acp",
        description="Pi agent via ACP",
        skill_paths=["$HOME/.pi/agent/skills", "$HOME/.agents/skills"],
        install_cmd=(
            f"{_NODE_INSTALL} && "
            "( command -v pi >/dev/null 2>&1 || "
            "npm install -g @mariozechner/pi-coding-agent@latest >/dev/null 2>&1 ) && "
            "( command -v pi-acp >/dev/null 2>&1 || "
            "npm install -g pi-acp@latest >/dev/null 2>&1 ) && "
            "command -v pi-acp >/dev/null 2>&1 && "
            # Deploy launch wrapper (bridges BENCHFLOW_PROVIDER_* → Pi config)
            + _install_python_script("/usr/local/bin/pi-acp-launcher", _PI_LAUNCHER)
        ),
        launch_cmd="python3 /usr/local/bin/pi-acp-launcher",
        protocol="acp",
        requires_env=[],  # inferred from --model at runtime
        # Pi is multi-protocol: speaks Anthropic natively and OpenAI via
        # models.json.  Empty lets the provider determine the protocol so
        # multi-endpoint providers (e.g. zai) route to the right URL.
        api_protocol="",
        # env_mapping intentionally empty — the launch wrapper handles
        # protocol-dependent translation (env vars for Anthropic,
        # models.json for OpenAI-compatible providers like vLLM).
    ),
    "openclaw": AgentConfig(
        name="openclaw",
        description="OpenClaw agent via ACP shim — model set at runtime via --model",
        skill_paths=["$HOME/.claude/skills", "$WORKSPACE/skills"],
        install_cmd=(
            f"{_NODE_INSTALL} && "
            "( command -v openclaw >/dev/null 2>&1 || "
            "npm install -g openclaw@latest >/dev/null 2>&1 ) && "
            "command -v openclaw >/dev/null 2>&1 && "
            # Configure: auto-approve tools (no model — set at runtime via ACP set_model)
            "mkdir -p ~/.openclaw && "
            'echo \'{"version":1,"defaults":{"allow_all":true}}\''
            " > ~/.openclaw/exec-approvals.json && "
            # Deploy ACP shim
            + _install_python_script("/usr/local/bin/openclaw-acp-shim", _OPENCLAW_SHIM)
        ),
        launch_cmd="python3 /usr/local/bin/openclaw-acp-shim",
        protocol="acp",
        requires_env=[],  # inferred from --model at runtime
        home_dirs=[".openclaw"],
    ),
    "codex-acp": AgentConfig(
        name="codex-acp",
        description="OpenAI Codex agent via ACP",
        skill_paths=["$HOME/.agents/skills"],
        install_cmd=(
            f"{_NODE_INSTALL} && "
            "( command -v codex-acp >/dev/null 2>&1 || "
            "npm install -g @zed-industries/codex-acp@latest >/dev/null 2>&1 ) && "
            "command -v codex-acp >/dev/null 2>&1"
        ),
        launch_cmd="codex-acp",
        protocol="acp",
        requires_env=["OPENAI_API_KEY"],
        api_protocol="openai-completions",
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "OPENAI_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "OPENAI_API_KEY",
        },
        credential_files=[
            CredentialFile(
                path="{home}/.codex/auth.json",
                env_source="OPENAI_API_KEY",
                template='{{"OPENAI_API_KEY": "{value}"}}',
            ),
        ],
        subscription_auth=SubscriptionAuth(
            replaces_env="OPENAI_API_KEY",
            detect_file="~/.codex/auth.json",
            files=[
                HostAuthFile("~/.codex/auth.json", "{home}/.codex/auth.json"),
            ],
        ),
    ),
    "gemini": AgentConfig(
        name="gemini",
        description="Google Gemini CLI via ACP",
        skill_paths=["$HOME/.gemini/skills"],
        install_cmd=(
            f"{_NODE_INSTALL} && "
            "( command -v gemini >/dev/null 2>&1 || "
            "npm install -g @google/gemini-cli@latest >/dev/null 2>&1 ) && "
            "command -v gemini >/dev/null 2>&1"
        ),
        launch_cmd="gemini --acp --yolo",
        protocol="acp",
        requires_env=["GOOGLE_API_KEY"],
        # api_protocol intentionally empty: Gemini speaks Google's native
        # GenerateContent format, which no current PROVIDERS entry exposes as
        # a multi-endpoint option. Set this when a Gemini-compatible provider
        # with multiple endpoints (e.g. OpenRouter) is added.
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "GEMINI_API_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "GOOGLE_API_KEY",
        },
        subscription_auth=SubscriptionAuth(
            replaces_env="GEMINI_API_KEY",
            detect_file="~/.gemini/oauth_creds.json",
            files=[
                HostAuthFile(
                    "~/.gemini/oauth_creds.json", "{home}/.gemini/oauth_creds.json"
                ),
                HostAuthFile("~/.gemini/settings.json", "{home}/.gemini/settings.json"),
                HostAuthFile(
                    "~/.gemini/google_accounts.json",
                    "{home}/.gemini/google_accounts.json",
                ),
            ],
        ),
    ),
    "openhands": AgentConfig(
        name="openhands",
        description="OpenHands agent via ACP (multi-model, Python-based)",
        skill_paths=[],
        install_cmd=(
            "( command -v openhands >/dev/null 2>&1 || "
            "pip install openhands >/dev/null 2>&1 ) && "
            "command -v openhands >/dev/null 2>&1"
        ),
        launch_cmd="openhands acp --always-approve --override-with-envs",
        protocol="acp",
        requires_env=["LLM_API_KEY"],
        api_protocol="",
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "LLM_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "LLM_API_KEY",
        },
    ),
}


# Derived lookup tables — install/launch commands by agent name.
# Updated by register_agent() when new agents are added at runtime.
AGENT_INSTALLERS: dict[str, str] = {name: a.install_cmd for name, a in AGENTS.items()}
AGENT_LAUNCH: dict[str, str] = {name: a.launch_cmd for name, a in AGENTS.items()}


def get_sandbox_home_dirs() -> set[str]:
    """Collect all dot-dirs under $HOME that sandbox user setup should copy.

    Derives from three sources across all registered agents:
    - skill_paths: $HOME/.foo/... → ".foo"
    - credential_files: {home}/.foo/... → ".foo"
    - home_dirs: explicit extras (e.g. ".openclaw")

    Always includes ".local" (pip scripts, etc.).
    """
    dirs: set[str] = {".local"}
    for cfg in AGENTS.values():
        for sp in cfg.skill_paths:
            if sp.startswith("$HOME/."):
                dirname = sp.removeprefix("$HOME/").split("/")[0]
                dirs.add(dirname)
        for cf in cfg.credential_files:
            # path uses {home}/.foo/... placeholder
            path = cf.path
            if path.startswith("{home}/."):
                dirname = path.removeprefix("{home}/").split("/")[0]
                dirs.add(dirname)
        if cfg.subscription_auth:
            for f in cfg.subscription_auth.files:
                if f.container_path.startswith("{home}/."):
                    dirname = f.container_path.removeprefix("{home}/").split("/")[0]
                    dirs.add(dirname)
        dirs.update(cfg.home_dirs)
    return dirs


def is_vertex_model(model: str) -> bool:
    """True if the model uses Vertex AI (GCP ADC auth, not API keys)."""
    from benchflow.agents.providers import find_provider

    result = find_provider(model)
    if result:
        _, cfg = result
        return cfg.auth_type == "adc"
    return False


def infer_env_key_for_model(model: str) -> str | None:
    """Infer the required API key environment variable from a model ID."""
    # Check custom providers first
    from benchflow.agents.providers import resolve_auth_env

    custom = resolve_auth_env(model)
    if custom is not None:
        return custom
    # ADC-based providers and built-in Vertex prefixes
    if is_vertex_model(model):
        return None
    # Fallback heuristics for well-known model names
    m = model.lower()
    if "gemini" in m:
        return "GEMINI_API_KEY"
    if "gpt" in m or m.startswith("o1") or m.startswith("o3"):
        return "OPENAI_API_KEY"
    if "claude" in m or "haiku" in m or "sonnet" in m or "opus" in m:
        return "ANTHROPIC_API_KEY"
    return None


AGENT_ALIASES: dict[str, str] = {
    "claude": "claude-agent-acp",
    "codex": "codex-acp",
    "gemini": "gemini",
    "pi": "pi-acp",
    "openclaw": "openclaw",
    "openhands": "openhands",
    "oh": "openhands",
}

VALID_PROTOCOLS = {"acp", "harbor"}


def parse_agent_spec(spec: str) -> tuple[str, str]:
    """Parse an agent spec like 'acp/claude-agent-acp' or 'claude'.

    Returns (protocol, agent_name) with alias resolution.
    Bare names default to 'acp' protocol.
    """
    if "/" in spec:
        protocol, name = spec.split("/", 1)
    else:
        protocol, name = "acp", spec

    name = AGENT_ALIASES.get(name, name)
    return protocol, name


def resolve_agent(spec: str) -> AgentConfig:
    """Resolve an agent spec to an AgentConfig.

    Supports: bare name, alias, protocol/name.
    Raises KeyError with suggestions for unknown agents.
    """
    protocol, name = parse_agent_spec(spec)

    if protocol not in VALID_PROTOCOLS:
        raise KeyError(
            f"Unknown protocol: {protocol!r}. Valid: {', '.join(sorted(VALID_PROTOCOLS))}"
        )

    if protocol == "harbor":
        return AgentConfig(
            name=name,
            install_cmd="",
            launch_cmd="",
            protocol="harbor",
            requires_env=[],
            description=f"Harbor agent: {name}",
        )

    if name in AGENTS:
        return AGENTS[name]

    # Fuzzy suggestion
    from difflib import get_close_matches

    close = get_close_matches(name, list(AGENTS.keys()), n=1, cutoff=0.6)
    if close:
        raise KeyError(f"Unknown agent: {name!r}. Did you mean: {close[0]!r}?")
    raise KeyError(
        f"Unknown agent: {name!r}. Available: {', '.join(sorted(AGENTS.keys()))}"
    )


def get_agent(name: str) -> tuple[AgentConfig, str]:
    """Get agent config by name.

    Returns (config, default_model) where default_model comes from config.default_model.
    Raises KeyError if not found.
    """
    if name not in AGENTS:
        available = ", ".join(sorted(AGENTS.keys()))
        raise KeyError(f"Unknown agent: {name!r}. Available: {available}")
    config = AGENTS[name]
    return config, config.default_model


def list_agents() -> list[AgentConfig]:
    """List all registered agents."""
    return list(AGENTS.values())


def register_agent(
    name: str,
    install_cmd: str,
    launch_cmd: str,
    *,
    protocol: str = "acp",
    requires_env: list[str] | None = None,
    description: str = "",
    skill_paths: list[str] | None = None,
    install_timeout: int = 900,
    env_mapping: dict[str, str] | None = None,
    credential_files: list[CredentialFile] | None = None,
    home_dirs: list[str] | None = None,
    subscription_auth: SubscriptionAuth | None = None,
) -> AgentConfig:
    """Register a custom agent at runtime.

    Usage:
        from benchflow.agents.registry import register_agent

        register_agent(
            name="my-agent",
            install_cmd="npm install -g my-agent",
            launch_cmd="my-agent --acp",
            requires_env=["MY_API_KEY"],
            skill_paths=["$HOME/.my-agent/skills"],
        )

    Returns the created AgentConfig.
    """
    config = AgentConfig(
        name=name,
        install_cmd=install_cmd,
        launch_cmd=launch_cmd,
        protocol=protocol,
        requires_env=requires_env or [],
        description=description,
        skill_paths=skill_paths or [],
        install_timeout=install_timeout,
        env_mapping=env_mapping or {},
        credential_files=credential_files or [],
        home_dirs=home_dirs or [],
        subscription_auth=subscription_auth,
    )
    AGENTS[name] = config
    AGENT_INSTALLERS[name] = install_cmd
    AGENT_LAUNCH[name] = launch_cmd
    return config
