"""Agent registry — supported agents and their configurations.

Each agent has:
- install_cmd: How to install in a sandbox (bash command)
- launch_cmd: How to start the agent (ACP command or binary)
- requires_env: Required environment variables
- protocol: "acp" (Agent Client Protocol) or "cli" (direct CLI execution)
"""

from dataclasses import dataclass, field
from pathlib import Path


# Node.js bootstrap — handles missing node, old node (<22), Ubuntu + Debian slim
_NODE_INSTALL = (
    "set -o pipefail; "
    "export DEBIAN_FRONTEND=noninteractive; "
    "NODE_OK=0; "
    "if command -v node >/dev/null 2>&1; then "
    "  NODE_VER=$(node -e 'console.log(process.versions.node.split(\".\")[0])' 2>/dev/null || echo 0); "
    "  [ \"$NODE_VER\" -ge 22 ] 2>/dev/null && NODE_OK=1; "
    "fi; "
    "if [ \"$NODE_OK\" = 0 ]; then "
    "  apt-get update -qq && "
    "  apt-get install -y -qq curl ca-certificates && "
    "  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && "
    "  apt-get install -y -qq nodejs; "
    "fi >/dev/null 2>&1"
)

# Path to the openclaw ACP shim script
_OPENCLAW_SHIM = (Path(__file__).parent / "openclaw_acp_shim.py").read_text()


@dataclass
class CredentialFile:
    """A file to write inside the container before agent launch."""

    path: str  # Target path in container (may use {home} placeholder)
    env_source: str  # Env var to read value from
    template: str = ""  # Template with {value} placeholder. Empty = raw value.
    mkdir: bool = True  # Create parent directory


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
    env_mapping: dict[str, str] = field(default_factory=dict)
    # Maps BENCHFLOW_PROVIDER_* → agent-native env var names.
    # Applied by SDK after provider resolution.
    credential_files: list[CredentialFile] = field(default_factory=list)
    # Files to write into container before agent launch (e.g. auth.json).
    home_dirs: list[str] = field(default_factory=list)
    # Extra dot-dirs under $HOME to copy to sandbox user (for dirs not
    # derivable from skill_paths or credential_files, e.g. ".openclaw").


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
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "ANTHROPIC_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "ANTHROPIC_AUTH_TOKEN",
        },
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
            "command -v pi-acp >/dev/null 2>&1"
        ),
        launch_cmd="pi-acp",
        protocol="acp",
        requires_env=["ANTHROPIC_API_KEY"],
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "ANTHROPIC_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "ANTHROPIC_AUTH_TOKEN",
        },
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
            # Ensure python3 for shim
            "( command -v python3 >/dev/null 2>&1 || "
            "(apt-get update -qq && apt-get install -y -qq python3 >/dev/null 2>&1) ) && "
            # Configure: auto-approve tools (no model — set at runtime via ACP set_model)
            "mkdir -p ~/.openclaw && "
            'echo \'{"version":1,"defaults":{"allow_all":true}}\''
            " > ~/.openclaw/exec-approvals.json && "
            # Deploy ACP shim
            "cat > /usr/local/bin/openclaw-acp-shim <<'SHIMEOF'\n"
            + _OPENCLAW_SHIM +
            "\nSHIMEOF\n"
            "chmod +x /usr/local/bin/openclaw-acp-shim"
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
        launch_cmd="gemini --acp",
        protocol="acp",
        requires_env=["GOOGLE_API_KEY"],
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "GEMINI_API_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "GOOGLE_API_KEY",
        },
    ),
}


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
    )
    AGENTS[name] = config
    # Update backwards-compat dicts
    from benchflow.sdk import AGENT_INSTALLERS, AGENT_LAUNCH
    AGENT_INSTALLERS[name] = install_cmd
    AGENT_LAUNCH[name] = launch_cmd
    return config
