"""Agent registry — supported agents and their configurations.

Each agent has:
- install_cmd: How to install in a sandbox (bash command)
- launch_cmd: How to start the agent (ACP command or binary)
- requires_env: Required environment variables
- protocol: "acp" (Agent Client Protocol) or "cli" (direct CLI execution)
"""

from dataclasses import dataclass, field


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


@dataclass
class AgentConfig:
    """Configuration for a supported agent."""

    name: str
    install_cmd: str
    launch_cmd: str
    protocol: str = "acp"  # "acp" or "cli"
    requires_env: list[str] = field(default_factory=list)
    description: str = ""


# Agent registry — all supported agents
AGENTS: dict[str, AgentConfig] = {
    "claude-agent-acp": AgentConfig(
        name="claude-agent-acp",
        description="Claude Code via ACP (Anthropic's Agent Client Protocol)",
        install_cmd=(
            f"{_NODE_INSTALL} && "
            "( command -v claude-agent-acp >/dev/null 2>&1 || "
            "npm install -g @zed-industries/claude-agent-acp@latest >/dev/null 2>&1 ) && "
            "command -v claude-agent-acp >/dev/null 2>&1"
        ),
        launch_cmd="claude-agent-acp",
        protocol="acp",
        requires_env=["ANTHROPIC_API_KEY"],
    ),
    "pi-acp": AgentConfig(
        name="pi-acp",
        description="Pi agent via ACP",
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
    ),
    # openclaw's ACP bridge requires sessions bound to chat threads via the gateway.
    # session/new succeeds but session/prompt returns ACP_SESSION_INIT_FAILED because
    # the gateway's agent has no ACP metadata for standalone sessions.
    # Needs openclaw to add headless/standalone ACP mode.
    "openclaw": AgentConfig(
        name="openclaw",
        description="OpenClaw via ACP (incompatible — needs headless ACP mode)",
        install_cmd=(
            f"{_NODE_INSTALL} && "
            "( command -v openclaw >/dev/null 2>&1 || "
            "npm install -g openclaw@latest >/dev/null 2>&1 ) && "
            "command -v openclaw >/dev/null 2>&1"
        ),
        launch_cmd="openclaw acp",
        protocol="acp",
        requires_env=["ANTHROPIC_API_KEY"],
    ),
    "codex-acp": AgentConfig(
        name="codex-acp",
        description="OpenAI Codex agent via ACP",
        install_cmd=(
            f"{_NODE_INSTALL} && "
            "( command -v codex-acp >/dev/null 2>&1 || "
            "npm install -g @zed-industries/codex-acp@latest >/dev/null 2>&1 ) && "
            "command -v codex-acp >/dev/null 2>&1"
        ),
        launch_cmd="codex-acp",
        protocol="acp",
        requires_env=["OPENAI_API_KEY"],
    ),
    "gemini": AgentConfig(
        name="gemini",
        description="Google Gemini CLI via ACP",
        install_cmd=(
            f"{_NODE_INSTALL} && "
            "( command -v gemini >/dev/null 2>&1 || "
            "npm install -g @google/gemini-cli@latest >/dev/null 2>&1 ) && "
            "command -v gemini >/dev/null 2>&1"
        ),
        launch_cmd="gemini --acp",
        protocol="acp",
        requires_env=["GOOGLE_API_KEY"],
    ),
}


def get_agent(name: str) -> AgentConfig:
    """Get agent config by name. Raises KeyError if not found."""
    if name not in AGENTS:
        available = ", ".join(sorted(AGENTS.keys()))
        raise KeyError(f"Unknown agent: {name!r}. Available: {available}")
    return AGENTS[name]


def list_agents() -> list[AgentConfig]:
    """List all registered agents."""
    return list(AGENTS.values())
