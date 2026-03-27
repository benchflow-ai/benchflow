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
class AgentConfig:
    """Configuration for a supported agent."""

    name: str
    install_cmd: str
    launch_cmd: str
    protocol: str = "acp"  # "acp" or "cli"
    requires_env: list[str] = field(default_factory=list)
    description: str = ""
    skill_paths: list[str] = field(default_factory=list)  # Where agent discovers skills


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
    ),
    "openclaw": AgentConfig(
        name="openclaw",
        description="OpenClaw agent via ACP shim (wraps openclaw agent --local)",
        skill_paths=["$HOME/.claude/skills", "$WORKSPACE/skills"],
        install_cmd=(
            f"{_NODE_INSTALL} && "
            "( command -v openclaw >/dev/null 2>&1 || "
            "npm install -g openclaw@latest >/dev/null 2>&1 ) && "
            "command -v openclaw >/dev/null 2>&1 && "
            # Ensure python3 for shim
            "( command -v python3 >/dev/null 2>&1 || "
            "(apt-get update -qq && apt-get install -y -qq python3 >/dev/null 2>&1) ) && "
            # Configure: auto-approve tools
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
        requires_env=["ANTHROPIC_API_KEY"],
    ),
    "openclaw-gemini": AgentConfig(
        name="openclaw-gemini",
        description="OpenClaw agent using Google Gemini API (via ACP shim)",
        skill_paths=["$HOME/.claude/skills", "$WORKSPACE/skills"],
        install_cmd=(
            f"{_NODE_INSTALL} && "
            "( command -v openclaw >/dev/null 2>&1 || "
            "npm install -g openclaw@latest >/dev/null 2>&1 ) && "
            "command -v openclaw >/dev/null 2>&1 && "
            "( command -v python3 >/dev/null 2>&1 || "
            "(apt-get update -qq && apt-get install -y -qq python3 >/dev/null 2>&1) ) && "
            # Configure: auto-approve tools + Gemini model
            "mkdir -p ~/.openclaw && "
            'echo \'{"version":1,"defaults":{"allow_all":true}}\''
            " > ~/.openclaw/exec-approvals.json && "
            # Write openclaw.json with Gemini config
            'cat > ~/.openclaw/openclaw.json <<\'CFGEOF\'\n'
            '{"agents":{"defaults":{"model":{"primary":"google/gemini-3.1-flash-lite-preview"}}}}\n'
            "CFGEOF\n"
            # Deploy ACP shim
            "cat > /usr/local/bin/openclaw-acp-shim <<'SHIMEOF'\n"
            + _OPENCLAW_SHIM +
            "\nSHIMEOF\n"
            "chmod +x /usr/local/bin/openclaw-acp-shim"
        ),
        launch_cmd="python3 /usr/local/bin/openclaw-acp-shim",
        protocol="acp",
        requires_env=["GEMINI_API_KEY"],
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


def register_agent(
    name: str,
    install_cmd: str,
    launch_cmd: str,
    *,
    protocol: str = "acp",
    requires_env: list[str] | None = None,
    description: str = "",
    skill_paths: list[str] | None = None,
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
    )
    AGENTS[name] = config
    # Update backwards-compat dicts
    from benchflow.sdk import AGENT_INSTALLERS, AGENT_LAUNCH
    AGENT_INSTALLERS[name] = install_cmd
    AGENT_LAUNCH[name] = launch_cmd
    return config
