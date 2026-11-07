"""The native harnesses BenchFlow ships, keyed by the agent they belong to.

``harness="native"`` is a run option of an existing agent entry: Claude Code
(``claude-agent-acp``) and Codex (``codex-acp``) run the same agent — same
registry entry, credentials, skills, model routing — through their own CLI
instead of the ACP adapter. An agent without an entry here has no native
harness, and asking for one fails before the sandbox starts.
"""

from __future__ import annotations

from collections.abc import Mapping

from benchflow.agents.protocol import AgentCapabilities
from benchflow.agents.registry import (
    CLAUDE_CODE_EXECUTABLE_PATH,
    CODEX_CLI_EXECUTABLE_PATH,
    codex_cli_install_cmd,
    pinned_npm_package,
)
from benchflow.native_harness import claude_code, codex
from benchflow.native_harness.spec import (
    HARNESS_ACP,
    HARNESS_NATIVE,
    HARNESSES,
    NativeHarness,
    NativeLaunch,
    NativeParser,
    NativeTurn,
)

# Multi-turn by resuming the CLI's session; no permission channel (the CLI
# runs with its approvals off), and token ids come from the gateway, not the
# agent, as for the ACP adapters.
_NATIVE_CAPABILITIES = AgentCapabilities(
    protocol="native-cli", nudges=True, ask_user=False, token_logprobs=False
)


def _package(name: str) -> str:
    package, version = pinned_npm_package(name)
    return f"{package}@{version}"


def _claude_launch(turn: NativeTurn, env: Mapping[str, str]) -> NativeLaunch:
    del env
    return claude_code.claude_code_launch(turn)


def _claude_parser(cwd: str, turn: int) -> NativeParser:
    del turn
    return claude_code.ClaudeCodeParser(cwd)


def _codex_launch(turn: NativeTurn, env: Mapping[str, str]) -> NativeLaunch:
    return codex.codex_launch(turn, codex.codex_config_from_env(dict(env)))


def _codex_parser(cwd: str, turn: int) -> NativeParser:
    return codex.CodexExecParser(cwd, turn=turn)


CLAUDE_CODE = NativeHarness(
    agent="claude-agent-acp",
    cli=claude_code.CLI,
    # The CLI claude-agent-acp's install already brings and pins; the adapter
    # runs this same binary (CLAUDE_CODE_EXECUTABLE).
    executable=CLAUDE_CODE_EXECUTABLE_PATH,
    package=_package("claude-code"),
    version_output=f"{pinned_npm_package('claude-code')[1]} (Claude Code)",
    install_cmd="",
    build_launch=_claude_launch,
    new_parser=_claude_parser,
    capabilities=_NATIVE_CAPABILITIES,
    efforts=claude_code.CLAUDE_CODE_EFFORTS,
    owns_model_via_env="ANTHROPIC_MODEL",
    accepts_session_id=True,
)

CODEX = NativeHarness(
    agent="codex-acp",
    cli=codex.CLI,
    executable=CODEX_CLI_EXECUTABLE_PATH,
    package=_package("codex"),
    version_output=f"codex-cli {pinned_npm_package('codex')[1]}",
    install_cmd=codex_cli_install_cmd(),
    build_launch=_codex_launch,
    new_parser=_codex_parser,
    capabilities=_NATIVE_CAPABILITIES,
    efforts=codex.CODEX_EFFORTS,
)

NATIVE_HARNESSES: dict[str, NativeHarness] = {
    harness.agent: harness for harness in (CLAUDE_CODE, CODEX)
}


def normalize_harness(value: object) -> str:
    """``"acp"`` or ``"native"``; None and empty mean ``"acp"``."""
    if value is None:
        return HARNESS_ACP
    if not isinstance(value, str):
        raise ValueError("harness must be a string: acp or native")
    normalized = value.strip().lower()
    if not normalized:
        return HARNESS_ACP
    if normalized not in HARNESSES:
        raise ValueError(f"harness must be one of: {', '.join(HARNESSES)}")
    return normalized


def native_harness_for(agent: str) -> NativeHarness:
    """The native harness of ``agent``; ValueError names the agents that have one."""
    harness = NATIVE_HARNESSES.get(agent)
    if harness is None:
        supported = ", ".join(sorted(NATIVE_HARNESSES))
        raise ValueError(
            f"agent {agent!r} has no native harness (harness='native' supports "
            f"{supported}); run it with harness='acp'"
        )
    return harness


def check_harness(harness: str, agents: list[str]) -> None:
    """Refuse a native run whose agents lack a native harness.

    ``agents`` are the run's agent names; scripted runners (oracle, nop) and
    the task runtime launch no agent and pass.
    """
    if harness != HARNESS_NATIVE:
        return
    for agent in agents:
        if agent in ("oracle", "nop", "task-runtime"):
            continue
        native_harness_for(agent)


__all__ = [
    "CLAUDE_CODE",
    "CODEX",
    "HARNESSES",
    "HARNESS_ACP",
    "HARNESS_NATIVE",
    "NATIVE_HARNESSES",
    "check_harness",
    "native_harness_for",
    "normalize_harness",
]
