"""Regression coverage for commit 6429743f (science-harness feature contracts)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.agents.providers import find_provider
from benchflow.agents.registry import AGENTS

GRID = Path(__file__).parents[1] / "docs/agent-feature-grid.md"


@pytest.mark.parametrize(
    ("agent", "skill_path", "effort_id", "mcp_transport"),
    [
        ("openscience", "$HOME/.claude/skills", "", "native-config"),
        (
            "deepseek-harness",
            "$HOME/.agents/skills",
            "reasoning_effort",
            "acp",
        ),
    ],
)
def test_science_harness_grid_matches_registry(
    agent: str, skill_path: str, effort_id: str, mcp_transport: str
) -> None:
    cfg = AGENTS[agent]
    text = GRID.read_text(encoding="utf-8")

    assert cfg.protocol == "acp"
    assert cfg.api_protocol == ""
    assert cfg.skill_paths == [skill_path]
    assert cfg.task_mcp_transport == mcp_transport
    assert cfg.acp_effort_config_id == effort_id
    assert f"`{agent}`" in text
    assert skill_path in text


def test_grid_pins_truthful_protocol_and_recovery_boundaries() -> None:
    text = GRID.read_text(encoding="utf-8")

    required = [
        "OpenAI Chat Completions; Anthropic Messages",
        "Unsupported; fails closed",
        "Native config HTTP, SSE, and stdio",
        "ACP HTTP",
        "Conditional on selected model/provider capability",
        "`session/resume` preferred; `session/load` also advertised",
        "Static install/path contract only; live run availability-gated",
        "Deferred until runtime credentials exist",
    ]
    for phrase in required:
        assert phrase in text


def test_grid_preserves_explicit_anthropic_route_and_legacy_behavior() -> None:
    direct = find_provider("anthropic-direct/claude-sonnet-4-6")
    assert direct is not None
    assert direct[1].api_protocol == "anthropic-messages"
    assert find_provider("anthropic/claude-sonnet-4-6") is None


@pytest.mark.parametrize("agent", ["openscience", "deepseek-harness"])
def test_science_harness_install_contract_is_sandbox_portable(agent: str) -> None:
    cfg = AGENTS[agent]

    assert cfg.launch_cmd.startswith("/opt/benchflow/bin/")
    assert all(path.startswith("$HOME/") for path in cfg.skill_paths)
    assert all(not path.startswith("/") for path in cfg.home_dirs)
    assert "/opt/benchflow" in cfg.install_cmd
    assert "sudo " not in cfg.install_cmd
