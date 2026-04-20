"""Tests for agent spec parsing — protocol/agent-name with aliases."""

import pytest

from benchflow.agents.registry import (
    AGENT_ALIASES,
    parse_agent_spec,
    resolve_agent,
)


class TestParseAgentSpec:
    """Test parse_agent_spec() protocol/name parsing."""

    def test_bare_name(self):
        assert parse_agent_spec("claude-agent-acp") == ("acp", "claude-agent-acp")

    def test_explicit_acp(self):
        assert parse_agent_spec("acp/claude-agent-acp") == ("acp", "claude-agent-acp")

    def test_harbor_protocol(self):
        assert parse_agent_spec("harbor/claude-code") == ("harbor", "claude-code")

    def test_alias_bare(self):
        assert parse_agent_spec("claude") == ("acp", "claude-agent-acp")

    def test_alias_with_protocol(self):
        assert parse_agent_spec("acp/claude") == ("acp", "claude-agent-acp")

    def test_alias_codex(self):
        assert parse_agent_spec("codex") == ("acp", "codex-acp")

    def test_alias_gemini(self):
        assert parse_agent_spec("gemini") == ("acp", "gemini")

    def test_unknown_name_passes_through(self):
        assert parse_agent_spec("my-custom-agent") == ("acp", "my-custom-agent")

    def test_harbor_unknown_agent(self):
        assert parse_agent_spec("harbor/openhands") == ("harbor", "openhands")


class TestResolveAgent:
    """Test resolve_agent() config lookup with fuzzy matching."""

    def test_resolve_registered_agent(self):
        config = resolve_agent("claude-agent-acp")
        assert config.name == "claude-agent-acp"
        assert config.protocol == "acp"

    def test_resolve_alias(self):
        config = resolve_agent("claude")
        assert config.name == "claude-agent-acp"

    def test_resolve_with_protocol(self):
        config = resolve_agent("acp/codex-acp")
        assert config.name == "codex-acp"

    def test_resolve_harbor_agent(self):
        config = resolve_agent("harbor/claude-code")
        assert config.protocol == "harbor"
        assert config.name == "claude-code"

    def test_resolve_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown agent"):
            resolve_agent("nonexistent-agent")

    def test_resolve_unknown_protocol_raises(self):
        with pytest.raises(KeyError, match="Unknown protocol"):
            resolve_agent("mcp/some-agent")

    def test_fuzzy_suggestion(self):
        with pytest.raises(KeyError, match="Did you mean"):
            resolve_agent("claude-acp")  # close to claude-agent-acp

    def test_all_aliases_resolve(self):
        """Every alias should resolve to a registered agent."""
        for alias, target in AGENT_ALIASES.items():
            config = resolve_agent(alias)
            assert config.name == target
