"""Tests for agent spec parsing — protocol/agent-name with aliases."""

import pytest

from benchflow.agents.registry import (
    AGENT_ALIASES,
    AGENT_INSTALLERS,
    AGENT_LAUNCH,
    AGENTS,
    parse_agent_spec,
    resolve_agent,
    resolve_agent_key,
)


class TestParseAgentSpec:
    """Test parse_agent_spec() protocol/name parsing."""

    def test_bare_name(self):
        assert parse_agent_spec("claude-agent-acp") == ("acp", "claude-agent-acp")

    def test_explicit_acp(self):
        assert parse_agent_spec("acp/claude-agent-acp") == ("acp", "claude-agent-acp")

    def test_acpx_protocol(self):
        assert parse_agent_spec("acpx/claude") == ("acpx", "claude-agent-acp")

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

    def test_acpx_unknown_agent(self):
        assert parse_agent_spec("acpx/openhands") == ("acpx", "openhands")


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

    def test_resolve_acpx_agent(self):
        config = resolve_agent("acpx/claude")
        assert config.protocol == "acp"
        assert config.name == "acpx:claude-agent-acp"
        assert "acpx" in config.launch_cmd

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


class TestResolveAgentKey:
    """resolve_agent_key() must register acpx-wrapped configs so the
    Rollout/Evaluation path (which keys lookups by bare name against
    AGENTS/AGENT_LAUNCH/AGENT_INSTALLERS) resolves the acpx launch/install
    commands instead of the literal spec string.
    """

    def test_plain_agent_returns_canonical_name(self):
        assert resolve_agent_key("claude-agent-acp") == "claude-agent-acp"
        assert resolve_agent_key("claude") == "claude-agent-acp"
        assert resolve_agent_key("acp/codex-acp") == "codex-acp"

    def test_acpx_agent_registers_wrapped_config(self):
        key = resolve_agent_key("acpx/claude")
        # Stable runtime key, distinct from the bare spec.
        assert key == "acpx:claude-agent-acp"
        assert key != "acpx/claude"

        # The key is registered in all three lookup tables.
        assert key in AGENTS
        assert key in AGENT_LAUNCH
        assert key in AGENT_INSTALLERS

        # The registered commands are the acpx commands — NOT the literal spec.
        launch = AGENT_LAUNCH[key]
        install = AGENT_INSTALLERS[key]
        assert launch != "acpx/claude"
        assert "acpx" in launch and "--approve-all" in launch
        assert "acpx" in install
        # The underlying claude install command is preserved (chained).
        assert AGENTS["claude-agent-acp"].install_cmd in install

    def test_acpx_lookup_does_not_fall_back_to_literal_string(self):
        """Regression: AGENT_LAUNCH.get(<resolved key>) must not return the
        bogus literal 'acpx/claude' default.
        """
        key = resolve_agent_key("acpx/claude")
        assert AGENT_LAUNCH.get(key, key) != "acpx/claude"
        assert AGENT_INSTALLERS.get(key) is not None

    def test_acpx_key_round_trips_through_resolve_agent(self):
        """A resolved acpx runtime key must resolve back to the same config."""
        key = resolve_agent_key("acpx/claude")
        config = resolve_agent(key)
        assert config.name == key
        assert "acpx" in config.launch_cmd

    def test_unknown_agent_passes_through(self):
        assert resolve_agent_key("totally-unknown-agent") == "totally-unknown-agent"

    def test_idempotent(self):
        first = resolve_agent_key("acpx/codex")
        second = resolve_agent_key("acpx/codex")
        assert first == second
        # Re-resolving the runtime key itself is stable too.
        assert resolve_agent_key(first) == first


class TestNormalizeAgentName:
    """normalize_agent_name() is the Rollout/Evaluation chokepoint that turns
    user-facing specs into registered runtime keys.
    """

    def test_acpx_spec_normalizes_to_registered_key(self):
        from benchflow._utils.config import normalize_agent_name

        key = normalize_agent_name("acpx/claude")
        assert key == "acpx:claude-agent-acp"
        assert key in AGENTS
        assert "acpx" in AGENT_LAUNCH[key]

    def test_plain_spec_normalizes_to_canonical(self):
        from benchflow._utils.config import normalize_agent_name

        assert normalize_agent_name("claude") == "claude-agent-acp"
        assert normalize_agent_name("acp/codex-acp") == "codex-acp"
