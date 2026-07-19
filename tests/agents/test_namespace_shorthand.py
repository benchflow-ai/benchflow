"""Namespace-shorthand agent specs — "<ns>:<id>" (design: #876).

Harbor-style colon prefixes over the existing flat names: the native/ACP path
owns bare names, other paths use their "<path>-" prefix, and the shorthand is a
deterministic candidate mapping onto those names (no new config fields).
Exact registered names and aliases always win — including ``acpx:`` runtime
keys, which share the colon and are exact-checked first.
"""

from __future__ import annotations

import pytest

from benchflow.agents import registry
from benchflow.agents.registry import AgentConfig, resolve_agent


def test_acp_shorthand_bare_name():
    assert resolve_agent("acp:mimo").name == "mimo"


def test_acp_shorthand_suffix_form():
    # no agent literally named "pi"; the -acp suffix candidate matches.
    assert resolve_agent("acp:pi").name == "pi-acp"


def test_acp_shorthand_via_alias():
    # "claude" is an alias for claude-agent-acp; candidates go through aliases.
    assert resolve_agent("acp:claude").name == "claude-agent-acp"
    assert resolve_agent("acp:codex").name == "codex-acp"


def test_prefix_namespaces(monkeypatch):
    fake = {
        "omnigent-probe": AgentConfig(
            name="omnigent-probe", install_cmd="true", launch_cmd="true"
        ),
        "ai-sdk-probe": AgentConfig(
            name="ai-sdk-probe", install_cmd="true", launch_cmd="true"
        ),
    }
    monkeypatch.setattr(registry, "AGENTS", {**registry.AGENTS, **fake})
    assert resolve_agent("omnigent:probe").name == "omnigent-probe"
    assert resolve_agent("ai-sdk:probe").name == "ai-sdk-probe"


def test_prefix_namespace_alias_candidate(monkeypatch):
    """A prefix candidate that is only an ALIAS still resolves (e.g. the
    historical ai-sdk-harness kept as an alias after a canonical rename)."""
    fake = {
        "some-canonical": AgentConfig(
            name="some-canonical", install_cmd="true", launch_cmd="true"
        ),
    }
    monkeypatch.setattr(registry, "AGENTS", {**registry.AGENTS, **fake})
    monkeypatch.setattr(
        registry,
        "AGENT_ALIASES",
        {**registry.AGENT_ALIASES, "ai-sdk-legacy": "some-canonical"},
    )
    assert resolve_agent("ai-sdk:legacy").name == "some-canonical"


def test_unknown_namespace_still_errors():
    with pytest.raises(KeyError):
        resolve_agent("zzz:mimo")


def test_unknown_id_in_namespace_errors():
    with pytest.raises(KeyError):
        resolve_agent("omnigent:does-not-exist")


def test_exact_names_and_bare_specs_unchanged():
    assert resolve_agent("mimo").name == "mimo"
    assert resolve_agent("pi-acp").name == "pi-acp"
    assert resolve_agent("pi").name == "pi-acp"  # existing alias


def test_acpx_protocol_composes_with_shorthand():
    cfg = resolve_agent("acpx/acp:pi")
    assert cfg.name.startswith("acpx:")
