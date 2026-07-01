"""Path-qualified agent specs — "<path>:<vendor>" resolution (Option A).

One vendor agent may be hosted via several adaptation paths (native ACP, the
AI-SDK harness, the Omnigent meta-harness). These lock the resolution contract:

* exact registry names and aliases ALWAYS win (full backwards compatibility,
  including ``acpx:`` runtime keys, which share the colon);
* ``"<path>:<vendor>"`` resolves the unique agent hosted by that path;
* a bare vendor id resolves iff exactly one agent carries it, and raises an
  error listing the qualified choices when several paths host it.
"""

from __future__ import annotations

import pytest

from benchflow.agents import registry
from benchflow.agents.registry import AgentConfig, resolve_agent


@pytest.fixture()
def _two_paths_one_vendor(monkeypatch):
    """Register the same vendor ('probe') via two paths + one unique vendor."""
    fake = {
        "probe-acp": AgentConfig(
            name="probe-acp",
            install_cmd="true",
            launch_cmd="true",
            vendor="probe",
            path="acp",
        ),
        "omnigent-probe": AgentConfig(
            name="omnigent-probe",
            install_cmd="true",
            launch_cmd="true",
            vendor="probe",
            path="omnigent",
        ),
        "solo-acp": AgentConfig(
            name="solo-acp",
            install_cmd="true",
            launch_cmd="true",
            vendor="solo",
            path="acp",
        ),
    }
    merged = {**registry.AGENTS, **fake}
    monkeypatch.setattr(registry, "AGENTS", merged)
    return fake


def test_qualified_spec_resolves_each_path(_two_paths_one_vendor):
    assert resolve_agent("acp:probe").name == "probe-acp"
    assert resolve_agent("omnigent:probe").name == "omnigent-probe"


def test_bare_vendor_unique_resolves(_two_paths_one_vendor):
    assert resolve_agent("solo").name == "solo-acp"


def test_bare_vendor_ambiguous_lists_qualified_choices(_two_paths_one_vendor):
    with pytest.raises(KeyError) as exc:
        resolve_agent("probe")
    msg = str(exc.value)
    assert "acp:probe" in msg and "omnigent:probe" in msg


def test_exact_name_beats_vendor_logic(_two_paths_one_vendor, monkeypatch):
    """A registered agent literally NAMED like a vendor id resolves verbatim —
    the vendor logic only fires for names that are not registered."""
    merged = {
        **registry.AGENTS,
        "probe": AgentConfig(name="probe", install_cmd="true", launch_cmd="true"),
    }
    monkeypatch.setattr(registry, "AGENTS", merged)
    assert resolve_agent("probe").name == "probe"  # no ambiguity error


def test_unknown_qualified_spec_still_errors(_two_paths_one_vendor):
    with pytest.raises(KeyError):
        resolve_agent("omnigent:does-not-exist")


def test_core_builtins_carry_metadata():
    """The core built-ins are backfilled — the marquee qualified names work.

    NB: asserted only for names no out-of-core plugin re-registers (an installed
    plugin that registers e.g. ``mimo`` without metadata overwrites core's entry
    — which is exactly why the agents-repo packages must declare vendor/path
    too; see the companion agents-repo change).
    """
    assert resolve_agent("acp:claude").name == "claude-agent-acp"
    assert resolve_agent("acp:codex").name == "codex-acp"
    assert resolve_agent("acp:harvey-lab").name == "harvey-lab-harness"
    assert resolve_agent("acp:openhands").name == "openhands"


def test_acpx_protocol_composes_with_qualified_spec(_two_paths_one_vendor):
    cfg = resolve_agent("acpx/acp:probe")
    assert cfg.name.startswith("acpx:")
