"""Directory-scan registration: turn a tree of manifest.toml files into AGENTS
entries (design decision #7, the eve-style filesystem discovery).

The env-gated entry point ``register_env_manifest_agents`` is a no-op when
``BENCHFLOW_AGENTS_DIR`` is unset — so importing core is unchanged by default;
the dual-source registry only lights up when a developer opts in. Registration
is fail-loud on collision (an agent or alias that already exists is an ambiguous
source of truth, not a silent shadow) unless ``override=True``, and writes every
name-keyed registry map (AGENTS / AGENT_INSTALLERS / AGENT_LAUNCH / AGENT_ALIASES)
the install + rollout paths read from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.agents.manifest import (
    AgentManifestError,
    discover_manifests,
    load_agents_from_dir,
    register_env_manifest_agents,
    register_manifest_agents,
)
from benchflow.agents.registry import AgentConfig

_BODY = """contract_version = "1.0"
name = "{name}"
install_cmd = "echo install"
launch_cmd = "echo launch"
"""


def _put(root: Path, subdir: str, name: str, extra: str = "") -> Path:
    d = root / subdir
    d.mkdir(parents=True)
    (d / "manifest.toml").write_text(_BODY.format(name=name) + extra)
    return d


def _maps():
    """Fresh throwaway copies of the four registry maps for hermetic tests."""
    return {
        "agents": {},
        "aliases": {},
        "installers": {},
        "launch": {},
    }


def test_discover_finds_nested_manifests_only(tmp_path: Path):
    _put(tmp_path, "alpha", "alpha")
    _put(tmp_path, "beta", "beta")
    (tmp_path / "manifest.toml").write_text("ignored = true\n")  # root-level: skip
    (tmp_path / "gamma").mkdir()
    (tmp_path / "gamma" / "notes.txt").write_text("no manifest here")
    found = discover_manifests(tmp_path)
    assert [p.parent.name for p in found] == ["alpha", "beta"]  # sorted, nested only


def test_discover_missing_dir_returns_empty(tmp_path: Path):
    assert discover_manifests(tmp_path / "nope") == []


def test_load_dir_keys_by_agent_name_not_dirname(tmp_path: Path):
    # eve-style: the directory ("mimo-acp") may differ from the agent ("mimo").
    _put(tmp_path, "mimo-acp", "mimo")
    loaded = load_agents_from_dir(tmp_path)
    assert set(loaded) == {"mimo"}
    assert loaded["mimo"].config.name == "mimo"


def test_load_dir_rejects_duplicate_agent_name(tmp_path: Path):
    _put(tmp_path, "one", "dup")
    _put(tmp_path, "two", "dup")
    with pytest.raises(AgentManifestError, match="dup"):
        load_agents_from_dir(tmp_path)


def test_register_writes_every_name_keyed_map(tmp_path: Path):
    _put(tmp_path, "demo", "demo", extra='aliases = ["demo-code"]\n')
    m = _maps()
    register_manifest_agents(load_agents_from_dir(tmp_path), **m)
    assert isinstance(m["agents"]["demo"], AgentConfig)
    # The installer/launch projections the install + rollout paths read MUST be
    # populated, not just AGENTS — else a manifest agent is uninstallable.
    assert m["installers"]["demo"] == "echo install"
    assert m["launch"]["demo"] == "echo launch"
    assert m["aliases"] == {"demo-code": "demo"}


def test_register_fails_loud_on_name_collision(tmp_path: Path):
    _put(tmp_path, "demo", "demo")
    m = _maps()
    m["agents"]["demo"] = AgentConfig(name="demo", install_cmd="x", launch_cmd="y")
    with pytest.raises(AgentManifestError, match="demo"):
        register_manifest_agents(load_agents_from_dir(tmp_path), **m)
    # collision is detected before any mutation: the batch is all-or-nothing.
    assert m["agents"]["demo"].install_cmd == "x"
    assert m["installers"] == {}


def test_register_override_replaces(tmp_path: Path):
    _put(tmp_path, "demo", "demo")
    m = _maps()
    m["agents"]["demo"] = AgentConfig(name="demo", install_cmd="OLD", launch_cmd="y")
    register_manifest_agents(load_agents_from_dir(tmp_path), **m, override=True)
    assert m["agents"]["demo"].install_cmd == "echo install"
    assert m["installers"]["demo"] == "echo install"


def test_register_fails_loud_on_alias_collision(tmp_path: Path):
    _put(tmp_path, "demo", "demo", extra='aliases = ["taken"]\n')
    m = _maps()
    m["aliases"]["taken"] = "someone-else"
    with pytest.raises(AgentManifestError, match="taken"):
        register_manifest_agents(load_agents_from_dir(tmp_path), **m)


def test_env_entry_is_noop_when_unset(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("BENCHFLOW_AGENTS_DIR", raising=False)
    m = _maps()
    registered = register_env_manifest_agents(**m)
    assert registered == []
    # importing core stays behavior-preserving: nothing touched.
    assert all(d == {} for d in m.values())


def test_env_entry_registers_when_set(tmp_path: Path, monkeypatch):
    _put(tmp_path, "demo", "demo", extra='aliases = ["demo-code"]\n')
    _put(tmp_path, "beta", "beta")
    monkeypatch.setenv("BENCHFLOW_AGENTS_DIR", str(tmp_path))
    m = _maps()
    registered = register_env_manifest_agents(**m)
    assert registered == ["beta", "demo"]
    assert set(m["agents"]) == {"beta", "demo"}
    assert set(m["installers"]) == {"beta", "demo"}
    assert set(m["launch"]) == {"beta", "demo"}
    assert m["aliases"] == {"demo-code": "demo"}


def test_merge_shim_only_keeps_core_shim_fields(tmp_path: Path):
    # Additive/compatible: a manifest reproducing an existing core agent overrides
    # it, taking DATA fields from the manifest but the host-side _SHIM_ONLY fields
    # (which the data-only manifest can't carry) from the existing core entry — so
    # the merged config equals the original.
    _put(
        tmp_path, "demo", "demo", extra='aliases = ["demo-code"]\n'
    )  # manifest: install_cmd="echo install", shim defaults
    m = _maps()
    m["agents"]["demo"] = AgentConfig(
        name="demo",
        install_cmd="CORE-INSTALL",
        launch_cmd="CORE-LAUNCH",
        acp_model_config_id="model",  # a _SHIM_ONLY field core owns
    )
    m["aliases"]["demo-code"] = "demo"
    register_manifest_agents(load_agents_from_dir(tmp_path), **m, merge_shim_only=True)
    merged = m["agents"]["demo"]
    assert merged.install_cmd == "echo install"  # DATA field comes from the manifest
    assert merged.acp_model_config_id == "model"  # _SHIM_ONLY preserved from core
    assert m["installers"]["demo"] == "echo install"
    assert m["aliases"]["demo-code"] == "demo"


def test_merge_shim_only_adds_new_agent_without_core_entry(tmp_path: Path):
    # An agent not already in core is just added (nothing to merge).
    _put(tmp_path, "brand-new", "brand-new")
    m = _maps()
    register_manifest_agents(load_agents_from_dir(tmp_path), **m, merge_shim_only=True)
    assert m["agents"]["brand-new"].install_cmd == "echo install"


def test_merge_shim_only_rejects_alias_collision(tmp_path: Path):
    """Guards PR #825: additive manifest mode must not remap existing aliases."""
    _put(tmp_path, "brand-new", "brand-new", extra='aliases = ["taken"]\n')
    m = _maps()
    m["agents"]["core"] = AgentConfig(
        name="core",
        install_cmd="CORE-INSTALL",
        launch_cmd="CORE-LAUNCH",
    )
    m["aliases"]["taken"] = "core"

    with pytest.raises(AgentManifestError, match="taken"):
        register_manifest_agents(
            load_agents_from_dir(tmp_path), **m, merge_shim_only=True
        )

    assert "brand-new" not in m["agents"]
    assert m["aliases"] == {"taken": "core"}


def test_register_fails_loud_on_name_vs_existing_alias(tmp_path: Path):
    # Alias-first resolution in registry.py (name = AGENT_ALIASES.get(name, name))
    # means a manifest agent whose NAME equals an existing alias for a *different*
    # agent is silently shadowed (unreachable). That must fail loud, not pass
    # through — the asymmetric gap this guards (BLOCKER 9).
    _put(tmp_path, "shadowed", "shadowed")
    m = _maps()
    m["aliases"]["shadowed"] = "other-agent"  # name 'shadowed' already maps elsewhere
    with pytest.raises(AgentManifestError, match="shadowed"):
        register_manifest_agents(load_agents_from_dir(tmp_path), **m)
    # all-or-nothing: a rejected batch leaves every map untouched.
    assert "shadowed" not in m["agents"]
    assert m["installers"] == {}
    assert m["aliases"] == {"shadowed": "other-agent"}


def test_register_fails_loud_on_same_batch_name_alias_collision(tmp_path: Path):
    # Same batch: 'beta' declares an alias 'alpha' equal to another incoming
    # agent's NAME. Registering both would make 'alpha' resolve to 'beta'
    # (alias-first), silently shadowing the real 'alpha' agent.
    _put(tmp_path, "alpha-dir", "alpha")
    _put(tmp_path, "beta-dir", "beta", extra='aliases = ["alpha"]\n')
    m = _maps()
    with pytest.raises(AgentManifestError, match="alpha"):
        register_manifest_agents(load_agents_from_dir(tmp_path), **m)
    # collision detected before any mutation: nothing is registered.
    assert m["agents"] == {}
    assert m["installers"] == {}


def test_discovery_walks_three_category_paths(tmp_path):
    """A manifest dropped under each of the 3 category paths (acp/, ai-sdk/,
    omnigent/) auto-registers. Discovery is recursive (rglob), so the eve-style
    ``acp/<agent>/manifest.toml`` layout works exactly like a flat
    ``<agent>/manifest.toml`` — adding a new agent is "drop it in the right path".
    """
    from benchflow.agents.manifest import load_agents_from_dir

    def write(category: str, agent: str) -> None:
        d = tmp_path / category / agent
        d.mkdir(parents=True)
        (d / "manifest.toml").write_text(
            f'contract_version = "1.0"\nname = "{agent}"\n'
            'install_cmd = "echo install"\nlaunch_cmd = "echo launch"\n'
        )

    write("acp", "cat-acp-agent")
    write("ai-sdk", "cat-aisdk-agent")
    write("omnigent", "cat-omnigent-agent")

    loaded = load_agents_from_dir(tmp_path)
    assert set(loaded) == {"cat-acp-agent", "cat-aisdk-agent", "cat-omnigent-agent"}
