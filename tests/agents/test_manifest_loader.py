"""The consumer half of the agent-decoupling contract: parse one manifest.toml
into the AgentConfig the registry already speaks (design decision #4).

These pin the loader's *consumer* contract — distinct from the agents-repo JSON
Schema, which is the stricter *author* contract. The loader enforces only the
four required keys, the contract-major gate, and the acp-only protocol gate, then
maps declared fields onto AgentConfig and ignores unknown keys (so a v1.x manifest
carrying a field added in a later minor still loads here). The partition test is
the drift guard: every AgentConfig field must be either declarable (_FIELD_MAP)
or deliberately shim-owned (_SHIM_ONLY) — a new field can never be silently lost.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from benchflow.agents.manifest import (
    _FIELD_MAP,
    _SHIM_ONLY,
    AgentManifestError,
    LoadedManifest,
    load_agent_manifest,
)
from benchflow.agents.registry import AgentConfig

_MINIMAL = """contract_version = "1.0"
name = "demo"
install_cmd = "echo install"
launch_cmd = "echo launch"
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "manifest.toml"
    p.write_text(body)
    return p


def test_loads_minimal_manifest(tmp_path: Path):
    loaded = load_agent_manifest(_write(tmp_path, _MINIMAL))
    assert isinstance(loaded, LoadedManifest)
    assert loaded.config.name == "demo"
    assert loaded.config.install_cmd == "echo install"
    assert loaded.config.launch_cmd == "echo launch"
    assert loaded.aliases == ()


def test_maps_all_declared_data_fields(tmp_path: Path):
    body = _MINIMAL + (
        'description = "a demo agent"\n'
        'protocol = "acp"\n'
        'api_protocol = "openai-completions"\n'
        'acp_model_format = "provider/model"\n'
        "supports_acp_set_model = false\n"
        'requires_env = ["OPENAI_API_KEY"]\n'
        "install_timeout = 1200\n"
        'default_model = "gpt-x"\n'
        'skill_paths = ["$HOME/.demo/skills"]\n'
        'home_dirs = [".demo"]\n'
        'aliases = ["demo-code", "demo2"]\n'
        "[env_mapping]\n"
        'BENCHFLOW_PROVIDER_API_KEY = "OPENAI_API_KEY"\n'
    )
    c = load_agent_manifest(_write(tmp_path, body)).config
    assert c.description == "a demo agent"
    assert c.api_protocol == "openai-completions"
    assert c.acp_model_format == "provider/model"
    assert c.supports_acp_set_model is False
    assert c.requires_env == ["OPENAI_API_KEY"]
    assert c.install_timeout == 1200
    assert c.default_model == "gpt-x"
    assert c.skill_paths == ["$HOME/.demo/skills"]
    assert c.home_dirs == [".demo"]
    assert c.env_mapping == {"BENCHFLOW_PROVIDER_API_KEY": "OPENAI_API_KEY"}


def test_aliases_parsed_separately_from_config(tmp_path: Path):
    body = _MINIMAL + 'aliases = ["demo-code", "demo2"]\n'
    assert load_agent_manifest(_write(tmp_path, body)).aliases == ("demo-code", "demo2")


@pytest.mark.parametrize(
    "missing", ["contract_version", "name", "install_cmd", "launch_cmd"]
)
def test_missing_required_key_is_loud(tmp_path: Path, missing: str):
    body = "\n".join(
        line for line in _MINIMAL.splitlines() if not line.startswith(missing + " ")
    )
    with pytest.raises(AgentManifestError, match=missing):
        load_agent_manifest(_write(tmp_path, body))


def test_unreadable_or_malformed_toml_is_loud(tmp_path: Path):
    with pytest.raises(AgentManifestError):
        load_agent_manifest(_write(tmp_path, "this is = = not toml"))
    with pytest.raises(AgentManifestError):
        load_agent_manifest(tmp_path / "does-not-exist.toml")


def test_wrong_contract_major_is_rejected(tmp_path: Path):
    body = _MINIMAL.replace('contract_version = "1.0"', 'contract_version = "2.0"')
    with pytest.raises(AgentManifestError, match="major"):
        load_agent_manifest(_write(tmp_path, body))


def test_malformed_contract_version_is_rejected(tmp_path: Path):
    body = _MINIMAL.replace('contract_version = "1.0"', 'contract_version = "v1"')
    with pytest.raises(AgentManifestError, match="MAJOR"):
        load_agent_manifest(_write(tmp_path, body))


def test_higher_minor_is_accepted_additive(tmp_path: Path):
    # SemVer: a minor bump within major 1 is a backward-compatible addition.
    body = _MINIMAL.replace('contract_version = "1.0"', 'contract_version = "1.7.3"')
    assert load_agent_manifest(_write(tmp_path, body)).config.name == "demo"


def test_unknown_keys_are_ignored(tmp_path: Path):
    # Forward compatibility: a field a later minor adds must not crash this loader.
    body = _MINIMAL + 'some_future_field = "whatever"\n'
    assert load_agent_manifest(_write(tmp_path, body)).config.name == "demo"


def test_rejects_non_acp_protocol(tmp_path: Path):
    # The contract is acp-only; non-ACP platforms ship an ACP-over-stdio shim.
    body = _MINIMAL + 'protocol = "session-factory"\n'
    with pytest.raises(AgentManifestError, match="acp"):
        load_agent_manifest(_write(tmp_path, body))


def test_install_timeout_coerced_to_int(tmp_path: Path):
    body = _MINIMAL + "install_timeout = 900\n"
    c = load_agent_manifest(_write(tmp_path, body)).config
    assert c.install_timeout == 900
    assert isinstance(c.install_timeout, int)


def test_field_map_and_shim_only_partition_agentconfig():
    """Drift guard: every AgentConfig field is either declarable in a manifest
    (_FIELD_MAP) or deliberately shim-owned (_SHIM_ONLY) — never neither (a
    silently dropped field) and never both (an ambiguous owner)."""
    declarable = set(_FIELD_MAP.values())
    shim = set(_SHIM_ONLY)
    all_fields = {f.name for f in fields(AgentConfig)}
    assert declarable & shim == set(), "a field is both declarable and shim-only"
    assert declarable | shim == all_fields, (
        "AgentConfig fields not partitioned: "
        f"unowned={all_fields - declarable - shim}, "
        f"stale={(declarable | shim) - all_fields}"
    )
