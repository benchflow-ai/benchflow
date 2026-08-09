"""Unit tests for the S-axis environment registry (benchflow._utils.env_registry)."""

from __future__ import annotations

import pytest

from benchflow._utils.env_registry import (
    EnvironmentRegistryError,
    looks_like_env_spec,
    resolve_environment,
    resolve_state,
)
from benchflow.environment.manifest import load_manifest


def _write_env(registry, name_version: str, base_image: str = "img:1"):
    p = registry / f"{name_version}.toml"
    p.write_text(f'[environment]\nname = "env0"\nbase_image = "{base_image}"\n')
    return p


def _write_env_yaml(registry, name_version: str, base_image: str = "img:1"):
    p = registry / f"{name_version}.yaml"
    p.write_text(f"environment:\n  name: env0\n  base_image: {base_image}\n")
    return p


def test_looks_like_env_spec_discriminates_spec_from_path():
    assert looks_like_env_spec("env0")
    assert looks_like_env_spec("env0@v2")
    assert not looks_like_env_spec("../_manifests/env0.toml")  # path → not a spec
    assert not looks_like_env_spec("a/b")
    assert not looks_like_env_spec("env0.toml")
    assert not looks_like_env_spec("env0.yaml")  # yaml path → not a spec
    assert not looks_like_env_spec("env0.yml")


def test_resolve_pinned_version(tmp_path):
    _write_env(tmp_path, "env0@v1")
    r = resolve_environment("env0@v1", registry=tmp_path)
    assert (r.name, r.version) == ("env0", "v1")
    assert r.manifest_path.name == "env0@v1.toml"
    assert r.env_hash.startswith("sha256:")
    assert r.spec == "env0@v1"


def test_resolve_bare_name_prefers_default_file(tmp_path):
    _write_env(tmp_path, "env0")  # env0.toml is the "default"
    _write_env(tmp_path, "env0@v1")
    assert resolve_environment("env0", registry=tmp_path).version == "default"


def test_resolve_bare_name_falls_back_to_newest(tmp_path):
    _write_env(tmp_path, "env0@v1")
    _write_env(tmp_path, "env0@v2")
    assert resolve_environment("env0", registry=tmp_path).version == "v2"


def test_resolve_content_addressed(tmp_path):
    _write_env(tmp_path, "env0@v1", base_image="img:1")
    _write_env(tmp_path, "env0@v2", base_image="img:2")
    h1 = resolve_environment("env0@v1", registry=tmp_path).env_hash
    h2 = resolve_environment("env0@v2", registry=tmp_path).env_hash
    assert h1 != h2


def test_resolve_missing_version_errors(tmp_path):
    _write_env(tmp_path, "env0@v1")
    with pytest.raises(EnvironmentRegistryError, match="not found"):
        resolve_environment("env0@v9", registry=tmp_path)


def test_resolve_unknown_name_errors(tmp_path):
    with pytest.raises(EnvironmentRegistryError, match="no versions"):
        resolve_environment("nope", registry=tmp_path)


def test_resolve_invalid_spec_errors(tmp_path):
    with pytest.raises(EnvironmentRegistryError, match="invalid environment spec"):
        resolve_environment("bad/spec@x", registry=tmp_path)


def test_resolve_no_registry_available_errors(tmp_path, monkeypatch):
    """Env var unset AND built-in registry missing → actionable error."""
    import benchflow._utils.env_registry as env_registry

    monkeypatch.delenv("BENCHFLOW_ENV_REGISTRY", raising=False)
    monkeypatch.setattr(env_registry, "_builtin_registry_dir", lambda: None)
    with pytest.raises(EnvironmentRegistryError, match="no environment registry"):
        resolve_environment("env0")


# ---- built-in registry (shipped inside the wheel) ---------------------------


def test_builtin_registry_resolves_env0_prod_when_env_unset(monkeypatch):
    """The acceptance contract: a bare pip install resolves ``env0@prod`` with
    no ``$BENCHFLOW_ENV_REGISTRY`` set — the pins ship inside the package."""
    monkeypatch.delenv("BENCHFLOW_ENV_REGISTRY", raising=False)
    r = resolve_environment("env0@prod")
    assert (r.name, r.version) == ("env0", "prod")
    assert r.manifest_path.name == "env0@prod.toml"
    assert r.env_hash.startswith("sha256:")


def test_builtin_registry_resolves_env0_outage_when_env_unset(monkeypatch):
    monkeypatch.delenv("BENCHFLOW_ENV_REGISTRY", raising=False)
    r = resolve_environment("env0@outage")
    assert (r.name, r.version) == ("env0", "outage")


def test_builtin_registry_manifests_are_importlib_resources(monkeypatch):
    """The pins are package data: importlib.resources must find them (that is
    what makes them ship in the wheel) and they must parse as manifests."""
    import importlib.resources

    root = importlib.resources.files("benchflow.environment") / "_registry"
    names = {entry.name for entry in root.iterdir()}
    assert {"env0@prod.toml", "env0@outage.toml"} <= names

    monkeypatch.delenv("BENCHFLOW_ENV_REGISTRY", raising=False)
    m = load_manifest("env0@prod")
    assert {s.name for s in m.services} >= {"mock-gmail", "mock-slack"}
    outage = load_manifest("env0@outage")
    assert "mock-gmail" not in {s.name for s in outage.services}


def test_env_var_registry_wins_over_builtin(tmp_path, monkeypatch):
    """When set, ``$BENCHFLOW_ENV_REGISTRY`` is the registry — the same spec
    resolves from it, not from the built-in pins."""
    _write_env(tmp_path, "env0@prod", base_image="img:override")
    monkeypatch.setenv("BENCHFLOW_ENV_REGISTRY", str(tmp_path))
    r = resolve_environment("env0@prod")
    assert r.manifest_path.parent == tmp_path
    assert load_manifest("env0@prod").base_image == "img:override"


def test_env_var_registry_wins_entirely_no_builtin_fallback(tmp_path, monkeypatch):
    """A name the env-var registry lacks does NOT fall back to the built-in."""
    monkeypatch.setenv("BENCHFLOW_ENV_REGISTRY", str(tmp_path))  # empty dir
    with pytest.raises(
        EnvironmentRegistryError, match=r"not found in .*available: none"
    ):
        resolve_environment("env0@prod")


def test_explicit_registry_argument_beats_set_env_var(tmp_path, monkeypatch):
    """Resolution order: an explicit ``registry=`` beats a SET env var."""
    env_dir = tmp_path / "from-env"
    arg_dir = tmp_path / "from-arg"
    env_dir.mkdir()
    arg_dir.mkdir()
    _write_env(env_dir, "env0@v1", base_image="img:env")
    _write_env(arg_dir, "env0@v1", base_image="img:arg")
    monkeypatch.setenv("BENCHFLOW_ENV_REGISTRY", str(env_dir))
    r = resolve_environment("env0@v1", registry=arg_dir)
    assert r.manifest_path.parent == arg_dir


def test_empty_env_var_counts_as_unset(monkeypatch):
    """``BENCHFLOW_ENV_REGISTRY=""`` resolves the built-in registry, exactly
    like an unset var (empty counts as unset, pinned by the docstring)."""
    monkeypatch.setenv("BENCHFLOW_ENV_REGISTRY", "")
    r = resolve_environment("env0@prod")
    assert (r.name, r.version) == ("env0", "prod")


def test_env_var_not_a_directory_error_names_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_ENV_REGISTRY", str(tmp_path / "nope"))
    with pytest.raises(
        EnvironmentRegistryError,
        match=r"from \$BENCHFLOW_ENV_REGISTRY.*is not a directory",
    ):
        resolve_environment("env0@prod")


def test_registry_argument_not_a_directory_error_names_the_argument(tmp_path):
    with pytest.raises(
        EnvironmentRegistryError,
        match=r"from the registry= argument.*is not a directory",
    ):
        resolve_environment("env0@prod", registry=tmp_path / "nope")


def test_builtin_unknown_name_error_lists_available(monkeypatch):
    monkeypatch.delenv("BENCHFLOW_ENV_REGISTRY", raising=False)
    with pytest.raises(
        EnvironmentRegistryError, match=r"available:.*env0@outage.*env0@prod"
    ):
        resolve_environment("nope")


def test_builtin_resolution_logs_sha256(monkeypatch, caplog):
    """Content-addressed provenance is preserved: resolving a built-in pin
    logs the manifest's sha256 exactly like a ``$BENCHFLOW_ENV_REGISTRY`` one."""
    import logging

    monkeypatch.delenv("BENCHFLOW_ENV_REGISTRY", raising=False)
    with caplog.at_level(logging.INFO, logger="benchflow.environment.manifest"):
        load_manifest("env0@prod")
    assert any("sha256:" in rec.getMessage() for rec in caplog.records)


# ---- load_manifest dispatch (spec vs file) --------------------------------


def test_load_manifest_resolves_spec_via_registry(tmp_path, monkeypatch):
    _write_env(tmp_path, "env0@v1", base_image="img:42")
    monkeypatch.setenv("BENCHFLOW_ENV_REGISTRY", str(tmp_path))
    m = load_manifest("env0@v1")
    assert m.base_image == "img:42"


def test_load_manifest_still_loads_a_real_file(tmp_path):
    p = _write_env(tmp_path, "plain", base_image="img:file")
    m = load_manifest(p)  # real path → loaded directly, no registry needed
    assert m.base_image == "img:file"


# ---- YAML manifests (canonical) alongside TOML (back-compat) ---------------


def test_resolve_yaml_manifest(tmp_path):
    _write_env_yaml(tmp_path, "env0@v1", base_image="img:yaml")
    r = resolve_environment("env0@v1", registry=tmp_path)
    assert r.manifest_path.name == "env0@v1.yaml"
    assert r.version == "v1"


def test_resolve_prefers_toml_for_back_compat_when_both_exist(tmp_path):
    _write_env(tmp_path, "env0@v1", base_image="img:toml")
    _write_env_yaml(tmp_path, "env0@v1", base_image="img:yaml")
    r = resolve_environment("env0@v1", registry=tmp_path)
    assert r.manifest_path.suffix == ".toml"


def test_load_manifest_yaml_spec_via_registry(tmp_path, monkeypatch):
    _write_env_yaml(tmp_path, "env0@v2", base_image="img:y")
    monkeypatch.setenv("BENCHFLOW_ENV_REGISTRY", str(tmp_path))
    assert load_manifest("env0@v2").base_image == "img:y"


def test_load_manifest_yaml_file_path(tmp_path):
    p = _write_env_yaml(tmp_path, "plain", base_image="img:yfile")
    assert load_manifest(p).base_image == "img:yfile"


# ---- resolve_state — the --state axis (inline JSON + tool subset) ----------


def _write_multi_service_env(registry, name="env0"):
    (registry / f"{name}.toml").write_text(
        '[environment]\nname = "env-0"\nbase_image = "img"\nowns_lifecycle = false\n'
        '[[environment.services]]\nname = "gmail"\ncommand = "gmail serve"\nport = 9001\n'
        '[[environment.services]]\nname = "slack"\ncommand = "slack serve"\nport = 9002\n'
        '[[environment.services]]\nname = "gcal"\ncommand = "gcal serve"\nport = 9003\n'
    )


def test_resolve_state_inline_json_filters_to_tool_subset(tmp_path):
    _write_multi_service_env(tmp_path)
    import os

    os.environ["BENCHFLOW_ENV_REGISTRY"] = str(tmp_path)
    try:
        m = resolve_state('{"name":"env0","tools":["gmail","gcal"]}')
        assert sorted(s.name for s in m.services) == ["gcal", "gmail"]  # slack dropped
    finally:
        del os.environ["BENCHFLOW_ENV_REGISTRY"]


def test_resolve_state_missing_tool_errors(tmp_path):
    _write_multi_service_env(tmp_path)
    import os

    os.environ["BENCHFLOW_ENV_REGISTRY"] = str(tmp_path)
    try:
        with pytest.raises(EnvironmentRegistryError, match="not in environment"):
            resolve_state('{"name":"env0","tools":["nope"]}')
    finally:
        del os.environ["BENCHFLOW_ENV_REGISTRY"]


def test_resolve_state_ref_form_no_filter(tmp_path):
    _write_env(tmp_path, "env0@v1", base_image="img:ref")
    import os

    os.environ["BENCHFLOW_ENV_REGISTRY"] = str(tmp_path)
    try:
        assert resolve_state("env0@v1").base_image == "img:ref"
    finally:
        del os.environ["BENCHFLOW_ENV_REGISTRY"]


def test_resolve_state_bad_json_errors():
    with pytest.raises(EnvironmentRegistryError, match=r"must be a mapping"):
        resolve_state('{"tools":["gmail"]}')  # no name (mapping without "name")
