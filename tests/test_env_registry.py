"""Unit tests for the S-axis environment registry (benchflow._utils.env_registry)."""

from __future__ import annotations

import pytest

from benchflow._utils.env_registry import (
    EnvironmentRegistryError,
    looks_like_env_spec,
    resolve_environment,
)
from benchflow.environment.manifest import load_manifest


def _write_env(registry, name_version: str, base_image: str = "img:1"):
    p = registry / f"{name_version}.toml"
    p.write_text(f'[environment]\nname = "env0"\nbase_image = "{base_image}"\n')
    return p


def test_looks_like_env_spec_discriminates_spec_from_path():
    assert looks_like_env_spec("env0")
    assert looks_like_env_spec("env0@v2")
    assert not looks_like_env_spec("../_manifests/env0.toml")  # path → not a spec
    assert not looks_like_env_spec("a/b")
    assert not looks_like_env_spec("env0.toml")


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


def test_resolve_no_registry_configured_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("BENCHFLOW_ENV_REGISTRY", raising=False)
    with pytest.raises(EnvironmentRegistryError, match="no environment registry"):
        resolve_environment("env0")


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
