"""Tests for EnvironmentManifest parsing.

The two fixtures below are the two real stateful-multi-service benchmarks:
ClawsBench/smolclaws (framework-started services, image-based task
selection) and chi-bench (entrypoint-owned lifecycle, env-var task
selection). The manifest schema must be honest to both.
"""

import pytest

from benchflow.environment.manifest import EnvironmentManifest, load_manifest

# smolclaws: no service-starting entrypoint, seed data baked per-image.
CLAWS_TOML = """
[environment]
name           = "clawsbench"
base_image     = "kywch/smolclaws-base:latest"
owns_lifecycle = false
isolation      = "per_task"

[[environment.services]]
name    = "gmail"
command = "claw-gmail --db /data/gmail.db serve --host 0.0.0.0 --port 9001 --no-mcp"
port    = 9001

[[environment.services]]
name    = "gcal"
command = "claw-gcal --db /data/gcal.db serve --host 0.0.0.0 --port 9003 --no-mcp"
port    = 9003

[environment.task_selection]
mechanism = "image"

[environment.forward_env]
keys = ["ANTHROPIC_API_KEY"]
"""

# chi-bench: single image, entrypoint owns the service lifecycle.
CHI_TOML = """
[environment]
name           = "chi-bench"
image          = "chi-bench:latest"
ports          = [8020, 8023]
owns_lifecycle = true

[environment.task_selection]
mechanism   = "env_var"
key         = "CHI_BENCH_TASK_ID"
inject_into = "entrypoint"

[environment.readiness]
http        = ["http://localhost:8023/health"]
timeout_sec = 120
"""


def test_parses_clawsbench_manifest():
    m = EnvironmentManifest.model_validate_toml(CLAWS_TOML)
    assert m.name == "clawsbench"
    assert m.base_image == "kywch/smolclaws-base:latest"
    assert m.image is None
    assert m.owns_lifecycle is False
    assert len(m.services) == 2
    assert m.services[0].name == "gmail"
    assert m.services[0].port == 9001
    assert m.services[0].health_path == "/health"  # default
    assert m.task_selection.mechanism == "image"
    assert m.forward_env.keys == ["ANTHROPIC_API_KEY"]


def test_parses_chibench_manifest():
    m = EnvironmentManifest.model_validate_toml(CHI_TOML)
    assert m.name == "chi-bench"
    assert m.image == "chi-bench:latest"
    assert m.owns_lifecycle is True
    assert m.services == []
    assert m.ports == [8020, 8023]
    assert m.task_selection.mechanism == "env_var"
    assert m.task_selection.key == "CHI_BENCH_TASK_ID"
    assert m.readiness.http == ["http://localhost:8023/health"]


def test_all_ports_unions_declared_and_service_ports():
    m = EnvironmentManifest.model_validate_toml(CLAWS_TOML)
    assert m.all_ports == [9001, 9003]
    m2 = EnvironmentManifest.model_validate_toml(CHI_TOML)
    assert m2.all_ports == [8020, 8023]


def test_effective_http_derived_from_services_when_unset():
    m = EnvironmentManifest.model_validate_toml(CLAWS_TOML)
    assert m.effective_http == [
        "http://localhost:9001/health",
        "http://localhost:9003/health",
    ]


def test_effective_http_explicit_wins():
    m = EnvironmentManifest.model_validate_toml(CHI_TOML)
    assert m.effective_http == ["http://localhost:8023/health"]


def test_rejects_owns_lifecycle_false_without_services():
    with pytest.raises(ValueError, match="non-empty"):
        EnvironmentManifest.model_validate_toml(
            '[environment]\nname="x"\nbase_image="x"\nowns_lifecycle=false\n'
        )


def test_rejects_owns_lifecycle_true_with_services():
    bad = """
[environment]
name = "x"
image = "x:latest"
owns_lifecycle = true

[[environment.services]]
name = "s"
command = "s serve"
port = 1
"""
    with pytest.raises(ValueError, match="entrypoint"):
        EnvironmentManifest.model_validate_toml(bad)


def test_rejects_manifest_with_neither_image():
    with pytest.raises(ValueError, match="image"):
        EnvironmentManifest.model_validate_toml('[environment]\nname="x"\n')


def test_rejects_unknown_isolation():
    with pytest.raises(ValueError):
        EnvironmentManifest.model_validate_toml(
            '[environment]\nname="x"\nimage="x"\nisolation="bogus"\n'
        )


def test_rejects_unknown_key():
    with pytest.raises(ValueError):
        EnvironmentManifest.model_validate_toml(
            '[environment]\nname="x"\nimage="x"\nbogus_key=1\n'
        )


def test_rejects_missing_environment_table():
    with pytest.raises(ValueError, match="environment"):
        EnvironmentManifest.model_validate_toml('[other]\nname="x"\n')


def test_load_manifest_from_file(tmp_path):
    p = tmp_path / "environment.toml"
    p.write_text(CHI_TOML)
    m = load_manifest(p)
    assert m.name == "chi-bench"
