"""Rollout integrates the Environment plane when a manifest is given.

Guards the vertical-slice wiring: RolloutConfig carries an optional
environment_manifest; Rollout provisions it, readiness-gates, and tears
it down. The full provision→readiness→teardown ordering against a real
container is exercised by tests/integration/test_clawsbench_slice.py.
"""

from benchflow.environment.manifest import EnvironmentManifest
from benchflow.rollout import RolloutConfig, _environment_manifest_metadata

_MANIFEST = EnvironmentManifest.model_validate_toml(
    '[environment]\nname = "x"\nimage = "x:latest"\n'
)


def test_rolloutconfig_accepts_environment_manifest():
    cfg = RolloutConfig(task_path="dummy", environment_manifest=_MANIFEST)
    assert cfg.environment_manifest is _MANIFEST


def test_rolloutconfig_environment_manifest_defaults_none():
    cfg = RolloutConfig(task_path="dummy")
    assert cfg.environment_manifest is None


def test_environment_manifest_metadata_is_config_artifact_safe():
    """Config artifacts should identify manifest-backed rollouts without secrets."""
    manifest = EnvironmentManifest.model_validate_toml(
        """
[environment]
name = "clawsbench"
base_image = "benchflow/clawsbench-base:latest"
owns_lifecycle = false

[[environment.services]]
name = "gmail"
command = "python -m claw_gmail.server"
port = 9001
"""
    )

    assert _environment_manifest_metadata(manifest) == {
        "name": "clawsbench",
        "image": None,
        "base_image": "benchflow/clawsbench-base:latest",
        "owns_lifecycle": False,
        "isolation": "per_task",
        "services": ["gmail"],
    }
