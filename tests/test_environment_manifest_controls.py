"""Manifest control points actually steer the sandbox (#395).

A RolloutConfig that carries an EnvironmentManifest must take effect:

1. ``image`` overrides ``task.config.environment.docker_image`` so the
   manifest — not the task's local Dockerfile — drives runtime image
   selection.
2. ``task_selection`` (mechanism=env_var, inject_into=entrypoint) binds
   the task id under the configured key so the image entrypoint and any
   subsequent ``sandbox.exec`` call observe it.
3. ``forward_env`` pulls the named host env vars into the sandbox so the
   environment can read declared host secrets/config.

Before this fix the manifest was accepted but ignored — the sandbox was
created from ``task.config.environment`` only, the env var never made it
into the container, and host-side ``forward_env`` was never resolved.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from benchflow.environment.manifest import (
    EnvironmentManifest,
    resolve_manifest_image,
    resolve_manifest_runtime_env,
)
from benchflow.sandbox.setup import _create_sandbox_environment
from benchflow.task import Task

# ── Manifests ────────────────────────────────────────────────────────────

_CHI_MANIFEST = EnvironmentManifest.model_validate_toml(
    """
[environment]
name           = "chi-bench"
image          = "manifest-image:latest"
ports          = [8023]
owns_lifecycle = true

[environment.task_selection]
mechanism   = "env_var"
key         = "TASK_ID"
inject_into = "entrypoint"

[environment.forward_env]
keys = ["MANIFEST_FWD_API_KEY", "MANIFEST_FWD_MISSING"]
"""
)

_CLAWS_MANIFEST = EnvironmentManifest.model_validate_toml(
    """
[environment]
name           = "clawsbench"
base_image     = "smolclaws-base:latest"
owns_lifecycle = false

[[environment.services]]
name    = "gmail"
command = "claw-gmail serve --port 9001"
port    = 9001
"""
)


# ── Fixtures ─────────────────────────────────────────────────────────────


def _write_task(root: Path, name: str = "demo-task") -> Path:
    """Materialize a minimal task tree with task.toml + Dockerfile."""
    task_dir = root / name
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Do it.\n")
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n'
        "[agent]\ntimeout_sec = 1\n"
        "[verifier]\ntimeout_sec = 1\n"
        '[environment]\ndocker_image = "task-toml-image:latest"\n'
    )
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    return task_dir


def _stub_rollout_paths(tmp_path: Path):
    """Minimal RolloutPaths stand-in — sandbox factories only read three dirs."""
    return type(
        "P",
        (),
        {
            "verifier_dir": tmp_path,
            "agent_dir": tmp_path,
            "artifacts_dir": tmp_path,
            "rollout_dir": tmp_path,
        },
    )()


# ── Unit-level: helper functions ─────────────────────────────────────────


def test_resolve_manifest_image_prefers_image_over_base_image():
    """Only a runnable ``image`` is returned; ``base_image`` alone yields None.

    ``base_image`` is what per-task images build FROM; it cannot be run on
    its own. Returning it would silently swap the runtime for a non-runnable
    image, which is worse than leaving the task path in effect.
    """
    assert resolve_manifest_image(_CHI_MANIFEST) == "manifest-image:latest"
    assert resolve_manifest_image(_CLAWS_MANIFEST) is None


def test_resolve_manifest_runtime_env_binds_task_id_and_forwards_present_host_vars():
    """Task selection key + forward_env keys are merged into one env dict.

    Forward_env keys that are unset on the host are silently skipped — the
    benchmark author can layer their own required-ness check upstream if
    the policy is strict.
    """
    host = {
        "MANIFEST_FWD_API_KEY": "secret-abc",
        # MANIFEST_FWD_MISSING intentionally absent
        "UNRELATED": "ignored",
    }
    env = resolve_manifest_runtime_env(_CHI_MANIFEST, task_id="task-42", host_env=host)
    assert env == {"TASK_ID": "task-42", "MANIFEST_FWD_API_KEY": "secret-abc"}


def test_resolve_manifest_runtime_env_skips_task_selection_when_inject_exec():
    """``inject_into = "exec"`` is the framework's later-layer concern.

    The setup-time helper only handles entrypoint-time injection; exec-time
    binding is a per-call decoration that lives elsewhere. The helper must
    not silently treat them as the same.
    """
    m = EnvironmentManifest.model_validate_toml(
        """
[environment]
name           = "x"
image          = "x:latest"
[environment.task_selection]
mechanism   = "env_var"
key         = "TASK_ID"
inject_into = "exec"
"""
    )
    assert resolve_manifest_runtime_env(m, task_id="task-1", host_env={}) == {}


# ── Sandbox-construction: the three control points end-to-end ───────────


def test_manifest_image_overrides_task_docker_image(tmp_path):
    """Control point 1: manifest ``image`` wins over task.toml docker_image."""
    task_dir = _write_task(tmp_path)
    task = Task(task_dir)
    assert task.config.environment.docker_image == "task-toml-image:latest"

    captured: dict = {}

    class _FakeSandbox:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("benchflow.sandbox.docker.DockerSandbox", _FakeSandbox):
        _create_sandbox_environment(
            "docker",
            task,
            task_dir,
            "rollout-1",
            _stub_rollout_paths(tmp_path),
            environment_manifest=_CHI_MANIFEST,
        )

    # The sandbox sees the manifest's image, not the task.toml's.
    assert captured["task_env_config"].docker_image == "manifest-image:latest"
    # And the task's config is not mutated — the override is per-rollout.
    assert task.config.environment.docker_image == "task-toml-image:latest"


def test_manifest_task_selection_lands_in_sandbox_persistent_env(tmp_path, monkeypatch):
    """Control point 2: task id binds under task_selection.key in the sandbox env.

    The persistent_env dict the helper builds is what Docker/Daytona/Modal
    forward into the container at compose-up + every ``exec`` call, so
    proving it appears there is proving the entrypoint sees it.
    """
    monkeypatch.delenv("MANIFEST_FWD_API_KEY", raising=False)
    task_dir = _write_task(tmp_path, name="task-xyz")
    task = Task(task_dir)

    captured: dict = {}

    class _FakeSandbox:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("benchflow.sandbox.docker.DockerSandbox", _FakeSandbox):
        _create_sandbox_environment(
            "docker",
            task,
            task_dir,
            "rollout-1",
            _stub_rollout_paths(tmp_path),
            environment_manifest=_CHI_MANIFEST,
        )

    # task_path.name is the task id; it must be bound under the manifest's key.
    assert captured["persistent_env"] == {"TASK_ID": "task-xyz"}


def test_manifest_forward_env_pulls_host_vars_into_sandbox(tmp_path, monkeypatch):
    """Control point 3: declared host env vars are forwarded into the sandbox."""
    monkeypatch.setenv("MANIFEST_FWD_API_KEY", "forwarded-value")
    monkeypatch.delenv("MANIFEST_FWD_MISSING", raising=False)
    task_dir = _write_task(tmp_path, name="task-42")
    task = Task(task_dir)

    captured: dict = {}

    class _FakeSandbox:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("benchflow.sandbox.docker.DockerSandbox", _FakeSandbox):
        _create_sandbox_environment(
            "docker",
            task,
            task_dir,
            "rollout-1",
            _stub_rollout_paths(tmp_path),
            environment_manifest=_CHI_MANIFEST,
        )

    persistent = captured["persistent_env"]
    assert persistent["MANIFEST_FWD_API_KEY"] == "forwarded-value"
    # Unset host vars are skipped — they don't appear as empty strings.
    assert "MANIFEST_FWD_MISSING" not in persistent
    # And task-selection still rides alongside.
    assert persistent["TASK_ID"] == "task-42"


# ── Negative: manifest=None preserves the legacy path ───────────────────


def test_no_manifest_means_no_persistent_env_and_no_image_override(tmp_path):
    """Without a manifest the sandbox is constructed exactly as before."""
    task_dir = _write_task(tmp_path)
    task = Task(task_dir)

    captured: dict = {}

    class _FakeSandbox:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("benchflow.sandbox.docker.DockerSandbox", _FakeSandbox):
        _create_sandbox_environment(
            "docker",
            task,
            task_dir,
            "rollout-1",
            _stub_rollout_paths(tmp_path),
            environment_manifest=None,
        )

    # task.toml's docker_image is untouched.
    assert captured["task_env_config"].docker_image == "task-toml-image:latest"
    # And no manifest-driven persistent_env overlay.
    assert captured["persistent_env"] is None


def test_base_image_only_manifest_does_not_override_image(tmp_path):
    """A manifest with only ``base_image`` leaves the task's image intact.

    ``base_image`` is a build-FROM marker, not a runnable target — overriding
    docker_image with it would silently break per-task images.
    """
    task_dir = _write_task(tmp_path)
    task = Task(task_dir)

    captured: dict = {}

    class _FakeSandbox:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("benchflow.sandbox.docker.DockerSandbox", _FakeSandbox):
        _create_sandbox_environment(
            "docker",
            task,
            task_dir,
            "rollout-1",
            _stub_rollout_paths(tmp_path),
            environment_manifest=_CLAWS_MANIFEST,
        )

    assert captured["task_env_config"].docker_image == "task-toml-image:latest"
