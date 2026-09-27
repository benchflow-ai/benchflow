"""Original-image recovery regressions for GH #1136."""

import asyncio
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.rollout import RolloutConfig
from benchflow.rollout._setup import _start_env_and_upload
from benchflow.rollout._verifier_recovery import (
    capture_original_docker_baseline,
    recovery_ineligible_reason,
)
from benchflow.sandbox import _recovery_baseline
from benchflow.sandbox._recovery_baseline import (
    DockerRecoveryBaseline,
    capture_baseline,
    config_digest,
    release_baseline,
    validate_task_identity,
)
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.lockdown import harden_before_verify
from benchflow.task import Task
from benchflow.task.config import SandboxConfig
from tests.test_sandbox_hardening import _make_env


@pytest.fixture
def env(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM example:mutable\n")
    item = DockerSandbox(tmp_path, "task", "baseline-test", None, SandboxConfig())
    item._main_container_id = AsyncMock(return_value="original")
    item._inspect_container = AsyncMock(return_value={"Image": "sha256:" + "a" * 64})

    async def cli(args, check=True):
        output = "daemon-one" if args[0] == "info" else "sha256:" + "a" * 64
        return SimpleNamespace(stdout=output, return_code=0, stderr="")

    item._docker_cli = AsyncMock(side_effect=cli)
    return item


@pytest.mark.asyncio
async def test_lease_uses_original_container_image_not_mutable_tag(env):
    """GH #1136: a mutable Dockerfile tag must resolve once before solver execution."""
    baseline = await capture_baseline(env, "task-digest", "b" * 64)
    env._docker_cli.assert_any_await(
        ["image", "tag", "sha256:" + "a" * 64, baseline.lease_tag]
    )
    assert baseline.sandbox_config_digest == config_digest(env.task_env_config)
    assert set(json.loads(baseline.model_dump_json())) == {
        "schema_version",
        "effective_allow_internet",
        "image_id",
        "lease_tag",
        "daemon_id",
        "task_digest",
        "effective_config_digest",
        "sandbox_config_digest",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["daemon", "image", "config", "missing"])
async def test_changed_baseline_fails_before_container_start(env, change):
    """GH #1136: no mutable tag or rebuild fallback after a missing/changed lease."""
    baseline = await capture_baseline(env, "task-digest", "b" * 64)
    if change == "config":
        env.task_env_config.cpus += 1
    else:

        async def cli(args, check=True):
            if change == "missing" and args[0] == "image":
                raise RuntimeError("not found")
            return SimpleNamespace(
                stdout=("other" if change == "daemon" else "daemon-one")
                if args[0] == "info"
                else "sha256:" + "c" * 64
            )

        env._docker_cli = AsyncMock(side_effect=cli)
    env.use_recovery_baseline(baseline)
    env._run_docker_compose_command = AsyncMock()
    with pytest.raises((ValueError, RuntimeError)):
        await env.start(force_build=False)
    env._run_docker_compose_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_fresh_recovery_never_builds_pulls_or_deletes_lease(env):
    """GH #1136: fresh recovery bypasses hooks/build and retains the baseline lease."""
    baseline = await capture_baseline(env, "task-digest", "b" * 64)
    env.use_recovery_baseline(baseline)
    env._run_docker_compose_command = AsyncMock()
    env._run_docker_compose_build = AsyncMock()
    env._run_pre_compose_hook = AsyncMock()
    env.exec = AsyncMock()
    env._probe_verifier_log_mount = AsyncMock()
    await env.start(force_build=False)
    env._run_docker_compose_build.assert_not_awaited()
    env._run_pre_compose_hook.assert_not_awaited()
    env._run_docker_compose_command.assert_any_await(
        ["up", "--detach", "--wait", "--no-build", "--pull", "never"]
    )
    assert env._env_vars.prebuilt_image_name == baseline.image_id
    await env.stop(delete=True)
    assert all(
        "--rmi" not in call.args[0]
        for call in env._run_docker_compose_command.await_args_list
    )


@pytest.mark.asyncio
async def test_async_started_callback_runs_before_upload(tmp_path):
    """GH #1136: image identity must be retained before any task file upload."""
    (tmp_path / "instruction.md").write_text("unchanged")
    order = []

    async def start(**kwargs):
        order.append("start")

    async def capture():
        order.append("baseline")

    async def upload(*args):
        order.append("upload")

    runtime = SimpleNamespace(start=start, upload_file=upload)
    await _start_env_and_upload(runtime, tmp_path, {}, on_started=capture)
    assert order == ["start", "baseline", "upload"]


@pytest.mark.asyncio
async def test_built_task_becomes_eligible_only_after_original_capture(env, tmp_path):
    """GH #1136: Dockerfile tasks are recoverable only with their original image lease."""
    task_path = tmp_path / "task"
    task_path.mkdir()
    (task_path / "instruction.md").write_text("Produce the submission")
    (task_path / "task.toml").write_text("[verifier]\nworkspace_recovery=true\n")
    root = tmp_path / "rollout"
    root.mkdir()
    (root / "config.json").write_text("{}")
    trial = SimpleNamespace(
        _config=RolloutConfig(task_path=task_path, task_digest="original-task"),
        _task=Task(task_path),
        _env=env,
        _require_rollout_dir=lambda: root,
    )
    assert "was not captured" in recovery_ineligible_reason(trial)
    await capture_original_docker_baseline(trial)
    assert recovery_ineligible_reason(trial) is None
    assert (
        json.loads((root / "config.json").read_text())["verifier_recovery"]["eligible"]
        is True
    )
    assert (
        json.loads((root / "docker-recovery-baseline.json").read_text())["task_digest"]
        == "original-task"
    )


@pytest.mark.asyncio
async def test_original_effective_network_setting_replayed_without_task_mutation(env):
    """GH #1136: oracle recovery retains original model-bootstrap network setting."""
    env.task_env_config.allow_internet = True
    baseline = await capture_baseline(env, "task-digest", "b" * 64)
    child_source_config = env.task_env_config.model_copy(deep=True)
    child_source_config.allow_internet = False
    env.task_env_config = child_source_config
    env.use_recovery_baseline(baseline)
    env._run_docker_compose_command = AsyncMock()
    env.exec = AsyncMock()
    env._probe_verifier_log_mount = AsyncMock()
    await env.start(force_build=False)
    assert env.task_env_config.allow_internet is True
    assert child_source_config.allow_internet is False


@pytest.mark.asyncio
async def test_pristine_identity_survives_real_hardening_but_rejects_override_drift(
    env, tmp_path
):
    """GH #1136: hardening-generated env must not invalidate original task identity."""
    task_path = tmp_path / "identity-task"
    task_path.mkdir()
    (task_path / "instruction.md").write_text("Produce a file")
    (task_path / "task.toml").write_text("[verifier]\nworkspace_recovery=true\n")
    parent = Task(task_path)
    baseline = await capture_baseline(env, "task-digest", config_digest(parent.config))
    await harden_before_verify(_make_env(), parent, None, workspace="/app")
    assert config_digest(parent.config) != baseline.effective_config_digest
    pristine = Task(task_path)
    validate_task_identity(baseline, "task-digest", pristine.config)
    pristine.config.verifier.timeout_sec += 1
    with pytest.raises(ValueError, match="identity changed"):
        validate_task_identity(baseline, "task-digest", pristine.config)


def _lease(image_id: str = "sha256:" + "a" * 64) -> DockerRecoveryBaseline:
    return DockerRecoveryBaseline(
        image_id=image_id,
        lease_tag="benchflow-recovery-lease:" + uuid.uuid4().hex,
        effective_allow_internet=False,
        daemon_id="daemon-one",
        task_digest="task-digest",
        effective_config_digest="b" * 64,
        sandbox_config_digest="c" * 64,
    )


@pytest.mark.asyncio
async def test_release_untags_only_the_lease_and_is_idempotent(monkeypatch):
    """Guards the fix for leases leaked by leased-image verifier recovery: remove by tag, never by id.

    ``docker image rm <tag>`` without ``--force`` only untags while other tags
    reference the image; a repeated release sees "No such image" and succeeds.
    """
    baseline = _lease()
    replies = iter(
        [
            (0, f"Untagged: {baseline.lease_tag}"),
            (1, f"Error response from daemon: No such image: {baseline.lease_tag}"),
        ]
    )
    calls = []

    async def docker(*args):
        calls.append(args)
        return next(replies)

    monkeypatch.setattr(_recovery_baseline, "_docker", docker)
    assert await release_baseline(baseline) is True
    assert await release_baseline(baseline) is True
    assert calls == [("image", "rm", baseline.lease_tag)] * 2


@pytest.mark.asyncio
async def test_release_never_forces_an_image_still_in_use(monkeypatch):
    """Guards the same fix: a refused release keeps the lease and does not raise."""
    docker = AsyncMock(
        return_value=(1, "conflict: unable to delete (must be forced) - container")
    )
    monkeypatch.setattr(_recovery_baseline, "_docker", docker)
    baseline = _lease()
    assert await release_baseline(baseline) is False
    docker.assert_awaited_once_with("image", "rm", baseline.lease_tag)


@pytest.mark.asyncio
async def test_failed_capture_validation_releases_its_new_lease(env, monkeypatch):
    """Guards the same fix: a lease tagged by a rejected capture is not orphaned."""
    released = AsyncMock(return_value=True)
    monkeypatch.setattr(_recovery_baseline, "release_baseline", released)
    monkeypatch.setattr(
        _recovery_baseline,
        "validate_baseline",
        AsyncMock(side_effect=ValueError("Recovery image lease is missing")),
    )
    with pytest.raises(ValueError):
        await capture_baseline(env, "task-digest", "b" * 64)
    tagged = next(
        call.args[0][3]
        for call in env._docker_cli.await_args_list
        if call.args[0][:2] == ["image", "tag"]
    )
    assert released.await_args.args[0].lease_tag == tagged


def _docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.mark.live
def test_live_release_keeps_other_tags_and_deletes_only_unreferenced(tmp_path):
    """Guards the lease-release fix against real Docker ``rmi <tag>`` semantics."""
    if os.environ.get("BENCHFLOW_DOCKER_SNAPSHOT_PROOF") != "1":
        pytest.skip("set BENCHFLOW_DOCKER_SNAPSHOT_PROOF=1 for the local Docker proof")
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not available")
    (tmp_path / "f").write_text(uuid.uuid4().hex)
    (tmp_path / "Dockerfile").write_text("FROM scratch\nCOPY f /f\n")
    other = "benchflow-lease-proof:" + uuid.uuid4().hex[:12]
    image_id = _docker("build", "-q", "-t", other, str(tmp_path))
    baseline = _lease(image_id)
    try:
        _docker("image", "tag", image_id, baseline.lease_tag)
        assert asyncio.run(release_baseline(baseline)) is True
        assert _docker("image", "inspect", "--format", "{{.Id}}", other) == image_id
        assert asyncio.run(release_baseline(baseline)) is True  # idempotent
        # With the lease as the last reference, releasing it frees the image.
        _docker("image", "tag", image_id, baseline.lease_tag)
        _docker("image", "rm", other)
        assert asyncio.run(release_baseline(baseline)) is True
        assert not _docker("images", "-q", "--filter", f"reference={other}")
        assert not _docker(
            "images", "-q", "--filter", f"reference={baseline.lease_tag}"
        )
    finally:
        for ref in (baseline.lease_tag, other):
            subprocess.run(["docker", "image", "rm", ref], capture_output=True)


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_scored_rollout_releases_its_lease(tmp_path):
    """Guards the lease-release fix end to end: a scored Docker rollout keeps no lease.

    Before the fix every eligible Docker rollout left a
    ``benchflow-recovery-lease:<id>`` tag (and so its image) behind.
    """
    if os.environ.get("BENCHFLOW_DOCKER_SNAPSHOT_PROOF") != "1":
        pytest.skip("set BENCHFLOW_DOCKER_SNAPSHOT_PROOF=1 for the local Docker proof")
    from benchflow.sdk import SDK

    task = tmp_path / "task"
    shutil.copytree(Path(__file__).parent / "examples" / "hello-world-task", task)
    (task / "task.md").unlink()
    (task / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nWORKDIR /app\n"
        "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts\n"
    )
    toml = task / "task.toml"
    toml.write_text(
        toml.read_text().replace(
            "[verifier]\n", "[verifier]\nworkspace_recovery = true\n", 1
        )
    )
    result = await SDK().run(
        task_path=task, agent="oracle", jobs_dir=tmp_path / "jobs", environment="docker"
    )
    assert result.rewards == {"reward": 1.0}
    (receipt,) = (tmp_path / "jobs").glob("*/*/docker-recovery-baseline.json")
    lease = json.loads(receipt.read_text())["lease_tag"]
    try:
        assert not _docker("images", "-q", "--filter", f"reference={lease}")
    finally:
        subprocess.run(["docker", "image", "rm", lease], capture_output=True)
