"""Local Docker mount-custody proof for selective PR #1046; no model/network calls."""

import asyncio
import os
import uuid

import pytest

from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths


@pytest.mark.live
@pytest.mark.parametrize("child_fails", [False, True])
async def test_live_bind_mounts_preserve_parent_and_archive_children(
    tmp_path, child_fails
):
    """PR #1046 custody: actual recreated Docker binds retain parent and child evidence."""
    if os.environ.get("BENCHFLOW_DOCKER_SNAPSHOT_PROOF") != "1":
        pytest.skip("explicit local Docker opt-in required")
    project = "bf-custody-" + uuid.uuid4().hex[:12]
    network = project + "_default"
    paths = RolloutPaths(tmp_path / "run")
    paths.mkdir()
    roots = (paths.agent_dir, paths.verifier_dir, paths.artifacts_dir)
    for root in roots:
        (root / "same.txt").write_text("parent")
    inodes = [root.stat().st_ino for root in roots]
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM python:3.12-slim\n")
    sandbox = DockerSandbox(
        environment_dir=environment_dir,
        environment_name=project,
        session_id=project,
        rollout_paths=paths,
        task_env_config=SandboxConfig(),
    )
    # The containers below are made by hand under this exact project name.
    sandbox._compose_project = project
    images = []
    original_snapshot = sandbox.snapshot

    async def snapshot(name=None):
        image = await original_snapshot(name)
        images.append(image)
        return image

    async def current_container():
        result = await sandbox._docker_cli(
            ["ps", "-q", "--filter", f"label=com.docker.compose.project={project}"]
        )
        return result.stdout.strip()

    async def execute(command, **kwargs):
        return await asyncio.wait_for(
            sandbox._docker_cli(
                ["exec", await current_container(), "sh", "-c", command], check=False
            ),
            30,
        )

    async def checked(command):
        result = await execute(command)
        assert result.return_code == 0, result.stderr
        return result.stdout.strip()

    sandbox._main_container_id = current_container
    sandbox.snapshot = snapshot
    sandbox.exec = execute
    try:
        await sandbox._docker_cli(["network", "create", network])
        command = [
            "run",
            "--pull=never",
            "-d",
            "--network",
            network,
            "--label",
            f"com.docker.compose.project={project}",
            "--label",
            "com.docker.compose.service=main",
        ]
        for root in roots:
            command += ["--mount", f"type=bind,src={root},dst=/logs/{root.name}"]
        await sandbox._docker_cli([*command, "python:3.12-slim", "sleep", "infinity"])
        await checked("echo parent > /state")
        rollout = Rollout(
            RolloutConfig(
                task_path=tmp_path / "task",
                scenes=[Scene.single(agent="dummy --agent")],
            )
        )
        rollout._env = sandbox
        rollout._rollout_dir = paths.rollout_dir
        rollout._rollout_paths = paths

        async def child(node):
            assert await checked("cat /state") == "parent"
            for name in ("agent", "verifier", "artifacts"):
                assert await checked(f"find /logs/{name} -mindepth 1 -maxdepth 1") == ""
                await checked(f"echo {node.id} > /logs/{name}/same.txt")
            await checked("echo child > /state")
            if child_fails:
                raise ValueError("intentional child failure")
            return 1

        if child_fails:
            with pytest.raises(ValueError, match="intentional child failure"):
                await rollout.branch(2, child, snapshot_layers={"sandbox"})
        else:
            assert await rollout.branch(2, child, snapshot_layers={"sandbox"}) == 1
        assert await checked("cat /state") == "parent"
        for root, inode in zip(roots, inodes, strict=True):
            assert root.stat().st_ino == inode
            assert (root / "same.txt").read_text() == "parent"
            assert await checked(f"cat /logs/{root.name}/same.txt") == "parent"
        bundles = list((paths.rollout_dir / "branches").glob("*/children/*"))
        assert len(bundles) == (1 if child_fails else 2)
        for bundle in bundles:
            for root in roots:
                assert (
                    bundle / "mounted" / root.name / "same.txt"
                ).read_text().strip() == bundle.name
    finally:
        result = await sandbox._docker_cli(
            ["ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"]
        )
        for container in result.stdout.split():
            await sandbox._docker_cli(["rm", "-f", container])
        for image in images:
            await sandbox._docker_cli(["image", "rm", image.ref])
        await sandbox._docker_cli(["network", "rm", network], check=False)
