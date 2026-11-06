"""A declared artifact is placed on Docker when the
controller's uid is not the sandbox user's.

The agent's turn hands the bind-mounted /logs/artifacts (the trial's own
artifacts/ folder) to the sandbox user, so the controller's rename of a staged
declared artifact into it failed with EACCES and the artifact was lost with the
container. Collection now hands the folder back to the host user first.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from benchflow.task.config import ArtifactConfig
from tests.test_artifacts_collection import LocalTransport, _collect, _layout


class _MountedSandbox(LocalTransport):
    """Docker's hook in miniature: the in-container chown of the mount point
    makes the host folder the controller's again."""

    is_mounted = True

    def __init__(self, host_folder: Path) -> None:
        super().__init__()
        self.host_folder = host_folder
        self.handed_back: list[tuple[str, bool]] = []

    async def _chown_to_host_user(
        self, path: str, recursive: bool = False, service: str = "main"
    ) -> None:
        self.handed_back.append((path, recursive))
        self.host_folder.chmod(0o755)


def _sandbox_owned(folder: Path) -> None:
    """As the agent's turn leaves it: not writable by this (controller) uid."""
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)


needs_permissions = pytest.mark.skipif(
    os.geteuid() == 0, reason="root writes the folder whatever its mode"
)


@needs_permissions
@pytest.mark.asyncio
async def test_a_declared_artifact_lands_in_a_mounted_folder_the_agent_held(
    tmp_path: Path,
) -> None:
    logs, workspace, trial = _layout(tmp_path)
    (workspace / "out").mkdir()
    (workspace / "out" / "answer.txt").write_text("hello\n")
    host_folder = trial / "artifacts"
    _sandbox_owned(host_folder)
    env = _MountedSandbox(host_folder)
    try:
        manifest = await _collect(
            env,
            logs,
            workspace,
            trial,
            artifacts=[
                ArtifactConfig(source=str(workspace / "out"), destination="out")
            ],
            mounted=True,
        )
    finally:
        host_folder.chmod(0o755)

    assert env.handed_back == [(str(logs), False)], "only the folder, not its files"
    assert (host_folder / "out" / "answer.txt").read_text() == "hello\n"
    declared = [c for c in manifest["collections"] if c["kind"] == "declared"]
    assert [c["status"] for c in declared] == ["collected"], declared


@pytest.mark.asyncio
async def test_nothing_changes_owner_without_a_declared_artifact(
    tmp_path: Path,
) -> None:
    logs, workspace, trial = _layout(tmp_path)
    env = _MountedSandbox(trial / "artifacts")
    await _collect(env, logs, workspace, trial, mounted=True)
    assert env.handed_back == []


@pytest.mark.asyncio
async def test_a_downloaded_logs_folder_is_never_handed_back(tmp_path: Path) -> None:
    """Daytona and the other downloading sandboxes copy files out as the
    controller; nothing in the sandbox changes owner."""
    logs, workspace, trial = _layout(tmp_path)
    (workspace / "report.txt").write_text("r\n")
    env = _MountedSandbox(trial / "artifacts")
    await _collect(env, logs, workspace, trial, artifacts=["report.txt"], mounted=False)
    assert env.handed_back == []
    assert (trial / "artifacts" / "report.txt").read_text() == "r\n"
