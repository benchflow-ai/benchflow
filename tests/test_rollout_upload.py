"""Tests for rollout startup uploads."""

from pathlib import Path

import pytest

from benchflow.rollout import _start_env_and_upload


class FakeUploadEnv:
    def __init__(self) -> None:
        self.started = False
        self.uploaded_files: list[tuple[Path, str]] = []
        self.uploaded_dirs: list[tuple[Path, str]] = []

    async def start(self, force_build: bool) -> None:
        self.started = force_build is False

    async def upload_file(self, source: Path, target: str) -> None:
        self.uploaded_files.append((source, target))

    async def upload_dir(self, source: Path, target: str) -> None:
        self.uploaded_dirs.append((source, target))


@pytest.mark.asyncio
async def test_start_env_uploads_task_environment_skills(tmp_path: Path) -> None:
    """Guards ENG-88: remote oracle tasks can import bundled task skills."""
    task = tmp_path / "task"
    (task / "environment" / "skills" / "jax-skills").mkdir(parents=True)
    (task / "solution").mkdir(parents=True)
    (task / "instruction.md").write_text("solve\n")
    (task / "solution" / "solve.sh").write_text("echo ok\n")
    env = FakeUploadEnv()
    timing: dict[str, float] = {}

    await _start_env_and_upload(env, task, timing)

    task_skills = task / "environment" / "skills"
    assert env.started is True
    assert (task / "instruction.md", "/instruction.md") in env.uploaded_files
    assert (task_skills, "/app/skills") in env.uploaded_dirs
    assert (task / "solution", "/solution") in env.uploaded_dirs
    assert "environment_setup" in timing
