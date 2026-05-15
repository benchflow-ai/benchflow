"""Tests for native Docker sandbox command construction."""

from __future__ import annotations

from pathlib import Path

from benchflow.sandboxes import DockerSandbox, DockerSandboxConfig, SandboxSpec


class FakeDockerSandbox(DockerSandbox):
    def __init__(self, config: DockerSandboxConfig, spec: SandboxSpec | None = None):
        super().__init__(config, spec)
        self.commands: list[list[str]] = []
        self.inputs: list[bytes | None] = []
        self.images: set[str] = set()

    async def _image_exists(self, image: str) -> bool:
        return image in self.images

    async def _run(self, cmd, *, stdin=None, timeout_sec=600):
        self.commands.append(list(cmd))
        self.inputs.append(stdin)

        class Completed:
            returncode = 0
            stdout = b"stdout"
            stderr = b""

        return Completed()


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    env = task / "environment"
    env.mkdir(parents=True)
    (env / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    return task


async def test_start_builds_and_runs_container(tmp_path: Path) -> None:
    sandbox = FakeDockerSandbox(
        DockerSandboxConfig(task_path=_task(tmp_path), session_id="abc"),
        SandboxSpec(provider="docker", allow_internet=False, cpus=2, memory_mb=512),
    )

    await sandbox.start()

    assert sandbox.commands[0][:3] == ["docker", "build", "-t"]
    run = sandbox.commands[1]
    assert run[:4] == ["docker", "run", "-d", "--name"]
    assert "--network" in run and "none" in run
    assert "--cpus" in run and "2" in run
    assert "--memory" in run and "512m" in run


async def test_exec_uses_docker_exec_with_env_and_user(tmp_path: Path) -> None:
    sandbox = FakeDockerSandbox(DockerSandboxConfig(task_path=_task(tmp_path)))

    result = await sandbox.exec(
        "echo hi",
        user="agent",
        cwd="/work",
        env={"A": "B"},
        timeout_sec=5,
    )

    cmd = sandbox.commands[-1]
    assert cmd[:3] == ["docker", "exec", "-i"]
    assert cmd[3:5] == ["--user", "agent"]
    assert "--workdir" in cmd and "/work" in cmd
    assert "--env" in cmd and "A=B" in cmd
    assert cmd[-3:] == ["bash", "-lc", "echo hi"]
    assert result.return_code == 0
    assert result.stdout == "stdout"


async def test_write_file_streams_base64_to_container(tmp_path: Path) -> None:
    sandbox = FakeDockerSandbox(DockerSandboxConfig(task_path=_task(tmp_path)))

    await sandbox.write_file("/app/data.txt", b"hello")

    assert sandbox.commands[-1][:3] == ["docker", "exec", sandbox.container_name]
    assert sandbox.inputs[-1] == b"aGVsbG8="
