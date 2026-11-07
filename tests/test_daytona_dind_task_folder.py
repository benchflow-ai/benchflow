"""Daytona DinD: a compose task that mounts its task folder finds its files (#1181).

On Docker, ``${CONTEXT_DIR}/..`` is the task folder on the host. ML-Dev-Bench's
``docker-compose.yaml`` mounts it read-only at ``/opt/task-src``, hides
``solution/`` and ``tests/`` under it with empty volumes, and mounts
``setup_workspace/`` beside it. The DinD path uploaded only ``environment/``
(to ``/benchflow/environment``), so ``/benchflow`` held no task files and
``compose up`` failed on the read-only mount point of ``solution``.

These tests run ``_DaytonaDinD.start`` against a fake Daytona sandbox: its
file system is a temporary folder, and its ``docker compose up`` resolves the
service's volumes the way Compose does (``${VAR}`` filled from the compose
environment, relative paths from the project directory, a missing bind
source made empty, a mount under a read-only bind needing its folder there).
"""

from __future__ import annotations

import json
import logging
import posixpath
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from benchflow.sandbox._base import ExecResult

pytest.importorskip("daytona")  # sandbox-daytona optional dependency

from benchflow.sandbox import daytona as daytona_module
from benchflow.sandbox.daytona import DaytonaSandbox
from benchflow.task.config import SandboxConfig
from tests._archive_fake import unpack_archive

# The shape of ML-Dev-Bench's compose file (toml-datasets
# ml_dev_bench_var_implementation), trimmed to what the mounts need.
_TASK_COMPOSE = """\
services:
  main:
    build:
      context: ${CONTEXT_DIR}
    image: ${MAIN_IMAGE_NAME}
    volumes:
      - ${CONTEXT_DIR}/..:/opt/task-src:ro
      - /opt/task-src/solution
      - /opt/task-src/tests
      - ${CONTEXT_DIR}/../setup_workspace:/opt/setup_workspace:ro
      - ${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}
      - ${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}
"""

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class _FakeDaytonaVM:
    """A Daytona DinD sandbox whose file system is ``root`` on this machine."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.mounts: dict[str, str] = {}
        self.fs = SimpleNamespace(
            upload_file=self._upload_file, upload_files=self._upload_files
        )
        self.id = "fake-dind"

    def path(self, vm_path: str) -> Path:
        assert vm_path.startswith("/"), vm_path
        return self.root / vm_path.lstrip("/")

    def files_under(self, vm_path: str) -> set[str]:
        base = self.path(vm_path)
        return {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}

    async def _upload_file(self, source: str, destination: str) -> None:
        target = self.path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source).read_bytes())

    async def _upload_files(self, files) -> None:
        for upload in files:
            await self._upload_file(upload.source, upload.destination)

    async def exec(
        self,
        command: str,
        cwd=None,
        env=None,
        timeout_sec=None,
        shell="bash -c",
        user=None,
        cleanup_session=False,
        route=None,
        system_shell=False,
    ) -> ExecResult:
        if unpack_archive(command, self.path):
            return ExecResult(return_code=0, stdout="", stderr="")
        if "network inspect" in command:
            return ExecResult(
                return_code=0,
                stdout=json.dumps(
                    [
                        {
                            "Driver": "bridge",
                            "Id": "a" * 64,
                            "Options": {"com.docker.network.bridge.name": "docker0"},
                        }
                    ]
                ),
                stderr="",
            )
        if command.startswith("mkdir -p "):
            for directory in shlex.split(command.split(" && ")[0])[2:]:
                self.path(directory).mkdir(parents=True, exist_ok=True)
        elif command.startswith("docker compose "):
            argv = shlex.split(command)
            if "--print" in argv:
                from tests.test_docker_prebuilt_choice import bake_plan

                return ExecResult(
                    stdout=json.dumps(
                        bake_plan(self._compose_config(argv, env or {}), "trial")
                    ),
                    return_code=0,
                )
            if argv[-3:] == ["config", "--format", "json"]:
                return ExecResult(
                    stdout=json.dumps(self._compose_config(argv, env or {})),
                    return_code=0,
                )
            if argv[-3:] == ["up", "-d", "--no-build"]:
                return self._compose_up(argv, env or {})
        # Also what the daemon guard looks for; other commands ignore it.
        return ExecResult(
            stdout="BF dind-daemon-guard status=0", stderr="", return_code=0
        )

    def _compose_config(self, argv, env):
        from tests.test_docker_prebuilt_choice import _Loader, _Reset

        result = {}

        def merge(old, new):
            for key, value in new.items():
                if isinstance(value, _Reset):
                    old.pop(key, None)
                elif isinstance(value, dict):
                    if not isinstance(old.get(key, {}), dict):
                        old[key] = {"context": old[key]}
                    merge(old.setdefault(key, {}), value)
                else:
                    old[key] = value

        for i, arg in enumerate(argv):
            if arg == "-f":
                text = self.path(argv[i + 1]).read_text()
                text = _VAR.sub(lambda m: env.get(m.group(1), m.group(2) or ""), text)
                merge(result, yaml.load(text, Loader=_Loader))
        return result

    def _compose_up(self, argv: list[str], env: dict[str, str]) -> ExecResult:
        from tests.test_docker_prebuilt_choice import _Loader

        project_dir = argv[argv.index("--project-directory") + 1]
        files = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-f"]
        volumes: list[str] = []
        for compose_file in files:
            document = yaml.load(self.path(compose_file).read_text(), Loader=_Loader)
            main = (document.get("services") or {}).get("main") or {}
            volumes.extend(main.get("volumes") or [])

        def fill(text: str) -> str:
            return _VAR.sub(lambda m: env.get(m.group(1), m.group(2) or ""), text)

        binds: dict[str, tuple[str, bool]] = {}
        targets: list[str] = []
        for volume in volumes:
            parts = fill(volume).split(":")
            if len(parts) == 1:  # an anonymous volume
                targets.append(parts[0])
                continue
            source, target = parts[0], parts[1]
            if not source.startswith("/"):
                source = posixpath.join(project_dir, source)
            source = posixpath.normpath(source)
            # Compose's short syntax makes a missing bind source.
            self.path(source).mkdir(parents=True, exist_ok=True)
            binds[target] = (source, parts[2:] == ["ro"])
            targets.append(target)
        for target in targets:
            for outer, (outer_source, read_only) in binds.items():
                if target == outer or not target.startswith(outer + "/"):
                    continue
                mount_point = posixpath.join(
                    outer_source, posixpath.relpath(target, outer)
                )
                if read_only and not self.path(mount_point).is_dir():
                    return ExecResult(
                        stdout="",
                        stderr=(
                            f'error mounting "{target}" to rootfs at '
                            f'"{target}": mkdirat {mount_point}: '
                            "read-only file system"
                        ),
                        return_code=1,
                    )
        self.mounts = {target: source for target, (source, _) in binds.items()}
        return ExecResult(stdout="", stderr="", return_code=0)


def _ml_dev_bench_like_task(root: Path) -> Path:
    task = root / "var_implementation"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "environment" / "docker-compose.yaml").write_text(_TASK_COMPOSE)
    (task / "setup_workspace").mkdir()
    (task / "setup_workspace" / "var.py").write_text("# the task's starting code\n")
    (task / "init_task.py").write_text("print('init')\n")
    (task / "instruction.md").write_text("Implement VAR.\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("echo the answer\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("echo grading\n")
    return task


def _dind_sandbox(
    environment_dir: Path, vm: _FakeDaytonaVM, monkeypatch
) -> DaytonaSandbox:
    sandbox = DaytonaSandbox(
        environment_dir=environment_dir,
        environment_name="var_implementation",
        session_id="trial-1",
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    sandbox.logger = logging.getLogger("test.daytona.dind.task_folder")

    async def create_sandbox(params) -> None:
        sandbox._sandbox = vm

    monkeypatch.setattr(sandbox, "_create_sandbox", create_sandbox)
    monkeypatch.setattr(sandbox, "_sandbox_exec", vm.exec)
    # These tests isolate upload/mount behavior. The fake VM has no Compose
    # config/realpath protocol; test_compose_oracle exercises the real checker.
    monkeypatch.setattr(sandbox._strategy, "_check_compose_project", AsyncMock())
    monkeypatch.setattr(sandbox._strategy, "_pin_or_verify_bind_targets", AsyncMock())
    monkeypatch.setattr(sandbox._strategy, "_prepare_inert_bootstrap", AsyncMock())
    monkeypatch.setattr("benchflow.sandbox._inert_bootstrap.release", AsyncMock())
    monkeypatch.setattr(
        daytona_module.DaytonaClientManager,
        "get_instance",
        AsyncMock(return_value=SimpleNamespace()),
    )
    return sandbox


@pytest.mark.asyncio
async def test_task_folder_mount_holds_the_uploaded_task_files(
    tmp_path: Path, monkeypatch
) -> None:
    """Guards #1181: compose up failed and /opt/task-src held no task files."""
    task = _ml_dev_bench_like_task(tmp_path / "tasks")
    vm = _FakeDaytonaVM(tmp_path / "vm")
    sandbox = _dind_sandbox(task / "environment", vm, monkeypatch)

    order = []
    sandbox._strategy._prepare_inert_bootstrap.side_effect = lambda: order.append(
        "inert"
    )
    sandbox._strategy._pin_or_verify_bind_targets.side_effect = lambda: order.append(
        "proof"
    )
    from benchflow.sandbox import _inert_bootstrap

    _inert_bootstrap.release.side_effect = lambda *args: order.append("release")
    await sandbox._strategy.start(force_build=True)
    assert order == ["inert", "proof", "release"]

    task_src = vm.mounts["/opt/task-src"]
    assert vm.files_under(task_src) == {
        "environment/Dockerfile",
        "environment/docker-compose.yaml",
        "init_task.py",
        "instruction.md",
        "setup_workspace/var.py",
        "task.toml",
    }
    # The oracle and verifier stay behind; their names are empty folders, the
    # mount points of the volumes that hide them.
    for hidden in ("solution", "tests"):
        assert vm.path(posixpath.join(task_src, hidden)).is_dir()
        assert vm.files_under(posixpath.join(task_src, hidden)) == set()
    assert vm.files_under(vm.mounts["/opt/setup_workspace"]) == {"var.py"}
    uploaded = {p.name for p in vm.root.rglob("*") if p.is_file()}
    assert "solve.sh" not in uploaded
    assert "test.sh" not in uploaded


@pytest.mark.asyncio
async def test_native_oracle_and_verifier_folders_stay_behind(
    tmp_path: Path, monkeypatch
) -> None:
    """task.md tasks name them oracle/ and verifier/; those are kept back too."""
    task = tmp_path / "tasks" / "native"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    volumes:\n"
        "      - ${CONTEXT_DIR}/..:/opt/task-src:ro\n"
        "      - /opt/task-src/oracle\n"
        "      - /opt/task-src/verifier\n"
    )
    (task / "task.md").write_text("---\nversion: 1\n---\nDo it.\n")
    (task / "oracle").mkdir()
    (task / "oracle" / "solve.sh").write_text("echo the answer\n")
    (task / "verifier").mkdir()
    (task / "verifier" / "verifier.md").write_text("grading\n")
    vm = _FakeDaytonaVM(tmp_path / "vm")
    sandbox = _dind_sandbox(task / "environment", vm, monkeypatch)

    await sandbox._strategy.start(force_build=True)

    task_src = vm.mounts["/opt/task-src"]
    assert vm.files_under(task_src) == {"environment/docker-compose.yaml", "task.md"}
    assert vm.path(posixpath.join(task_src, "oracle")).is_dir()
    assert vm.path(posixpath.join(task_src, "verifier")).is_dir()
    # A folder the task does not have gets no stand-in.
    assert not vm.path(posixpath.join(task_src, "solution")).exists()


@pytest.mark.asyncio
async def test_environment_outside_a_task_folder_uploads_only_itself(
    tmp_path: Path, monkeypatch
) -> None:
    """An environment folder with no task.toml or task.md beside it does not
    upload its parent, which could be any folder of the caller's."""
    parent = tmp_path / "somewhere"
    (parent / "environment").mkdir(parents=True)
    (parent / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: x\n"
    )
    (parent / "unrelated.txt").write_text("not the task's\n")
    vm = _FakeDaytonaVM(tmp_path / "vm")
    sandbox = _dind_sandbox(parent / "environment", vm, monkeypatch)

    await sandbox._strategy.start(force_build=True)

    assert vm.files_under("/runtime/task") == {"environment/docker-compose.yaml"}


@pytest.mark.asyncio
async def test_symlinks_in_the_task_folder_are_not_followed(
    tmp_path: Path, monkeypatch
) -> None:
    """The task-folder upload keeps #411's symlink defence."""
    secret = tmp_path / "victim-a.txt"
    secret.write_text("host secret\n")
    task = _ml_dev_bench_like_task(tmp_path / "tasks")
    (task / "linked.txt").symlink_to(secret)
    (task / "linked_dir").symlink_to(tmp_path)
    vm = _FakeDaytonaVM(tmp_path / "vm")
    sandbox = _dind_sandbox(task / "environment", vm, monkeypatch)

    await sandbox._strategy.start(force_build=True)

    uploaded = vm.files_under("/runtime/task")
    assert "linked.txt" not in uploaded
    assert not any(name.startswith("linked_dir/") for name in uploaded)


@pytest.mark.asyncio
async def test_symlinked_environment_cannot_expand_the_upload(tmp_path, monkeypatch):
    """Keep the caller's root."""
    private = _ml_dev_bench_like_task(tmp_path / "private")
    (private / "private-data.txt").write_text("private sentinel")
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "task.toml").write_text('version = "1.0"\n')
    (selected / "environment").symlink_to(private / "environment")
    vm = _FakeDaytonaVM(tmp_path / "vm")
    sandbox = _dind_sandbox(selected / "environment", vm, monkeypatch)
    with pytest.raises(ValueError, match="symlink"):
        await sandbox._strategy._upload_task_folder()
    assert not vm.files_under("/runtime/task")


@pytest.mark.asyncio
async def test_materialized_private_package_is_not_uploaded(tmp_path, monkeypatch):
    """Guards the task-folder upload from 06b8a04a, using the real materializer."""
    from benchflow.task.formats import materialize_task_dir
    from tests._taskmd_helpers import EXAMPLES

    source = tmp_path / "source"
    shutil.copytree(EXAMPLES / "hello-world", source)
    (source / "verifier" / "private.txt").write_text("private grader sentinel")
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "cache"))
    task = materialize_task_dir(source)
    assert (task / ".taskmd/package/verifier/private.txt").is_file()
    (task / "environment/docker-compose.yaml").write_text(_TASK_COMPOSE)
    vm = _FakeDaytonaVM(tmp_path / "vm")
    sandbox = _dind_sandbox(task / "environment", vm, monkeypatch)
    sandbox._sandbox = vm
    await sandbox._strategy._upload_task_folder()
    assert not any(p.startswith(".taskmd/") for p in vm.files_under("/runtime/task"))


@pytest.mark.asyncio
async def test_task_git_database_is_not_uploaded(tmp_path, monkeypatch):
    """Guards the task-folder upload from 06b8a04a: no recoverable held-out files."""
    task = _ml_dev_bench_like_task(tmp_path / "tasks")
    for args in (
        ["init"],
        ["add", "solution", "tests"],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-m",
            "Fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(task), *args], check=True, capture_output=True)
    vm = _FakeDaytonaVM(tmp_path / "vm")
    sandbox = _dind_sandbox(task / "environment", vm, monkeypatch)
    sandbox._sandbox = vm
    await sandbox._strategy._upload_task_folder()
    assert not any(p.startswith(".git/") for p in vm.files_under("/runtime/task"))


@pytest.mark.asyncio
async def test_task_executable_modes_are_restored_and_failure_is_fatal(
    tmp_path, monkeypatch
):
    """SDK uploads do not copy modes (the
    file-by-file path, where the sandbox has no tar)."""
    task = _ml_dev_bench_like_task(tmp_path / "tasks")
    script = task / "setup_workspace/start.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o6755)
    vm = _FakeDaytonaVM(tmp_path / "vm")
    sandbox = _dind_sandbox(task / "environment", vm, monkeypatch)
    calls = []
    sandbox._sandbox = vm

    async def permissions(command, **kwargs):
        calls.append(command)
        if "tar -x -f" in command:  # no tar: the files one SDK batch at a time
            return ExecResult(stdout="", stderr="", return_code=127)
        return ExecResult(
            stdout="", stderr="mode failure", return_code=int("chmod 755" in command)
        )

    monkeypatch.setattr(sandbox, "_sandbox_exec", permissions)
    with pytest.raises(RuntimeError, match="permissions"):
        await sandbox._strategy._upload_task_folder()
    assert any("chmod 755" in c and "setup_workspace/start.sh" in c for c in calls)
    assert any("chmod 644" in c for c in calls)


async def test_task_executable_modes_travel_in_the_archive(tmp_path, monkeypatch):
    """DAYTONA-GAPS rule 1: the one archive carries each file's mode by the
    upload's rule (755 when any execute bit is set, else 644, never set-id),
    and root's tar writes it."""
    task = _ml_dev_bench_like_task(tmp_path / "tasks")
    script = task / "setup_workspace/start.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o6755)
    vm = _FakeDaytonaVM(tmp_path / "vm")
    sandbox = _dind_sandbox(task / "environment", vm, monkeypatch)
    sandbox._sandbox = vm
    modes: dict[str, int] = {}
    upload_file = vm.fs.upload_file

    async def recording_upload(source, destination):
        with tarfile.open(source) as tar:
            modes.update({member.name: member.mode for member in tar.getmembers()})
        await upload_file(source, destination)

    vm.fs.upload_file = recording_upload
    await sandbox._strategy._upload_task_folder()
    (start,) = [name for name in modes if name.endswith("setup_workspace/start.sh")]
    assert modes[start] == 0o755
    assert set(modes.values()) <= {0o644, 0o755}
