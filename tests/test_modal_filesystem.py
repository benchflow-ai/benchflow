"""The Modal backend uses Modal's current Sandbox API (files, exec timeouts).

Modal's servers refuse the legacy Sandbox filesystem API (``Sandbox.open``,
``mkdir``, ``ls``) with ``UnimplementedError`` ("The legacy Sandbox filesystem
API is no longer supported"). On the 0.9 candidate every Modal rollout
therefore failed right after its sandbox started, on the ``mkdir`` of the log
directories (a smoke run on a VM, 2026-10-26). Guards the port of commits
3171a48b0, 0077c7948 and fe31cc233 against a return to
that API: the fake sandbox below has only ``filesystem`` and ``exec``, so any
legacy call fails the test. ``Sandbox.exec`` also takes only an integer
``timeout``; a task's ``600.0`` fails there.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("tenacity")

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.modal_impl import ModalSandbox, modal_exec_timeout
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import SandboxPaths
from tests._trust_admission import admitted


class _Method:
    """One ``Sandbox.filesystem`` method: records its calls, ``.aio`` style."""

    def __init__(self, effect=None) -> None:
        self.calls: list[tuple] = []
        self._effect = effect

    async def aio(self, *args):
        self.calls.append(args)
        if self._effect is not None:
            self._effect(*args)


def _copy_to_local(remote_path: str, local_path) -> None:
    Path(local_path).write_text(f"contents of {remote_path}")


class SandboxFilesystemNotFoundError(Exception):
    """Named as modal.exception's: the backend maps it by name."""


def _sandbox(tmp_path: Path) -> tuple[ModalSandbox, SimpleNamespace, AsyncMock]:
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n")
    env = ModalSandbox(
        environment_dir=environment,
        environment_name="modal-fs",
        session_id="modal-fs__1",
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    filesystem = SimpleNamespace(
        make_directory=_Method(),
        copy_from_local=_Method(),
        copy_to_local=_Method(_copy_to_local),
    )
    # The current API only: no open, ls or mkdir attribute exists.
    env._sandbox = SimpleNamespace(filesystem=filesystem)
    exec_mock = AsyncMock(return_value=ExecResult(return_code=0, stdout="", stderr=""))
    env.exec = exec_mock
    return env, filesystem, exec_mock


@pytest.mark.asyncio
async def test_log_directories_are_made_with_the_filesystem_api(tmp_path) -> None:
    env, filesystem, _ = _sandbox(tmp_path)

    await env._make_log_dirs()

    assert filesystem.make_directory.calls == [
        (str(SandboxPaths.agent_dir),),
        (str(SandboxPaths.verifier_dir),),
    ]


@pytest.mark.asyncio
async def test_upload_copies_from_local_and_applies_the_mode(tmp_path) -> None:
    env, filesystem, exec_mock = _sandbox(tmp_path)
    source = tmp_path / "secret.env"
    source.write_text("KEY=value\n")

    await env.upload_file(
        source, "/root/.config/secret.env", mode="600", admission=admitted(env)
    )

    assert filesystem.copy_from_local.calls == [
        (str(source), "/root/.config/secret.env")
    ]
    command = exec_mock.await_args.args[0]
    assert command == f"chmod 600 {shlex.quote('/root/.config/secret.env')}"


@pytest.mark.asyncio
async def test_download_copies_to_local_and_publishes_the_file(tmp_path) -> None:
    env, filesystem, _ = _sandbox(tmp_path)
    target = tmp_path / "rollout" / "verifier" / "reward.txt"

    await env.download_file("/logs/verifier/reward.txt", target)

    assert [call[0] for call in filesystem.copy_to_local.calls] == [
        "/logs/verifier/reward.txt"
    ]
    assert target.read_text() == "contents of /logs/verifier/reward.txt"


@pytest.mark.asyncio
async def test_a_missing_remote_file_is_still_file_not_found(tmp_path) -> None:
    """The legacy API raised FileNotFoundError; callers still get it."""
    env, filesystem, _ = _sandbox(tmp_path)

    def missing(remote_path, local_path):
        raise SandboxFilesystemNotFoundError(f"No such file: {remote_path}")

    filesystem.copy_to_local = _Method(missing)
    with pytest.raises(FileNotFoundError, match=r"reward\.json"):
        await env.download_file("/logs/verifier/reward.json", tmp_path / "out.json")


@pytest.mark.asyncio
async def test_upload_dir_stages_one_archive_through_the_filesystem_api(
    tmp_path,
) -> None:
    env, filesystem, _ = _sandbox(tmp_path)
    env.exec_route = AsyncMock(
        return_value=SimpleNamespace(shell="/bin/sh", pin="/usr/bin:/bin")
    )
    tests = tmp_path / "tests"
    (tests / "data").mkdir(parents=True)
    (tests / "test.sh").write_text("#!/bin/sh\n")
    (tests / "data" / "ref.json").write_text("{}\n")

    await env.upload_dir(tests, "/tests", admission=admitted(env))

    assert len(filesystem.copy_from_local.calls) == 1
    staged = filesystem.copy_from_local.calls[0][1]
    assert staged.startswith("/opt/benchflow/run/stage-") and staged.endswith(
        "/dir.tar"
    )


@pytest.mark.asyncio
async def test_is_dir_and_is_file_test_the_path_through_exec(tmp_path) -> None:
    env, _, exec_mock = _sandbox(tmp_path)

    assert await env.is_dir("/logs/verifier", user="agent") is True
    assert exec_mock.await_args.args[0] == "test -d /logs/verifier"
    assert exec_mock.await_args.kwargs["user"] == "agent"

    exec_mock.return_value = ExecResult(return_code=1, stdout="", stderr="")
    assert await env.is_file("/logs/verifier/reward.txt") is False
    assert exec_mock.await_args.args[0] == "test -f /logs/verifier/reward.txt"


def test_exec_timeouts_are_whole_seconds() -> None:
    """Modal's exec takes an integer timeout; a task's ``600.0`` fails there."""
    assert modal_exec_timeout(None) is None
    assert modal_exec_timeout(600.0) == 600
    assert modal_exec_timeout(90.5) == 91
    assert modal_exec_timeout(0.2) == 1


@pytest.mark.asyncio
async def test_exec_hands_modal_an_integer_timeout(monkeypatch) -> None:
    pytest.importorskip("modal")
    import modal

    # Secret.from_dict needs no network: the dict is what reaches the process.
    monkeypatch.setattr(modal.Secret, "from_dict", staticmethod(lambda d: dict(d)))
    env = object.__new__(ModalSandbox)
    env.default_user = None
    env._persistent_env = {}
    seen: dict = {}

    async def exec_aio(*argv, **kwargs):
        seen.update(kwargs)
        process = MagicMock()
        process.stdout.read.aio = AsyncMock(return_value="ok\n")
        process.stderr.read.aio = AsyncMock(return_value="")
        process.wait.aio = AsyncMock(return_value=0)
        return process

    env._sandbox = MagicMock()
    env._sandbox.exec.aio = exec_aio

    result = await env.exec("true", timeout_sec=600.0, system_shell=True)

    assert seen["timeout"] == 600 and isinstance(seen["timeout"], int)
    assert (result.return_code, result.stdout) == (0, "ok\n")


class SandboxFilesystemFileTooLargeError(Exception):
    """Named as modal.exception's: the backend maps it by name."""


@pytest.mark.asyncio
async def test_a_file_over_modals_transfer_limit_fails_with_that_reason(tmp_path):
    """BUILD-NOTES: a 12.8 GB workspace archive met Modal's 5 GiB
    per-file limit. The capture fails naming the limit; nothing is published."""
    from benchflow.sandbox.modal_impl import (
        MODAL_FILE_TRANSFER_LIMIT,
        ModalFileTooLargeError,
    )

    env, filesystem, _ = _sandbox(tmp_path)

    def too_large(remote_path, local_path):
        raise SandboxFilesystemFileTooLargeError(
            "file is 12770631680 bytes, which exceeds the 5368709120 byte limit"
        )

    filesystem.copy_to_local = _Method(too_large)
    target = tmp_path / "out" / "workspace.tar"
    with pytest.raises(ModalFileTooLargeError) as refused:
        await env.download_file("/tmp/benchflow-evidence-x.tar", target)
    message = str(refused.value)
    assert "/tmp/benchflow-evidence-x.tar" in message
    assert str(MODAL_FILE_TRANSFER_LIMIT) in message and "5 GiB" in message
    assert "no part of it was copied" in message
    assert not target.exists()

    filesystem.copy_from_local = _Method(too_large)
    source = tmp_path / "transfer.tar"
    source.write_bytes(b"x")
    with pytest.raises(ModalFileTooLargeError, match="into the Modal sandbox"):
        await env.upload_file(source, "/tmp/transfer.tar", admission=admitted(env))


def test_modal_sandbox_names_fit_modals_rules():
    """PR #6 (d19f2caa9 / cb59bd70f): a long task name plus ``-verifier`` was
    refused at create; a shortened or cleaned name keeps a hash of the whole."""
    from benchflow.sandbox.modal_impl import modal_sandbox_name

    assert modal_sandbox_name("hello-modal__ed7795d2") == "hello-modal__ed7795d2"
    long = "fpv-" + "x" * 70 + "__4cbf3da7-verifier"
    shortened = modal_sandbox_name(long)
    assert len(shortened) == 64 and shortened.startswith("fpv-xxx")
    assert shortened != modal_sandbox_name(long.replace("4cbf3da7", "4cbf3da8"))
    cleaned = modal_sandbox_name("task with spaces/and:colons")
    assert all(c.isalnum() or c in "._-" for c in cleaned)
    assert cleaned != "task with spaces/and:colons"
