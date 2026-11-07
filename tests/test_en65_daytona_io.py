"""Guards Daytona capture and transport regressions present at ac3bedaa1.

The real SDK schema case also guards the dependency fix at 764325430.
All provider calls are fake; the final test uses the existing local Docker
transport fixture to execute the complete nested command.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shlex
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.reachable_write import UnsafeDownloadEntryError
from benchflow.sandbox import _transport
from benchflow.sandbox._base import SYSTEM_PATH, ExecResult, ExecRoute
from benchflow.sandbox.daytona import DaytonaSandbox, _load_daytona_sdk
from benchflow.sandbox.daytona_dind import _DaytonaDinD
from benchflow.task.paths import RolloutPaths, SandboxPaths
from benchflow.task.verifier import Verifier
from tests import test_exec_transport_contract as transport

transport_box = transport.box


@pytest.fixture(autouse=True)
def daytona_sdk():
    pytest.importorskip("daytona")
    _load_daytona_sdk()


@pytest.mark.parametrize("directory", [False, True])
async def test_sdk207_optional_timestamp_preserves_readable_output(tmp_path, directory):
    """Guard 764325430: actual SDK decoding must retain null-timestamp files."""
    from daytona_toolbox_api_client_async.models.file_info import FileInfo

    metadata = {
        "group": "root",
        "isDir": False,
        "modTime": "2026-10-26 00:00:00 +0000 UTC",
        "mode": "-rw-r--r--",
        "modifiedAt": None,
        "name": "reward.txt",
        "owner": "root",
        "permissions": "644",
        "size": 1,
    }

    async def download(source, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text("1")

    async def download_many(*, files):
        replies = []
        for request in files:
            await download(request.source, request.destination)
            replies.append(
                SimpleNamespace(
                    source=request.source,
                    result=request.destination,
                    error=None,
                )
            )
        return replies

    sandbox = object.__new__(DaytonaSandbox)
    sandbox.logger = logging.getLogger(__name__)
    sandbox._sandbox = SimpleNamespace(
        fs=SimpleNamespace(
            get_file_info=AsyncMock(side_effect=lambda _: FileInfo.from_dict(metadata)),
            list_files=AsyncMock(side_effect=lambda _: [FileInfo.from_dict(metadata)]),
            search_files=AsyncMock(
                return_value=SimpleNamespace(files=["/logs/verifier/reward.txt"])
            ),
            download_file=download,
            download_files=download_many,
        )
    )
    target = tmp_path / "capture"
    if directory:
        await sandbox._sdk_download_dir("/logs/verifier", target)
        target /= "reward.txt"
    else:
        await sandbox._sdk_download_file("/logs/verifier/reward.txt", target)
    assert target.read_text() == "1"


def _archive_reply(metadata):
    if metadata is None:
        return "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
    encoded = base64.b64encode(json.dumps(metadata).encode()).decode()
    return f"HTTP/1.1 200 OK\r\nX-Docker-Container-Path-Stat: {encoded}\r\n\r\n"


def _dind_files(files, *, metadata=None, copy_error=False, reply_error=None):
    strategy = object.__new__(_DaytonaDinD)
    staged = {}
    copies = []
    queries = []

    async def compose(command, **kwargs):
        if command == ["ps", "--all", "-q", "main"]:
            return ExecResult(stdout="container-id\n", return_code=0)
        assert command[0] == "cp"
        source = command[1].removeprefix("main:")
        copies.append(source)
        if copy_error or source not in files:
            return ExecResult(stdout="", stderr="copy failed", return_code=1)
        staged[command[2]] = files[source]
        return ExecResult(stdout="", return_code=0)

    async def vm(command, **kwargs):
        if command.startswith("rm -f "):
            return ExecResult(stdout="", return_code=0)
        assert command == "docker system dial-stdio"
        request = kwargs["stdin_payload"].decode()
        url = request.split()[1]
        source = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["path"][0]
        queries.append(source)
        if reply_error:
            return ExecResult(stdout=reply_error, return_code=0)
        info = (metadata or {}).get(source)
        if info is None and source in files:
            info = {"mode": 0o644}
        elif info is None and source in ("/logs/verifier", "/logs", "/"):
            info = {"mode": 0x80000000 | 0o755}
        return ExecResult(stdout=_archive_reply(info), return_code=0)

    async def download(source, target):
        Path(target).write_text(staged[source])

    strategy._compose_exec = AsyncMock(side_effect=compose)
    strategy._vm_exec = AsyncMock(side_effect=vm)
    strategy._env = SimpleNamespace(
        _sdk_download_file=AsyncMock(side_effect=download),
        logger=logging.getLogger(__name__),
    )
    return strategy, copies, queries


@pytest.mark.parametrize(
    "name,value", [("reward.txt", "1"), ("reward.json", '{"reward": 1}')]
)
async def test_dind_recovers_reward_with_absent_optional_companion(
    tmp_path, name, value
):
    """Guard ac3bedaa1: proven absence must not abort canonical reward recovery."""
    strategy, copies, queries = _dind_files({f"/logs/verifier/{name}": value})
    paths = RolloutPaths(rollout_dir=tmp_path / "trial")
    paths.mkdir()
    verifier = object.__new__(Verifier)
    verifier._rollout_paths = paths
    verifier._sandbox = strategy
    verifier._logger = logging.getLogger(__name__)
    assert await verifier._recover_main_verifier_outputs(SandboxPaths())
    assert (paths.verifier_dir / name).read_text() == value
    assert copies == [f"/logs/verifier/{name}"]
    assert "/logs/verifier" in queries


@pytest.mark.parametrize(
    "mode,link", [(0x80000000, ""), (0x08000000, "target"), (0x02000000, "")]
)
async def test_dind_rejects_unsafe_reward_companion(tmp_path, mode, link):
    """Guard ac3bedaa1: unsafe entries cannot become absent optional rewards."""
    source = "/logs/verifier/reward.json"
    strategy, copies, _ = _dind_files(
        {}, metadata={source: {"mode": mode, "linkTarget": link}}
    )
    with pytest.raises(UnsafeDownloadEntryError):
        await strategy.download_file(source, tmp_path / "reward")
    assert copies == []


async def test_dind_existing_file_copy_failure_remains_an_error(tmp_path):
    """Guard ac3bedaa1: losing a known file during copy is not optional absence."""
    strategy, copies, _ = _dind_files(
        {"/logs/verifier/reward.txt": "1"}, copy_error=True
    )
    with pytest.raises(RuntimeError, match="compose cp failed"):
        await strategy.download_file("/logs/verifier/reward.txt", tmp_path / "reward")
    assert copies == ["/logs/verifier/reward.txt"]


@pytest.mark.parametrize(
    "reply", ["", "HTTP/1.1 500 Server Error\r\n\r\n", "HTTP/1.1 200 OK\r\n\r\n"]
)
async def test_dind_metadata_transport_error_remains_an_error(tmp_path, reply):
    """Guard ac3bedaa1: an unavailable daemon cannot prove optional absence."""
    strategy, copies, _ = _dind_files({}, reply_error=reply)
    if not reply:
        strategy._vm_exec = AsyncMock(return_value=ExecResult(stdout="", return_code=1))
    with pytest.raises(RuntimeError):
        await strategy.download_file("/logs/verifier/reward.json", tmp_path / "reward")
    assert copies == []


async def test_dind_missing_container_is_not_an_absent_file(tmp_path):
    """Guard ac3bedaa1: a daemon-wide 404 cannot authorize dropping a reward."""
    strategy, copies, _ = _dind_files({}, reply_error=_archive_reply(None))
    with pytest.raises(RuntimeError, match="container"):
        await strategy.download_file("/logs/verifier/reward.json", tmp_path / "reward")
    assert copies == []


@pytest.mark.parametrize("mode", [-1, 0x100000000, True, "33188", None])
async def test_dind_invalid_metadata_cannot_prove_a_regular_file(tmp_path, mode):
    """Guard ac3bedaa1: malformed daemon metadata must fail closed."""
    source = "/logs/verifier/reward.txt"
    strategy, copies, _ = _dind_files({}, metadata={source: {"mode": mode}})
    with pytest.raises(RuntimeError, match="metadata is invalid"):
        await strategy.download_file(source, tmp_path / "reward")
    assert copies == []


async def test_dind_dangling_parent_is_not_optional_absence(tmp_path):
    """Guard ac3bedaa1: a missing child of an unsafe parent is not absent evidence."""
    strategy, copies, _ = _dind_files(
        {},
        metadata={
            "/logs/verifier": {"mode": 0x08000000, "linkTarget": "/missing"},
        },
    )
    with pytest.raises(UnsafeDownloadEntryError, match="unsafe remote file parent"):
        await strategy.download_file("/logs/verifier/reward.json", tmp_path / "reward")
    assert copies == []


async def test_dind_absence_proof_against_a_native_daemon(transport_box):
    """Guard ac3bedaa1: real daemon HEAD distinguishes files, folders and absence."""
    strategy = object.__new__(_DaytonaDinD)
    strategy._compose_exec = AsyncMock(
        return_value=ExecResult(
            stdout=transport_box.cid + "\n",
            return_code=0,
        )
    )

    async def vm(command, **kwargs):
        process = await asyncio.create_subprocess_exec(
            *shlex.split(command),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(kwargs["stdin_payload"]),
            timeout=20,
        )
        return ExecResult(
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            return_code=process.returncode,
        )

    strategy._vm_exec = vm
    await strategy._check_compose_download_file("/etc/passwd")
    with pytest.raises(FileNotFoundError):
        await strategy._check_compose_download_file("/tmp/no-such-en65-reward")
    with pytest.raises(UnsafeDownloadEntryError):
        await strategy._check_compose_download_file("/tmp")


@pytest.mark.parametrize(
    "size", [12_000, 14_000, 14_500, 15_000, 15_500, 16_000, 16_500, 20_000]
)
async def test_nested_staging_includes_the_final_proof_in_its_bound(size):
    """Guard ac3bedaa1: staging overhead must not create a rejected size gap."""
    commands = []

    async def execute(session, request, **kwargs):
        commands.append(request.command)
        return SimpleNamespace(cmd_id="command")

    sandbox = object.__new__(DaytonaSandbox)
    sandbox.logger = logging.getLogger(__name__)
    sandbox._sandbox = SimpleNamespace(
        process=SimpleNamespace(
            create_session=AsyncMock(),
            execute_session_command=execute,
        )
    )
    sandbox._poll_response = AsyncMock(
        return_value=ExecResult(stdout="OK", return_code=0)
    )
    sandbox._sdk_upload_bytes = AsyncMock()
    sandbox._sdk_delete_tree = AsyncMock()
    strategy = object.__new__(_DaytonaDinD)
    strategy._env = sandbox
    strategy._compose_cmd = lambda args: shlex.join(["docker", "compose", *args])
    strategy._compose_env_vars = lambda: {"TASK_DATA": "x" * size}
    result = await strategy._exec(
        "true #" + "x" * 66_000,
        cwd=None,
        env=None,
        user=None,
        service="main",
        timeout_sec=10,
        cleanup_session=False,
        route=ExecRoute(pin=SYSTEM_PATH, task_root=False, explicit_root=True),
    )
    assert result.return_code == 0
    assert len(commands) == 1
    assert _transport.size(commands[0]) <= _transport.COMMAND_BOUND
    assert sandbox._sdk_delete_tree.await_count == 1


@pytest.mark.parametrize("cut_outer_payload", [False, True])
async def test_nested_staging_really_executes_and_preserves_stdin(
    transport_box, cut_outer_payload
):
    """Guard ac3bedaa1: the overflow repair preserves both nested payloads."""
    sandbox = transport._daytona(transport_box, dind=True)
    sandbox._strategy._compose_env_vars = lambda: {"TASK_DATA": "x" * 15_000}
    upload = sandbox._sdk_upload_bytes

    async def upload_with_optional_cut(data, target):
        if cut_outer_payload and target.endswith("/command"):
            data = data[:-1]
        await upload(data, target)

    sandbox._sdk_upload_bytes = upload_with_optional_cut
    result = await sandbox._strategy._exec(
        'printf "%s:%s" "${#TASK_DATA}" complete # ' + "x" * 66_000,
        cwd=None,
        env=None,
        user=None,
        service="main",
        timeout_sec=60,
        cleanup_session=False,
        route=ExecRoute(pin=SYSTEM_PATH, task_root=False, explicit_root=True),
    )
    if cut_outer_payload:
        assert result.return_code == _transport.READ_FAILED
        assert "complete" not in result.stdout
        assert "incomplete" in result.stderr
    else:
        assert result.return_code == 0, result.stderr
        assert result.stdout.strip() == "15000:complete"
