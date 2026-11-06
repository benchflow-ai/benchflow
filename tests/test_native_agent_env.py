"""Guards the bounded transport fix for K1-R17-1 at 02648360."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.agent_exec import agent_exec_call
from benchflow.sandbox.native_agent_env import uploaded_agent_env


def command():
    return agent_exec_call(
        ":", "agent", sandbox=SimpleNamespace(agent_exec_shell="/bin/bash")
    )[0]


@pytest.mark.asyncio
async def test_command_size_does_not_depend_on_names_values_or_count():
    """Guards K1-R18-1 at 5f1525ba for every private-file backend."""
    lengths = []
    for count, width in [(1, 1), (500, 400), (3000, 1), (1, 100000)]:
        async with uploaded_agent_env(
            command(),
            {f"NAME_{i}": "x" * width for i in range(count)},
            root_exec=AsyncMock(return_value=ExecResult(return_code=0)),
            upload=AsyncMock(),
        ) as (body, opening):
            lengths.append(len(body + opening))
            assert "x" * 100 not in body + opening
    assert len(set(lengths)) == 1


@pytest.mark.asyncio
async def test_private_upload_contains_data_and_is_removed_after_dispatch():
    root = AsyncMock(return_value=ExecResult(return_code=0))
    payloads = []

    async def upload(source, target):
        payloads.append((json.loads(source.read_text()), target))

    async with uploaded_agent_env(
        command(), {"BASH_ENV": "$(dummy)", "ODD": "\n'"}, root_exec=root, upload=upload
    ) as (restore, opening):
        assert "$(dummy)" not in restore + opening
        assert "exec 9<" in opening
        assert "/bin/rmdir" in opening
        assert root.await_args.args[0].startswith(
            "umask 077; /bin/mkdir -- /.runtime-agent-env-"
        )
    assert payloads[0][0] == {"BASH_ENV": "$(dummy)", "ODD": "\n'"}
    assert "/bin/rm -f -- " + payloads[0][1] in root.await_args.args[0]


@pytest.mark.asyncio
async def test_existing_directory_is_neither_uploaded_into_nor_removed():
    root = AsyncMock(return_value=ExecResult(return_code=1))
    upload = AsyncMock()
    with pytest.raises(RuntimeError, match="private native agent"):
        async with uploaded_agent_env(
            command(), {"A": "1"}, root_exec=root, upload=upload
        ):
            pytest.fail("must refuse the existing path")
    upload.assert_not_awaited()
    assert root.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["upload", "dispatch"])
async def test_failed_upload_or_dispatch_cleans_only_its_private_file(failure):
    root = AsyncMock(return_value=ExecResult(return_code=0))
    upload = AsyncMock(
        side_effect=RuntimeError("dummy") if failure == "upload" else None
    )
    with pytest.raises(RuntimeError, match="dummy"):
        async with uploaded_agent_env(
            command(), {"A": "1"}, root_exec=root, upload=upload
        ):
            raise RuntimeError("dummy")
    assert root.await_count == 2
    cleanup = root.await_args.args[0]
    assert "/bin/rm -f -- /.runtime-agent-env-" in cleanup
    assert "rm -rf" not in cleanup
