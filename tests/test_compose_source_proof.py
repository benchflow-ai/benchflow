"""Guards S1-CONF-4/5 at 8484e852: unknown is not disjoint, including subtrees."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from benchflow.task import compose_env
from benchflow.task.compose_env import _ProjectCheck, check_compose_in_vm
from tests.test_compose_mutable_sources import _project
from tests.test_compose_path_aliases import CONSUMERS

WRITERS = ["bind", "sidecar", "volume", "volume-rbind", "inherited"]


@pytest.fixture(autouse=True)
def _no_host_optin(monkeypatch):
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)


def _fixture(tmp_path, writer, consumer):
    root = tmp_path / "work"
    (root / "sub/nested").mkdir(parents=True)
    (tmp_path / "export/data").mkdir(parents=True)
    (tmp_path / "build").mkdir()
    return root, _project(root, tmp_path / "export/data", writer, consumer)


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("unavailable", ["root", "parent"])
def test_unavailable_identity_never_proves_disjointness(
    tmp_path, monkeypatch, writer, consumer, unavailable
):
    """Guards CONF-4: missing/unreadable/unstatable roots and ancestors refuse."""
    root, project = _fixture(tmp_path, writer, consumer)
    missing = str(root if unavailable == "root" else tmp_path / "export")
    identity = compose_env._local_identity
    monkeypatch.setattr(
        compose_env, "_local_identity", lambda p: None if p == missing else identity(p)
    )
    refused = _ProjectCheck(str(tmp_path), (str(tmp_path),)).check(project)
    assert refused, "an unavailable identity was accepted as disjoint"
    assert any("read-only" in r.describe() for r in refused)


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("alias", ["sub", "sub/nested"])
def test_bind_alias_of_any_writable_subtree_is_refused(
    tmp_path, monkeypatch, writer, consumer, alias
):
    """Guards CONF-5: a host bind can name any descendant, not only the root."""
    root, project = _fixture(tmp_path, writer, consumer)
    identity = compose_env._local_identity
    # Model the kernel's identity at a bind mountpoint without CAP_SYS_ADMIN.
    monkeypatch.setattr(
        compose_env,
        "_local_identity",
        lambda p: (
            identity(str(root / alias))
            if p == str(tmp_path / "export")
            else identity(p)
        ),
    )
    refused = _ProjectCheck(str(tmp_path), (str(tmp_path),)).check(project)
    assert any("writable over the launch source" in r.describe() for r in refused)


@pytest.mark.parametrize("writer", WRITERS)
def test_missing_writable_root_is_refused_on_the_real_filesystem(tmp_path, writer):
    """Guards CONF-4 without mocks: Compose must not create an unchecked root."""
    (tmp_path / "input/data").mkdir(parents=True)
    project = _project(tmp_path / "missing", tmp_path / "input/data", writer, "bind")
    refused = _ProjectCheck(str(tmp_path), (str(tmp_path),)).check(project)
    assert refused
    assert any(
        "missing" in r.describe() and "read-only" in r.describe() for r in refused
    )


@pytest.mark.parametrize("backend", ["docker", "vm"])
@pytest.mark.parametrize("fault", ["unreadable", "unstattable"])
@pytest.mark.asyncio
async def test_incomplete_writable_tree_is_refused(
    tmp_path, monkeypatch, backend, fault
):
    """Guards CONF-5: a failed subtree inspection cannot produce clearance."""
    root, project = _fixture(tmp_path, "bind", "bind")
    if backend == "docker":
        real = os.scandir

        def scandir(path):
            if str(path) == str(root / "sub"):
                raise PermissionError("dummy unreadable subtree")
            return real(path)

        monkeypatch.setattr(os, "scandir", scandir)
        if fault == "unstattable":
            identity = compose_env._local_identity
            monkeypatch.setattr(
                compose_env,
                "_local_identity",
                lambda p: None if p == str(root / "sub") else identity(p),
            )
        refused = _ProjectCheck(str(tmp_path), (str(tmp_path),)).check(project)
    else:

        async def run(command):
            argv = shlex.split(command)
            if "config" in argv:
                return SimpleNamespace(return_code=0, stdout=json.dumps(project))
            if argv[:2] == ["sh", "-c"]:
                if "find" in argv[2]:
                    return SimpleNamespace(return_code=1, stdout="1:1\n")
                result = subprocess.run(argv, capture_output=True, text=True)
                return SimpleNamespace(
                    return_code=result.returncode, stdout=result.stdout
                )
            return SimpleNamespace(return_code=0, stdout="x$$y")

        refused = await check_compose_in_vm(
            run, ["docker", "compose"], str(tmp_path), [str(tmp_path)]
        )
    assert refused, "incomplete writable-tree inspection must fail closed"


@pytest.mark.parametrize("missing", ["root", "ancestor"])
@pytest.mark.asyncio
async def test_vm_individual_missing_stat_is_refused(tmp_path, monkeypatch, missing):
    """Guards CONF-4 on DinD even when the batch exits zero and has every row."""
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    root, project = _fixture(tmp_path, "bind", "bind")
    missing_path = str(root if missing == "root" else tmp_path / "export")

    async def run(command):
        argv = shlex.split(command)
        if "config" in argv:
            return SimpleNamespace(return_code=0, stdout=json.dumps(project))
        if argv[:2] == ["sh", "-c"]:
            if "find" not in argv[2] and "stat" in argv[2]:
                values = [
                    "-"
                    if p == missing_path
                    else ":".join(map(str, compose_env._local_identity(p)))
                    for p in argv[4:]
                ]
                return SimpleNamespace(return_code=0, stdout="\n".join(values))
            result = subprocess.run(argv, capture_output=True, text=True)
            return SimpleNamespace(return_code=result.returncode, stdout=result.stdout)
        return SimpleNamespace(return_code=0, stdout="x$$y")

    assert await check_compose_in_vm(
        run, ["docker", "compose"], str(tmp_path), [str(tmp_path)]
    )


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_existing_disjoint_trees_still_pass(tmp_path, consumer):
    """Guards 8484e852's honest independent paths while proving their identities."""
    _, project = _fixture(tmp_path, "bind", consumer)
    assert not _ProjectCheck(str(tmp_path), (str(tmp_path),)).check(project)


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.asyncio
async def test_vm_bind_alias_of_writable_subtree_is_refused(tmp_path, writer, consumer):
    """Guards CONF-5 through the final DinD check, with real subtree inspection."""
    root, project = _fixture(tmp_path, writer, consumer)

    async def run(command):
        argv = shlex.split(command)
        if "config" in argv:
            return SimpleNamespace(return_code=0, stdout=json.dumps(project))
        if argv[:2] == ["sh", "-c"]:
            if "find" not in argv[2] and "stat" in argv[2]:

                def identity(path):
                    if path == str(tmp_path / "export"):
                        path = str(root / "sub")
                    return ":".join(map(str, compose_env._local_identity(path)))

                return SimpleNamespace(
                    return_code=0, stdout="\n".join(map(identity, argv[4:]))
                )
            result = subprocess.run(argv, capture_output=True, text=True)
            return SimpleNamespace(return_code=result.returncode, stdout=result.stdout)
        return SimpleNamespace(return_code=0, stdout="x$$y")

    refused = await check_compose_in_vm(
        run, ["docker", "compose"], str(tmp_path), [str(tmp_path)]
    )
    assert any("writable over the launch source" in r.describe() for r in refused)


def test_vm_realpath_keeps_the_root_identity():
    """Guards CONF-4's fail-closed ancestry proof: / must not become empty."""
    result = subprocess.run(
        ["sh", "-c", compose_env._VM_REALPATH, "realpath", "/"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "/\n"


@pytest.mark.parametrize(
    "fault",
    [
        "failed",
        "short",
        "malformed",
        "incomplete",
        "dropped-prefix",
        "dropped-middle",
        "dropped-tail",
    ],
)
@pytest.mark.asyncio
async def test_vm_requires_complete_filesystem_receipts(tmp_path, fault):
    """Guards CONF-4/5: a partial resolver or tree receipt never clears a writer."""
    _, project = _fixture(tmp_path, "bind", "bind")

    async def run(command):
        argv = shlex.split(command)
        if "config" in argv:
            return SimpleNamespace(return_code=0, stdout=json.dumps(project))
        if argv[:2] == ["sh", "-c"]:
            if "readlink" in argv[2] and fault in ("failed", "short"):
                return SimpleNamespace(return_code=int(fault == "failed"), stdout="")
            if "find" in argv[2] and fault in ("malformed", "incomplete"):
                return SimpleNamespace(
                    return_code=0,
                    stdout="not-an-identity\ncomplete\n"
                    if fault == "malformed"
                    else "1:1\n",
                )
            result = subprocess.run(argv, capture_output=True, text=True)
            if "find" in argv[2] and fault.startswith("dropped-"):
                lines = result.stdout.splitlines()
                index = {"dropped-prefix": 0, "dropped-middle": 1, "dropped-tail": -2}[
                    fault
                ]
                lines.pop(index)
                result.stdout = "\n".join(lines) + "\n"
            return SimpleNamespace(return_code=result.returncode, stdout=result.stdout)
        return SimpleNamespace(return_code=0, stdout="x$$y")

    assert await check_compose_in_vm(
        run, ["docker", "compose"], str(tmp_path), [str(tmp_path)]
    )


def test_local_tree_budget_limits_queued_directories(tmp_path, monkeypatch):
    """Guards the CONF-5 scan at 42e70417: bound the population before queuing it."""
    for i in range(8):
        (tmp_path / str(i)).mkdir()
    monkeypatch.setattr(compose_env, "_SOURCE_DIRECTORY_LIMIT", 4, raising=False)
    assert compose_env._local_tree_identities(str(tmp_path)) is None


def test_vm_tree_budget_limits_output_at_the_producer(tmp_path):
    """Guards the CONF-5 VM scan at 42e70417: oversized receipts fail, not truncate."""
    for i in range(8):
        (tmp_path / str(i)).mkdir()
    result = subprocess.run(
        ["sh", "-c", compose_env._VM_TREE, "tree", str(tmp_path), "4"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert len(result.stdout.splitlines()) <= 4
    assert not any(line.startswith("complete") for line in result.stdout.splitlines())


@pytest.mark.parametrize("character", [" ", "\t", "\r", "\u0085", "\u2028", "\u2029"])
@pytest.mark.asyncio
async def test_vm_honest_path_whitespace_is_not_a_record_separator(tmp_path, character):
    """Guards the CONF-4 proof at 42e70417: preserve every non-LF path character."""
    folder = tmp_path / f"operator{character}root"
    _, project = _fixture(folder, "bind", "bind")

    async def run(command):
        argv = shlex.split(command)
        if "config" in argv:
            return SimpleNamespace(return_code=0, stdout=json.dumps(project))
        if argv[:2] == ["sh", "-c"]:
            result = subprocess.run(argv, capture_output=True, text=False)
            return SimpleNamespace(
                return_code=result.returncode, stdout=result.stdout.decode()
            )
        return SimpleNamespace(return_code=0, stdout="x$$y")

    assert not await check_compose_in_vm(
        run, ["docker", "compose"], str(folder), [str(folder)]
    )
