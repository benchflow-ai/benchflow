"""A channel interpreter never replaces verifier Python."""

from __future__ import annotations

import logging
import runpy
import shlex
import textwrap
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._runtime_tree import KIT, KIT_CHANNEL_PYTHON, KIT_PYTHON
from benchflow.sandbox import _channel_python as source
from benchflow.sandbox import daytona_channel as channel
from benchflow.sandbox._base import KIT_PROBE_ROUTE
from benchflow.sandbox._exec_channel import helper_python_argv
from tests._root_stand_in import run_sh
from tests.daytona_channel_fakes import _lifecycle_sandbox
from tests.test_en229_python_layer import tree as tree

pytestmark = pytest.mark.timeout(30)
FIXTURES = Path(__file__).parent / "fixtures/en229_channel"
DIRECT = f'#!{KIT}/bin/sh\nexec {KIT}/libexec/python3 -I -S -B "$@"\n'
LOADER = DIRECT.replace(
    "exec ", f"exec /lib64/ld-linux-x86-64.so.2 --library-path {KIT}/lib "
)


@pytest.mark.parametrize(
    "wrapper",
    [
        DIRECT,
        LOADER,
        DIRECT.rstrip("\n"),
        LOADER.replace("/lib64/", "/srv/"),
        DIRECT.replace("-I -S -B", "-S -B"),
        DIRECT.replace("libexec", "../workspace"),
        DIRECT + "echo injected\n",
        DIRECT + "\x00",
        "",
        source.wrapper(),
        source.wrapper().replace("-I -S -B", "-S -B"),
        source.wrapper().replace("python3.12", "python3.12;echo injected"),
    ],
)
def test_legacy_wrapper_parser_is_equivalent(wrapper):
    """Run the frozen integ3 parser and the new default on identical input."""
    old = runpy.run_path(str(FIXTURES / "legacy_parser.py.txt"))["helper_python_argv"]
    assert helper_python_argv(wrapper) == old(wrapper)
    if wrapper == source.wrapper():
        assert helper_python_argv(wrapper, channel_python=True) == [
            source.CHANNEL_BINARY,
            "-I",
            "-S",
            "-B",
        ]
    else:
        assert helper_python_argv(wrapper, channel_python=True) == old(wrapper)


@pytest.fixture
def kit(tree, monkeypatch):
    kit = tree.parent / "kit"
    (kit / "py").mkdir(parents=True)
    (kit / "RECORD").write_text("kit=1\npython=0\nbash=1\ntrusted=/usr/bin\n")
    monkeypatch.setattr(source, "KIT_PYTHON", str(kit / "py/python3"))
    monkeypatch.setattr(source, "KIT_CHANNEL_PYTHON", str(kit / "py/channel-python3"))
    monkeypatch.setattr(source, "CHANNEL_TREE", str(kit / "channel"))
    monkeypatch.setattr(
        source, "CHANNEL_BINARY", str(kit / "channel/python/bin/python3.12")
    )
    return kit


@pytest.mark.parametrize(
    "state", ["valid", "absent", "bytes", "mode", "link", "image-python", "copy-fails"]
)
def test_preparation_preserves_old_records_and_never_runs_the_source(tree, kit, state):
    """Authentication failure has the old no-channel result and records."""
    binary = tree / "python/bin/python3.12"
    if state == "absent":
        binary.unlink()
    elif state == "bytes":
        binary.write_text("#!/bin/sh\nexit 0\n")
    elif state == "mode":
        binary.chmod(0o777)
    elif state == "link":
        binary.unlink()
        binary.symlink_to("/usr/bin/python3")
    elif state == "image-python":
        image = kit / "py/python3"
        image.write_text(DIRECT)
        image.chmod(0o755)
    before = {p.relative_to(kit): p.read_bytes() for p in kit.rglob("*") if p.is_file()}
    # Integ3 did no channel preparation at this boundary.
    old = run_sh(":", capture_output=True, text=True)
    command = source.prepare_command()
    if state == "copy-fails":
        command = "cp() { return 1; }; " + command
    new = run_sh(command, capture_output=True, text=True)
    assert (new.returncode, new.stdout, new.stderr) == (
        old.returncode,
        old.stdout,
        old.stderr,
    )
    assert not (tree.parent / "executed").exists()
    assert {p: (kit / p).read_bytes() for p in before} == before
    if state == "valid":
        assert Path(source.KIT_CHANNEL_PYTHON).read_text() == source.wrapper()
        assert Path(source.CHANNEL_BINARY).read_bytes() == binary.read_bytes()
        assert not Path(source.KIT_PYTHON).exists()
    else:
        assert not Path(source.KIT_CHANNEL_PYTHON).exists()
        assert not Path(source.CHANNEL_TREE).exists()


def test_new_preparation_is_only_inside_the_original_build_boundary():
    """No source probe or copy is moved after the agent or into kit loading."""
    from benchflow._runtime_tree import kit_load_command, kit_prepare_command

    added = source.prepare_command()
    assert added in kit_prepare_command("build")
    assert added in kit_prepare_command("all")
    assert added not in kit_prepare_command("seal")
    assert added not in kit_prepare_command(
        "restore", stage="/.runtime-kit-" + "a" * 32
    )
    assert added not in kit_load_command()
    build = kit_prepare_command("build")
    assert build.index(added) < build.index('echo "BF kit-built"')


async def _startup(
    monkeypatch, *, old, kit_wrapper=DIRECT, fallback=None, failure=None
):
    box, _ = _lifecycle_sandbox(monkeypatch)
    provider = box._sandbox
    provider.fs = SimpleNamespace(upload_file=AsyncMock())
    calls = []

    async def control(self, command):
        calls.append(command)
        self.sandbox._exec_counts().channel_setup_requests += 1
        if failure == "read":
            raise OSError("/private/path/with-sensitive-data")
        if command == f"{channel.KIT_BIN}/cat {KIT_PYTHON}":
            return channel.ExecResult(
                return_code=1 if kit_wrapper is None else 0, stdout=kit_wrapper or ""
            )
        if command == f"{channel.KIT_BIN}/cat {KIT_CHANNEL_PYTHON}":
            return channel.ExecResult(
                return_code=1 if fallback is None else 0, stdout=fallback or ""
            )
        return channel.ExecResult(return_code=1 if failure == "capture" else 0)

    launches = []

    async def connect(self, command, route):
        launches.append(command)
        if failure == "connect":
            raise OSError("/private/path/with-sensitive-data")
        self.pid = 123
        self.ready.set_result(123)

    monkeypatch.setattr(channel.DaytonaChannels, "_control", control)
    monkeypatch.setattr(channel.DaytonaChannel, "connect", connect)
    monkeypatch.setattr(channel.uuid, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    monkeypatch.setattr(
        channel._transport,
        "stage_folder",
        lambda: "/opt/runtime/run/stage-" + "a" * 32,
    )
    if old:
        namespace = vars(channel).copy()
        exec(textwrap.dedent((FIXTURES / "legacy_start.py.txt").read_text()), namespace)
        monkeypatch.setattr(channel.DaytonaChannels, "_start", namespace["_start"])
    pool = channel.DaytonaChannels(box, KIT_PROBE_ROUTE)
    started = await pool.starting
    result = None
    if started is None:
        assert await pool.run("echo answer", KIT_PROBE_ROUTE, 5) is None
        # The unchanged session path supplies the same result and diagnostics.
        result = await box._sandbox_session_exec(
            "echo answer", timeout_sec=5, route=KIT_PROBE_ROUTE, cleanup_session=False
        )
    counts = asdict(box._exec_counts())
    pool.finalize()
    await pool.join()
    return started is not None, calls, launches, result, counts


@pytest.mark.parametrize("wrapper", [DIRECT, LOADER, "malformed"])
async def test_existing_startup_matches_frozen_integ3(monkeypatch, wrapper):
    """Usable and refused image wrappers keep the same requests and answers."""
    with monkeypatch.context() as patch:
        old = await _startup(patch, old=True, kit_wrapper=wrapper)
    with monkeypatch.context() as patch:
        new = await _startup(patch, old=False, kit_wrapper=wrapper)
    assert old[:4] == new[:4]
    for counts in (old[4], new[4]):
        counts.pop("exec_seconds")
    assert old[4] == new[4]


async def test_verified_channel_wrapper_starts_only_the_channel(monkeypatch):
    """Missing image Python selects the separate sealed interpreter."""
    started, calls, launches, _, counts = await _startup(
        monkeypatch, old=False, kit_wrapper=None, fallback=source.wrapper()
    )
    assert started and counts["channel_starts"] == 1
    assert calls[:2] == [
        f"{channel.KIT_BIN}/cat {KIT_PYTHON}",
        f"{channel.KIT_BIN}/cat {KIT_CHANNEL_PYTHON}",
    ]
    assert shlex.join([source.CHANNEL_BINARY, "-I", "-S", "-B"]) in launches[0]
    assert KIT_PYTHON not in launches[0]


@pytest.mark.parametrize("fallback", [None, "tampered", source.wrapper() + "extra\n"])
async def test_failed_or_tampered_fallback_keeps_the_old_session_result(
    monkeypatch, fallback, caplog
):
    """No verified interpreter means sessions, with unchanged record fields."""
    caplog.set_level(logging.INFO)
    with monkeypatch.context() as patch:
        old = await _startup(patch, old=True, kit_wrapper=None)
    with monkeypatch.context() as patch:
        new = await _startup(patch, old=False, kit_wrapper=None, fallback=fallback)
    assert not old[0] and not new[0] and old[3] == new[3]
    assert old[4].keys() == new[4].keys()
    for name in old[4]:
        if name not in {"channel_setup_requests", "exec_seconds"}:
            assert old[4][name] == new[4][name]
    assert "Daytona command channel did not start:" in caplog.text
    assert KIT not in caplog.text


@pytest.mark.parametrize("failure", ["read", "capture", "connect"])
async def test_start_failure_logs_fixed_reason_without_exception_paths(
    monkeypatch, caplog, failure
):
    """Startup failures are visible at info level without private paths."""
    caplog.set_level(logging.INFO)
    await _startup(monkeypatch, old=False, failure=failure)
    assert "Daytona command channel did not start:" in caplog.text
    assert "/private/" not in caplog.text and KIT not in caplog.text
