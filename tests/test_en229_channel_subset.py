"""Guards the fix against the full-tree staging cost in commit 0567f965."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._runtime_tree import (
    KIT,
    KIT_CHANNEL_PYTHON,
    KIT_PYTHON,
    kit_load_command,
    kit_prepare_command,
)
from benchflow.sandbox import _channel_python as channel_python
from benchflow.sandbox import _delegate_disarm as stage
from benchflow.sandbox import daytona_channel as channel
from benchflow.sandbox._base import KIT_PROBE_ROUTE, ExecResult
from benchflow.sandbox._channel_python_files import DIRECTORIES_BY_TREE, FILES_BY_TREE
from benchflow.sandbox._channel_python_probe import startup_source
from benchflow.sandbox._exec_channel import HELPER_SOURCE, VERDICT_SOURCE, ExecChannel
from benchflow.sandbox._staged_python_layer import TREE_SHA256
from tests.test_daytona_channel_contract import _exec_box
from tests.test_en229_layer_build_failures_docker import _docker
from tests.test_en229_python_layer import _digest
from tools.channel_python_files import derive, extract

pytestmark = pytest.mark.timeout(30)
PIN = stage.STAGED_PYTHON_SHA256["x86_64", "gnu"]
FILES = FILES_BY_TREE[TREE_SHA256[PIN]]


@pytest.fixture(scope="module")
def pinned_tree(tmp_path_factory):
    archive = os.environ.get("BENCHFLOW_TEST_CHANNEL_ARCHIVE")
    if not archive or platform.system() != "Linux" or platform.machine() != "x86_64":
        pytest.skip(
            "set BENCHFLOW_TEST_CHANNEL_ARCHIVE to the pinned x86_64 GNU archive"
        )
    root = tmp_path_factory.mktemp("channel-source")
    extract(Path(archive), root)
    assert _digest(root, "--owner=0", "--group=0") == TREE_SHA256[PIN]
    return root


@pytest.fixture
def subset(pinned_tree, tmp_path, monkeypatch):
    root = tmp_path / "subset"
    monkeypatch.setattr(channel_python, "CHANNEL_TREE", str(root))
    monkeypatch.setattr(stage, "staged_python_dir", lambda *_: str(pinned_tree))
    subprocess.run(
        ["/bin/sh", "-c", channel_python.copy_command("x86_64", "gnu")], check=True
    )
    return root


@pytest.mark.parametrize("locale", ["C", "C.UTF-8"])
def test_file_set_is_derived_from_pinned_startup_and_actual_helper(
    pinned_tree, subset, locale
):
    """Guards 0567f965: derive, do not guess, Python startup/import/loaded-library files."""
    binary = "python/bin/python3.12"
    expected = {"files": FILES, "directories": DIRECTORIES_BY_TREE[TREE_SHA256[PIN]]}
    assert derive(pinned_tree / binary, pinned_tree, locale) == expected
    assert derive(subset / binary, subset, locale) == expected
    actual = tuple(
        sorted(str(p.relative_to(subset)) for p in subset.rglob("*") if p.is_file())
    )
    assert actual == FILES
    for file in FILES:
        assert (subset / file).read_bytes() == (pinned_tree / file).read_bytes()
    assert not list(subset.rglob("__pycache__"))


@pytest.mark.parametrize("preflight", [False, True])
async def test_full_and_subset_helper_have_identical_results(
    pinned_tree, subset, tmp_path, preflight
):
    """Guards 0567f965: less copying preserves commands, streams, cwd and environment."""
    results = []
    for tree in (pinned_tree, subset):
        source = HELPER_SOURCE
        if tree == subset and preflight:
            source = (
                startup_source((channel._PTY_PRELUDE, HELPER_SOURCE, VERDICT_SOURCE))
                + source
            )
        helper = ExecChannel(
            [
                str(tree / "python/bin/python3.12"),
                "-I",
                "-S",
                "-B",
                "-c",
                source,
            ],
            {"PATH": "/usr/bin:/bin", "LANG": "C", "BF_DATA": "unchanged"},
            workdir=str(tmp_path),
            logger=logging.getLogger(__name__),
        )
        await helper.start()
        try:
            result = await helper.run(
                [
                    "/bin/sh",
                    "-c",
                    "env | sort; pwd; printf 'a\\000b'; printf err >&2; exit 7",
                ],
                cwd=str(tmp_path),
                stdin_data=None,
            )
            results.append((result.return_code, result.stream(1), result.stream(2)))
        finally:
            await helper.close()
    assert results[0] == results[1]
    assert results[0][0] == 7 and results[0][2] == b"err"
    assert results[0][1].endswith(b"a\0b")


@pytest.mark.parametrize("missing", ["selectors.py", "tty.py", "encodings/__init__.py"])
async def test_failed_import_precedes_channel_acceptance_and_one_session_runs(
    subset,
    tmp_path,
    monkeypatch,
    missing,
):
    """Guards 0567f965: real import failure accepts no channel command or duplicate."""
    output = await _expect_failed_start(
        subset, tmp_path, monkeypatch, "python/lib/python3.12/" + missing
    )
    if missing.startswith("encodings/"):
        assert b"Fatal Python error" in output and b"encoding" in output
    else:
        assert b"ModuleNotFoundError" in output


async def test_review_missing_json_initializer_fails_before_startup(
    subset, tmp_path, monkeypatch
):
    """Guards d055ca6d's namespace JSON regression before READY."""
    await _expect_failed_start(
        subset, tmp_path, monkeypatch, "python/lib/python3.12/json/__init__.py"
    )


@pytest.mark.parametrize("missing", FILES)
async def test_every_derived_file_is_required_before_ready(
    subset, tmp_path, monkeypatch, missing
):
    """Guards d055ca6d: every missing derived file selects exactly one session."""
    await _expect_failed_start(subset, tmp_path, monkeypatch, missing)


async def _expect_failed_start(subset, tmp_path, monkeypatch, missing):
    if shutil.which("script") is None:
        pytest.skip("util-linux script required for a real controlling terminal")
    (subset / missing).unlink()
    box, _old_pool, _old_channel = _exec_box(monkeypatch)
    provider = box._sandbox
    payload = tmp_path / "payload"
    side_effect = tmp_path / "accepted"
    command = f"printf 'once\\n' >> {shlex.quote(str(side_effect))}"
    launches, sends, readers, processes, output = [], [], [], [], bytearray()

    async def stage_source(self, source):
        payload.write_bytes(source)
        Path(str(payload) + ".env").write_bytes(b"PATH=/usr/bin:/bin\0LANG=C\0")
        Path(str(payload) + ".cwd").write_text(str(tmp_path) + "\n")
        Path(str(payload) + ".mask").write_text("0022\n")
        return str(payload)

    async def control(self, command):
        if command == f"{channel.KIT_BIN}/cat {KIT_PYTHON}":
            return ExecResult(return_code=1)
        if command == f"{channel.KIT_BIN}/cat {KIT_CHANNEL_PYTHON}":
            return ExecResult(return_code=0, stdout=channel_python.wrapper())
        return ExecResult(return_code=0)

    async def connect(self, command, route):
        launches.append(command)
        argv = [
            str(subset / "python/bin/python3.12"),
            "-I",
            "-S",
            "-B",
            str(payload),
            self.journal,
            str(payload),
        ]
        process = await asyncio.create_subprocess_exec(
            "script",
            "-qefc",
            shlex.join(argv),
            "/dev/null",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(process)

        async def send(data):
            sends.append(data)
            assert process.stdin is not None
            process.stdin.write(data)
            await process.stdin.drain()

        async def pump():
            assert process.stdout is not None
            while data := await process.stdout.read(65536):
                output.extend(data)
                await self.on_data(data)
            await process.wait()
            self.die("helper exited")

        self.pty = SimpleNamespace(
            send_input=send, kill=AsyncMock(), disconnect=AsyncMock()
        )
        readers.append(asyncio.create_task(pump()))
        await asyncio.wait_for(asyncio.shield(self.ready), 5)

    execute = provider.process.execute_session_command

    async def accept(session_id, request, timeout=None):
        result = await execute(session_id, request, timeout)
        assert not side_effect.exists()
        subprocess.run(["/bin/sh", "-c", command], check=True)
        return result

    monkeypatch.setattr(channel.DaytonaChannels, "_stage_source", stage_source)
    monkeypatch.setattr(channel.DaytonaChannels, "_control", control)
    monkeypatch.setattr(channel.DaytonaChannel, "connect", connect)
    monkeypatch.setattr(provider.process, "execute_session_command", accept)
    pool = channel.DaytonaChannels(box, KIT_PROBE_ROUTE)
    box._command_channels = pool
    try:
        assert await pool.starting is None
        assert all(c.marker not in output for c in pool.channels)
        assert not Path(str(payload) + ".started").exists()
        assert not side_effect.exists()
        assert not sends
        result = await box.exec(command, user="root", timeout_sec=5)
        assert result.return_code == 0
        assert side_effect.read_text() == "once\n"
        assert len(provider.process.executed) == len(provider.process.created) == 1
        assert len(launches) == 1 and not sends
        assert box._exec_counts().channel_starts == 0
        assert all(not c.alive and not c.pending for c in pool.channels)
    finally:
        pool.finalize()
        await pool.join()
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()
        await asyncio.gather(*readers)
    return output


def test_startup_probe_derives_attribute_checks_from_helper(pinned_tree):
    """Guards d055ca6d: a future module call is checked before it can be accepted."""
    source = startup_source(
        ("import os, sys, json\ndef later():\n    json.missing_channel_api()\n",)
    )
    result = subprocess.run(
        [str(pinned_tree / "python/bin/python3.12"), "-I", "-S", "-B", "-c", source],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "AttributeError" in result.stderr and "missing_channel_api" in result.stderr


def test_startup_probe_rejects_lazy_imports():
    """Guards d055ca6d: derivation and startup share the eager-import requirement."""
    with pytest.raises(ValueError, match="precede command acceptance"):
        startup_source(("def later():\n    import pathlib\n",))


@pytest.mark.parametrize("damage", ["namespace", "not-callable"])
def test_startup_probe_requires_real_modules_and_callable_attributes(subset, damage):
    """Guards d055ca6d: successful imports alone do not establish functionality."""
    initializer = subset / "python/lib/python3.12/json/__init__.py"
    if damage == "namespace":
        initializer.unlink()
        sources = ("import os, sys, json\n",)
    else:
        with initializer.open("a") as stream:
            stream.write("\nloads = None\n")
        sources = (channel._PTY_PRELUDE, HELPER_SOURCE, VERDICT_SOURCE)
    result = subprocess.run(
        [
            str(subset / "python/bin/python3.12"),
            "-I",
            "-S",
            "-B",
            "-c",
            startup_source(sources),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ImportError: nonfunctional channel dependency" in result.stderr


def test_underived_platforms_keep_the_full_authenticated_copy(monkeypatch):
    """Guards 0567f965: a different tree cannot inherit a guessed subset."""
    monkeypatch.setattr(channel_python, "FILES_BY_TREE", {})
    assert channel_python.copy_command("x86_64", "gnu") == (
        f"cp -a {stage.staged_python_dir('x86_64', 'gnu')} {channel_python.CHANNEL_TREE}"
    )


@pytest.mark.timeout(120)
@pytest.mark.parametrize("tamper_source", [False, True])
def test_full_proof_and_seal_cover_source_and_every_copied_file(
    pinned_tree,
    tmp_path,
    monkeypatch,
    tamper_source,
):
    """Guards 0567f965: full proof still rejects omitted bytes; subset stays sealed."""
    if shutil.which("docker") is None:
        pytest.skip("Docker required")
    image = _docker(
        "image", "inspect", "--format", "{{.Id}}", "ubuntu:24.04", check=False
    )
    if image.returncode:
        pytest.skip("cached Ubuntu fixture required; never pulls")
    source = stage.staged_python_dir("x86_64", "gnu")
    setup = f"mkdir -p {source}; cp -a /input/. {source}; chown -R 0:0 {source}; chmod 700 {source};\n"
    if tamper_source:
        omitted = "python/lib/python3.12/email/message.py"
        assert omitted not in FILES
        setup += f"echo changed >> {source}/{omitted}\n"
    commands = [kit_prepare_command()]
    with monkeypatch.context() as patch:
        patch.setattr(channel_python, "FILES_BY_TREE", {})
        commands.append(kit_prepare_command())
    results = []
    output = tmp_path / "output"
    output.mkdir()
    for index, command in enumerate(commands):
        tail = f"\ntest {'! -e' if tamper_source else '-x'} {KIT_CHANNEL_PYTHON} || exit 91\n"
        if not tamper_source:
            tail += (
                f"find {channel_python.CHANNEL_TREE} -type f | sort > /output/files-{index}\n"
                f"cp {KIT}/MANIFEST /output/manifest-{index}\n"
                f"echo changed >> {channel_python.CHANNEL_TREE}/python/lib/python3.12/selectors.py\n"
            )
        else:
            tail += f"echo changed >> {KIT}/IMAGE_PYTHON_GUARD\n"
        script = output / f"kit-{index}.sh"
        script.write_text(setup + command + tail + kit_load_command())
        result = _docker(
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "root",
            "--name",
            f"en229-l13-subset-{uuid.uuid4().hex[:12]}",
            "-v",
            f"{pinned_tree}:/input:ro",
            "-v",
            f"{output}:/output",
            image.stdout.strip(),
            "sh",
            f"/output/{script.name}",
            check=False,
        )
        assert result.returncode not in (0, 91), result.stdout + result.stderr
        assert "changed since BenchFlow made" in result.stderr
        results.append((result.returncode, result.stdout, result.stderr))
    assert results[0] == results[1]
    if not tamper_source:
        actual = (output / "files-0").read_text().splitlines()
        assert actual == [f"{channel_python.CHANNEL_TREE}/{file}" for file in FILES]
        manifest = {
            line.split("  ", 1)[1]
            for line in (output / "manifest-0").read_text().splitlines()
        }
        assert {"./channel/" + file for file in FILES} <= manifest
        assert len((output / "files-1").read_text().splitlines()) > len(actual)
