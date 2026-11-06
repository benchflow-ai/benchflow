"""Guards benchflow-ai/benchflow#1199: a verifier timeout kept nothing.

Tests still running at the verifier's deadline score 0, and the rollout
cancelled only its own await: test.sh and its children ran on in the
sandbox, and ``/logs/verifier`` was never brought in, so a Daytona rollout
kept an empty ``verifier/`` and nothing showed how far the tests had got.
The rollout now kills the tests' process tree and keeps what they wrote,
with each reward file renamed ``timeout-<name>`` (a timeout is no verdict),
and records it in ``verifier_timeout_info``. The score stays 0.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
from pathlib import Path

import pytest

from benchflow.rollout import _setup
from benchflow.sandbox._base import ExecResult
from tests._receipt_fakes import receipt_aware
from tests.test_verifier_lost_exec import _Sandbox, _score
from tests.test_verifier_lost_exec import _setup as _lost_setup

_SCRIPT = (
    "echo '{\"steps\": 3}' > /logs/verifier/partial.json\n"
    "echo tests-started\n"
    "echo 1 > /logs/verifier/reward.txt\n"
    "sleep 60\n"
)


#: chown where the test's shell (PATH /usr/bin:/bin) may not find it (macOS).
_CHOWN = shutil.which("chown", path="/usr/bin:/bin:/usr/sbin:/sbin") or "chown"


class _Remote(_Sandbox):
    """The real shell, with ``/logs/verifier`` in the sandbox, not mounted:
    the verifier downloads it, as from a Daytona sandbox. The test's user
    stands in for the sandbox's root: root's ``chown root:root`` of the
    folder it clears (#84) gives it to that user."""

    is_mounted = False

    def __init__(self, mapping, *, download_fails=False, download_hangs=False):
        super().__init__(mapping)
        self.download_fails = download_fails
        self.download_hangs = download_hangs

    def _map(self, text: str) -> str:
        return (
            super()
            ._map(text)
            .replace("chown root:root ", f"{_CHOWN} {os.getuid()}:{os.getgid()} ")
        )

    async def download_dir(self, source_dir, target_dir, service="main", **_kw):
        if self.download_hangs:
            await asyncio.Event().wait()
        if self.download_fails:
            raise ConnectionError("dummy: the folder download broke")
        shutil.copytree(
            self._map(str(source_dir)), target_dir, symlinks=True, dirs_exist_ok=True
        )

    async def download_file(self, source_path, target_path, service="main", **_kw):
        source = Path(self._map(str(source_path)))
        if not source.exists():
            raise FileNotFoundError(source_path)
        shutil.copyfile(source, target_path)


def _remote(tmp_path, script, cls=_Remote, **sandbox):
    task, paths, mounted = _lost_setup(tmp_path, script)
    mapping = dict(mounted._mapping)
    mapping["/logs/verifier"] = str(tmp_path / "sandbox-logs-verifier")
    return task, paths, cls(mapping, **sandbox)


def _names(paths):
    return sorted(entry.name for entry in paths.verifier_dir.iterdir())


async def test_timed_out_tests_keep_their_output_on_a_downloaded_sandbox(tmp_path):
    task, paths, env = _remote(tmp_path, _SCRIPT)

    rewards, error, diagnostic = await _score(task, paths, env)

    assert (rewards, error) == ({"reward": 0.0}, None)
    assert diagnostic is not None and diagnostic.scored is True
    verifier = paths.verifier_dir
    assert (verifier / "partial.json").read_text() == '{"steps": 3}\n'
    assert "tests-started" in (verifier / "test-stdout.txt").read_text()
    # A reward written before the deadline is kept, never as a verdict.
    assert "reward.txt" not in _names(paths)
    assert (verifier / "timeout-reward.txt").read_text() == "1\n"
    assert diagnostic.partial_output_files == 3
    assert diagnostic.partial_output_bytes > 0
    assert diagnostic.partial_output_error is None
    assert diagnostic.processes_killed == 0  # the fake's cancel took them
    assert diagnostic.to_dict()["partial_output_files"] == 3


async def test_timed_out_tests_on_mounted_outputs_keep_no_verdict(tmp_path):
    task, paths, env = _lost_setup(tmp_path, _SCRIPT)

    rewards, error, diagnostic = await _score(task, paths, env)

    assert (rewards, error) == ({"reward": 0.0}, None)
    assert "reward.txt" not in _names(paths)
    assert (paths.verifier_dir / "timeout-reward.txt").read_text() == "1\n"
    assert (paths.verifier_dir / "partial.json").exists()
    assert diagnostic.partial_output_files == 3


async def test_a_failed_folder_download_keeps_the_files_one_by_one(tmp_path):
    task, paths, env = _remote(tmp_path, _SCRIPT, download_fails=True)

    rewards, error, diagnostic = await _score(task, paths, env)

    assert (rewards, error) == ({"reward": 0.0}, None)
    assert "tests-started" in (paths.verifier_dir / "test-stdout.txt").read_text()
    assert (paths.verifier_dir / "timeout-reward.txt").read_text() == "1\n"
    assert "partial.json" not in _names(paths)  # not one of the named outputs
    assert diagnostic.partial_output_error is None


async def test_a_download_with_no_reward_is_kept_one_by_one_too(tmp_path):
    script = "echo tests-started\nsleep 60\n"
    task, paths, env = _remote(tmp_path, script, download_fails=True)

    rewards, error, diagnostic = await _score(task, paths, env)

    assert (rewards, error) == ({"reward": 0.0}, None)
    assert "tests-started" in (paths.verifier_dir / "test-stdout.txt").read_text()
    assert diagnostic.partial_output_error is None


async def test_keeping_the_output_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(_setup, "_TIMED_OUT_OUTPUT_BUDGET_SEC", 0.5)
    task, paths, env = _remote(tmp_path, _SCRIPT, download_hangs=True)

    rewards, error, diagnostic = await _score(task, paths, env)

    assert (rewards, error) == ({"reward": 0.0}, None)
    assert diagnostic.partial_output_error == "not done within 0.5s"
    # Whatever the download did not bring, no reward lands as a verdict.
    assert "reward.txt" not in _names(paths)


async def test_a_reward_whose_timeout_name_is_taken_still_leaves_its_name(tmp_path):
    # The test command's own code puts a folder where the reward would go.
    script = (
        "mkdir -p /logs/verifier/timeout-reward.txt/x\n"
        "echo 1 > /logs/verifier/reward.txt\n"
        "sleep 60\n"
    )
    task, paths, env = _remote(tmp_path, script)

    rewards, error, diagnostic = await _score(task, paths, env)

    assert (rewards, error) == ({"reward": 0.0}, None)
    names = _names(paths)
    assert "reward.txt" not in names
    (aside,) = [
        n for n in names if n.startswith("timeout-") and n != "timeout-reward.txt"
    ]
    assert aside.endswith("-reward.txt")
    assert (paths.verifier_dir / aside).read_text() == "1\n"
    assert diagnostic.partial_output_error is None


async def test_an_unanswered_sandbox_is_not_killed_or_copied(tmp_path):
    task, paths, env = _lost_setup(
        tmp_path,
        "echo 1 > /logs/verifier/reward.txt\n",
        lose_exec=True,
        probe_fails=True,
    )

    rewards, _error, diagnostic = await _score(task, paths, env)

    assert rewards is None
    assert diagnostic is not None and diagnostic.scored is False
    assert diagnostic.processes_killed is None
    assert diagnostic.partial_output_files is None
    assert not any("killed $#" in command for command in env.commands)


class _Detached(_Remote):
    """A sandbox whose processes outlive a cancelled exec, as a remote
    sandbox's do: cancelling the host's await leaves test.sh running."""

    def __init__(self, mapping, **kwargs):
        super().__init__(mapping, **kwargs)
        self.processes = []

    @receipt_aware
    async def exec(self, command, env=None, **kwargs):
        self.commands.append(command)
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            self._map(command),
            env={"PATH": "/usr/bin:/bin", **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self.processes.append(process)
        stdout, stderr = await process.communicate()
        return ExecResult(
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            return_code=process.returncode,
        )


class _SentAsData(_Detached):
    """A sandbox that sends each command as data, on a shell's stdin, as one
    too long for a command line is sent: no process names the receipt."""

    @receipt_aware
    async def exec(self, command, env=None, **kwargs):
        self.commands.append(command)
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-s",
            env={"PATH": "/usr/bin:/bin", **(env or {})},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self.processes.append(process)
        stdout, stderr = await process.communicate(self._map(command).encode())
        return ExecResult(
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            return_code=process.returncode,
        )


@pytest.mark.skipif(
    not Path("/proc/self/status").exists(), reason="the kill reads /proc"
)
@pytest.mark.parametrize("cls", [_Detached, _SentAsData])
async def test_the_kill_takes_the_tests_whole_process_tree(tmp_path, cls):
    # test.sh's own child runs past the deadline; only the start receipt's
    # unique name, in the command line or (a command sent as data) in the
    # environment of the processes the command starts, leads the kill to
    # this test's tree.
    pidfile = tmp_path / "child.pid"
    script = f"sleep 300 &\necho $! > {pidfile}\nwait\n"
    task, paths, env = _remote(tmp_path, script, cls=cls)
    try:
        rewards, error, diagnostic = await _score(task, paths, env)
        assert (rewards, error) == ({"reward": 0.0}, None)
        child = int(pidfile.read_text())
        for _ in range(50):
            if not Path(f"/proc/{child}").exists():
                break
            await asyncio.sleep(0.1)
        assert not Path(f"/proc/{child}").exists()
        # test.sh and its sleep at least, beside the exec's own shells.
        assert diagnostic.processes_killed is not None
        assert diagnostic.processes_killed >= 2
    finally:
        for process in env.processes:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()


def test_the_kill_without_a_receipt_matches_the_capture_file():
    assert _setup._kill_verifier_command(None) == _setup._KILL_VERIFIER_SH
    assert _setup._kill_verifier_command(("/x/not-a-receipt", "main")) == (
        _setup._KILL_VERIFIER_SH
    )
    name = "verifier-started-" + "0" * 31 + "f"
    command = _setup._kill_verifier_command((f"/run/r/{name}", "main"))
    assert "grep -l 'verifier-started-" + "0" * 31 + "[f]'" in command
    assert "grep -qx 'BENCHFLOW_VERIFIER_ATTEMPT=verifier-started-" in command
    # Only this attempt's processes: never every test.sh, never the capture
    # file any attempt writes, and never the receipt's name as written, which
    # this script's own command line would match.
    assert "/logs/verifie[r]" not in command
    assert "test[.]sh" not in command
    assert name not in command
