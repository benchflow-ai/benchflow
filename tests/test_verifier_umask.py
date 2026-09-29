"""The verifier's commands run under umask 022, whatever the runtime's mask.

Guards the fix for the pytest plugin guard's dependence on the container
runtime (review of sdk-update-2026-09-27 at c5b75fcc, must-fix 3). The guard
trusts plugin code only when it and every parent directory are root-owned and
not group- or world-writable. BenchFlow set no mask for test.sh, and
``docker exec`` on Docker-in-Docker (Docker 29.8.1, runc 1.5.1) runs with
umask 0000, so uv's cache under ``/root/.cache/uv`` came out 0777/0666, the
guard refused the ``pytest-json-ctrf`` that Terminal-Bench 2's test.sh
installs with ``uvx``, and the correct oracle's trial ended unscored. On the
VM's own Docker (umask 0022) the same oracle scored 1.0.

The sandbox model here runs the verifier's real command string through
``/bin/sh`` in a process whose mask is 0000, as that runtime does.
"""

import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchflow.rollout._separate_verifier import _install_script
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.lockdown import VERIFIER_UMASK, with_verifier_umask
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig

# What test.sh does that matters here: it creates a directory and a file, the
# way uv lays out its cache or pip installs a package.
TEST_SH = """#!/bin/sh
mkdir -p "$OUT/cache/site-packages"
printf 'x' > "$OUT/cache/site-packages/plugin.py"
echo 1 > "$VERIFIER_DIR/reward.txt"
"""


class UmaskZeroSandbox:
    """Run commands through ``/bin/sh`` under umask 0000, sandbox paths mapped here."""

    is_mounted = True

    def __init__(self, tmp_path: Path, paths: RolloutPaths):
        self.tests = tmp_path / "sandbox-tests"
        self.out = tmp_path / "sandbox-out"
        self.out.mkdir()
        self.paths = paths
        self.commands: list[str] = []

    async def upload_dir(self, source_dir, target_dir, **kwargs):
        subprocess.run(["cp", "-R", str(source_dir), str(self.tests)], check=True)

    def local(self, command: str) -> str:
        return command.replace("/logs/verifier", str(self.paths.verifier_dir)).replace(
            "/tests/", f"{self.tests}/"
        )

    async def exec(self, command, env=None, **kwargs):
        self.commands.append(command)
        result = subprocess.run(
            ["/bin/sh", "-c", self.local(command)],
            env={
                "PATH": os.environ["PATH"],
                "OUT": str(self.out),
                "VERIFIER_DIR": str(self.paths.verifier_dir),
            },
            capture_output=True,
            text=True,
            timeout=30,
            preexec_fn=lambda: os.umask(0),
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )


def _task(tmp_path: Path) -> MagicMock:
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text(TEST_SH)
    task = MagicMock()
    task.name = "task"
    task.task_dir = tests_dir.parent
    task.paths.task_dir = tests_dir.parent
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
    return task


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.asyncio
async def test_test_sh_creates_files_0644_under_a_umask_0000_runtime(tmp_path):
    """What test.sh installs is not group- or world-writable on a 0000 runtime."""
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    sandbox = UmaskZeroSandbox(tmp_path, paths)
    # The model really runs commands under 0000, as Docker-in-Docker does.
    probe = await sandbox.exec(f"mkdir {sandbox.out}/probe && umask")
    assert probe.stdout.strip() in ("0000", "000", "0")
    assert _mode(sandbox.out / "probe") == 0o777

    result = await Verifier(
        task=_task(tmp_path), rollout_paths=paths, sandbox=sandbox
    ).verify()

    assert result.rewards == {"reward": 1.0}
    (command,) = [c for c in sandbox.commands if "test-stdout.txt" in c]
    assert command.startswith(f"umask {VERIFIER_UMASK} && ")
    site = sandbox.out / "cache" / "site-packages"
    assert _mode(sandbox.out / "cache") == 0o755
    assert _mode(site) == 0o755
    assert _mode(site / "plugin.py") == 0o644


@pytest.mark.asyncio
async def test_start_receipt_and_test_command_share_the_mask(tmp_path):
    """A recovery-eligible verifier's start receipt runs under the mask too."""
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    commands = []

    async def upload_dir(**kwargs):
        return None

    async def exec_(command, **kwargs):
        commands.append(command)
        if "test-stdout.txt" in command:
            paths.reward_text_path.write_text("1")
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    sandbox = SimpleNamespace(is_mounted=True, exec=exec_, upload_dir=upload_dir)
    verifier = Verifier(
        task=_task(tmp_path),
        rollout_paths=paths,
        sandbox=sandbox,
        execution_receipt=True,
    )

    await verifier.verify()

    (command,) = [c for c in commands if "test-stdout.txt" in c]
    assert command.startswith("umask 022 && printf started > /run/benchflow/")


def test_with_verifier_umask_prefixes_the_command():
    assert with_verifier_umask("true") == "umask 022 && true"


def test_separate_verifier_unpacks_under_the_mask():
    """The separate verifier's unpack creates a missing workspace 0755, not 0777."""
    script = _install_script("/tmp/transfer.tar", "/app", "0" * 64).splitlines()

    assert script[:2] == ["set -e", f"umask {VERIFIER_UMASK}"]
    assert script.index(f"umask {VERIFIER_UMASK}") < next(
        i for i, line in enumerate(script) if line.startswith("mkdir")
    )
