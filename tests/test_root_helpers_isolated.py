"""Root helpers never import modules from the agent's workspace.

BenchFlow runs small ``python3 -c`` helpers as root after the agent stops:
disk reclaim, the verifier PATH check, pytest-plugin discovery, the symlink
purge, workspace evidence capture. Their working directory is the sandbox's
default one, which is the task's WORKDIR, which is the agent's workspace. A
plain ``python3 -c`` puts that directory first on ``sys.path``, so a
``json.py`` or ``glob.py`` the agent left there ran as root before the
verifier, in the shared sandbox and in a separate verifier sandbox alike
(the frozen workspace is that sandbox's WORKDIR too). ``python3 -I`` keeps the
working directory, the user site and ``PYTHON*`` variables out.

Guards PR #NNNN against the planted-module path it closes.
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow.sandbox.protocol import ExecResult

# Stdlib modules the helpers import that are not loaded at interpreter start.
PLANTED = ("json", "glob", "shutil", "pathlib", "fnmatch", "hashlib", "tarfile")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An agent workspace holding stdlib-named modules that report their import."""
    root = tmp_path / "workspace"
    root.mkdir()
    marker = tmp_path / "imported.txt"
    for name in PLANTED:
        (root / f"{name}.py").write_text(
            f"open({str(marker)!r}, 'a').write({name!r} + '\\n')\nraise SystemExit(0)\n"
        )
    (root / "answer.txt").write_text("42\n")
    return root


def _imported(workspace: Path) -> list[str]:
    marker = workspace.parent / "imported.txt"
    return marker.read_text().split() if marker.exists() else []


def _run_from(workspace: Path, command: str) -> subprocess.CompletedProcess:
    """Run a helper command as the sandbox would, from the agent's workspace."""
    python = shlex.quote(sys.executable)
    if command.startswith("python3 "):
        command = python + command[len("python3") :]
    command = command.replace("&& python3 ", f"&& {python} ")
    return subprocess.run(
        ["sh", "-c", command], cwd=workspace, capture_output=True, text=True
    )


def test_the_planted_modules_would_run_under_a_plain_python3_c(
    workspace: Path,
) -> None:
    """Control: without -I the workspace's json.py wins over the stdlib's."""
    _run_from(workspace, "python3 -c 'import json'")
    assert _imported(workspace) == ["json"]


def test_trusted_path_check_is_isolated(workspace: Path) -> None:
    from benchflow.sandbox.lockdown import _trusted_path_extras_cmd

    command = _trusted_path_extras_cmd("/usr/bin:/bin", ("/tmp",))
    result = _run_from(workspace, command)
    assert result.returncode == 0, result.stderr
    assert _imported(workspace) == []


def test_reclaim_is_isolated(workspace: Path, tmp_path: Path) -> None:
    from benchflow.sandbox._cache_reclaim import build_reclaim_caches_cmd

    command = build_reclaim_caches_cmd(str(workspace))
    # The production command deletes caches under /root, /home/* and /tmp;
    # its second argument roots every path in an empty folder instead.
    tail = " 2>/dev/null; true"
    assert command.count(tail) == 1
    prefix = tmp_path / "fake-root"
    prefix.mkdir()
    command = command.replace(tail, f" {shlex.quote(str(prefix))}{tail}")
    _run_from(workspace, command)
    assert _imported(workspace) == []


def test_external_symlink_purge_is_isolated(workspace: Path) -> None:
    from benchflow.sandbox.lockdown import _purge_external_symlinks_cmd

    (workspace / "escape").symlink_to("/etc/hostname")
    _run_from(workspace, _purge_external_symlinks_cmd(str(workspace)))
    assert _imported(workspace) == []
    # It still did its job.
    assert not (workspace / "escape").is_symlink()


class _Recorder:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.commands: list[str] = []

    async def exec(self, cmd: str, **kwargs) -> ExecResult:
        self.commands.append(cmd)
        result = _run_from(self.workspace, cmd)
        return ExecResult(result.returncode, result.stdout, result.stderr)


def test_pytest_plugin_discovery_is_isolated(workspace: Path) -> None:
    from types import SimpleNamespace

    from benchflow.sandbox.lockdown import _discover_pytest_plugin_flags
    from benchflow.task.config import TaskConfig

    env = _Recorder(workspace)
    task = SimpleNamespace(config=TaskConfig(), task_dir=None)
    asyncio.run(_discover_pytest_plugin_flags(env, task))
    assert env.commands and env.commands[0].startswith("cd / && python3 -I -c")
    assert _imported(workspace) == []


def test_workspace_capture_is_isolated(workspace: Path, tmp_path: Path) -> None:
    import shutil

    from benchflow.review.evidence import capture_task_evidence

    class Box(_Recorder):
        async def download_file(self, src: str, dst: Path) -> None:
            shutil.copyfile(src, dst)

    env = Box(workspace)
    manifest = asyncio.run(
        capture_task_evidence(env, str(workspace), tmp_path / "evidence")
    )
    assert _imported(workspace) == []
    # The planted modules are the agent's files and are captured as such.
    assert {"json.py", "answer.txt"} <= {entry.path for entry in manifest.entries}
    assert all(
        shlex.split(c)[:3] == ["python3", "-I", "-c"]
        for c in env.commands
        if c.startswith("python3")
    )


def test_reviewer_side_evidence_scripts_are_isolated() -> None:
    from benchflow.review import evidence

    commands: list[str] = []

    class Env:
        async def exec(self, cmd: str, **kwargs) -> ExecResult:
            commands.append(cmd)
            return ExecResult(0, "", "")

    asyncio.run(evidence.validate_uploaded_workspace(Env(), "/app", "/m.json"))
    asyncio.run(evidence.install_uploaded_workspace(Env(), "/a.tar", "/app", "/m.json"))
    assert commands
    assert all(shlex.split(c)[:3] == ["python3", "-I", "-c"] for c in commands)


def test_cleanup_and_restore_keep_out_of_the_workspace() -> None:
    from benchflow.sandbox.lockdown import _build_cleanup_cmd, _restore_workspace_cmd

    # Not run here: the cleanup walks / deleting conftest.py files. Its purge
    # stays non-isolated on purpose (it must see PYTHONPATH and user-site
    # directories) but runs from /, and imports nothing but sys and os.
    cleanup = _build_cleanup_cmd()
    assert 'cd / && python3 -c "import sys,os;' in cleanup
    assert "python3 -I -c 'import shutil,sys;" in _restore_workspace_cmd("/app")
