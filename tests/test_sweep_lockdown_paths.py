"""Guards the fix at 363d815b: every path lock must succeed."""

import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.lockdown import lockdown_paths


@pytest.mark.parametrize("code_key", ["return_code", "exit_code"])
async def test_failed_lock_refuses_agent_launch(code_key):
    env = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(
                **{code_key: 1}, stderr="/oracle: read-only filesystem", stdout=""
            )
        )
    )
    with pytest.raises(RuntimeError, match=r"[Ll]ock.*paths|[Ll]ocking"):
        await lockdown_paths(env, ["/oracle", "/solution"])


async def test_lock_explicitly_uses_root():
    env = SimpleNamespace(
        exec=AsyncMock(return_value=ExecResult(stdout="", stderr="", return_code=0))
    )
    await lockdown_paths(env, ["/oracle"])
    assert env.exec.call_args.kwargs.get("user") == "root"


@pytest.mark.parametrize("utility", ["chown", "chmod"])
@pytest.mark.parametrize("pattern", [False, True], ids=["paths", "glob"])
async def test_each_lock_failure_survives_later_success(tmp_path, utility, pattern):
    """Execute only dummy lock utilities; never change the ownership under test."""
    for name in ("a", "b"):
        (tmp_path / ("lock-" + name)).mkdir()
    # The first match fails, the second succeeds: the loop's last status
    # must never conceal a failed lock, for either utility or path spelling.
    functions = (
        f'{utility}() {{ case "$*" in *lock-a*) return 1;; *) return 0;; esac; }}; '
        + ("chmod() { :; }; " if utility == "chown" else "chown() { :; }; ")
    )

    async def execute(command, **kwargs):
        result = subprocess.run(
            ["sh", "-c", functions + command], capture_output=True, text=True
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    paths = (
        [str(tmp_path / "lock-*")]
        if pattern
        else [str(tmp_path / "lock-a"), str(tmp_path / "lock-b")]
    )
    with pytest.raises(RuntimeError):
        await lockdown_paths(SimpleNamespace(exec=execute), paths)


async def test_missing_paths_are_safe_and_links_cannot_expose_targets(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "link").symlink_to(target)

    async def execute(command, **kwargs):
        result = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    with pytest.raises(RuntimeError):
        await lockdown_paths(SimpleNamespace(exec=execute), [str(tmp_path / "link")])
    await lockdown_paths(SimpleNamespace(exec=execute), [str(tmp_path / "missing")])
