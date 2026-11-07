"""Guards the fix at 363d815b: preserve answers."""

import subprocess

import pytest

from benchflow.sandbox import lockdown
from tests.test_sandbox_hardening import _make_env, _make_task


@pytest.mark.parametrize("parent", ["tmp", "var-tmp"])
@pytest.mark.parametrize(
    "leaf",
    [
        "project",
        "project with ' quotes",
        "",
        "project[1]",
        "project*name",
        "project?name",
        r"project\name",
        "project\n",
        "project\t",
    ],
)
@pytest.mark.parametrize("cleanup_conftests", [True, False])
@pytest.mark.parametrize("alias", [False, True], ids=["direct", "symlink"])
async def test_hardening_preserves_python_answers_in_temporary_workspaces(
    tmp_path, monkeypatch, parent, leaf, cleanup_conftests, alias
):
    roots = [tmp_path / "tmp", tmp_path / "var-tmp"]
    for root in roots:
        root.mkdir()
    workspace = tmp_path / parent / leaf
    workspace.mkdir(exist_ok=True)
    answer = workspace / "answer.py"
    answer.write_text("print('correct')\n")
    lookalike = (
        leaf.replace("[1]", "1").replace("*", "X").replace("?", "X").replace("\\", "")
    )
    shadow = None
    if lookalike != leaf:
        neighbor = workspace.parent / lookalike
        neighbor.mkdir()
        shadow = neighbor / "shadow.py"
        shadow.write_text("dummy injection outside the workspace\n")
    if alias:
        link = tmp_path / "workspace-link"
        link.symlink_to(workspace)
        workspace = link
    junk = next(root for root in roots if root != tmp_path / parent) / "shadow.py"
    junk.write_text("dummy injection\n")
    monkeypatch.setattr(
        lockdown,
        "_read_hardening_config",
        lambda _: {"cleanup_conftests": cleanup_conftests},
    )
    env = _make_env()
    await lockdown.harden_before_verify(
        env, _make_task(), sandbox_user=None, workspace=str(workspace)
    )
    cleanup = next(
        c.args[0] for c in env.exec.call_args_list if "find /tmp /var/tmp" in c.args[0]
    )
    # Keep the physical-workspace assignment even when the shared root
    # route prefixes its first statement. This fixture executes only the
    # temporary-directory sweep, with both roots replaced below.
    cleanup = cleanup[cleanup.index("bf_cleanup_workspace=") :]
    purge = "; ".join(
        part
        for part in cleanup.split("; ")
        if part.startswith(("find /tmp /var/tmp", "bf_cleanup_workspace="))
    )
    purge = purge.replace("find /tmp /var/tmp", "find " + " ".join(map(str, roots)))
    result = subprocess.run(["sh", "-c", purge], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert answer.read_text() == "print('correct')\n"
    assert not junk.exists()
    if shadow is not None:
        assert not shadow.exists()


async def test_soft_cleanup_receives_the_workspace(monkeypatch):
    """The shared helper must keep the same answer-preservation contract."""
    env = _make_env()
    await lockdown.cleanup_verifier_python_hooks(env, None, workspace="/tmp/project")
    assert "/tmp/project" in env.exec.call_args.args[0]
    assert "workspace" not in env.exec.call_args.kwargs
