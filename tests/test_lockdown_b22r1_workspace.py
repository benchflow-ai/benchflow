"""Guard CONFIRM-lockdown-3 at 8c11e9a4: preserve workspace mount aliases."""

import shlex
import subprocess
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from tests.test_sandbox_hardening import _make_env, _make_task


@pytest.mark.parametrize("soft", [False, True], ids=["final", "soft"])
@pytest.mark.parametrize(
    "leaf", ["project", "project[1]", "project*?\\name", "space ' \n"]
)
@pytest.mark.parametrize("backend", ["overlay", "ext4", "tmpfs"])
async def test_temporary_cleanup_preserves_only_workspace_mount_aliases(
    tmp_path, monkeypatch, soft, leaf, backend
):
    workspace = tmp_path / "app"
    alias = tmp_path / "tmp" / leaf
    nested = tmp_path / "var-tmp" / leaf
    for path in (workspace, alias, nested):
        path.mkdir(parents=True)
        (path / "answer.py").write_text("correct\n")
    junk = tmp_path / "tmp" / "junk.py"
    junk.write_text("injection\n")

    def escaped(path):
        return (
            str(path)
            .replace("\\", "\\134")
            .replace(" ", "\\040")
            .replace("\t", "\\011")
            .replace("\n", "\\012")
        )

    info = "1 0 0:1 / / rw - overlay dummy rw\n"
    for i, path in enumerate((workspace, alias, nested), 2):
        root = "/project" if path != nested else "/project/subtree"
        info += f"{i} 1 0:2 {root} {escaped(path)} rw - {backend} dummy rw\n"

    def execute(command, **kwargs):
        if "/proc/self/mountinfo" in command:
            return SimpleNamespace(
                stdout=info + "# end\n\0", stderr="", return_code=0, exit_code=0
            )
        if "find /tmp /var/tmp" in command:
            purge = "; ".join(
                part
                for part in command.split("; ")
                if part.startswith(("find /tmp /var/tmp", "bf_cleanup_workspace="))
            )
            purge = purge.replace(
                "find /tmp /var/tmp",
                "find "
                + " ".join(shlex.quote(str(tmp_path / p)) for p in ("tmp", "var-tmp")),
            )
            result = subprocess.run(["sh", "-c", purge], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
        return SimpleNamespace(stdout="", stderr="", return_code=0, exit_code=0)

    monkeypatch.setattr(
        lockdown, "_read_hardening_config", lambda _: {"cleanup_conftests": False}
    )
    env = _make_env(execute)
    if soft:
        await lockdown.cleanup_verifier_python_hooks(
            env, None, workspace=str(workspace)
        )
    else:
        await lockdown.harden_before_verify(
            env, _make_task(), sandbox_user=None, workspace=str(workspace)
        )
    for path in (workspace, alias, nested):
        assert (path / "answer.py").read_text() == "correct\n"
    assert not junk.exists()
