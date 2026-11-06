"""Guard FF-install-C12 at a0c81002: staging retains parent custody."""

import re
import shutil
from pathlib import Path

import pytest

from benchflow.agents.install import _run_installer
from benchflow.sandbox._base import ExecResult
from tests._root_stand_in import run_sh


class LocalInstallerEnv:
    def __init__(self, root, *, upload_error=False):
        self.root = root
        self.upload_error = upload_error
        self.uploads = []
        self.commands = []
        self.modes = []

    def local(self, value):
        return value.replace("/opt/benchflow", str(self.root / "protected")).replace(
            "/tmp/benchflow-agent-install",
            str(self.root / "workspace/benchflow-agent-install"),
        )

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        result = run_sh(self.local(command), text=True, capture_output=True)
        stdout = result.stdout.replace(
            str(self.root / "protected"), "/opt/benchflow"
        ).replace(
            str(self.root / "workspace/benchflow-agent-install"),
            "/tmp/benchflow-agent-install",
        )
        return ExecResult(
            return_code=result.returncode, stdout=stdout, stderr=result.stderr
        )

    async def upload_file(self, source, destination, **_kwargs):
        self.uploads.append(destination)
        self.modes.append(Path(self.local(destination)).parent.stat().st_mode & 0o777)
        if self.upload_error:
            raise OSError("dummy upload failure")
        shutil.copyfile(source, self.local(destination))


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "upload-error"])
async def test_staged_script_uses_protected_parent_and_cleans_up(tmp_path, outcome):
    """Guards a0c81002: every backend uses protected staging, including errors."""
    (tmp_path / "workspace").mkdir(mode=0o777)
    env = LocalInstallerEnv(tmp_path, upload_error=outcome == "upload-error")
    command = (
        "#"
        + "x" * 50_000
        + "\nprintf 'dummy installer'\nexit "
        + ("23" if outcome == "failure" else "0")
    )
    if outcome == "upload-error":
        with pytest.raises(OSError, match="dummy upload"):
            await _run_installer(env, command, 30)
    else:
        result = await _run_installer(env, command, 30)
        assert result.return_code == (23 if outcome == "failure" else 0)
        assert result.stdout == "dummy installer"
    assert len(env.uploads) == 1
    assert env.modes == [0o700]
    assert (tmp_path / "protected").stat().st_mode & 0o777 == 0o755
    assert env.uploads[0].startswith("/opt/benchflow/agent-install/"), (
        "staging has an agent-replaceable parent"
    )
    assert not Path(env.local(env.uploads[0])).parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["protected", "protected/agent-install"])
@pytest.mark.parametrize("kind", ["symlink", "writable", "sticky-writable"])
async def test_staging_refuses_untrusted_parent_before_upload(tmp_path, path, kind):
    """Guards a0c81002: neither ancestor nor staging root can be substituted."""
    # The agent's workspace, which it may write.
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace").chmod(0o777)
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "symlink":
        target.symlink_to(tmp_path / "workspace", target_is_directory=True)
    else:
        target.mkdir()
        target.chmod(0o777 if kind == "writable" else 0o1777)
    env = LocalInstallerEnv(tmp_path)
    result = await _run_installer(env, "#" + "x" * 50_000 + "\ntrue", 30)
    assert result.return_code != 0, "unsafe staging was accepted"
    assert not env.uploads
    assert "is not root's alone" in result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipt",
    [
        "/tmp/benchflow-agent-install.ABC123xyz0",
        "/opt/benchflow/agent-install/../agent-install.ABC",
        "/opt/benchflow/agent-install/install.ABC/extra",
        "/opt/benchflow/agent-install/install.ABC\n/tmp/other",
    ],
)
async def test_invalid_staging_receipt_never_uploads_or_cleans(receipt):
    """Guards a0c81002: cleanup cannot express a backend-selected outside path."""

    class Env:
        def __init__(self):
            self.calls = []

        async def exec(self, command, **kwargs):
            self.calls.append(command)
            return ExecResult(return_code=0, stdout=receipt, stderr="")

        async def upload_file(self, *args, **_kwargs):
            pytest.fail("unvalidated staging receipt reached upload")

    env = Env()
    with pytest.raises(RuntimeError, match="invalid directory"):
        await _run_installer(env, "#" + "x" * 50_000, 30)
    assert len(env.calls) == 1
    assert not re.search(r"\brm |\brmdir ", env.calls[0])
