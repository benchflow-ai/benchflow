"""Guard FF-install-C2 against agent-owned runtime parents at cd4d492d."""

from pathlib import Path

import pytest

from benchflow.providers import litellm_runtime as runtime
from tests.test_root_only_tools import _LocalSandbox


class UploadSandbox(_LocalSandbox):
    async def upload_file(self, source, destination, **kwargs):
        Path(destination).write_bytes(Path(source).read_bytes())


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["writable-parent", "symlink-parent", "existing-child"])
async def test_runtime_is_private_before_any_upload(tmp_path, bad):
    parent = tmp_path / "runtime"
    parent.mkdir()
    child = parent / "dummy-role"
    if bad == "writable-parent":
        parent.chmod(0o777)
    elif bad == "symlink-parent":
        # The link sits in a folder the agent may write.
        (tmp_path / "workspace").mkdir()
        (tmp_path / "workspace").chmod(0o777)
        link = tmp_path / "workspace" / "alias"
        link.symlink_to(parent, target_is_directory=True)
        child = link / "dummy-role"
    else:
        child.mkdir()
    with pytest.raises(RuntimeError, match="prepare LiteLLM"):
        await runtime._upload_runtime_files_to_sandbox(
            UploadSandbox(tmp_path), runtime_dir=str(child), config={}
        )
    assert not (child / "config.yaml").exists()


@pytest.mark.asyncio
async def test_runtime_uploads_work_beneath_a_protected_parent(tmp_path):
    child = tmp_path / "runtime/role"
    await runtime._upload_runtime_files_to_sandbox(
        UploadSandbox(tmp_path), runtime_dir=str(child), config={}
    )
    assert (child / "launcher.py").is_file()
    assert child.stat().st_mode & 0o077 == 0
