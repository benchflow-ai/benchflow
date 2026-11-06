"""PT R11-3: later-role requirements transport regressed in fe895b04."""

import shutil
from pathlib import Path

import pytest

from benchflow.agents import install, registry
from benchflow.sandbox._base import ExecResult


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["deepagents", "harvey-lab-harness"])
async def test_later_agent_uploads_requirements_without_sandbox_python(
    tmp_path, monkeypatch, agent
):
    """Guards fe895b04: a python=0 kit must not run a requirements decoder."""
    uploads = {}

    class Sandbox:
        async def exec(self, command, **kwargs):
            return ExecResult(return_code=0)

        async def upload_file(self, source, destination, **_kwargs):
            uploads[destination] = Path(source).read_bytes()

    async def run(_env, command, timeout_sec):
        assert "zlib.decompress" not in command, "python=0 refuses this root decoder"
        return ExecResult(return_code=0)

    monkeypatch.setattr(install, "_run_installer", run)
    monkeypatch.setenv("BENCHFLOW_FRESH_INSTALL", "1")
    await install.install_agent(Sandbox(), agent, tmp_path)
    name = "deepagents" if agent == "deepagents" else "harvey-lab"
    requirements = (
        registry.DEEPAGENTS_REQUIREMENTS
        if agent == "deepagents"
        else registry.HARVEY_LAB_REQUIREMENTS
    )
    assert (
        uploads[f"/opt/runtime/agents/{name}-requirements.txt"]
        == ("\n".join(requirements) + "\n").encode()
    )


def test_python_free_kit_still_refuses_python_helpers(tmp_path, monkeypatch):
    """The transport fix must not weaken fe895b04's protected-Python rule."""
    import subprocess

    from benchflow import _root_python

    record = tmp_path / "RECORD"
    record.write_text("kit=1\npython=0\n")
    monkeypatch.setattr(_root_python, "KIT_RECORD", str(record))
    monkeypatch.setattr(_root_python, "KIT_PYTHON", str(tmp_path / "absent-python"))
    result = subprocess.run(
        [
            "sh",
            "-c",
            _root_python.root_python_sh(
                "print('wrong')", fallback=shutil.which("python3")
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 127 and "wrong" not in result.stdout
