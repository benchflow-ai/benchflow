"""Guards the fix at 363d815b for shell-only images."""

import os
import subprocess
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from tests.test_verifier_uv_state import Layout, make_task


@pytest.mark.parametrize(
    "key", ["PIP_CONFIG_FILE", "XDG_CONFIG_DIRS", "XDG_CONFIG_HOME", "HOME"]
)
async def test_shell_only_image_cannot_retain_inherited_pip_configuration(
    tmp_path, monkeypatch, key
):
    layout = Layout(tmp_path, monkeypatch, workspace_name="app")
    inherited = {key: str(layout.workspace / "agent-config")}

    async def execute(command, **kwargs):
        if command.startswith("cd / && python3 -c"):
            return ExecResult(stdout="", stderr="python3: not found", return_code=127)
        if "command -v python3" in command:
            return ExecResult(stdout="", stderr="", return_code=1)
        result = subprocess.run(
            ["sh", "-c", command],
            env={"PATH": "/usr/bin:/bin", **inherited},
            capture_output=True,
            text=True,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    found = await lockdown._isolate_verifier_tool_state(
        SimpleNamespace(exec=execute),
        make_task(layout),
        {},
        "agent",
        str(layout.workspace),
    )
    assert found.get("PIP_CONFIG_FILE", inherited.get("PIP_CONFIG_FILE")) == os.devnull
