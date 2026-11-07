"""Guards PY8 FF-sw-rollout fixes against the 363d815b artifact regressions."""

import asyncio
import os
import shlex
import sys

import pytest

from benchflow.review.evidence import (
    capture_task_evidence,
    capture_workspace,
    install_uploaded_workspace,
    prepare_workspace_upload,
    validate_uploaded_workspace,
)
from benchflow.sandbox.protocol import ExecResult
from tests.test_artifacts_collection import LocalTransport, _collect, _layout


class PoisonTransport(LocalTransport):
    def __init__(self, poison, route):
        super().__init__()
        self.poison, self.route = poison, route

    async def exec(self, cmd, *, user="root", timeout_sec=30):
        argv = shlex.split(cmd)
        if argv[0] == "python3":
            argv[0] = sys.executable
        env = dict(os.environ)
        if self.route == "pythonpath":
            env["PYTHONPATH"] = str(self.poison)
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.poison if self.route == "cwd" else self.poison.parent,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        return ExecResult(process.returncode, stdout.decode(), stderr.decode())


@pytest.mark.parametrize("route", ["cwd", "pythonpath"])
@pytest.mark.parametrize(
    "module", ["json", "pathlib", "tarfile", "hashlib", "sitecustomize"]
)
@pytest.mark.parametrize(
    "operation", ["collect", "capture", "admission", "unpack", "task-evidence"]
)
async def test_privileged_helpers_ignore_agent_imports(
    tmp_path, route, module, operation
):
    logs, workspace, trial = _layout(tmp_path)
    (workspace / "answer.txt").write_text("42")
    (logs / "answer.txt").write_text("42")
    bundle = tmp_path / "clean-bundle"
    await capture_workspace(LocalTransport(), str(workspace), bundle)
    archive = tmp_path / "clean.tar"
    prepare_workspace_upload(bundle, archive)
    poison = tmp_path / "poison"
    poison.mkdir()
    marker = tmp_path / "dummy-import-marker"
    (poison / f"{module}.py").write_text(
        f"open({str(marker)!r}, 'w').write('agent code')\nraise RuntimeError('dummy poison')\n"
    )
    env = PoisonTransport(poison, route)
    try:
        if operation == "collect":
            result = await _collect(env, logs, workspace, trial)
            assert result["collections"][0]["status"] == "collected"
        elif operation == "capture":
            await capture_workspace(env, str(workspace), tmp_path / "captured")
        elif operation == "admission":
            await validate_uploaded_workspace(
                env, str(workspace), str(bundle / "manifest.json")
            )
        elif operation == "unpack":
            await install_uploaded_workspace(
                env,
                str(archive),
                str(tmp_path / "installed"),
                str(bundle / "manifest.json"),
            )
        else:
            await capture_task_evidence(
                env, str(workspace), tmp_path / "evidence", artifacts=[str(logs)]
            )
    finally:
        assert not marker.exists(), "privileged helper imported agent code"
