"""Guards the fix at f4324120."""

import asyncio
import contextlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request

from benchflow.providers import litellm_runtime as runtime
from benchflow.providers.litellm_request_guard import SingleRouteGuard


@pytest.fixture
def boundary(monkeypatch):
    app = FastAPI()

    @app.post("/{rest:path}")
    async def echo(request: Request):
        return {
            "path": request.url.path,
            "body": await request.json(),
            "content_type": request.headers.get("content-type"),
        }

    monkeypatch.setenv(
        "BENCHFLOW_LITELLM_SERVED_MODELS", '["allowed", "gemini/gemini-2.5-flash"]'
    )
    app.add_middleware(SingleRouteGuard)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("module", ["urllib", "sysconfig", "sitecustomize"])
async def test_health_probe_cannot_import_workspace(tmp_path, module):
    marker = tmp_path / "dummy-marker"
    (tmp_path / f"{module}.py").write_text(
        f'open({str(marker)!r}, "w").write("dummy")\nraise SystemExit(0)\n'
    )

    # The real probe runs as a child in an agent-writable directory. A refused
    # connection is fine; executing the planted module is the regression.
    class Sandbox:
        async def exec(self, command, **kwargs):
            import os

            child = await asyncio.create_subprocess_shell(
                command,
                cwd=tmp_path,
                env={**os.environ, "PYTHONPATH": str(tmp_path)},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await child.communicate()
            return SimpleNamespace(
                return_code=child.returncode, stdout=out.decode(), stderr=err.decode()
            )

    with contextlib.suppress(RuntimeError):
        await runtime._poll_sandbox_health(
            Sandbox(),
            python=sys.executable,
            port=1,
            stderr_path=str(tmp_path / "missing"),
            deadline_s=0.05,
        )
    assert not marker.exists()
