"""Guard FF-install-C10 at 4f757bb7: cleanup accepts the launcher's own files."""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.providers import litellm_runtime as runtime
from tests._root_stand_in import run_sh


class _LocalSandbox:
    def __init__(self, root):
        self.root = root

    async def exec(self, command, **_kwargs):
        done = run_sh(
            command,
            env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
            capture_output=True,
            text=True,
            timeout=20,
        )
        return SimpleNamespace(
            return_code=done.returncode,
            stdout=done.stdout,
            stderr=done.stderr,
        )


def _wait_for(path: Path) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def launched_proxy(tmp_path, request):
    """Run the real launcher with a dummy proxy, without provider credentials."""
    folder = tmp_path / "runtime"
    folder.mkdir(mode=0o700)
    stopped = tmp_path / "dummy-stopped"
    callback = folder / "callback.jsonl"
    proxy = tmp_path / "dummy-litellm"
    proxy.write_text(
        f"#!{sys.executable}\n"
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        f"stopped = Path({str(stopped)!r})\n"
        "def stop(*_):\n"
        "    stopped.write_text('dummy stopped')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "print('dummy stdout', flush=True)\n"
        "print('dummy stderr', file=sys.stderr, flush=True)\n"
        f"Path({str(callback)!r}).write_text('dummy callback')\n"
        "while True:\n"
        "    time.sleep(0.1)\n"
    )
    proxy.chmod(0o700)
    launcher = folder / "launcher.py"
    launcher.write_text(runtime._sandbox_launcher_source())
    paths = {
        name: str(folder / file)
        for name, file in {
            "stdout": "stdout.log",
            "stderr": "stderr.log",
            "pid": "litellm.pid",
            "state": "state.json",
        }.items()
    }
    config = folder / "launch.json"
    config.write_text(
        json.dumps(
            {
                **paths,
                "litellm": str(proxy),
                "config": "dummy-config",
                "runtime_dir": str(folder),
                "env": {},
            }
        )
    )
    command = shlex.join([sys.executable, str(launcher), str(config)])
    done = run_sh(
        f"umask {request.param:03o}; exec {command}",
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert done.returncode == 0, done.stderr
    pid = int(json.loads(done.stdout)["pid"])
    try:
        assert _wait_for(callback), "dummy proxy did not start"
        yield folder, paths, stopped
    finally:
        # Only this fixture's dummy child; a broken production stop must not
        # leave it alive after the assertion fails.
        if not stopped.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
        assert _wait_for(stopped), "dummy proxy did not terminate"


@pytest.mark.parametrize("launched_proxy", [0o000, 0o002, 0o022], indirect=True)
def test_launcher_and_child_create_private_files(launched_proxy):
    """Guard FF-install-C10: even child callback files inherit a private umask."""
    folder, paths, _ = launched_proxy
    for path in [*map(Path, paths.values()), folder / "callback.jsonl"]:
        assert path.stat().st_mode & 0o777 == 0o600, path.name


@pytest.mark.asyncio
@pytest.mark.parametrize("launched_proxy", [0o000, 0o002, 0o022], indirect=True)
@pytest.mark.parametrize("cleanup", ["role-stop", "failed-start"])
async def test_cleanup_stops_proxy_and_removes_runtime(launched_proxy, cleanup):
    """Guard FF-install-C10 in both guarded cleanup paths, not just launch mode."""
    folder, paths, stopped = launched_proxy
    sandbox = _LocalSandbox(folder.parent)
    if cleanup == "failed-start":
        await runtime._terminate_sandbox_litellm(
            sandbox,
            pid_path=paths["pid"],
            runtime_dir=str(folder),
        )
    else:
        process = runtime.SandboxLiteLLMProcess(
            sandbox=sandbox,
            route=None,
            runtime_dir=str(folder),
            endpoint=runtime.LiteLLMEndpoint("dummy", "dummy"),
            log_path=str(folder / "callback.jsonl"),
            pid_path=paths["pid"],
            stdout_path=paths["stdout"],
            stderr_path=paths["stderr"],
            session_id="dummy-session",
            agent_name="dummy-agent",
        )
        await process.stop()
    assert _wait_for(stopped), "guard rejected the launcher's own PID file"
    assert not folder.exists(), "guard left the runtime behind"


@pytest.mark.asyncio
@pytest.mark.parametrize("launched_proxy", [0o077], indirect=True)
@pytest.mark.parametrize("cleanup", ["role-stop", "failed-start"])
async def test_cleanup_still_refuses_a_writable_pid_file(launched_proxy, cleanup):
    """Guard FF-install-C10 without weakening the existing PID custody check."""
    folder, paths, stopped = launched_proxy
    Path(paths["pid"]).chmod(0o666)
    sandbox = _LocalSandbox(folder.parent)
    if cleanup == "failed-start":
        await runtime._terminate_sandbox_litellm(
            sandbox,
            pid_path=paths["pid"],
            runtime_dir=str(folder),
        )
    else:
        process = runtime.SandboxLiteLLMProcess(
            sandbox=sandbox,
            route=None,
            runtime_dir=str(folder),
            endpoint=runtime.LiteLLMEndpoint("dummy", "dummy"),
            log_path=str(folder / "callback.jsonl"),
            pid_path=paths["pid"],
            stdout_path=paths["stdout"],
            stderr_path=paths["stderr"],
            session_id="dummy-session",
            agent_name="dummy-agent",
        )
        await process.stop()
    assert not stopped.exists()
    assert folder.exists()
