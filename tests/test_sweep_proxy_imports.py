"""Guards the fix at 363d815b: private proxy imports."""

import json

import pytest

from benchflow.providers import litellm_runtime as runtime
from benchflow.providers.litellm_config import resolve_litellm_route
from tests.test_litellm_hardening import _FakeSandbox


@pytest.mark.parametrize(
    "pythonpath",
    [None, "", ":", ".", "/tmp/dummy-workspace", "/trusted::/tmp/dummy-workspace"],
)
@pytest.mark.asyncio
async def test_proxy_launch_uses_only_private_import_directory(pythonpath, monkeypatch):
    async def layer(*args, **kwargs):
        return "/opt/runtime/litellm/bin/python", None

    monkeypatch.setattr(runtime, "_proxy_from_layer", layer)
    sandbox = _FakeSandbox()
    env = {"ANTHROPIC_API_KEY": "dummy-upstream"}
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    route = resolve_litellm_route("anthropic/claude-sonnet-4-6", env)
    proc = await runtime._start_sandbox_litellm(
        sandbox=sandbox,
        route=route,
        master_key="dummy-gateway",
        agent_env=env,
        session_id="s",
        agent_name="claude-agent-acp",
    )
    launch = json.loads(
        next(v for k, v in sandbox.uploaded.items() if k.endswith("launch_config.json"))
    )
    assert launch["env"]["PYTHONPATH"] == proc.runtime_dir
    assert launch["cwd"] == proc.runtime_dir
    assert launch["runtime_dir"] == proc.runtime_dir
    assert 'cwd=cfg["runtime_dir"]' in runtime._sandbox_launcher_source()


def test_install_import_checks_ignore_cwd_and_python_environment():
    script = runtime.sandbox_litellm_install_command("/tmp/dummy-venv")
    assert "/bin/python -I -c" in script


@pytest.mark.asyncio
async def test_launcher_cannot_import_an_agent_workspace_module(tmp_path, monkeypatch):
    """The generated launcher runs a real child; the marker is entirely synthetic."""
    import asyncio
    import os
    import subprocess
    import sys

    private = tmp_path / "private"
    private.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = tmp_path / "planted-import"
    done = tmp_path / "finished"
    (workspace / "fractions.py").write_text(
        f'open({str(marker)!r}, "w").write("dummy")\n'
    )
    entry = private / "entry"
    entry.write_text(
        f'#!{sys.executable}\nimport fractions\nopen({str(done)!r}, "w").write("done")\n'
    )
    entry.chmod(0o700)
    launcher = private / "launcher.py"
    launcher.write_text(runtime._sandbox_launcher_source())

    # Drive production launch configuration rather than constructing the fixed
    # environment in the test itself.
    async def layer(*args, **kwargs):
        return "/opt/runtime/litellm/bin/python", None

    monkeypatch.setattr(runtime, "_proxy_from_layer", layer)
    sandbox = _FakeSandbox()
    env = {"ANTHROPIC_API_KEY": "dummy-upstream"}
    route = resolve_litellm_route("anthropic/claude-sonnet-4-6", env)
    await runtime._start_sandbox_litellm(
        sandbox=sandbox,
        route=route,
        master_key="dummy-gateway",
        agent_env=env,
        session_id="s",
        agent_name="claude-agent-acp",
    )
    cfg = json.loads(
        next(v for k, v in sandbox.uploaded.items() if k.endswith("launch_config.json"))
    )
    cfg.update(
        litellm=str(entry),
        stdout=str(private / "stdout"),
        stderr=str(private / "stderr"),
        pid=str(private / "pid"),
        state=str(private / "state"),
        cwd=str(private),
        runtime_dir=str(private),
    )
    cfg["env"]["PYTHONPATH"] = cfg["env"]["PYTHONPATH"].replace(
        cfg["config"].rsplit("/", 1)[0], str(private)
    )
    config = private / "launch.json"
    config.write_text(json.dumps(cfg))
    result = subprocess.run(
        [sys.executable, "-I", str(launcher), str(config)],
        cwd=workspace,
        env={**os.environ, "PYTHONPATH": str(workspace)},
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    for _ in range(100):
        if done.exists():
            break
        await asyncio.sleep(0.01)
    assert done.exists(), (private / "stderr").read_text()
    assert not marker.exists(), "proxy imported the planted workspace module"


@pytest.mark.parametrize(
    "name",
    [
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PYTHONSTARTUP",
        "PYTHONPATH",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "BASH_ENV",
        "ENV",
    ],
)
@pytest.mark.asyncio
async def test_proxy_never_restores_agent_startup_controls(name, monkeypatch):
    async def layer(*args, **kwargs):
        return "/opt/runtime/litellm/bin/python", None

    monkeypatch.setattr(runtime, "_proxy_from_layer", layer)
    sandbox = _FakeSandbox()
    env = {"ANTHROPIC_API_KEY": "dummy-upstream", name: "/tmp/dummy-agent-code"}
    route = resolve_litellm_route("anthropic/claude-sonnet-4-6", env)
    proc = await runtime._start_sandbox_litellm(
        sandbox=sandbox,
        route=route,
        master_key="dummy-gateway",
        agent_env=env,
        session_id="s",
        agent_name="claude-agent-acp",
    )
    cfg = json.loads(
        next(v for k, v in sandbox.uploaded.items() if k.endswith("launch_config.json"))
    )
    if name == "PYTHONPATH":
        assert cfg["env"][name] == proc.runtime_dir
    else:
        assert name not in cfg["env"]
