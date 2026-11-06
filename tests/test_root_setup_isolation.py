"""Guard FF-install-C14 and FF-install-C15 at f2a5ed5f: root agent and proxy
setup runs no tool, interpreter or module an earlier role could have placed.

Root finds git, python3 and node by PATH, and an image's PATH or PYTHONPATH
can name folders the agent owns (``ENV PATH=/app/node_modules/.bin:...``,
``ENV PYTHONPATH=/app``, ``$HOME/.local/bin`` when HOME is the workspace).
Every test plants a dummy that only writes a marker file.

Since PT's C8 route (py8/parity-triage), root's setup commands carry no
environment prefix of their own: they go to ``exec`` with no user, which on
every backend is root's plumbing route (a root-only PATH, every ``PYTHON*``
and loader name unset). The installer tests here send what install.py sends
through a Docker sandbox whose exec replays BenchFlow's real argv locally
with the image's environment (PT's harness in ``test_c8_uid_routes.py``).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow import _uv_pin
from benchflow.agents import install as install_mod
from benchflow.agents.registry import AGENTS
from benchflow.models import AgentInstallError
from benchflow.providers import litellm_bedrock_preflight as preflight_mod
from benchflow.providers import litellm_runtime as runtime_mod

needs_sh = pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")


def _plant(folder: Path, name: str, marker: Path) -> None:
    tool = folder / name
    tool.write_text(f"#!/bin/sh\nprintf dummy > {shlex.quote(str(marker))}\n")
    tool.chmod(0o755)


def _agent_dir(tmp_path: Path, name: str = "agent-bin") -> Path:
    """A PATH folder anyone can change, as a workspace on PATH is to root."""
    folder = tmp_path / name
    folder.mkdir()
    folder.chmod(0o777)
    return folder


class _LocalEnv:
    """A Docker sandbox whose image environment is ``env``: each exec runs
    the argv BenchFlow's Docker backend builds, locally (no container)."""

    def __init__(self, env: dict[str, str], cwd: Path):
        from tests.test_c8_uid_routes import _docker_sandbox

        self.env = env
        self.cwd = cwd
        self.commands: list[str] = []
        self.sandbox = _docker_sandbox(env)

    async def exec(self, command, timeout_sec=None, user=None, **kwargs):
        self.commands.append(command)
        return await self.sandbox.exec(
            command, timeout_sec=timeout_sec, user=user, **kwargs
        )

    async def upload_file(self, source, target, **_kwargs):
        raise AssertionError("small commands run inline")


@needs_sh
@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["git", "python3", "node", "curl", "uv"])
async def test_installer_never_runs_a_tool_from_an_agent_changeable_path_folder(
    tmp_path, tool
):
    """Guards f2a5ed5f: the installer ran the first ``tool`` on the exec's PATH."""
    planted = _agent_dir(tmp_path)
    marker = tmp_path / "planted-tool-ran"
    _plant(planted, tool, marker)
    env = {"PATH": f"{planted}:/usr/bin:/bin", "HOME": str(tmp_path)}
    result = await install_mod._run_installer(
        _LocalEnv(env, tmp_path),
        f"command -v {tool} || true; {tool} --version >/dev/null 2>&1 || true; "
        'echo "PATH=$PATH"',
        30,
    )
    assert not marker.exists(), f"root ran the planted {tool}"
    assert result.return_code == 0, result.stderr
    assert str(planted) not in result.stdout


@needs_sh
@pytest.mark.asyncio
async def test_installer_path_is_the_routes_never_the_images(tmp_path):
    """Root's install PATH is the route's root-only system folders: no image
    folder, root's or not, relative or empty, is kept (PT's C8 route; it
    replaced FF-install's filter of the image's PATH)."""
    kept = tmp_path / "opt-tools"
    kept.mkdir(mode=0o755)
    tmp_path.chmod(0o755)
    planted = _agent_dir(tmp_path)
    env = {
        "PATH": f"{planted}:relative/bin::{kept}:/usr/bin:/bin",
        "HOME": str(tmp_path),
    }
    result = await install_mod._run_installer(
        _LocalEnv(env, tmp_path), 'echo "PATH=$PATH"', 30
    )
    assert result.return_code == 0, result.stderr
    path = result.stdout.strip().removeprefix("PATH=").split(":")
    assert path and all(entry.startswith("/") for entry in path)
    assert str(planted) not in path and str(kept) not in path


@needs_sh
@pytest.mark.asyncio
@pytest.mark.parametrize("variable", ["PYTHONPATH", "PYTHONUSERBASE", "PYTHONSTARTUP"])
async def test_installer_python_reads_no_python_variable(tmp_path, variable):
    """Guards f2a5ed5f: an image's PYTHON* settings reached root's Python."""
    planted = tmp_path / "app"
    planted.mkdir()
    marker = tmp_path / "planted-module-ran"
    payload = f"open({str(marker)!r}, 'w').write('dummy')\n"
    for name in ("json.py", "sitecustomize.py", "startup.py"):
        (planted / name).write_text(payload)
    site = planted / f"lib/python{sys.version_info[0]}.{sys.version_info[1]}"
    (site / "site-packages").mkdir(parents=True)
    (site / "site-packages/usercustomize.py").write_text(payload)
    value = {
        "PYTHONPATH": str(planted),
        "PYTHONUSERBASE": str(planted),
        "PYTHONSTARTUP": str(planted / "startup.py"),
    }[variable]
    base = getattr(sys, "_base_executable", sys.executable)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), variable: value}
    result = await install_mod._run_installer(
        _LocalEnv(env, tmp_path),
        f"{shlex.quote(base)} -c 'import json' && env | grep ^PYTHON || true",
        30,
    )
    assert result.return_code == 0, result.stderr
    assert not marker.exists(), f"root's Python honoured {variable}"
    assert "PYTHON" not in result.stdout


@needs_sh
@pytest.mark.asyncio
async def test_failed_install_diagnostics_run_no_planted_node(tmp_path, monkeypatch):
    """Guards f2a5ed5f: the failure diagnostics ran ``node --version`` by PATH."""
    planted = _agent_dir(tmp_path)
    marker = tmp_path / "planted-node-ran"
    _plant(planted, "node", marker)
    env = _LocalEnv(
        {"PATH": f"{planted}:/usr/bin:/bin", "HOME": str(tmp_path)}, tmp_path
    )
    monkeypatch.setitem(install_mod.AGENT_INSTALLERS, "claude-agent-acp", "exit 7")
    with pytest.raises(AgentInstallError):
        await install_mod.install_agent(env, "claude-agent-acp", tmp_path / "rollout")
    assert any("node --version" in c for c in env.commands)
    assert not marker.exists()


@needs_sh
@pytest.mark.asyncio
async def test_web_tool_policy_setup_runs_no_planted_interpreter(tmp_path):
    """Guards f2a5ed5f: the settings merge ran python3 or node by PATH as root."""
    planted = _agent_dir(tmp_path)
    marker = tmp_path / "planted-interpreter-ran"
    for name in ("python3", "node"):
        _plant(planted, name, marker)
    home = tmp_path / "home"
    home.mkdir()
    env = _LocalEnv({"PATH": f"{planted}:/usr/bin:/bin", "HOME": str(home)}, tmp_path)
    agent = (
        "mimocode"
        if "mimocode" in AGENTS
        else next(
            name for name, cfg in AGENTS.items() if cfg.disallow_web_tools_setup_cmd
        )
    )
    # The image may lack python3; only what ran matters.
    with contextlib.suppress(Exception):
        await install_mod.apply_web_tool_policy(
            env, agent, AGENTS[agent], str(home), disallow=True
        )
    assert env.commands
    assert not marker.exists()


@needs_sh
@pytest.mark.parametrize("through_installer", [False, True])
def test_openhands_uv_git_child_never_runs_homes_git(tmp_path, through_installer):
    """Guards f2a5ed5f (FF-install-C15): the recipe put $HOME/.local/bin first
    on root's PATH, so uv's Git child ran a git the agent left there."""
    home = tmp_path / "root-home"
    (home / ".local/bin").mkdir(parents=True)
    marker = tmp_path / "planted-git-ran"
    _plant(home / ".local/bin", "git", marker)
    tools = tmp_path / "tools"
    tools.mkdir()
    trace = tmp_path / "uv-git"
    # uv's stand-in runs Git by name, as uv does for a git+ source.
    (tools / "uv").write_text(
        f"#!/bin/sh\ncommand -v git > {shlex.quote(str(trace))}\ngit --version >/dev/null\nexit 19\n"
    )
    (tools / "uv").chmod(0o755)
    (tools / "curl").write_text("#!/bin/sh\nexit 0\n")
    (tools / "curl").chmod(0o755)
    for folder in (tmp_path, tools):
        folder.chmod(0o755)
    command = AGENTS["openhands"].install_cmd
    command = command.replace(_uv_pin.uv_install_command(), ":").replace(
        _uv_pin.managed_python_install_command(shlex.quote(_uv_pin.UV_BIN)), ":"
    )
    command = command.replace(_uv_pin.UV_DIR, str(tools))
    command = command.replace(_uv_pin.BENCHFLOW_ROOT, str(tmp_path / "protected"))
    image = {"PATH": f"{tools}:{home}/.local/bin:/usr/bin:/bin", "HOME": str(home)}
    if through_installer:
        # What install.py sends: root's plumbing route, whose PATH is the
        # route's; the recipe's own PATH puts uv's folder first.
        import asyncio

        result = asyncio.run(_LocalEnv(image, home).exec(command, timeout_sec=60))
    else:
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=home,
            env={"PATH": f"{tools}:/usr/bin:/bin", "HOME": str(home)},
            capture_output=True,
            text=True,
            timeout=60,
        )
    assert trace.exists(), result.stderr
    assert not marker.exists(), "root's uv ran the git in HOME"
    assert not trace.read_text().startswith(str(home))


def test_no_install_recipe_puts_a_home_or_relative_folder_on_path():
    """Enumerate the class over every agent: an install-time PATH entry under
    HOME, ~ or a relative folder is the agent's to fill."""
    for name, config in AGENTS.items():
        for match in re.finditer(
            r'(?:export )?PATH="([^"]*)"', config.install_cmd or ""
        ):
            for entry in match.group(1).split(":"):
                if entry == "$PATH":
                    continue
                assert entry.startswith("/") and "$HOME" not in entry, (name, entry)


@needs_sh
@pytest.mark.asyncio
async def test_every_root_setup_exec_in_install_takes_the_plumbing_route(tmp_path):
    """Enumerate install.py's root execs that run installers or interpreters:
    each goes to ``exec`` as root (no user), so the backend gives it root's
    plumbing route, and none sets its own PATH or PYTHON* first."""
    seen: list[str] = []

    class _Recorder:
        async def exec(self, command, timeout_sec=None, user=None, **kwargs):
            assert user in (None, "root"), (user, command[:200])
            seen.append(command)
            return SimpleNamespace(return_code=1, stdout="", stderr="dummy")

    with pytest.raises(AgentInstallError):
        await install_mod.install_agent(_Recorder(), "openhands", tmp_path / "rollout")
    policy_agents = [n for n, c in AGENTS.items() if c.disallow_web_tools_setup_cmd]
    for name in policy_agents:
        # The recorder fails every exec; only the commands matter.
        with contextlib.suppress(Exception):
            await install_mod.apply_web_tool_policy(
                _Recorder(), name, AGENTS[name], "/root", disallow=True
            )
    assert len(seen) >= 2 + len(policy_agents)
    for command in seen:
        assert not re.match(r"\s*(export )?(PATH|PYTHON\w*)=", command), command[:200]


# --- the proxy (FF-install-C14) -------------------------------------------


def _production_launch_flags() -> list[str]:
    """The interpreter flags the production launch command passes."""
    import asyncio

    from benchflow.providers.litellm_config import resolve_litellm_route
    from tests.test_litellm_hardening import _FakeSandbox

    route = resolve_litellm_route(
        "minimax/MiniMax-M3",
        {"MINIMAX_API_KEY": "k", "MINIMAX_BASE_URL": "https://api.minimax.io/v1"},
    )
    sandbox = _FakeSandbox()
    asyncio.run(
        runtime_mod._start_sandbox_litellm(
            sandbox=sandbox,
            route=route,
            master_key="sk-master",
            agent_env={"MINIMAX_API_KEY": "k", "PYTHONPATH": "/app"},
            session_id="s",
            agent_name="openhands",
        )
    )
    launch = next(c for c in sandbox.exec_calls if "launcher.py" in c)
    match = re.search(r"/venv/bin/python((?: -\w+)*) \S*launcher\.py", launch)
    assert match, launch
    return match.group(1).split()


@needs_sh
def test_proxy_launcher_and_child_read_no_inherited_python_settings(tmp_path):
    """Guards f2a5ed5f: with an image's ENV PYTHONPATH=/app, the root launcher
    imported /app/json.py, and the proxy child kept that PYTHONPATH."""
    flags = _production_launch_flags()
    app = tmp_path / "app"
    app.mkdir()
    marker = tmp_path / "planted-json-ran"
    (app / "json.py").write_text(f"open({str(marker)!r}, 'w').write('dummy')\n")
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    (runtime / "launcher.py").write_text(runtime_mod._sandbox_launcher_source())
    seen = tmp_path / "child-env.json"
    child = tmp_path / "dummy-litellm"
    child.write_text(
        f"#!{sys.executable} -I\nimport json, os\n"
        f"open({str(seen)!r}, 'w').write(json.dumps("
        "{k: v for k, v in os.environ.items() if k.startswith('PYTHON')}))\n"
    )
    child.chmod(0o755)
    config = {
        "python": sys.executable,
        "litellm": str(child),
        "config": str(runtime / "config.yaml"),
        "stdout": str(runtime / "stdout.log"),
        "stderr": str(runtime / "stderr.log"),
        "pid": str(runtime / "litellm.pid"),
        "state": str(runtime / "state.json"),
        "runtime_dir": str(runtime),
        "env": {"PYTHONPATH": str(runtime), "PYTHONNOUSERSITE": "1"},
    }
    (runtime / "launch_config.json").write_text(json.dumps(config))
    done = subprocess.run(
        [
            sys.executable,
            *flags,
            str(runtime / "launcher.py"),
            str(runtime / "launch_config.json"),
        ],
        cwd=app,
        env={**os.environ, "PYTHONPATH": str(app), "PYTHONWARNINGS": "ignore"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert not marker.exists(), "root's launcher imported the workspace's json"
    assert done.returncode == 0, done.stderr
    for _ in range(100):
        if seen.exists() and seen.read_text():
            break
        import time

        time.sleep(0.05)
    assert json.loads(seen.read_text()) == {
        "PYTHONPATH": str(runtime),
        "PYTHONNOUSERSITE": "1",
        "PYTHON_DOTENV_DISABLED": "1",
    }


@needs_sh
@pytest.mark.asyncio
async def test_bedrock_preflight_reads_no_inherited_user_site(tmp_path):
    """Guards f2a5ed5f: the root preflight kept the exec's PYTHONUSERBASE."""
    planted = tmp_path / "userbase"
    site = (
        planted / f"lib/python{sys.version_info[0]}.{sys.version_info[1]}/site-packages"
    )
    site.mkdir(parents=True)
    marker = tmp_path / "planted-usercustomize-ran"
    (site / "usercustomize.py").write_text(f"open({str(marker)!r}, 'w').write('x')\n")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    script = runtime / "bedrock_patch_preflight.py"
    script.write_text("print('ok')\n")
    base = getattr(sys, "_base_executable", sys.executable)
    env = _LocalEnv(
        {
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "PYTHONUSERBASE": str(planted),
        },
        tmp_path,
    )
    # A dummy preflight prints no verdict; only what it imported matters.
    with contextlib.suppress(Exception):
        await preflight_mod.preflight_sandbox_bedrock_patch(
            env, python=base, runtime_dir=str(runtime), preflight_path=str(script)
        )
    assert env.commands
    assert not marker.exists(), "root's preflight honoured PYTHONUSERBASE"
