"""Guard CONFIRM-lockdown-1/2 at 8c11e9a4: every pip uses fixed config policy."""

import os
import subprocess
import sys

import pytest

from benchflow.sandbox import _verifier_tool_state as state
from benchflow.sandbox import lockdown
from tests.test_sandbox_hardening import _make_env, _make_task


@pytest.mark.parametrize(
    "selector,value",
    [
        ("XDG_CONFIG_DIRS", value)
        for value in (":", ":/etc/xdg", "/etc/xdg:", "/etc::/etc/xdg", ".", "relative")
    ]
    + [("XDG_CONFIG_HOME", value) for value in ("relative", ".", "../config")],
)
@pytest.mark.parametrize("explicit", [None, "safe", "/dev/null"])
def test_pip_config_spellings_cannot_authorize_agent_files(
    tmp_path, monkeypatch, selector, value, explicit
):
    """Relative/empty pip XDG entries are cwd-based, unlike uv's XDG default."""
    monkeypatch.setattr(state, "owned_safely", lambda st: True)
    home = tmp_path / "protected"
    home.mkdir()
    trusted = home / "pip.conf"
    trusted.write_text("")
    env = {"HOME": str(home), selector: value}
    if explicit:
        env["PIP_CONFIG_FILE"] = str(trusted) if explicit == "safe" else explicit
    result = state.overrides(env, ("/app",), str(tmp_path / "fresh"))
    assert result.get("PIP_CONFIG_FILE", env.get("PIP_CONFIG_FILE")) == os.devnull


@pytest.mark.parametrize("site", [False, True], ids=["probe-python", "other-venv"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("isolated", [False, True])
def test_every_installer_interpreter_ignores_file_configuration(
    tmp_path, monkeypatch, site, explicit, isolated
):
    """The verifier can choose a venv unknown to the hardening probe."""
    monkeypatch.setattr(state, "owned_safely", lambda st: True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    malicious = workspace / "pip.conf"
    malicious.write_text(
        "[global]\nno-index = true\nfind-links = /dummy-agent-wheels\n"
    )
    home = tmp_path / "protected"
    home.mkdir()
    env = {"HOME": str(home), "XDG_CONFIG_DIRS": str(home)}
    executable = sys._base_executable
    if site:
        venv = tmp_path / "checkenv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        (venv / "pip.conf").symlink_to(malicious)
        executable = str(venv / "bin/python")
    else:
        env["XDG_CONFIG_DIRS"] = str(workspace)
        (workspace / "pip").mkdir()
        (workspace / "pip/pip.conf").symlink_to(malicious)
    if explicit:
        config = home / "pip.conf"
        config.write_text("")
        env["PIP_CONFIG_FILE"] = str(config)
    overrides = state.overrides(env, (str(workspace),), str(tmp_path / "fresh"))
    child_env = {"PATH": os.environ["PATH"], **env, **overrides}
    result = subprocess.run(
        [
            executable,
            "-I",
            "-m",
            "pip",
            *(["--isolated"] if isolated else []),
            "config",
            "list",
        ],
        env=child_env,
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "dummy-agent-wheels" not in result.stdout
    assert "global.no-index" not in result.stdout


def test_trusted_pip_environment_survives_fixed_file_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "owned_safely", lambda st: True)
    env = {"HOME": str(tmp_path), "PIP_INDEX_URL": "https://mirror.invalid/simple"}
    result = state.overrides(env, ("/app",), str(tmp_path / "fresh"))
    assert {**env, **result}["PIP_INDEX_URL"] == env["PIP_INDEX_URL"]
    assert result.get("PIP_CONFIG_FILE") == os.devnull


@pytest.mark.parametrize("user", [None, "root", "0", "1000"])
@pytest.mark.parametrize("service", ["main", "checker"])
@pytest.mark.parametrize("agent_paths", [None, ("/app",)])
async def test_fixed_pip_policy_survives_every_verifier_route(
    user, service, agent_paths
):
    """A failed/absent probe, non-root user or sidecar must keep the same policy."""
    task = _make_task(user=user)
    task.config.verifier.service = service
    task.config.verifier.env = {
        "PIP_CONFIG_FILE": "/trusted/pip.conf",
        "PIP_INDEX_URL": "https://mirror.invalid/simple",
    }
    env = _make_env()
    env._benchflow_path_inspectors[service] = "/etc/fake-inspector"
    result = await lockdown._build_verifier_env(
        env, task, "agent", "/app", agent_paths=agent_paths
    )
    assert result["PIP_CONFIG_FILE"] == os.devnull
    assert result["PIP_INDEX_URL"] == "https://mirror.invalid/simple"
