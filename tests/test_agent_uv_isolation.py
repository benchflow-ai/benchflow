"""Guard FF-install-C3: cd4d492d isolated proxy uv but not agent installers."""

import shlex

import pytest

from benchflow import _uv_pin
from benchflow.agents.registry import AGENTS
from tests._root_stand_in import run_sh


@pytest.mark.parametrize("agent", ["deepagents", "openhands", "harvey-lab-harness"])
def test_agent_uv_drops_workspace_catalog_and_inherited_settings(tmp_path, agent):
    tools = tmp_path / "tools"
    tools.mkdir()
    witness = tmp_path / "untrusted-catalog-visible"
    trace = tmp_path / "uv-env"
    (tmp_path / "uv.toml").write_text(
        'python-downloads-json-url="file:///tmp/dummy-catalog.json"\n'
    )
    uv = tools / "uv"
    uv.write_text(
        '#!/bin/sh\nif [ "${UV_NO_CONFIG:-}" != 1 ] || '
        '[ -n "${UV_PYTHON_DOWNLOADS_JSON_URL:-}" ]; then '
        f"touch {shlex.quote(str(witness))}; fi\n"
        f"env > {shlex.quote(str(trace))}\nexit 19\n"
    )
    uv.chmod(0o755)
    for name in ("curl", "git"):
        tool = tools / name
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o755)
    # The checked uv downloader has its own real-shell tests. Intercept only
    # provisioning so this exercises the subsequent privileged uv caller.
    command = AGENTS[agent].install_cmd.replace(_uv_pin.uv_install_command(), ":")
    command = command.replace(_uv_pin.UV_DIR, str(tools))
    command = command.replace(
        getattr(_uv_pin, "BENCHFLOW_ROOT", "/opt/runtime"),
        str(tmp_path / "protected"),
    )
    result = run_sh(
        command,
        cwd=tmp_path,
        env={
            "PATH": f"{tools}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "UV_PYTHON_DOWNLOADS_JSON_URL": "file:///tmp/dummy-catalog.json",
            "UV_TOOL_DIR": str(tmp_path / "agent-owned"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert trace.exists(), result.stderr
    assert not witness.exists()
    settings = dict(line.split("=", 1) for line in trace.read_text().splitlines())
    assert settings["UV_PYTHON_PREFERENCE"] == "only-managed"
    assert "UV_NO_CACHE" not in settings
    assert settings["UV_CACHE_DIR"].startswith(str(tmp_path / "protected"))
    assert settings["UV_PYTHON_INSTALL_DIR"].startswith(str(tmp_path / "protected"))


_UV_AGENTS = sorted(
    name
    for name, config in AGENTS.items()
    if _uv_pin.uv_install_command() in (config.install_cmd or "")
)


def test_uv_agents_are_enumerated():
    assert {"deepagents", "openhands", "harvey-lab-harness"} <= set(_UV_AGENTS)


@pytest.mark.parametrize("agent", _UV_AGENTS)
@pytest.mark.parametrize("state", ["fresh", "existing", "agent-writable"])
def test_agent_uv_temp_files_stay_in_a_root_only_folder(tmp_path, agent, state):
    """Guards a0c81002 (FF-install-C12's class): uv unpacks hash-checked
    downloads in TMPDIR, so an agent-owned /tmp could swap them before root
    installs them. Every root uv call gets a checked private TMPDIR."""
    tools = tmp_path / "tools"
    tools.mkdir()
    trace = tmp_path / "uv-env"
    uv = tools / "uv"
    uv.write_text(f"#!/bin/sh\nenv > {shlex.quote(str(trace))}\nexit 19\n")
    uv.chmod(0o755)
    for name in ("curl", "git"):
        tool = tools / name
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o755)
    agent_tmp = tmp_path / "agent-owned-tmp"
    agent_tmp.mkdir()
    protected = tmp_path / "protected"
    private = protected / "agents/uv-tmp"
    if state != "fresh":
        private.mkdir(parents=True)
        # Root-only as in a sandbox, whatever the test runner's umask.
        protected.chmod(0o755)
        private.parent.chmod(0o755)
        private.chmod(0o777 if state == "agent-writable" else 0o700)
    command = AGENTS[agent].install_cmd.replace(_uv_pin.uv_install_command(), ":")
    command = command.replace(_uv_pin.UV_DIR, str(tools))
    command = command.replace(_uv_pin.BENCHFLOW_ROOT, str(protected))
    result = run_sh(
        command,
        cwd=tmp_path,
        env={
            "PATH": f"{tools}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "TMPDIR": str(agent_tmp),
            "TMP": str(agent_tmp),
            "TEMP": str(agent_tmp),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    if state == "agent-writable":
        assert not trace.exists(), "uv ran with an agent-changeable TMPDIR"
        # Refused before uv runs, by the check of the agent uv tree that
        # holds the folder, or by the folder's own.
        assert "is not root's alone" in result.stderr
        return
    assert trace.exists(), result.stderr
    settings = dict(
        line.split("=", 1) for line in trace.read_text().splitlines() if "=" in line
    )
    for name in ("TMPDIR", "TMP", "TEMP"):
        assert settings[name] == str(private), name
    assert private.stat().st_mode & 0o777 == 0o700
