"""Guards FF-sw-launchers against the surviving PTY shell at 363d815b."""

import subprocess

import pytest

from benchflow.sandbox.process.agentcore import AgentCoreProcess


@pytest.mark.parametrize("status", [0, 7, 127])
@pytest.mark.parametrize("environment", [False, True])
def test_agent_exit_ends_owning_shell(tmp_path, status, environment):
    env = tmp_path / "launch-env"
    env.write_text("export DUMMY=ok\n")
    command = AgentCoreProcess._launch_command(
        f"exit {status}", str(env) if environment else None, str(tmp_path)
    )
    result = subprocess.run(
        ["/bin/bash", "-c", command + "; printf SURVIVED"],
        capture_output=True,
        timeout=5,
    )
    assert result.stdout == b""
    assert result.returncode == status
    if environment:
        assert not env.exists()
