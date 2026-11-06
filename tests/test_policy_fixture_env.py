"""Guards FF-test-env at 7a2387ac: policy checks select their own tools."""

import pytest

from tests.test_agent_antigravity import _run_policy_cmd
from tests.test_hosted_search_policy import _run_hosted_search_cmd, _run_no_web_cmd
from tests.test_internet_policy import _run_setup_cmd


@pytest.mark.parametrize("path", ["", "/missing-tools", "hostile"])
@pytest.mark.parametrize("runner", ["internet", "hosted", "no-web", "antigravity"])
def test_policy_settings_do_not_depend_on_the_callers_path(
    path, runner, tmp_path, monkeypatch
):
    """7a2387ac: use the pytest interpreter even outside the system PATH."""
    hostile = tmp_path / "host-tools"
    hostile.mkdir()
    marker = tmp_path / "host-tool-ran"
    for name in ("python3", "node", "bash", "sh", "mkdir", "cat"):
        tool = hostile / name
        tool.write_text(f"#!/bin/sh\n: > '{marker}'\nexit 99\n")
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(hostile) if path == "hostile" else path)
    monkeypatch.setenv("HOME", str(tmp_path / "unused-home"))
    monkeypatch.setenv("BASH_ENV", str(hostile / "bash"))
    home = tmp_path / "agent home"
    home.mkdir()
    if runner == "internet":
        _run_setup_cmd("claude-agent-acp", home)
    elif runner == "hosted":
        _run_hosted_search_cmd("gemini", home)
    elif runner == "no-web":
        _run_no_web_cmd("opencode", home)
    else:
        from benchflow.agents.registry import AGENTS

        _run_policy_cmd(AGENTS["antigravity"].disallow_web_tools_setup_cmd, home)
    assert not marker.exists()
