"""The claude agent's settings are written without python3 (REPORT B8).

``FROM ubuntu:24.04``, the base ``bench new`` writes, has no python3, and the
claude agent's no-web and hosted-search policies wrote
``~/.claude/settings.json`` with ``python3 -c``, so the trial failed before any
model call ("BenchFlow agent settings setup needs python3 in the task image").
The settings are now written with the Node runtime the agent install brings.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.agents.registry import AGENTS

POLICIES = ["disallow_web_tools_setup_cmd", "disallow_hosted_search_setup_cmd"]


def _stub(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


def _run(cmd: str, home: Path, bin_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/sh", "-c", cmd],
        env={"PATH": str(bin_dir), "BENCHFLOW_AGENT_HOME": str(home)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_settings_are_written_by_node_and_never_by_python3(tmp_path, policy):
    """With a node and a python3 both on PATH, node writes the file."""
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    bin_dir.mkdir()
    home.mkdir()
    calls = tmp_path / "calls.txt"
    for name in ("node", "python3"):
        _stub(bin_dir, name, f'echo "{name} $1 $3" >> {calls}\nexit 0\n')

    result = _run(getattr(AGENTS["claude-agent-acp"], policy), home, bin_dir)

    assert result.returncode == 0, result.stderr
    lines = calls.read_text().splitlines()
    assert lines == [f"node -e {home}/.claude/settings.json"]


def _real_node_bin(tmp_path: Path) -> Path:
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs a node binary (the deterministic tier runs the real agent)")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Only node: no python3 anywhere on PATH.
    (bin_dir / "node").symlink_to(node)
    return bin_dir


def test_node_writes_the_no_web_settings_and_keeps_existing_ones(tmp_path):
    bin_dir, home = _real_node_bin(tmp_path), tmp_path / "home"
    settings = home / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"model": "x", "permissions": {"deny": ["Bash"]}}))
    cmd = AGENTS["claude-agent-acp"].disallow_web_tools_setup_cmd

    for _ in range(2):
        result = _run(cmd, home, bin_dir)
        assert result.returncode == 0, result.stderr

    data = json.loads(settings.read_text())
    assert data == {
        "model": "x",
        "permissions": {"deny": ["Bash", "WebSearch", "WebFetch"]},
    }
    assert settings.read_text().endswith("}\n")


def test_node_writes_the_hosted_search_settings_into_a_new_home(tmp_path):
    bin_dir, home = _real_node_bin(tmp_path), tmp_path / "home"
    cmd = AGENTS["claude-agent-acp"].disallow_hosted_search_setup_cmd

    result = _run(cmd, home, bin_dir)

    assert result.returncode == 0, result.stderr
    data = json.loads((home / ".claude" / "settings.json").read_text())
    assert data == {"permissions": {"deny": ["WebSearch"]}}


@pytest.mark.parametrize("policy", POLICIES)
def test_python3_still_writes_them_when_no_node_is_there(tmp_path, policy):
    """A manifest or plugin install without BenchFlow's Node keeps working."""
    python3 = shutil.which("python3")
    if python3 is None:
        pytest.skip("needs python3")
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(python3)

    result = _run(getattr(AGENTS["claude-agent-acp"], policy), home, bin_dir)

    assert result.returncode == 0, result.stderr
    deny = json.loads((home / ".claude" / "settings.json").read_text())["permissions"][
        "deny"
    ]
    assert "WebSearch" in deny


def test_neither_runtime_names_both(tmp_path):
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    bin_dir.mkdir()

    # An empty PATH no longer hides the fixed, isolated Python fallback.
    # Model an image without either installed runtime explicitly.
    command = (
        AGENTS["claude-agent-acp"]
        .disallow_hosted_search_setup_cmd.replace(
            "/usr/bin/python3", "/missing/dummy-python3"
        )
        .replace("/usr/local/bin/python3", "/missing/dummy-local-python3")
        .replace("/opt/benchflow/node/bin/node", "/missing/dummy-node")
    )
    result = _run(command, home, bin_dir)

    assert result.returncode == 127
    assert "needs node" in result.stderr
    assert "or python3 in the task image" in result.stderr
