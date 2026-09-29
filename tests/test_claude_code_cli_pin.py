"""claude-agent-acp runs a Claude Code CLI pinned apart from the adapter (#1137).

claude-opus-5-5 refuses Claude Code older than 2.1.280, and the CLI the
adapter's SDK bundles only moves with an adapter release. The launcher hands
the pinned CLI to the adapter through ``CLAUDE_CODE_EXECUTABLE``, which
claude-agent-acp 0.81.2 reads before it looks for its SDK's bundled binary
(``claudeCliPath`` and ``pathToClaudeCodeExecutable`` in ``acp-agent.ts``).
These tests run the generated shell against fakes; nothing is installed.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.agents import registry
from benchflow.agents.registry import (
    AGENTS,
    CLAUDE_AGENT_ACP_LAUNCHER,
    CLAUDE_CODE_EXECUTABLE_PATH,
    _claude_code_version_check,
    pinned_npm_package,
)

NODE = "/opt/benchflow/node/bin/node"


def _fake(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def test_pins_name_the_adapter_and_the_cli_apart():
    assert pinned_npm_package("claude-agent-acp") == (
        "@agentclientprotocol/claude-agent-acp",
        "0.81.2",
    )
    assert pinned_npm_package("claude-code") == ("@anthropic-ai/claude-code", "2.1.280")


def test_install_adds_the_pinned_cli_beside_the_adapter():
    cmd = AGENTS["claude-agent-acp"].install_cmd
    npm = "/opt/benchflow/node/bin/npm install -g --prefix /opt/benchflow/js-agents"
    adapter = f"{npm} @agentclientprotocol/claude-agent-acp@0.81.2 )"
    # Its own npm call: the package's postinstall places the native binary.
    cli = f"{npm} @anthropic-ai/claude-code@2.1.280 )"
    assert adapter in cmd
    assert cli in cmd
    launcher = cmd.index("printf '%s\\n'")
    assert cmd.index(adapter) < cmd.index(cli) < launcher
    assert cmd.index(f"{CLAUDE_CODE_EXECUTABLE_PATH} --version") > launcher
    assert cmd.endswith(_claude_code_version_check())


def test_other_js_agents_keep_their_launcher_byte_for_byte():
    """The launcher generalization must not rewrite any other agent's install."""
    cmd = AGENTS["gemini"].install_cmd
    assert (
        "printf '%s\\n' '#!/bin/sh' 'exec /opt/benchflow/node/bin/node "
        '/opt/benchflow/js-agents/bin/gemini "$@"\' > /opt/benchflow/bin/gemini'
    ) in cmd
    assert "CLAUDE_CODE_EXECUTABLE" not in cmd


def test_launcher_hands_the_pinned_cli_to_the_adapter(tmp_path):
    """The adapter sees the pinned CLI even when the caller names another one."""
    node = _fake(
        tmp_path / "node",
        'printf "%s\\n" "$CLAUDE_CODE_EXECUTABLE" "$@"\n',
    )
    launcher = tmp_path / "claude-agent-acp"
    launcher.write_text(CLAUDE_AGENT_ACP_LAUNCHER.replace(NODE, str(node)))
    env = {**os.environ, "CLAUDE_CODE_EXECUTABLE": "/tmp/unpinned-claude"}
    out = subprocess.run(
        ["sh", str(launcher), "--flag", "two words"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.splitlines()
    assert out == [
        CLAUDE_CODE_EXECUTABLE_PATH,
        "/opt/benchflow/js-agents/bin/claude-agent-acp",
        "--flag",
        "two words",
    ]


@pytest.mark.parametrize(
    "output,ok",
    [
        ("2.1.280 (Claude Code)", True),
        ("2.1.257 (Claude Code)", False),
        # The package's bin placeholder, left when postinstall did not run.
        ("Error: claude native binary not installed.", False),
    ],
)
def test_install_fails_unless_the_cli_reports_its_pin(tmp_path, output, ok):
    cli = _fake(tmp_path / "claude", f'echo "{output}"\n')
    check = _claude_code_version_check().replace(CLAUDE_CODE_EXECUTABLE_PATH, str(cli))
    result = subprocess.run(["sh", "-c", check], capture_output=True, text=True)
    assert (result.returncode == 0) is ok
    if not ok:
        assert "is not Claude Code 2.1.280" in result.stderr
        assert output in result.stderr


@pytest.mark.skipif(shutil.which("dash") is None, reason="dash not installed")
def test_version_check_runs_under_dash(tmp_path):
    cli = _fake(tmp_path / "claude", 'echo "2.1.280 (Claude Code)"\n')
    check = _claude_code_version_check().replace(CLAUDE_CODE_EXECUTABLE_PATH, str(cli))
    assert subprocess.run(["dash", "-c", check]).returncode == 0


def test_cli_pin_bump_retargets_the_install_check(monkeypatch):
    monkeypatch.setattr(
        registry, "_CLAUDE_CODE_PACKAGE", "@anthropic-ai/claude-code@2.1.290"
    )
    assert re.search(r"'2\.1\.290 \(Claude Code\)'", _claude_code_version_check())
