"""Guard the FF-install merge of 8b03a596: U4 pins retain installer custody."""

import re
import shlex

import pytest

from benchflow import _uv_pin
from benchflow.agents.registry import AGENTS
from tests._root_stand_in import run_sh


@pytest.mark.parametrize(
    "agent,selector",
    [
        ("deepagents", r"-r (/\S+)"),
        ("harvey-lab-harness", r"-r (/\S+)"),
        ("harvey-lab-harness", r"uv venv .*?--python \S+ (\S+)"),
        ("harvey-lab-harness", r"git (?:-c \S+ )?-C (\S+)"),
        ("openhands", r"--overrides (\S+)"),
    ],
)
@pytest.mark.parametrize("unsafe", ["writable", "symlink", "safe"])
def test_agent_installer_refuses_replaceable_inputs_before_uv(
    tmp_path, agent, selector, unsafe
):
    """Guards cf58fc0e + 8b03a596: a safe uv must not bless unsafe inputs."""
    command = AGENTS[agent].install_cmd
    command = command.replace(_uv_pin.uv_install_command(), ":")
    match = re.search(selector, command)
    assert match, "the pinned recipe must name its inputs"
    original = match[1]
    # Only provisioning is replaced. Execute the recipe's real custody checks
    # and control flow against temporary paths, before the first agent uv use.
    provision = getattr(_uv_pin, "managed_python_install_command", None)
    if provision is not None:
        command = command.replace(provision(shlex.quote(_uv_pin.UV_BIN)), ":")
    tools = tmp_path / "tools"
    tools.mkdir()
    trace = tmp_path / "uv-used"
    for name, body in {
        "curl": "exit 0",
        "git": "exit 0",
        "uv": f"touch {shlex.quote(str(trace))}; exit 19",
    }.items():
        tool = tools / name
        tool.write_text(f"#!/bin/sh\n{body}\n")
        tool.chmod(0o755)
    command = command.replace(_uv_pin.UV_DIR, str(tools))
    roots = ("/opt", "/root", "/tmp/oh-sdk-overrides.txt")
    target = original
    for root in roots:
        command = command.replace(root, str(tmp_path) + root)
        target = target.replace(root, str(tmp_path) + root)
    from pathlib import Path

    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The link's target sits in a folder the agent may write.
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace").chmod(0o777)
    sentinel = tmp_path / "workspace" / "untouched"
    sentinel.write_text("dummy input\n")
    if unsafe == "symlink":
        path.symlink_to(sentinel)
    elif path.suffix == ".txt":
        path.write_text("dummy input\n")
        path.chmod(0o600 if unsafe == "safe" else 0o666)
    else:
        path.mkdir()
        path.chmod(0o700 if unsafe == "safe" else 0o777)
    result = run_sh(
        command,
        cwd=tmp_path,
        env={"PATH": f"{tools}:/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    if unsafe == "safe":
        assert trace.exists(), result.stderr
        assert result.returncode == 19
        return
    assert not trace.exists(), "uv ran with an agent-replaceable installer input"
    assert "is not root's alone" in result.stderr
    assert sentinel.read_text() == "dummy input\n"
