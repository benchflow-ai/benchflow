"""Guard FF-install-C13 at a0c81002: pinned installs ignore ambient Git inputs."""

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow import _uv_pin
from benchflow.agents import registry
from tests._root_stand_in import run_sh


def _git(*args, cwd=None):
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
        },
    ).stdout.strip()


@pytest.mark.parametrize("agent", ["harvey-lab-harness", "openhands"])
@pytest.mark.parametrize(
    "source",
    [
        "home",
        "xdg",
        "include",
        "env-global",
        "env-system",
        "env-count",
        "env-parameters",
        "template",
        "repository",
        "fsmonitor",
    ],
)
@pytest.mark.parametrize("git_version", ["current", "before-2.32"])
def test_installer_git_and_uv_children_ignore_agent_configuration(
    tmp_path, agent, source, git_version
):
    """Guards a0c81002: exercise Git checkout under every ambient config source.

    Git before 2.32 (ubuntu:20.04 has 2.25) ignores GIT_CONFIG_GLOBAL and
    GIT_CONFIG_COUNT and reads $HOME/.gitconfig: "before-2.32" runs the real
    Git through a wrapper that drops them, as such a Git would.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git("init", "-q", str(origin))
    (origin / "harness").mkdir()
    (origin / "harness/tools.py").write_text("trusted = True\n")
    _git("add", ".", cwd=origin)
    _git("commit", "-qm", "fixture", cwd=origin)
    commit = _git("rev-parse", "HEAD", cwd=origin)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    hooks = workspace / "hooks"
    hooks.mkdir()
    marker = tmp_path / "hook-ran"
    hook = hooks / "post-checkout"
    hook.write_text(f"#!/bin/sh\nprintf dummy > {shlex.quote(str(marker))}\n")
    hook.chmod(0o755)
    config = workspace / "config"
    config.write_text(f"[core]\n\thooksPath = {hooks}\n")
    env = {
        **os.environ,
        "HOME": str(workspace),
        "XDG_CONFIG_HOME": str(workspace / "xdg"),
    }
    if source == "home":
        (workspace / ".gitconfig").write_text(config.read_text())
    elif source == "xdg":
        xdg = workspace / "xdg/git/config"
        xdg.parent.mkdir(parents=True)
        xdg.write_text(config.read_text())
    elif source == "include":
        (workspace / ".gitconfig").write_text(f"[include]\n\tpath = {config}\n")
    elif source in ("env-global", "env-system"):
        env["GIT_CONFIG_" + source.split("-")[1].upper()] = str(config)
    elif source == "env-count":
        env.update(
            GIT_CONFIG_COUNT="1",
            GIT_CONFIG_KEY_0="core.hooksPath",
            GIT_CONFIG_VALUE_0=str(hooks),
        )
    elif source == "env-parameters":
        env["GIT_CONFIG_PARAMETERS"] = shlex.quote(f"core.hooksPath={hooks}")
    elif source == "template":
        env["GIT_TEMPLATE_DIR"] = str(workspace)
    elif source == "fsmonitor":
        # core.fsmonitor names a program Git runs when it refreshes the index.
        (workspace / ".gitconfig").write_text(f"[core]\n\tfsmonitor = {hook}\n")
    else:
        # These can redirect even an explicit `git -C protected` operation.
        env["GIT_DIR"] = str(origin / ".git")
        env["GIT_WORK_TREE"] = str(workspace)
        _git("config", "core.hooksPath", str(hooks), cwd=origin)
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "curl").write_text("#!/bin/sh\nexit 0\n")
    (tools / "curl").chmod(0o755)
    if git_version == "before-2.32":
        real_git = shutil.which("git")
        assert real_git
        (tools / "git").write_text(
            "#!/bin/sh\nunset GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM GIT_CONFIG_COUNT "
            "GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0 GIT_CONFIG_KEY_1 GIT_CONFIG_VALUE_1\n"
            f'exec {shlex.quote(real_git)} "$@"\n'
        )
        (tools / "git").chmod(0o755)
    protected = tmp_path / "protected"
    checkout = protected / "agents/harvey-labs"
    trace = tmp_path / "uv-child-env"
    child_checkout = protected / "uv-child-checkout"
    # Replace downloads only. Harvey executes the real recipe's Git sequence;
    # OpenHands' uv stand-in spawns real Git with the inherited installer env.
    child = (
        f"git init -q {child_checkout} && "
        f"git -C {child_checkout} fetch -q {origin} {commit} && "
        f"git -C {child_checkout} checkout -q {commit}"
    )
    (tools / "uv").write_text(
        f"#!/bin/sh\nenv > {shlex.quote(str(trace))}\n"
        + (child + " || exit $?\n" if agent == "openhands" else "")
        + "exit 19\n"
    )
    (tools / "uv").chmod(0o755)
    command = registry.AGENTS[agent].install_cmd
    command = command.replace(_uv_pin.uv_install_command(), ":").replace(
        _uv_pin.managed_python_install_command(shlex.quote(_uv_pin.UV_BIN)), ":"
    )
    command = command.replace(_uv_pin.UV_DIR, str(tools))
    command = command.replace(_uv_pin.BENCHFLOW_ROOT, str(protected))
    command = command.replace(registry.HARVEY_LAB_REPO, str(origin)).replace(
        registry.HARVEY_LAB_COMMIT, commit
    )
    env["PATH"] = f"{tools}:{os.environ['PATH']}"
    result = run_sh(
        command,
        cwd=workspace,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert not marker.exists(), "installer executed an agent-supplied Git hook"
    assert result.returncode == 19, result.stderr
    installed = checkout if agent == "harvey-lab-harness" else child_checkout
    assert (installed / "harness/tools.py").read_text() == "trusted = True\n"
    assert trace.exists()
    if source == "repository":
        assert "GIT_DIR=" not in trace.read_text()
        assert "GIT_WORK_TREE=" not in trace.read_text()


@pytest.mark.parametrize("agent", ["deepagents", "harvey-lab-harness", "openhands"])
def test_git_environment_is_a_fixed_allowlist(tmp_path, agent):
    """Guards a0c81002: future GIT_* overrides cannot bypass the fixed policy."""
    tools = tmp_path / "tools"
    tools.mkdir()
    trace = tmp_path / "env"
    for name in ("curl", "git", "uv"):
        path = tools / name
        path.write_text(
            f"#!/bin/sh\nenv > {trace}\nexit 19\n"
            if name == "uv"
            else "#!/bin/sh\nexit 0\n"
        )
        path.chmod(0o755)
    command = registry.AGENTS[agent].install_cmd.replace(
        _uv_pin.uv_install_command(), ":"
    )
    command = command.replace(_uv_pin.UV_DIR, str(tools)).replace(
        _uv_pin.BENCHFLOW_ROOT, str(tmp_path / "protected")
    )
    hostile = {
        key: "dummy"
        for key in (
            "GIT_CONFIG",
            "GIT_CONFIG_PARAMETERS",
            "GIT_CONFIG_KEY_9",
            "GIT_CONFIG_VALUE_9",
            "GIT_TEMPLATE_DIR",
            "GIT_EXEC_PATH",
            "GIT_SSH",
            "GIT_SSH_COMMAND",
            "GIT_ASKPASS",
            "GIT_EXTERNAL_DIFF",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_NAMESPACE",
            "GIT_FUTURE_SETTING",
        )
    }
    result = run_sh(
        command,
        env={"PATH": f"{tools}:{os.environ['PATH']}", "HOME": str(tmp_path), **hostile},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    observed = {
        line.split("=", 1)[0]
        for line in trace.read_text().splitlines()
        if line.startswith("GIT_")
    }
    assert observed.isdisjoint(hostile), (
        f"ambient Git overrides survived: {observed & hostile.keys()}"
    )


def test_openhands_real_uv_git_child_ignores_workspace_config(tmp_path):
    """Guards a0c81002 with real uv and Git, using a local, dependency-free wheel."""
    import shutil
    import sys

    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n'
    )
    (origin / "backend.py").write_text(
        "import pathlib, zipfile\n"
        "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
        "    name = 'openhands-0.0.1-py3-none-any.whl'\n"
        "    with zipfile.ZipFile(pathlib.Path(wheel_directory) / name, 'w') as z:\n"
        "        z.writestr('openhands.py', 'def main(): print(42)\\n')\n"
        "        z.writestr('openhands-0.0.1.dist-info/METADATA', 'Metadata-Version: 2.1\\nName: openhands\\nVersion: 0.0.1\\n')\n"
        "        z.writestr('openhands-0.0.1.dist-info/WHEEL', 'Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n')\n"
        "        z.writestr('openhands-0.0.1.dist-info/entry_points.txt', '[console_scripts]\\nopenhands=openhands:main\\n')\n"
        "        z.writestr('openhands-0.0.1.dist-info/RECORD', '')\n"
        "    return name\n"
    )
    _git("init", "-q", str(origin))
    _git("add", ".", cwd=origin)
    _git("commit", "-qm", "fixture", cwd=origin)
    commit = _git("rev-parse", "HEAD", cwd=origin)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".gitconfig").write_text(
        f'[url "file:///tmp/FF-install-missing-origin/"]\n\tinsteadOf = {origin.as_uri()}\n'
    )
    uv = shutil.which("uv")
    assert uv, "run through uv run pytest"
    command = registry.AGENTS["openhands"].install_cmd
    command = command.replace(_uv_pin.uv_install_command(), ":").replace(
        _uv_pin.managed_python_install_command(shlex.quote(_uv_pin.UV_BIN)), ":"
    )
    command = command.replace(_uv_pin.UV_DIR, str(Path(uv).parent))
    command = command.replace(_uv_pin.BENCHFLOW_ROOT, str(tmp_path / "protected"))
    # Retain the production setup and uv tool invocation; only the remote
    # package, its overrides and Python provisioning become local fixtures.
    import re

    command = re.sub(r"--overrides \S+ ", "--no-build-isolation ", command)
    command = re.sub(
        r"--from '[^']+'",
        "--from " + shlex.quote(f"git+{origin.as_uri()}@{commit}"),
        command,
    )
    command = re.split(r" &&\s+uv tool list", command, maxsplit=1)[0]
    command = command.replace(
        f"--python {_uv_pin.PYTHON_VERSION}", f"--python {shlex.quote(sys.executable)}"
    )
    # The OpenHands recipe wraps this final chain in a subshell.
    command += " )"
    result = run_sh(
        command,
        cwd=workspace,
        env={**os.environ, "HOME": str(workspace)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    launcher = tmp_path / "protected/bin/openhands"
    assert subprocess.check_output([str(launcher)], text=True).strip() == "42"


def _run_words(command: str) -> list[int]:
    """Offsets where ``git`` runs as a command (not ``command -v git``, a
    package name or a ``git+https`` URL)."""
    return [
        m.start(1)
        for m in re.finditer(
            r"(?:^|[;&|(]\s*|\$\(\s*|=\S*\s+)(git)\s+-?[A-Za-z]", command
        )
        if not command[: m.start(1)].rstrip().endswith("-v")
    ]


@pytest.mark.parametrize("agent", sorted(registry.AGENTS))
def test_every_installer_git_and_uv_git_child_has_no_ambient_config(agent):
    """Guards a0c81002: enumerate the class over every agent recipe. Each Git
    command runs with HOME and XDG_CONFIG_HOME at /dev/null and hooks off,
    and so does each uv call that fetches a ``git+`` source."""
    command = registry.AGENTS[agent].install_cmd or ""
    for offset in _run_words(command):
        assert command[:offset].endswith(registry._AGENT_GIT_HOME + " "), (
            f"{agent}: git at {offset} reads root's ambient config"
        )
        assert command[offset:].startswith("git -c core.hooksPath=/dev/null")
    for match in re.finditer(r"\buv (?:tool|pip) install\b[^&|;]*", command):
        if "git+" in match.group(0):
            assert command[: match.start()].endswith(registry._AGENT_GIT_HOME + " ")
    if _run_words(command) or "git+" in command:
        assert registry._agent_git_exports() in command


def test_the_git_enumeration_sees_harvey_and_openhands():
    harvey = registry.AGENTS["harvey-lab-harness"].install_cmd
    assert len(_run_words(harvey)) == 6
    assert "git+" in registry.AGENTS["openhands"].install_cmd
