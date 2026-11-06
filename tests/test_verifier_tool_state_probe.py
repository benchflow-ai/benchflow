"""Unit checks for the uv-state fix: the uv/pip state probe and refusal classification.

Guards the fix for the pytest plugin guard refusing plugins test.sh installs
with uvx under WORKDIR /root, and the later fix that moves the state always,
into a directory the guard trusts by path, because ``/root/.cache/uv`` itself
came out world-writable on a runtime whose exec mask is 0000 (review of
sdk-update-2026-09-27 at c5b75fcc, must-fix 3). The end-to-end scenarios live
in test_verifier_uv_state.py.
"""

import os
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import _pytest_plugin_guard as guard_module
from benchflow.sandbox import _verifier_tool_state as tool_state
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from tests.test_verifier_uv_state import CTRF_PLUGIN, Layout, install_plugin, make_task


@pytest.fixture(autouse=True)
def _no_host_system_config(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_state, "SYSTEM_CONFIG_DIRS", str(tmp_path / "etc-xdg"))
    monkeypatch.setattr(
        tool_state, "SYSTEM_UV_CONFIG", str(tmp_path / "etc-uv/uv.toml")
    )


def test_refusal_classification_needs_every_untrusted_file_to_be_new(
    tmp_path, monkeypatch
):
    """Guards the uv-state scoring rule: old untrusted files, or nothing new, score.

    Old and new are by hardening's record of the agent's trees; without
    one nothing is evidence (V2-R1-B-1).
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    site = workspace / "plugins"
    install_plugin(site, CTRF_PLUGIN)
    monkeypatch.setattr(sys, "path", [str(site), *sys.path])
    monkeypatch.setattr(guard_module, "_BENCHFLOW_RECORD", "")
    monkeypatch.setattr(
        guard_module, "_RECORD", {guard_module._identity(os.lstat(workspace)): {None}}
    )
    blocked = (str(workspace),)
    new = guard_module._installed_during_verification(["ctrf"], True, blocked, ())
    assert str(site / "ctrf_model.py") in new
    assert str(site / "ctrf_model-0.3.5.dist-info") in new

    everything = {}
    for path in [workspace, *workspace.rglob("*")]:
        everything.setdefault(guard_module._identity(os.lstat(path)), set()).add(None)
    monkeypatch.setattr(guard_module, "_RECORD", everything)
    assert (
        guard_module._installed_during_verification(["ctrf"], True, blocked, ()) == []
    )
    monkeypatch.setattr(guard_module, "_RECORD", False)
    assert (
        guard_module._installed_during_verification(["ctrf"], True, blocked, ()) == []
    )
    # A plugin that does not exist anywhere was never installed.
    monkeypatch.setattr(
        guard_module, "_RECORD", {guard_module._identity(os.lstat(workspace)): {None}}
    )
    assert (
        guard_module._installed_during_verification(
            ["absent_plugin"], True, blocked, ()
        )
        == []
    )


# The probe's decisions, directly.


def _probe(tmp_path, env, blocked=(), monkeypatch=None, *, lexical=False):
    return tool_state.overrides(env, blocked, str(tmp_path / "fresh"), lexical=lexical)


MOVED_ALWAYS = {
    "UV_CACHE_DIR": "uv-cache",
    "UV_TOOL_DIR": "uv-tools",
    "UV_PYTHON_INSTALL_DIR": "uv-python",
    "PIP_CACHE_DIR": "pip-cache",
}


def _moved(tmp_path):
    return {key: str(tmp_path / "fresh" / entry) for key, entry in MOVED_ALWAYS.items()}


def test_image_cache_in_a_runtime_directory_moves_with_the_uv_config(
    tmp_path, monkeypatch
):
    """An image ``ENV UV_CACHE_DIR=/tmp/uv`` is agent-writable even with a safe $HOME."""
    monkeypatch.setattr(tool_state, "owned_safely", lambda st: True)
    runtime = tmp_path / "tmp"
    found = _probe(
        tmp_path,
        {"HOME": str(tmp_path / "root"), "UV_CACHE_DIR": str(runtime / "uv")},
        (str(runtime),),
    )
    assert found == {
        **_moved(tmp_path),
        # Workspace configuration must not decide what lands in the new cache.
        "UV_CONFIG_FILE": str(tmp_path / "fresh/uv.toml"),
        "PIP_CONFIG_FILE": os.devnull,
    }


def test_every_location_moves_where_the_agent_could_not_write(tmp_path, monkeypatch):
    """A root-only $HOME moves too, so the guard need not judge its modes.

    With ``WORKDIR /app``, ``/root/.cache/uv`` was left in place and trusted by
    ownership and mode, and on Docker-in-Docker (exec mask 0000) uv created it
    world-writable: the guard refused the verifier's own ctrf.
    """
    monkeypatch.setattr(tool_state, "owned_safely", lambda st: True)
    found = _probe(tmp_path, {"HOME": str(tmp_path / "root")}, (str(tmp_path / "app"),))
    assert found == {
        **_moved(tmp_path),
        "UV_CONFIG_FILE": str(tmp_path / "fresh/uv.toml"),
        "PIP_CONFIG_FILE": os.devnull,
    }


def test_safe_user_uv_config_is_kept_and_workspace_config_is_not(tmp_path, monkeypatch):
    """A mirror in root's own ~/.config/uv/uv.toml still reaches uv; the workspace's cannot.

    uv reads only ``UV_CONFIG_FILE`` when it is set, which keeps the project
    configuration in the working directory (the workspace) from choosing
    packages that land in the trusted directory.
    """
    monkeypatch.setattr(tool_state, "owned_safely", lambda st: True)
    home = tmp_path / "root"
    (home / ".config/uv").mkdir(parents=True)
    (home / ".config/uv/uv.toml").write_text('index-url = "https://mirror"\n')
    system = tmp_path / "etc-xdg"
    (system / "uv").mkdir(parents=True)
    (system / "uv/uv.toml").write_text('index-url = "https://system"\n')
    env = {"HOME": str(home), "XDG_CONFIG_DIRS": str(system)}

    found = _probe(tmp_path, env, (str(tmp_path / "app"),))
    assert found["UV_CONFIG_FILE"] == str(home / ".config/uv/uv.toml")
    # Under WORKDIR /root the user file is the agent's: the system one is used.
    found = _probe(tmp_path, env, (str(home),))
    assert found["UV_CONFIG_FILE"] == str(system / "uv/uv.toml")


def test_trusted_system_uv_config_and_explicit_choices_are_kept(tmp_path, monkeypatch):
    """Image-level uv configuration (a mirror in /etc/uv) survives the move."""
    monkeypatch.setattr(tool_state, "owned_safely", lambda st: True)
    home = tmp_path / "root"
    system = tmp_path / "etc-xdg"
    (system / "uv").mkdir(parents=True)
    (system / "uv/uv.toml").write_text('index-url = "https://mirror.example"\n')
    blocked = (str(home),)
    env = {"HOME": str(home), "XDG_CONFIG_DIRS": str(system)}
    assert _probe(tmp_path, env, blocked)["UV_CONFIG_FILE"] == str(
        system / "uv/uv.toml"
    )
    # UV_NO_CONFIG already reads nothing; /dev/null pip configuration is kept.
    env.update(UV_NO_CONFIG="1", PIP_CONFIG_FILE=os.devnull)
    found = _probe(tmp_path, env, blocked)
    assert "UV_CONFIG_FILE" not in found and found["PIP_CONFIG_FILE"] == os.devnull


def test_agent_writable_config_home_alone_neutralises_configuration(
    tmp_path, monkeypatch
):
    """``XDG_CONFIG_HOME`` in the workspace: uv and pip read an empty configuration."""
    monkeypatch.setattr(tool_state, "owned_safely", lambda st: True)
    workspace = tmp_path / "app"
    found = _probe(
        tmp_path,
        {"HOME": str(tmp_path / "root"), "XDG_CONFIG_HOME": str(workspace / ".config")},
        (str(workspace),),
    )
    assert found == {
        **_moved(tmp_path),
        "UV_CONFIG_FILE": str(tmp_path / "fresh/uv.toml"),
        "PIP_CONFIG_FILE": os.devnull,
    }


@pytest.mark.parametrize("lexical", [False, True], ids=["probe", "shell-only"])
def test_pip_reading_no_configuration_is_left_alone(tmp_path, lexical):
    """``PIP_CONFIG_FILE=/dev/null`` (main #1188) stays /dev/null.

    pip then reads no file at all, the global one included; replacing it with
    an empty file in the fresh directory, as for an unsafe user file, would
    bring /etc/pip.conf back (PY8 main sync).
    """
    workspace = tmp_path / "root"
    env = {"HOME": str(workspace), "PIP_CONFIG_FILE": os.devnull}
    found = tool_state.overrides(
        env, (str(workspace),), str(tmp_path / "fresh"), lexical=lexical
    )
    assert found["PIP_CONFIG_FILE"] == os.devnull
    assert found["PIP_CACHE_DIR"] == str(tmp_path / "fresh" / "pip-cache")


def test_group_writable_ancestor_is_not_safe(tmp_path, monkeypatch):
    """A cache below a directory other users can write is not root's alone."""
    shared = tmp_path / "shared"
    (shared / "cache").mkdir(parents=True)
    real_stat = os.stat

    def image_stat(path, *args, **kwargs):
        result = list(real_stat(path, *args, **kwargs))
        result[4] = 0
        result[0] = (result[0] & ~0o022) | (0o002 if str(path) == str(shared) else 0)
        return os.stat_result(result)

    monkeypatch.setattr(os, "stat", image_stat)
    assert tool_state.safe(str(tmp_path / "other"), ())
    assert not tool_state.safe(str(shared / "cache" / "uv"), ())


@pytest.mark.parametrize(
    "verifier", [{"user": "agent"}, {"service": "target"}], ids=["non-root", "service"]
)
async def test_only_a_root_verifier_in_main_is_moved(tmp_path, monkeypatch, verifier):
    """Hardening owns ``main`` only (#248); a non-root verifier cannot use root's directory."""
    layout = Layout(tmp_path, monkeypatch)
    task = make_task(layout)
    for key, value in verifier.items():
        setattr(task.config.verifier, key, value)
    env = await lockdown._isolate_verifier_tool_state(
        layout, task, {"HOME": str(layout.home)}, "agent", str(layout.workspace)
    )
    assert env == {}


async def test_shell_only_image_decides_from_paths_and_creates_in_sh(
    tmp_path, monkeypatch
):
    """No python3 at hardening (test.sh installs uv, which downloads one)."""
    layout = Layout(tmp_path, monkeypatch)
    commands = []

    async def shell_only(command, **kwargs):
        commands.append(command)
        if command.startswith("cd / && python3 -I -c"):
            return ExecResult(stdout="", stderr="python3: not found", return_code=127)
        if "command -v python3" in command:
            return ExecResult(stdout="", stderr="", return_code=1)
        return layout.run(command)

    env = await lockdown._isolate_verifier_tool_state(
        SimpleNamespace(exec=shell_only),
        make_task(layout),
        {"HOME": str(layout.home)},
        "agent",
        str(layout.workspace),
    )
    (directory,) = layout.fs.glob("_benchflow_verifier_*")
    assert env["UV_CACHE_DIR"] == str(directory / "uv-cache")
    assert env["UV_CONFIG_FILE"] == str(directory / "uv.toml")
    assert (directory / "uv.toml").read_text() == ""
    assert (directory / "pip.conf").stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize(
    "result",
    [
        ExecResult(stdout="", stderr="boom", return_code=1),
        ExecResult(stdout='{"LD_PRELOAD": "/x"}', stderr="", return_code=0),
        ExecResult(stdout='{"UV_CACHE_DIR": "relative"}', stderr="", return_code=0),
        ExecResult(stdout="not json", stderr="", return_code=0),
        # The guard trusts the new directory by path: a location elsewhere,
        # or one left out, is not what hardening asked for.
        ExecResult(
            stdout='{"UV_CACHE_DIR": "/root/.cache/uv"}', stderr="", return_code=0
        ),
        ExecResult(stdout="{}", stderr="", return_code=0),
    ],
    ids=["failed", "foreign-key", "relative", "garbage", "elsewhere", "nothing"],
)
async def test_failed_probe_moves_nothing(tmp_path, monkeypatch, result):
    """A probe that fails or answers oddly sets nothing; the guard still refuses.
    (HOME here is outside the agent's folders: its workspace is elsewhere.)"""
    layout = Layout(tmp_path, monkeypatch, workspace_name="app")
    env = await lockdown._isolate_verifier_tool_state(
        SimpleNamespace(exec=AsyncMock(return_value=result)),
        make_task(layout),
        {"HOME": str(layout.home)},
        "agent",
        str(layout.workspace),
    )
    assert env == {}


@pytest.mark.parametrize(
    "result",
    [
        ExecResult(stdout="", stderr="boom", return_code=1),
        ExecResult(stdout="not json", stderr="", return_code=0),
        ExecResult(stdout="{}", stderr="", return_code=0),
    ],
    ids=["failed", "garbage", "nothing"],
)
async def test_failed_probe_still_moves_a_home_the_agent_could_write(
    tmp_path, monkeypatch, result
):
    """HOME is the agent's workspace (WORKDIR /root). A failed probe
    leaves uv and pip where they were, but the verifier's HOME moves anyway:
    a planted ~/.local/bin/env or ~/.curlrc would run as root while it grades."""
    layout = Layout(tmp_path, monkeypatch)
    made: list[str] = []

    async def exec_(command, *args, **kwargs):
        if "mkdir -m 755" in command:
            made.append(command)
            return ExecResult(stdout="", stderr="", return_code=0)
        return result

    task = make_task(layout)
    task.config.verifier.env = {}  # HOME is the image's, not the task's choice
    env = await lockdown._isolate_verifier_tool_state(
        SimpleNamespace(exec=AsyncMock(side_effect=exec_)),
        task,
        {"HOME": str(layout.home)},
        "agent",
        str(layout.workspace),
    )
    home = env["HOME"]
    assert home.startswith(str(layout.fs / "_benchflow_verifier_"))
    assert home.endswith("/home")
    assert env == {
        "HOME": home,
        "XDG_CONFIG_HOME": f"{home}/.config",
        "XDG_CACHE_HOME": f"{home}/.cache",
        "XDG_DATA_HOME": f"{home}/.local/share",
        "XDG_STATE_HOME": f"{home}/.local/state",
    }
    (command,) = made
    assert "mkdir -m 700" in command and ".curlrc" in command and ".wgetrc" in command


async def test_a_task_that_names_its_home_keeps_it(tmp_path, monkeypatch):
    """A task whose tests read the agent's output through ~ sets HOME in its
    [verifier] env: the move leaves it, and what the agent left there."""
    layout = Layout(tmp_path, monkeypatch)

    async def exec_(command, *args, **kwargs):
        return ExecResult(stdout="", stderr="boom", return_code=1)

    env = await lockdown._isolate_verifier_tool_state(
        SimpleNamespace(exec=AsyncMock(side_effect=exec_)),
        make_task(layout),  # names HOME in its [verifier] env
        {"HOME": str(layout.home)},
        "agent",
        str(layout.workspace),
    )
    assert "HOME" not in env and "XDG_CONFIG_HOME" not in env


def test_a_home_in_the_agents_folders_moves(tmp_path):
    blocked = ("/root",)
    moved = tool_state.home_overrides({"HOME": "/root"}, blocked, "/d", lexical=True)
    assert moved["HOME"] == "/d/home"
    assert moved["XDG_CONFIG_HOME"] == "/d/home/.config"
    # Elsewhere, judged by the paths alone, it stays: the image's own files
    # there (a model cache, a tool's settings) still count.
    assert (
        tool_state.home_overrides({"HOME": "/root"}, ("/app",), "/d", lexical=True)
        == {}
    )


def test_a_home_with_an_entry_the_agent_could_write_moves(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".local").mkdir(parents=True)
    (home / ".bashrc").write_text("")
    writable = os.lstat(home / ".local").st_ino
    monkeypatch.setattr(tool_state, "owned_safely", lambda st: st.st_ino != writable)
    assert tool_state.home_is_safe(str(home), ()) is False
    monkeypatch.setattr(tool_state, "owned_safely", lambda st: True)
    assert tool_state.home_is_safe(str(home), ()) is True
    assert tool_state.home_overrides({"HOME": str(home)}, (), str(tmp_path / "d")) == {}


def test_the_new_home_is_roots_with_empty_rc_files(tmp_path):
    directory = tmp_path / ("_benchflow_verifier_" + uuid.uuid4().hex)
    home = directory / "home"
    tool_state.create(str(directory), str(home))
    assert home.stat().st_mode & 0o777 == 0o700
    for name in (".curlrc", ".wgetrc"):
        assert (home / name).read_text() == ""
        assert (home / name).stat().st_mode & 0o777 == 0o600


def test_directory_name_is_fresh_for_every_verification(tmp_path, monkeypatch):
    """The directory must not exist beforehand: the probe refuses to reuse one."""
    directory = tmp_path / ("_benchflow_verifier_" + uuid.uuid4().hex)
    tool_state.create(str(directory))
    with pytest.raises(FileExistsError):
        tool_state.create(str(directory))
