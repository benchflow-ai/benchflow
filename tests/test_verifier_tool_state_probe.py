"""Unit checks for the uv-state fix: the uv/pip state probe and refusal classification.

Guards the fix for the pytest plugin guard refusing plugins test.sh installs
with uvx under WORKDIR /root. The end-to-end
scenarios live in test_verifier_uv_state.py.
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
    """Guards the uv-state scoring rule: old untrusted files, or nothing new, score."""
    site = tmp_path / "workspace"
    install_plugin(site, CTRF_PLUGIN)
    monkeypatch.setattr(sys, "path", [str(site), *sys.path])
    monkeypatch.setattr(guard_module, "_armed_ns", lambda: 0)
    blocked = (str(site),)
    new = guard_module._installed_during_verification(["ctrf"], True, blocked, ())
    assert str(site / "ctrf_model.py") in new
    assert str(site / "ctrf_model-0.3.5.dist-info") in new

    future = 1 << 62
    monkeypatch.setattr(guard_module, "_armed_ns", lambda: future)
    assert (
        guard_module._installed_during_verification(["ctrf"], True, blocked, ()) == []
    )
    # A plugin that does not exist anywhere was never installed.
    monkeypatch.setattr(guard_module, "_armed_ns", lambda: 0)
    assert (
        guard_module._installed_during_verification(
            ["absent_plugin"], True, blocked, ()
        )
        == []
    )


# The probe's decisions, directly.


def _probe(tmp_path, env, blocked=(), monkeypatch=None, *, lexical=False):
    return tool_state.overrides(env, blocked, str(tmp_path / "fresh"), lexical=lexical)


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
        "UV_CACHE_DIR": str(tmp_path / "fresh/uv-cache"),
        # Workspace configuration must not decide what lands in the new cache.
        "UV_CONFIG_FILE": str(tmp_path / "fresh/uv.toml"),
    }


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
    assert "UV_CONFIG_FILE" not in found and "PIP_CONFIG_FILE" not in found


def test_agent_writable_config_home_alone_neutralises_configuration(
    tmp_path, monkeypatch
):
    """``XDG_CONFIG_HOME`` in the workspace: caches stay, configuration moves."""
    monkeypatch.setattr(tool_state, "owned_safely", lambda st: True)
    workspace = tmp_path / "app"
    found = _probe(
        tmp_path,
        {"HOME": str(tmp_path / "root"), "XDG_CONFIG_HOME": str(workspace / ".config")},
        (str(workspace),),
    )
    assert set(found) == {"UV_CONFIG_FILE", "PIP_CONFIG_FILE"}


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
        if command.startswith("cd / && python3 -c"):
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
    ],
    ids=["failed", "foreign-key", "relative", "garbage"],
)
async def test_failed_probe_moves_nothing(tmp_path, monkeypatch, result):
    """A probe that fails or answers oddly sets nothing; the guard still refuses."""
    layout = Layout(tmp_path, monkeypatch)
    env = await lockdown._isolate_verifier_tool_state(
        SimpleNamespace(exec=AsyncMock(return_value=result)),
        make_task(layout),
        {"HOME": str(layout.home)},
        "agent",
        str(layout.workspace),
    )
    assert env == {}


def test_directory_name_is_fresh_for_every_verification(tmp_path, monkeypatch):
    """The directory must not exist beforehand: the probe refuses to reuse one."""
    directory = tmp_path / ("_benchflow_verifier_" + uuid.uuid4().hex)
    tool_state.create(str(directory))
    with pytest.raises(FileExistsError):
        tool_state.create(str(directory))
