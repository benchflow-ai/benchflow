"""Guards the fix at 363d815b: pip must not read agent config."""

import os

import pytest

from benchflow.sandbox import _verifier_tool_state as state


@pytest.mark.parametrize(
    "source", ["xdg-first", "xdg-last", "xdg-relative", "system", "site"]
)
@pytest.mark.parametrize("explicit", [None, "/dev/null", "trusted"])
def test_unsafe_pip_global_configuration_disables_all_config_files(
    tmp_path, monkeypatch, source, explicit
):
    monkeypatch.setattr(state, "owned_safely", lambda st: True)
    blocked = tmp_path / "workspace"
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    conf = trusted / "pip.conf"
    conf.write_text("")
    env = {"HOME": str(trusted), "XDG_CONFIG_DIRS": str(trusted)}
    if source.startswith("xdg"):
        choices = [str(blocked), str(trusted)]
        if source == "xdg-last":
            choices.reverse()
        if source == "xdg-relative":
            choices = ["relative", str(trusted)]
        env["XDG_CONFIG_DIRS"] = ":".join(choices)
    elif source == "system":
        monkeypatch.setattr(
            state, "SYSTEM_PIP_CONFIG", str(blocked / "pip.conf"), raising=False
        )
    else:
        monkeypatch.setattr(state.sys, "prefix", str(blocked))
    if explicit:
        env["PIP_CONFIG_FILE"] = str(conf) if explicit == "trusted" else explicit
    found = state.overrides(env, (str(blocked),), str(tmp_path / "fresh"))
    # /dev/null suppresses every pip configuration source; an empty file
    # suppresses user config alone, leaving globals active.
    assert found.get("PIP_CONFIG_FILE", env.get("PIP_CONFIG_FILE")) == os.devnull


def test_protected_global_pip_file_is_also_disabled(tmp_path, monkeypatch):
    """CONFIRM-lockdown-2: another interpreter's site file cannot be inventoried."""
    monkeypatch.setattr(state, "owned_safely", lambda st: True)
    found = state.overrides(
        {"HOME": str(tmp_path), "XDG_CONFIG_DIRS": "/etc/xdg"}, ("/app",), "/fresh"
    )
    assert found["PIP_CONFIG_FILE"] == os.devnull
