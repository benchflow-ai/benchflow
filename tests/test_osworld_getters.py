"""Tests for the in-guest OSWorld getter shim + vendored-getter registry."""

from __future__ import annotations

import base64

import pytest

from benchflow.adapters.osworld_getters import (
    GETTER_MODULE,
    ShimEnv,
    _ShimController,
    resolve_vendored_getter,
)


def _fake_run(captured: list):
    def run(command, shell):  # mirrors RunCommand(command, shell) -> str
        captured.append((command, shell))
        if command.startswith("python3 -c"):
            return "printed-output\n"
        if command.startswith("base64 -w0"):
            return base64.b64encode(b"file-bytes").decode()
        return ""

    return run


class TestShimController:
    def test_execute_python_command_shape_and_routing(self):
        cap: list = []
        ctl = _ShimController(_fake_run(cap))
        res = ctl.execute_python_command("import os; print(os.getcwd())")
        assert res["status"] == "success"
        assert res["output"] == "printed-output\n"
        assert res["returncode"] == 0
        assert cap and cap[0][0].startswith("python3 -c ")

    def test_get_file_base64_roundtrip(self):
        ctl = _ShimController(_fake_run([]))
        assert ctl.get_file("/some/path") == b"file-bytes"

    def test_get_file_missing_returns_none(self):
        ctl = _ShimController(lambda c, s: "")  # empty -> no file
        assert ctl.get_file("/nope") is None

    def test_live_state_controller_call_raises(self):
        ctl = _ShimController(lambda c, s: "")
        with pytest.raises(NotImplementedError):
            ctl.get_accessibility_tree()


class TestRegistry:
    def test_known_getter_types_mapped(self):
        for gtype in ("bookmarks", "vlc_config", "accessibility_tree", "vm_file"):
            assert gtype in GETTER_MODULE

    def test_unknown_getter_returns_none(self):
        assert resolve_vendored_getter("totally_made_up_getter") is None

    def test_shim_env_surface(self):
        env = ShimEnv(lambda c, s: "", "/tmp/cache")
        assert env.vm_platform == "Linux"
        assert env.cache_dir == "/tmp/cache"
        assert hasattr(env.controller, "execute_python_command")
