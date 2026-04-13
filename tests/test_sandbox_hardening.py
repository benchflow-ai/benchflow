"""Tests for harden_before_verify and the sandbox hardening helpers.

Covers three tiers of reward-forge mitigations:
  Tier 1 — wipe /logs/verifier/ before verification
  Tier 2 — snapshot and restore build-config files
  Tier 3 — dedicated verifier OS user, pip isolation, workspace refresh
"""

import json
import shlex
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Shared helpers ────────────────────────────────────────────────────────────

_ALL_BUILD_FILES = (
    "setup.py",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "noxfile.py",
    "hatch.toml",
    "flit.ini",
    "MANIFEST.in",
)


def _blank_manifest() -> dict[str, bool]:
    return {f: False for f in _ALL_BUILD_FILES}


def _manifest_env(manifest: dict[str, bool]):
    """Return an async side_effect that serves a manifest for cat calls."""
    from benchflow._sandbox import _SNAPSHOT_MANIFEST

    def side_effect(cmd, **kwargs):
        if f"cat {_SNAPSHOT_MANIFEST}" in cmd:
            return MagicMock(stdout=json.dumps(manifest), stderr="", exit_code=0)
        return MagicMock(stdout="", stderr="", exit_code=0)

    return side_effect


def _make_env(side_effect=None):
    env = MagicMock()
    if side_effect:
        env.exec = AsyncMock(side_effect=side_effect)
    else:
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
    return env


def _make_task(user=None):
    task = MagicMock()
    task.config.verifier.env = None
    task.config.verifier.user = user
    task.config.verifier.pytest_plugins = None
    return task


# ── TestHardenSequence ────────────────────────────────────────────────────────


class TestHardenSequence:
    """End-to-end hardening sequence through sdk._verify."""

    @pytest.fixture
    def harness(self, tmp_path):
        from benchflow.sdk import SDK

        sdk = SDK()
        task = MagicMock()
        task.config.verifier.timeout_sec = 5
        task.config.verifier.env = None
        task.config.verifier.user = None
        tp = MagicMock()
        tp.verifier_dir = tmp_path / "verifier"
        env = _make_env()
        return sdk, env, task, tp

    @pytest.mark.asyncio
    async def test_with_sandbox_user(self, harness):
        """pkill → wipe → restore → refresh → cleanup → env injection, full path."""
        sdk, env, task, tp = harness
        # Use a manifest-aware env so _restore_build_config can parse the cat call.
        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        mock_v = MagicMock()
        mock_v.verify = AsyncMock(return_value=MagicMock(rewards={"reward": 1.0}))
        with patch("benchflow.sdk.Verifier", return_value=mock_v):
            await sdk._verify(
                env, task, tp, {}, sandbox_user="agent", workspace="/testbed"
            )

        cmds = [c.args[0] for c in env.exec.call_args_list]
        assert "pkill -u agent" in cmds[0]
        wipe_idx = next(
            (i for i, c in enumerate(cmds) if "rm -rf /logs/verifier" in c), None
        )
        restore_idx = next(
            (i for i, c in enumerate(cmds) if "rm -f /testbed/setup.py" in c), None
        )
        cleanup_idx = next((i for i, c in enumerate(cmds) if "conftest.py" in c), None)
        assert wipe_idx is not None
        assert restore_idx is not None, (
            "restore file op not found — workspace path not exercised"
        )
        assert cleanup_idx is not None
        assert wipe_idx < restore_idx < cleanup_idx
        assert any("mkdir -p /logs/verifier" in c for c in cmds)
        cleanup_cmd = next(c for c in cmds if "conftest.py" in c)
        assert "sitecustomize.py" in cleanup_cmd and ".pth" in cleanup_cmd
        assert "-not -path '/tests/*'" in cleanup_cmd
        injected = task.config.verifier.env
        assert (
            injected["PATH"]
            == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        assert "--rootdir=/tests" in injected["PYTEST_ADDOPTS"]
        assert "-p no:cacheprovider" in injected["PYTEST_ADDOPTS"]
        assert injected["PYTHONPATH"] == ""
        assert "PYTHONHOME" not in injected  # breaks Py_Initialize if set to ""
        assert injected["PYTHONDONTWRITEBYTECODE"] == "1"

    @pytest.mark.asyncio
    async def test_without_sandbox_user(self, harness):
        """No pkill when sandbox_user is None; cleanup and env injection still run."""
        sdk, env, task, tp = harness
        mock_v = MagicMock()
        mock_v.verify = AsyncMock(return_value=MagicMock(rewards={"reward": 1.0}))
        with patch("benchflow.sdk.Verifier", return_value=mock_v):
            await sdk._verify(env, task, tp, {}, sandbox_user=None)

        cmds = [c.args[0] for c in env.exec.call_args_list]
        assert all("pkill" not in c for c in cmds)
        assert any("conftest.py" in c for c in cmds)
        addopts = task.config.verifier.env["PYTEST_ADDOPTS"]
        assert "--rootdir=/tests" in addopts
        assert "-p no:cacheprovider" in addopts

    @pytest.mark.asyncio
    async def test_task_env_overrides_win(self, harness):
        """Task-level verifier env vars override VERIFIER_ENV defaults."""
        sdk, env, task, tp = harness
        task.config.verifier.env = {"PATH": "/custom/bin", "MY_VAR": "hello"}
        mock_v = MagicMock()
        mock_v.verify = AsyncMock(return_value=MagicMock(rewards={"reward": 1.0}))
        with patch("benchflow.sdk.Verifier", return_value=mock_v):
            await sdk._verify(env, task, tp, {})
        injected = task.config.verifier.env
        assert injected["PATH"] == "/custom/bin"
        assert injected["MY_VAR"] == "hello"
        assert injected["PYTHONPATH"] == ""  # non-overridden defaults kept


# ── TestVerifierDirWipe ───────────────────────────────────────────────────────


class TestVerifierDirWipe:
    """Tier 1: /logs/verifier/ is wiped and recreated before the verifier runs."""

    @pytest.mark.asyncio
    async def test_wipe_recreates_verifier_dir(self):
        """rm -rf, mkdir -p, and chmod 777 are all in one atomic call; it runs as root."""
        from benchflow._sandbox import harden_before_verify

        env = _make_env()
        await harden_before_verify(env, _make_task(), sandbox_user=None)

        match = next(
            (
                c
                for c in env.exec.call_args_list
                if "rm -rf /logs/verifier" in c.args[0]
                and "mkdir -p /logs/verifier" in c.args[0]
                and "chmod 777 /logs/verifier" in c.args[0]
            ),
            None,
        )
        assert match is not None, (
            "expected a single call with rm -rf, mkdir -p, and chmod 777 for /logs/verifier"
        )
        assert match.kwargs.get("user") == "root"


# ── TestBuildConfigSnapshot ───────────────────────────────────────────────────


class TestBuildConfigSnapshot:
    """Tier 2: build-config files are snapshotted before the agent and restored before verification."""

    @pytest.mark.asyncio
    async def test_absent_file_recorded_as_false(self):
        """Absent file → false in manifest (no __ABSENT__ string in content)."""
        from benchflow._sandbox import _snapshot_build_config

        env = _make_env(
            side_effect=[
                MagicMock(stdout="", stderr="", exit_code=0),  # mkdir
                *[
                    MagicMock(stdout="absent\n", stderr="", exit_code=0)
                    for _ in range(8)
                ],
                MagicMock(stdout="", stderr="", exit_code=0),  # manifest write
            ]
        )

        await _snapshot_build_config(env, workspace="/testbed")

        calls = [c.args[0] for c in env.exec.call_args_list]
        manifest_call = next(c for c in calls if "manifest.json" in c)
        json_str = shlex.split(manifest_call)[1]
        manifest = json.loads(json_str)
        assert manifest["setup.py"] is False
        assert "__ABSENT__" not in json_str

    @pytest.mark.asyncio
    async def test_present_file_recorded_as_true(self):
        """Present file → true in manifest; cp command was issued."""
        from benchflow._sandbox import _snapshot_build_config

        env = _make_env(
            side_effect=[
                MagicMock(stdout="", stderr="", exit_code=0),  # mkdir
                MagicMock(stdout="present\n", stderr="", exit_code=0),  # setup.py
                *[
                    MagicMock(stdout="absent\n", stderr="", exit_code=0)
                    for _ in range(7)
                ],
                MagicMock(stdout="", stderr="", exit_code=0),  # manifest write
            ]
        )

        await _snapshot_build_config(env, workspace="/testbed")

        calls = [c.args[0] for c in env.exec.call_args_list]
        assert any("cp --preserve=all /testbed/setup.py" in c for c in calls)
        manifest_call = next(c for c in calls if "manifest.json" in c)
        assert json.loads(shlex.split(manifest_call)[1])["setup.py"] is True

    @pytest.mark.asyncio
    async def test_restore_removes_absent_file(self):
        """Absent entry in manifest → rm -f for destination; runs as root."""
        from benchflow._sandbox import _restore_build_config

        manifest = _blank_manifest()
        env = _make_env(
            side_effect=[
                MagicMock(stdout=json.dumps(manifest), stderr="", exit_code=0),
                *[MagicMock(stdout="", stderr="", exit_code=0) for _ in range(8)],
            ]
        )

        await _restore_build_config(env, workspace="/testbed")

        rm_call = next(
            (
                c
                for c in env.exec.call_args_list
                if "rm -f /testbed/setup.py" in c.args[0]
            ),
            None,
        )
        assert rm_call is not None
        assert rm_call.kwargs.get("user") == "root"

    @pytest.mark.asyncio
    async def test_restore_overwrites_agent_modified_file(self):
        """Present entry in manifest → cp + chown root:root + chmod 644; runs as root."""
        from benchflow._sandbox import _restore_build_config

        manifest = {**_blank_manifest(), "setup.py": True}
        env = _make_env(
            side_effect=[
                MagicMock(stdout=json.dumps(manifest), stderr="", exit_code=0),
                *[MagicMock(stdout="", stderr="", exit_code=0) for _ in range(8)],
            ]
        )

        await _restore_build_config(env, workspace="/testbed")

        cp_call = next(
            (
                c
                for c in env.exec.call_args_list
                if "setup.py" in c.args[0] and "cp" in c.args[0]
            ),
            None,
        )
        assert cp_call is not None
        assert "chown root:root" in cp_call.args[0]
        assert "chmod 644" in cp_call.args[0]
        assert cp_call.kwargs.get("user") == "root"

    @pytest.mark.parametrize("fname", _ALL_BUILD_FILES)
    @pytest.mark.asyncio
    async def test_restore_severs_symlink_before_cp(self, fname):
        """rm -f dst must precede cp for every build-config file so a symlink the agent
        planted is severed, not followed. Parametrized over all 8 tracked files."""
        from benchflow._sandbox import _restore_build_config

        manifest = {**_blank_manifest(), fname: True}
        env = _make_env(
            side_effect=[
                MagicMock(stdout=json.dumps(manifest), stderr="", exit_code=0),
                *[MagicMock(stdout="", stderr="", exit_code=0) for _ in range(8)],
            ]
        )

        await _restore_build_config(env, workspace="/testbed")

        calls = [c.args[0] for c in env.exec.call_args_list]
        cp_call = next((c for c in calls if fname in c and "cp" in c), None)
        assert cp_call is not None, f"no cp call for {fname!r} found"
        # The rm -f must appear in the same command, before cp.
        assert "rm -f" in cp_call, (
            f"rm -f must precede cp for {fname!r} to sever any agent-planted symlink at dst"
        )
        rm_pos = cp_call.index("rm -f")
        cp_pos = cp_call.index("cp ")
        assert rm_pos < cp_pos, (
            f"rm -f must come before cp in the command for {fname!r}"
        )

    @pytest.mark.asyncio
    async def test_harden_calls_restore_before_cleanup(self):
        """All restore ops (manifest read + per-file deletes) complete before CLEANUP_CMD."""
        from benchflow._sandbox import _SNAPSHOT_MANIFEST, harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        task = _make_task()
        await harden_before_verify(env, task, sandbox_user=None, workspace="/testbed")

        calls = [c.args[0] for c in env.exec.call_args_list]
        restore_manifest_idx = next(
            (i for i, c in enumerate(calls) if _SNAPSHOT_MANIFEST in c), None
        )
        # With a blank manifest every file is absent → rm -f calls are the restore ops.
        restore_file_idx = next(
            (i for i, c in enumerate(calls) if "rm -f /testbed/setup.py" in c), None
        )
        cleanup_idx = next((i for i, c in enumerate(calls) if "conftest.py" in c), None)
        assert restore_manifest_idx is not None, "manifest read not found"
        assert restore_file_idx is not None, "per-file restore op not found"
        assert cleanup_idx is not None, "CLEANUP_CMD not found"
        assert restore_manifest_idx < restore_file_idx < cleanup_idx

    @pytest.mark.asyncio
    async def test_harden_skips_restore_without_workspace(self):
        """No restore calls when workspace=None."""
        from benchflow._sandbox import _SNAPSHOT_MANIFEST, harden_before_verify

        env = _make_env()
        await harden_before_verify(env, _make_task(), sandbox_user=None, workspace=None)

        calls = [c.args[0] for c in env.exec.call_args_list]
        assert not any(_SNAPSHOT_MANIFEST in c for c in calls)

    @pytest.mark.asyncio
    async def test_snapshot_dir_chmod_700(self):
        """Snapshot dir is created with chmod 700 so sandbox_user cannot tamper."""
        from benchflow._sandbox import _snapshot_build_config

        env = _make_env(
            side_effect=[
                MagicMock(stdout="", stderr="", exit_code=0),  # mkdir + chmod
                *[
                    MagicMock(stdout="absent\n", stderr="", exit_code=0)
                    for _ in range(8)
                ],
                MagicMock(stdout="", stderr="", exit_code=0),  # manifest write
            ]
        )

        await _snapshot_build_config(env, workspace="/testbed")

        calls = [c.args[0] for c in env.exec.call_args_list]
        assert any("chmod 700" in c and ".benchflow_build_snapshot" in c for c in calls)


# ── TestVerifierUserHarden ────────────────────────────────────────────────────


class TestVerifierUserHarden:
    """Tier 3: harden_before_verify sets the verifier OS user and pip isolation."""

    @pytest.mark.asyncio
    async def test_verifier_user_set_when_none(self):
        """verifier.user is set to 'verifier' when the task leaves it unset."""
        from benchflow._sandbox import _VERIFIER_USER, harden_before_verify

        env = _make_env()
        task = _make_task(user=None)
        await harden_before_verify(env, task, sandbox_user=None)

        assert task.config.verifier.user == _VERIFIER_USER

    @pytest.mark.asyncio
    async def test_verifier_user_not_overridden_when_root(self):
        """task opt-out user='root' is preserved."""
        from benchflow._sandbox import harden_before_verify

        env = _make_env()
        task = _make_task(user="root")
        await harden_before_verify(env, task, sandbox_user=None)

        assert task.config.verifier.user == "root"

    @pytest.mark.asyncio
    async def test_verifier_user_not_overridden_when_uid_zero(self):
        """task opt-out with integer UID 0 is preserved."""
        from benchflow._sandbox import harden_before_verify

        env = _make_env()
        task = _make_task(user=0)
        await harden_before_verify(env, task, sandbox_user=None)

        assert task.config.verifier.user == 0

    def test_verifier_env_contains_pip_isolation_vars(self):
        """VERIFIER_ENV includes pip isolation vars and HOME=/nonexistent."""
        from benchflow._sandbox import VERIFIER_ENV

        assert VERIFIER_ENV["PYTHONNOUSERSITE"] == "1"
        assert VERIFIER_ENV["PIP_USER"] == "0"
        assert VERIFIER_ENV["PIP_NO_USER_CONFIG"] == "1"
        assert VERIFIER_ENV["HOME"] == "/nonexistent"

    @pytest.mark.asyncio
    async def test_refresh_workspace_called_after_restore_before_cleanup(self):
        """_refresh_verifier_workspace runs after restore and before CLEANUP_CMD."""
        from benchflow._sandbox import _SNAPSHOT_MANIFEST, harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        task = _make_task(user=None)
        await harden_before_verify(env, task, sandbox_user=None, workspace="/testbed")

        calls = [c.args[0] for c in env.exec.call_args_list]
        restore_idx = next(
            (i for i, c in enumerate(calls) if _SNAPSHOT_MANIFEST in c), None
        )
        refresh_idx = next(
            (i for i, c in enumerate(calls) if "/testbed_verify/" in c), None
        )
        cleanup_idx = next((i for i, c in enumerate(calls) if "conftest.py" in c), None)
        assert restore_idx is not None, "restore not found"
        assert refresh_idx is not None, "_refresh_verifier_workspace not found"
        assert cleanup_idx is not None, "CLEANUP_CMD not found"
        assert restore_idx < refresh_idx < cleanup_idx


# ── TestVerifierEnv ───────────────────────────────────────────────────────────


class TestVerifierEnv:
    """VERIFIER_ENV contract: every key must be intentional."""

    def test_env_contract(self):
        """Closed-set check — any new key must be added here deliberately."""
        from benchflow._sandbox import VERIFIER_ENV

        addopts = VERIFIER_ENV["PYTEST_ADDOPTS"]

        assert set(VERIFIER_ENV.keys()) == {
            "PATH",
            "PYTEST_ADDOPTS",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONSAFEPATH",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "PYTHONNOUSERSITE",
            "PIP_USER",
            "PIP_NO_USER_CONFIG",
            "HOME",
        }

        assert "-c /dev/null" in addopts
        assert "--confcutdir=/tests" in addopts
        assert "--rootdir=/tests" in addopts
        assert "-p no:cacheprovider" in addopts
        assert VERIFIER_ENV["PYTHONSAFEPATH"] == "1"
        assert VERIFIER_ENV["PYTHONSTARTUP"] == ""
        assert VERIFIER_ENV["LD_PRELOAD"] == ""
        assert VERIFIER_ENV["LD_LIBRARY_PATH"] == ""
        assert VERIFIER_ENV["PYTHONPATH"] == ""
        assert VERIFIER_ENV["PYTHONDONTWRITEBYTECODE"] == "1"
        assert (
            VERIFIER_ENV["PATH"]
            == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )

    def test_plugin_autoload_disabled(self):
        """PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 must be set in VERIFIER_ENV source."""
        from benchflow._sandbox import VERIFIER_ENV

        assert VERIFIER_ENV.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"

    @pytest.mark.asyncio
    async def test_plugin_autoload_disabled_survives_task_env_override(self):
        """A task that sets PYTEST_DISABLE_PLUGIN_AUTOLOAD=0 in verifier.env must not win.

        Task env is applied via dict.update(), which would normally overwrite the key.
        The production code must re-pin it to '1' after the merge.
        """
        from benchflow._sandbox import harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        task = _make_task()
        # Simulate a hostile or misconfigured task env that tries to re-enable autoload.
        task.config.verifier.env = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "0"}
        await harden_before_verify(env, task, sandbox_user=None, workspace=None)

        assert task.config.verifier.env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1", (
            "task env must not be able to override PYTEST_DISABLE_PLUGIN_AUTOLOAD"
        )

    @pytest.mark.asyncio
    async def test_per_task_plugins_appended_to_addopts(self):
        """pytest_plugins are translated to -p flags; PYTEST_DISABLE_PLUGIN_AUTOLOAD must survive."""
        from benchflow._sandbox import harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        task = _make_task()
        task.config.verifier.pytest_plugins = ["pytest-json-ctrf", "myplug"]
        await harden_before_verify(env, task, sandbox_user=None, workspace=None)

        final_env = task.config.verifier.env
        addopts = final_env.get("PYTEST_ADDOPTS", "")
        assert "-p pytest-json-ctrf" in addopts
        assert "-p myplug" in addopts
        # The security flag must still be present after the plugin flags are appended.
        assert final_env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("plugins", [None, []])
    async def test_no_extra_addopts_when_no_plugins(self, plugins):
        """PYTEST_ADDOPTS is not modified when pytest_plugins is None or empty list."""
        from benchflow._sandbox import VERIFIER_ENV, harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        task = _make_task()
        task.config.verifier.pytest_plugins = plugins
        await harden_before_verify(env, task, sandbox_user=None, workspace=None)

        assert (
            task.config.verifier.env["PYTEST_ADDOPTS"] == VERIFIER_ENV["PYTEST_ADDOPTS"]
        )
        assert task.config.verifier.env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"

    def test_pythonhome_not_set(self):
        """PYTHONHOME must not be set — even "" breaks Py_Initialize."""
        from benchflow._sandbox import VERIFIER_ENV

        assert "PYTHONHOME" not in VERIFIER_ENV

    def test_devnull_blocks_hostile_pyproject(self, tmp_path):
        """Real pytest under -c /dev/null ignores agent-written pyproject.toml."""
        import os
        import subprocess
        import sys

        plugin_marker = "benchflow_test_nonexistent_plugin_xyz123"
        (tmp_path / "pyproject.toml").write_text(
            f'[tool.pytest.ini_options]\naddopts = "-p {plugin_marker}"\n'
        )
        (tmp_path / "test_dummy.py").write_text("def test_pass():\n    assert True\n")

        clean_env = {
            k: os.environ[k]
            for k in ("PATH", "HOME", "LANG", "LC_ALL")
            if k in os.environ
        }

        unhardened = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "test_dummy.py"],
            cwd=tmp_path,
            env=clean_env,
            capture_output=True,
            text=True,
        )
        assert unhardened.returncode != 0, (
            "Sanity check failed: hostile pyproject.toml should crash unhardened pytest. "
            f"stdout: {unhardened.stdout}\nstderr: {unhardened.stderr}"
        )
        assert plugin_marker in unhardened.stdout + unhardened.stderr, (
            "Sanity check passed for the wrong reason: hostile plugin marker not in output. "
            f"stdout: {unhardened.stdout}\nstderr: {unhardened.stderr}"
        )

        hardened = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-c",
                "/dev/null",
                "--collect-only",
                "test_dummy.py",
            ],
            cwd=tmp_path,
            env=clean_env,
            capture_output=True,
            text=True,
        )
        assert hardened.returncode == 0, (
            "-c /dev/null should block hostile pyproject.toml discovery. "
            f"stdout: {hardened.stdout}\nstderr: {hardened.stderr}"
        )
        assert "test_pass" in hardened.stdout, (
            "Hardened branch returned 0 but did not collect test_pass. "
            f"stdout: {hardened.stdout}\nstderr: {hardened.stderr}"
        )
        assert plugin_marker not in hardened.stdout + hardened.stderr, (
            f"-c /dev/null did not suppress hostile pyproject.toml — "
            f"plugin marker {plugin_marker!r} leaked into hardened output."
        )
