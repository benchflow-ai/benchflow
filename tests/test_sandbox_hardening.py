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
    "requirements.txt",
    "requirements-dev.txt",
    "Makefile",
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

    def test_cleanup_cmd_no_maxdepth(self):
        """CLEANUP_CMD must not limit find depth so deeply nested conftest.py is caught."""
        from benchflow._sandbox import CLEANUP_CMD

        assert "-maxdepth" not in CLEANUP_CMD, (
            "CLEANUP_CMD has a -maxdepth limit — conftest.py nested beyond that "
            "depth escapes the sweep"
        )

    @pytest.mark.asyncio
    async def test_cleanup_cmd_runs_as_root(self):
        """CLEANUP_CMD must run as root so find can traverse all dirs."""
        from benchflow._sandbox import harden_before_verify

        env = _make_env()
        await harden_before_verify(env, _make_task(), sandbox_user=None)

        cleanup = next(
            (c for c in env.exec.call_args_list if "conftest.py" in c.args[0]),
            None,
        )
        assert cleanup is not None
        assert cleanup.kwargs.get("user") == "root"


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
                    for _ in range(len(_ALL_BUILD_FILES))
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
                    for _ in range(len(_ALL_BUILD_FILES) - 1)
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
                *[
                    MagicMock(stdout="", stderr="", exit_code=0)
                    for _ in range(len(_ALL_BUILD_FILES))
                ],
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
                *[
                    MagicMock(stdout="", stderr="", exit_code=0)
                    for _ in range(len(_ALL_BUILD_FILES))
                ],
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
                *[
                    MagicMock(stdout="", stderr="", exit_code=0)
                    for _ in range(len(_ALL_BUILD_FILES))
                ],
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
    async def test_workspace_frozen_after_restore(self):
        """After restore, workspace is chowned root and made read-only (a-w).

        This closes the editable-install source modification vector: the agent
        may have changed /testbed/src/pkg/utils.py, but it is frozen before
        the verifier runs, so imports of that package execute the canonical code.
        """
        from benchflow._sandbox import _SNAPSHOT_MANIFEST, harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        task = _make_task()
        await harden_before_verify(env, task, sandbox_user=None, workspace="/testbed")

        freeze_call_obj = next(
            (
                c
                for c in env.exec.call_args_list
                if "chown -R root:root" in c.args[0]
                and "chmod -R a-w" in c.args[0]
                and "/testbed" in c.args[0]
            ),
            None,
        )
        assert freeze_call_obj is not None, (
            "workspace freeze (chown root + chmod a-w) not found — "
            "editable-install source tampering is not mitigated"
        )
        assert freeze_call_obj.kwargs.get("user") == "root", (
            "workspace freeze must run as root"
        )
        # Freeze must come after restore so the canonical files are written first.
        calls = [c.args[0] for c in env.exec.call_args_list]
        restore_idx = next(i for i, c in enumerate(calls) if _SNAPSHOT_MANIFEST in c)
        freeze_idx = next(
            i
            for i, c in enumerate(calls)
            if "chown -R root:root" in c and "chmod -R a-w" in c
        )
        assert restore_idx < freeze_idx

    @pytest.mark.asyncio
    async def test_workspace_not_frozen_when_workspace_none(self):
        """No freeze call when workspace=None (nothing to freeze)."""
        from benchflow._sandbox import harden_before_verify

        env = _make_env()
        await harden_before_verify(env, _make_task(), sandbox_user=None, workspace=None)

        assert not any("chmod -R a-w" in c.args[0] for c in env.exec.call_args_list)

    @pytest.mark.asyncio
    async def test_full_workspace_restore_from_testbed_verify(self):
        """When workspace is set, a full restore from /testbed_verify is attempted.

        This closes F2: agent-modified source files (e.g. /testbed/src/pkg/utils.py)
        are reset to pre-agent canonical state from the snapshot copy, not just the
        11-file build-config subset.  rsync is tried first; cp -a is the fallback.
        """
        from benchflow._sandbox import harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        await harden_before_verify(
            env, _make_task(), sandbox_user=None, workspace="/testbed"
        )

        restore_call = next(
            (
                c
                for c in env.exec.call_args_list
                if "/testbed_verify" in c.args[0]
                and "rsync" in c.args[0]
                and "/testbed" in c.args[0]
            ),
            None,
        )
        assert restore_call is not None, (
            "full workspace restore from /testbed_verify not found — "
            "agent-modified source files survive to the verifier"
        )
        assert restore_call.kwargs.get("user") == "root"
        # Full restore must run before the freeze so canonical files get locked.
        calls = [c.args[0] for c in env.exec.call_args_list]
        restore_idx = next(
            i for i, c in enumerate(calls) if "/testbed_verify" in c and "rsync" in c
        )
        freeze_idx = next(i for i, c in enumerate(calls) if "chmod -R a-w" in c)
        assert restore_idx < freeze_idx, "full restore must precede workspace freeze"

    def test_build_config_files_matches_test_constant(self):
        """_ALL_BUILD_FILES in this test file must mirror _BUILD_CONFIG_FILES in the implementation.

        If they diverge, the parametrized symlink-sever test silently skips new files.
        """
        from benchflow._sandbox import _BUILD_CONFIG_FILES

        assert set(_ALL_BUILD_FILES) == set(_BUILD_CONFIG_FILES), (
            "Update _ALL_BUILD_FILES at the top of this test file to match "
            f"_BUILD_CONFIG_FILES: {sorted(_BUILD_CONFIG_FILES)}"
        )
        assert "requirements.txt" in _BUILD_CONFIG_FILES
        assert "requirements-dev.txt" in _BUILD_CONFIG_FILES
        assert "Makefile" in _BUILD_CONFIG_FILES

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
                    for _ in range(len(_ALL_BUILD_FILES))
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
            "PYTHONPYCACHEPREFIX",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONSAFEPATH",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "PYTHONNOUSERSITE",
            "PIP_USER",
            "PIP_NO_USER_CONFIG",
            "HOME",
            "PYTHONBREAKPOINT",
            "COVERAGE_PROCESS_START",
            "DJANGO_SETTINGS_MODULE",
            "CELERY_CONFIG_MODULE",
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

    @pytest.mark.asyncio
    async def test_pytest_addopts_hardened_when_task_env_none(self):
        """PYTEST_ADDOPTS is the hardened base even when task.config.verifier.env is None.

        The rebuild must happen unconditionally — not only when task env is populated.
        Without this, a None task env would leave PYTEST_ADDOPTS unset or lost.
        """
        from benchflow._sandbox import VERIFIER_ENV, harden_before_verify

        env = _make_env()
        task = _make_task()
        task.config.verifier.env = None
        await harden_before_verify(env, task, sandbox_user=None)

        assert (
            task.config.verifier.env["PYTEST_ADDOPTS"] == VERIFIER_ENV["PYTEST_ADDOPTS"]
        )
        assert "-c /dev/null" in task.config.verifier.env["PYTEST_ADDOPTS"]
        assert "--confcutdir=/tests" in task.config.verifier.env["PYTEST_ADDOPTS"]

    @pytest.mark.asyncio
    async def test_pytest_addopts_not_overridable_by_task_env(self):
        """A task that sets PYTEST_ADDOPTS in verifier.env must not win.

        Without the re-pin the task could strip -c /dev/null and --confcutdir,
        re-enabling pyproject.toml discovery and conftest walk-up.
        """
        from benchflow._sandbox import VERIFIER_ENV, harden_before_verify

        env = _make_env()
        task = _make_task()
        task.config.verifier.env = {"PYTEST_ADDOPTS": "--rootdir=/testbed"}
        await harden_before_verify(env, task, sandbox_user=None)

        assert (
            task.config.verifier.env["PYTEST_ADDOPTS"] == VERIFIER_ENV["PYTEST_ADDOPTS"]
        )

    @pytest.mark.asyncio
    async def test_pytest_addopts_task_override_with_plugins(self):
        """Even when the task overrides PYTEST_ADDOPTS, plugins are appended to the hardened base."""
        from benchflow._sandbox import VERIFIER_ENV, harden_before_verify

        env = _make_env()
        task = _make_task()
        task.config.verifier.env = {"PYTEST_ADDOPTS": "--rootdir=/evil"}
        task.config.verifier.pytest_plugins = ["pytest-json-ctrf"]
        await harden_before_verify(env, task, sandbox_user=None)

        addopts = task.config.verifier.env["PYTEST_ADDOPTS"]
        assert addopts.startswith(VERIFIER_ENV["PYTEST_ADDOPTS"])
        assert "-p pytest-json-ctrf" in addopts
        assert "--rootdir=/evil" not in addopts

    def test_pythonpycacheprefix_set_to_nonexistent(self):
        """PYTHONPYCACHEPREFIX must redirect .pyc lookups away from __pycache__ dirs.

        Without this, an agent can pre-compile a malicious payload into
        workspace/__pycache__/*.pyc and have it execute despite PYTHONDONTWRITEBYTECODE=1
        (which only blocks new writes, not reading existing bytecode).
        """
        from benchflow._sandbox import VERIFIER_ENV

        assert VERIFIER_ENV.get("PYTHONPYCACHEPREFIX") == "/nonexistent"

    @pytest.mark.asyncio
    async def test_symlinks_purged_before_workspace_freeze(self):
        """Symlinks in the workspace must be deleted before freezing.

        Without this, a symlink planted by the agent (e.g. utils.py -> /tmp/evil.py)
        survives the freeze; the target is outside the frozen area and remains
        writable, so agent code still executes on import.
        """
        from benchflow._sandbox import harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        await harden_before_verify(
            env, _make_task(), sandbox_user=None, workspace="/testbed"
        )

        symlink_purge = next(
            (
                c
                for c in env.exec.call_args_list
                if "find" in c.args[0]
                and "-type l" in c.args[0]
                and "-delete" in c.args[0]
                and "/testbed" in c.args[0]
            ),
            None,
        )
        assert symlink_purge is not None, (
            "symlink purge not found — agent symlinks pointing to external "
            "writable targets survive the workspace freeze"
        )
        assert symlink_purge.kwargs.get("user") == "root"
        # Symlink purge must run before the freeze.
        calls = [c.args[0] for c in env.exec.call_args_list]
        symlink_idx = next(
            i for i, c in enumerate(calls) if "-type l" in c and "-delete" in c
        )
        freeze_idx = next(i for i, c in enumerate(calls) if "chmod -R a-w" in c)
        assert symlink_idx < freeze_idx, "symlink purge must precede workspace freeze"

    @pytest.mark.asyncio
    async def test_pycache_purged_during_workspace_freeze(self):
        """__pycache__ directories must be deleted before the workspace is frozen.

        Defense-in-depth against PYTHONPYCACHEPREFIX bypass: even if the prefix
        redirect is circumvented, pre-staged .pyc files are physically gone.
        """
        from benchflow._sandbox import harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        await harden_before_verify(
            env, _make_task(), sandbox_user=None, workspace="/testbed"
        )

        purge_call = next(
            (
                c
                for c in env.exec.call_args_list
                if "__pycache__" in c.args[0] and "rm -rf" in c.args[0]
            ),
            None,
        )
        assert purge_call is not None, (
            "__pycache__ purge not found — pre-compiled .pyc bytecode not mitigated"
        )
        assert purge_call.kwargs.get("user") == "root"
        # Purge must happen before the chown/chmod freeze
        calls = [c.args[0] for c in env.exec.call_args_list]
        purge_idx = next(
            i for i, c in enumerate(calls) if "__pycache__" in c and "rm -rf" in c
        )
        freeze_idx = next(i for i, c in enumerate(calls) if "chown -R root:root" in c)
        assert purge_idx < freeze_idx, "pycache purge must run before workspace freeze"

    def test_code_execution_env_vars_cleared(self):
        """Env vars that trigger arbitrary code execution must be neutralised.

        PYTHONBREAKPOINT: any value other than "0" imports an arbitrary callable.
        COVERAGE_PROCESS_START: coverage.py executes plugins/config on startup.
        DJANGO_SETTINGS_MODULE: Django imports the named module at startup.
        CELERY_CONFIG_MODULE: Celery imports and executes the named module.
        """
        from benchflow._sandbox import VERIFIER_ENV

        assert VERIFIER_ENV["PYTHONBREAKPOINT"] == "0"
        assert VERIFIER_ENV["COVERAGE_PROCESS_START"] == ""
        assert VERIFIER_ENV["DJANGO_SETTINGS_MODULE"] == ""
        assert VERIFIER_ENV["CELERY_CONFIG_MODULE"] == ""

    @pytest.mark.asyncio
    async def test_code_execution_env_vars_repinned_after_task_merge(self):
        """Task env must not be able to override code-execution env vars."""
        from benchflow._sandbox import harden_before_verify

        env = _make_env()
        task = _make_task()
        task.config.verifier.env = {
            "PYTHONBREAKPOINT": "os:system",
            "COVERAGE_PROCESS_START": "/testbed/.coveragerc",
            "DJANGO_SETTINGS_MODULE": "evil.settings",
            "CELERY_CONFIG_MODULE": "evil.celeryconfig",
        }
        await harden_before_verify(env, task, sandbox_user=None)

        result = task.config.verifier.env
        assert result["PYTHONBREAKPOINT"] == "0"
        assert result["COVERAGE_PROCESS_START"] == ""
        assert result["DJANGO_SETTINGS_MODULE"] == ""
        assert result["CELERY_CONFIG_MODULE"] == ""

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
