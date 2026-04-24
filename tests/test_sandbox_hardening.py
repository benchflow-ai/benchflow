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
    task.task_dir = None
    return task


def _snapshot_side_effect(present: frozenset = frozenset()) -> list:
    """Build side_effect list for _snapshot_build_config: mkdir -> per-file probes -> manifest write.

    present: which _BUILD_CONFIG_FILES names exist in the sandbox (rest are absent).
    Ordering mirrors _BUILD_CONFIG_FILES declaration order — that ordering IS the
    contract under test, so we iterate _ALL_BUILD_FILES directly.
    """
    probes = [
        MagicMock(
            stdout="present\n" if fname in present else "absent\n",
            stderr="",
            exit_code=0,
        )
        for fname in _ALL_BUILD_FILES
    ]
    return [
        MagicMock(stdout="", stderr="", exit_code=0),  # mkdir
        *probes,
        MagicMock(stdout="", stderr="", exit_code=0),  # manifest write
    ]


def _restore_side_effect(manifest: dict[str, bool]) -> list:
    """Build side_effect list for _restore_build_config: manifest read -> per-file ops.

    One empty result per file in _BUILD_CONFIG_FILES declaration order.
    """
    return [
        MagicMock(stdout=json.dumps(manifest), stderr="", exit_code=0),
        *[
            MagicMock(stdout="", stderr="", exit_code=0)
            for _ in range(len(_ALL_BUILD_FILES))
        ],
    ]


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
        """pkill → wipe → workspace freeze → cleanup → env injection."""
        sdk, env, task, tp = harness
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
        chown_idx = next(
            (i for i, c in enumerate(cmds) if "chown -R root:root /testbed" in c),
            None,
        )
        cleanup_idx = next((i for i, c in enumerate(cmds) if "conftest.py" in c), None)
        assert wipe_idx is not None
        assert chown_idx is not None, "workspace chown not found"
        assert cleanup_idx is not None
        assert wipe_idx < chown_idx < cleanup_idx
        assert not any("rm -f /testbed/setup.py" in c for c in cmds)
        assert not any("rsync -a --delete /testbed_verify/" in c for c in cmds)
        assert any("mkdir -p /logs/verifier" in c for c in cmds)
        cleanup_cmd = next(c for c in cmds if "conftest.py" in c)
        assert "sitecustomize.py" in cleanup_cmd and ".pth" in cleanup_cmd
        assert "-not -path '/tests/*'" in cleanup_cmd
        injected = task.config.verifier.env
        assert "--rootdir" not in injected["PYTEST_ADDOPTS"]
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
        assert "--rootdir" not in addopts
        assert "-p no:cacheprovider" in addopts

    @pytest.mark.asyncio
    async def test_task_env_overrides_win(self, harness):
        """Task-level verifier env vars override defaults except pinned invariants."""
        from benchflow._sandbox import VERIFIER_ENV

        sdk, env, task, tp = harness
        task.config.verifier.env = {"PATH": "/custom/bin", "MY_VAR": "hello"}
        mock_v = MagicMock()
        mock_v.verify = AsyncMock(return_value=MagicMock(rewards={"reward": 1.0}))
        with patch("benchflow.sdk.Verifier", return_value=mock_v):
            await sdk._verify(env, task, tp, {})
        injected = task.config.verifier.env
        assert injected["PATH"] == VERIFIER_ENV["PATH"]
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

    def test_cleanup_cmd_purges_py_from_tmp(self):
        """CLEANUP_CMD must delete *.py from /tmp and /var/tmp (module-shadow via non-workspace cwd)."""
        from benchflow._sandbox import CLEANUP_CMD

        assert "find /tmp /var/tmp -name '*.py' -delete" in CLEANUP_CMD

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

        env = _make_env(side_effect=_snapshot_side_effect())

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
            side_effect=_snapshot_side_effect(present=frozenset({"setup.py"}))
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
        env = _make_env(side_effect=_restore_side_effect(manifest))

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
        env = _make_env(side_effect=_restore_side_effect(manifest))

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
        env = _make_env(side_effect=_restore_side_effect(manifest))

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
        await harden_before_verify(
            env,
            task,
            sandbox_user=None,
            workspace="/testbed",
            restore_workspace=True,
        )

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
    async def test_workspace_chowned_after_restore(self):
        """After restore, workspace is chowned to root (belt-and-suspenders against
        zombie sandbox-user processes writing during the verify phase).

        chmod -R a-w is intentionally absent: the verifier runs as root and needs
        to write build artifacts (pip install -e ., setup.py install).  Content
        integrity is guaranteed by the rsync restore, not by read-only permissions.
        """
        from benchflow._sandbox import _SNAPSHOT_MANIFEST, harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        task = _make_task()
        await harden_before_verify(
            env,
            task,
            sandbox_user=None,
            workspace="/testbed",
            restore_workspace=True,
        )

        chown_call = next(
            (
                c
                for c in env.exec.call_args_list
                if "chown -R root:root" in c.args[0] and "/testbed" in c.args[0]
            ),
            None,
        )
        assert chown_call is not None, (
            "workspace chown (root:root) not found — "
            "zombie sandbox-user writes not mitigated"
        )
        assert chown_call.kwargs.get("user") == "root"
        assert "chmod -R a-w" not in chown_call.args[0], (
            "chmod -R a-w must not be present — it breaks pip install as root verifier"
        )
        # chown must come after restore so canonical files are in place first.
        calls = [c.args[0] for c in env.exec.call_args_list]
        restore_idx = next(i for i, c in enumerate(calls) if _SNAPSHOT_MANIFEST in c)
        chown_idx = next(
            i
            for i, c in enumerate(calls)
            if "chown -R root:root" in c and "/testbed" in c
        )
        assert restore_idx < chown_idx

    @pytest.mark.asyncio
    async def test_workspace_ops_skipped_when_workspace_none(self):
        """No workspace chown or chmod when workspace=None (nothing to operate on)."""
        from benchflow._sandbox import harden_before_verify

        env = _make_env()
        await harden_before_verify(env, _make_task(), sandbox_user=None, workspace=None)

        assert not any(
            "chown -R root:root" in c.args[0] for c in env.exec.call_args_list
        )
        assert not any("chmod -R a-w" in c.args[0] for c in env.exec.call_args_list)

    @pytest.mark.asyncio
    async def test_full_workspace_restore_from_testbed_verify_when_enabled(self):
        """When enabled, a full restore from /testbed_verify is attempted.

        This closes F2: agent-modified source files (e.g. /testbed/src/pkg/utils.py)
        are reset to pre-agent canonical state from the snapshot copy, not just the
        11-file build-config subset.  rsync is tried first; cp -a is the fallback.
        """
        from benchflow._sandbox import harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        await harden_before_verify(
            env,
            _make_task(),
            sandbox_user=None,
            workspace="/testbed",
            restore_workspace=True,
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
        # Full restore must run before the chown so canonical files are in place first.
        calls = [c.args[0] for c in env.exec.call_args_list]
        restore_idx = next(
            i for i, c in enumerate(calls) if "/testbed_verify" in c and "rsync" in c
        )
        chown_idx = next(i for i, c in enumerate(calls) if "chown -R root:root" in c)
        assert restore_idx < chown_idx, "full restore must precede workspace chown"

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
    async def test_harden_skips_restore_by_default(self):
        """No destructive workspace restore unless restore_workspace=True."""
        from benchflow._sandbox import _SNAPSHOT_MANIFEST, harden_before_verify

        env = _make_env()
        await harden_before_verify(
            env, _make_task(), sandbox_user=None, workspace="/testbed"
        )

        calls = [c.args[0] for c in env.exec.call_args_list]
        assert not any(_SNAPSHOT_MANIFEST in c for c in calls)
        assert not any("rsync -a --delete /testbed_verify/" in c for c in calls)

    @pytest.mark.asyncio
    async def test_snapshot_dir_chmod_700(self):
        """Snapshot dir is created with chmod 700 so sandbox_user cannot tamper."""
        from benchflow._sandbox import _snapshot_build_config

        env = _make_env(side_effect=_snapshot_side_effect())

        await _snapshot_build_config(env, workspace="/testbed")

        calls = [c.args[0] for c in env.exec.call_args_list]
        assert any("chmod 700" in c and ".benchflow_build_snapshot" in c for c in calls)


# ── TestVerifierUserHarden ────────────────────────────────────────────────────


class TestVerifierUserHarden:
    """harden_before_verify pip isolation and env hardening (verifier OS user removed)."""

    def test_verifier_env_contains_pip_isolation_vars(self):
        """VERIFIER_ENV includes pip isolation vars and HOME=/root."""
        from benchflow._sandbox import VERIFIER_ENV

        assert VERIFIER_ENV["PYTHONNOUSERSITE"] == "1"
        assert VERIFIER_ENV["PIP_USER"] == "0"
        assert VERIFIER_ENV["PIP_NO_USER_CONFIG"] == "1"
        assert VERIFIER_ENV["HOME"] == "/root"

    @pytest.mark.asyncio
    async def test_refresh_workspace_called_after_restore_before_cleanup(self):
        """_refresh_verifier_workspace runs after restore and before CLEANUP_CMD."""
        from benchflow._sandbox import _SNAPSHOT_MANIFEST, harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        task = _make_task(user=None)
        await harden_before_verify(
            env,
            task,
            sandbox_user=None,
            workspace="/testbed",
            restore_workspace=True,
        )

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

    @pytest.mark.asyncio
    async def test_workspace_restore_is_opt_in(self):
        """Default verification keeps legitimate workspace-answer changes."""
        from benchflow._sandbox import harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        task = _make_task(user=None)
        await harden_before_verify(
            env,
            task,
            sandbox_user=None,
            workspace="/testbed",
        )

        calls = [c.args[0] for c in env.exec.call_args_list]
        assert not any("rsync -a --delete /testbed_verify/" in c for c in calls)
        assert not any("rm -f /testbed/setup.py" in c for c in calls)
        assert any("conftest.py" in c for c in calls)


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
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "PYTHONNOUSERSITE",
            "PIP_USER",
            "PIP_NO_USER_CONFIG",
            "PIP_BREAK_SYSTEM_PACKAGES",
            "HOME",
            "PYTHONBREAKPOINT",
            "COVERAGE_PROCESS_START",
            "DJANGO_SETTINGS_MODULE",
            "CELERY_CONFIG_MODULE",
        }

        assert "-c /dev/null" in addopts
        assert "--confcutdir=/tests" in addopts
        assert "--rootdir" not in addopts
        assert "-p no:cacheprovider" in addopts
        assert (
            "PYTHONSAFEPATH" not in VERIFIER_ENV
        )  # removed: Tier 4 freeze covers cwd vector
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
    async def test_distro_pip_env_fedora(self):
        """Fedora-like ID triggers PIP_PREFIX=/usr/local."""
        from benchflow._sandbox import _distro_pip_env

        env = _make_env(
            side_effect=lambda *a, **kw: MagicMock(
                stdout='ID=fedora\nID_LIKE="rhel centos"\n', stderr="", exit_code=0
            )
        )
        assert await _distro_pip_env(env) == {"PIP_PREFIX": "/usr/local"}

    @pytest.mark.asyncio
    async def test_container_plugin_discovery_merged_into_addopts(self):
        """Plugins discovered from root-owned container packages appear as -p flags."""
        from benchflow._sandbox import harden_before_verify

        def side_effect(cmd, **kwargs):
            if "_DISCOVER_PYTEST" in str(cmd) or "importlib.metadata" in str(cmd):
                return MagicMock(
                    stdout='["benchmark", "xdist"]', stderr="", exit_code=0
                )
            return MagicMock(stdout="", stderr="", exit_code=0)

        env = _make_env(side_effect=side_effect)
        task = _make_task()
        await harden_before_verify(env, task, sandbox_user=None)

        addopts = task.config.verifier.env["PYTEST_ADDOPTS"]
        assert "-p benchmark" in addopts
        assert "-p xdist" in addopts

    @pytest.mark.asyncio
    async def test_plugin_discovery_failure_graceful(self):
        """If container-side discovery fails, hardening proceeds without extra plugins."""
        from benchflow._sandbox import VERIFIER_ENV, harden_before_verify

        def side_effect(cmd, **kwargs):
            if "importlib.metadata" in str(cmd):
                raise RuntimeError("no python3")
            return MagicMock(stdout="", stderr="", exit_code=0)

        env = _make_env(side_effect=side_effect)
        task = _make_task()
        await harden_before_verify(env, task, sandbox_user=None)

        assert task.config.verifier.env["PYTEST_ADDOPTS"] == VERIFIER_ENV["PYTEST_ADDOPTS"]

    @pytest.mark.asyncio
    async def test_distro_pip_env_ubuntu(self):
        """Ubuntu must NOT get PIP_PREFIX (their downstream pip already prefixes)."""
        from benchflow._sandbox import _distro_pip_env

        env = _make_env(
            side_effect=lambda *a, **kw: MagicMock(
                stdout="ID=ubuntu\nID_LIKE=debian\n", stderr="", exit_code=0
            )
        )
        assert await _distro_pip_env(env) == {}

    def test_trusted_path_merge_keeps_validated_extras(self):
        """Validated image PATH entries are prepended once to the safe base."""
        from benchflow._sandbox import _merge_trusted_verifier_path

        merged = _merge_trusted_verifier_path(
            [
                "/root/.local/bin",
                "/opt/tool/bin",
                "/usr/local/bin",
                "/root/.local/bin",
            ]
        )

        assert merged == (
            "/root/.local/bin:/opt/tool/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )

    def test_blocked_path_prefixes_include_runtime_and_sandbox_paths(self):
        """Runtime, workspace, and sandbox-user dirs are excluded from PATH extras."""
        from benchflow._sandbox import _blocked_verifier_path_prefixes

        blocked = _blocked_verifier_path_prefixes("agent", "/workspace")

        assert "/tmp" in blocked
        assert "/var/tmp" in blocked
        assert "/logs" in blocked
        assert "/testbed" in blocked
        assert "/workspace" in blocked
        assert "/home/agent" in blocked

    def test_trusted_path_extras_cmd_passes_json_args(self):
        """Container-side PATH validation receives JSON-encoded policy inputs."""
        import shlex

        from benchflow._sandbox import _trusted_path_extras_cmd

        cmd = _trusted_path_extras_cmd("/root/.local/bin:/tmp/bin", ("/tmp",))
        parts = shlex.split(cmd)

        assert parts[:2] == ["python3", "-c"]
        assert json.loads(parts[3]) == "/root/.local/bin:/tmp/bin"
        assert "/usr/local/bin" in json.loads(parts[4])
        assert json.loads(parts[5]) == ["/tmp"]

    @pytest.mark.asyncio
    async def test_harden_preserves_trusted_container_path_extras(self):
        """Verifier PATH includes trusted image-level additions from the container."""
        from benchflow._sandbox import harden_before_verify

        def side_effect(cmd, **kwargs):
            if cmd == "printenv PATH":
                return MagicMock(
                    stdout="/root/.local/bin:/tmp/pwn:/usr/local/bin:/opt/uv/bin\n",
                    stderr="",
                    exit_code=0,
                )
            if cmd.startswith("python3 -c"):
                return MagicMock(
                    stdout='["/root/.local/bin", "/opt/uv/bin"]',
                    stderr="",
                    exit_code=0,
                )
            return MagicMock(stdout="", stderr="", exit_code=0)

        task = _make_task()
        await harden_before_verify(
            _make_env(side_effect=side_effect), task, sandbox_user=None
        )

        assert task.config.verifier.env["PATH"] == (
            "/root/.local/bin:/opt/uv/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )

    @pytest.mark.asyncio
    async def test_task_env_path_cannot_override_hardened_path(self):
        """Task env keeps ordinary vars but cannot replace verifier PATH."""
        from benchflow._sandbox import harden_before_verify

        def side_effect(cmd, **kwargs):
            if cmd == "printenv PATH":
                return MagicMock(
                    stdout="/root/.local/bin:/tmp/pwn:/usr/local/bin\n",
                    stderr="",
                    exit_code=0,
                )
            if cmd.startswith("python3 -c"):
                return MagicMock(
                    stdout='["/root/.local/bin"]',
                    stderr="",
                    exit_code=0,
                )
            return MagicMock(stdout="", stderr="", exit_code=0)

        task = _make_task()
        task.config.verifier.env = {"PATH": "/custom/bin", "MY_VAR": "hello"}
        await harden_before_verify(
            _make_env(side_effect=side_effect), task, sandbox_user="agent"
        )

        assert task.config.verifier.env["PATH"] == (
            "/root/.local/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        assert task.config.verifier.env["MY_VAR"] == "hello"

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
        """pytest_plugins from task.toml are translated to -p flags."""
        from benchflow._sandbox import harden_before_verify

        env = _make_env(side_effect=_manifest_env(_blank_manifest()))
        task = _make_task()
        task.config.verifier.pytest_plugins = ["ctrf", "myplug"]
        await harden_before_verify(env, task, sandbox_user=None, workspace=None)

        final_env = task.config.verifier.env
        addopts = final_env.get("PYTEST_ADDOPTS", "")
        assert "-p ctrf" in addopts
        assert "-p myplug" in addopts
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
        task.config.verifier.pytest_plugins = ["ctrf"]
        await harden_before_verify(env, task, sandbox_user=None)

        addopts = task.config.verifier.env["PYTEST_ADDOPTS"]
        assert addopts.startswith(VERIFIER_ENV["PYTEST_ADDOPTS"])
        assert "-p ctrf" in addopts
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
    async def test_symlinks_purged_before_workspace_chown(self):
        """Symlinks in the workspace must be deleted before the workspace chown.

        Without this, a symlink planted by the agent (e.g. utils.py -> /tmp/evil.py)
        survives; the target is outside the workspace and remains writable,
        so agent code still executes on import during the verify phase.
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
                if "is_symlink()" in c.args[0]
                and "rglob" in c.args[0]
                and "/testbed" in c.args[0]
            ),
            None,
        )
        assert symlink_purge is not None, (
            "symlink purge not found — agent symlinks pointing to external "
            "writable targets survive into the verify phase"
        )
        assert symlink_purge.kwargs.get("user") == "root"
        # Purge resolves each symlink and skips it unless its realpath escapes
        # the workspace, so in-tree fixtures (e.g. OTP cert symlinks) survive.
        assert (
            "resolve()" in symlink_purge.args[0]
            and "startswith" in symlink_purge.args[0]
        )
        # Symlink purge must run before the chown.
        calls = [c.args[0] for c in env.exec.call_args_list]
        symlink_idx = next(
            i for i, c in enumerate(calls) if "is_symlink()" in c and "rglob" in c
        )
        chown_idx = next(i for i, c in enumerate(calls) if "chown -R root:root" in c)
        assert symlink_idx < chown_idx, "symlink purge must precede workspace chown"

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
        # Baseline-aware: dirs present in /testbed_verify must survive so tasks
        # whose verifiers diff workspace against the baseline don't break.
        assert "/testbed_verify" in purge_call.args[0]
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


class TestSandboxFailureModes:
    """Recovery paths when untrusted inputs (task.toml, PATH extras) are malformed."""

    @pytest.mark.asyncio
    async def test_plugin_discovery_bad_json_graceful(self):
        """Malformed JSON from container plugin discovery falls back gracefully."""
        from benchflow._sandbox import _discover_pytest_plugin_flags

        env = _make_env(side_effect=lambda cmd, **kw: MagicMock(
            stdout="not valid json", stderr="", exit_code=0
        ))
        task = _make_task()
        flags = await _discover_pytest_plugin_flags(env, task)
        assert flags == ""

    @pytest.mark.asyncio
    async def test_trusted_path_extras_malformed_json_falls_back(self):
        """Malformed JSON from the container-side PATH probe falls back to SAFE_VERIFIER_PATH."""
        from benchflow._sandbox import _SAFE_VERIFIER_PATH, _trusted_verifier_path

        async def fake_exec(cmd, user=None, timeout_sec=None):
            result = MagicMock()
            if "printenv PATH" in cmd:
                result.stdout = "/usr/local/bin:/usr/bin:/bin"
            else:
                result.stdout = "not json"
            return result

        env = MagicMock()
        env.exec = AsyncMock(side_effect=fake_exec)

        path = await _trusted_verifier_path(env, sandbox_user=None, workspace=None)
        # Malformed JSON ⇒ extras treated as empty ⇒ result equals safe PATH
        assert path == _SAFE_VERIFIER_PATH
