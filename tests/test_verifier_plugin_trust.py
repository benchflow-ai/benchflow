"""Which pytest11 registrations the verifier is willing to load.

Discovery output becomes the verifier's ``-p`` allowlist, so a name that
reaches it is imported inside pytest with the power to rewrite a test report.
The decision lives in a container-side script rather than in Python, so these
run the real script through an interpreter with a controlled ``sys.path``
instead of mocking ``env.exec`` -- a mock can show that a name was passed
along, not whether it should have been.

``OLD_SCRIPT`` is the form these replace, kept so the contrast is asserted
rather than described.

These all guard PR #1117.
"""

import json
import os
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

OLD_SCRIPT = r"""
import json, sys
try:
    from importlib.metadata import entry_points
    try:
        eps = list(entry_points(group='pytest11'))
    except TypeError:
        eps = list(entry_points().get('pytest11', []))
    names = sorted(set(ep.name for ep in eps))
    print(json.dumps(names))
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    print("[]")
""".strip()

needs_root = pytest.mark.skipif(
    os.geteuid() != 0,
    reason="needs root to create the root-owned tree a trusted plugin lives in",
)


def _plant_registration(root, dist_name, plugin_name, module):
    """Register ``plugin_name`` as a pytest11 entry point rooted at ``root``.

    This is what ``pip`` leaves behind, and what ``importlib.metadata`` reads:
    a ``*.dist-info`` directory holding ``entry_points.txt``. Creating one
    needs no pip and no root, which is the point.
    """
    info = root / f"{dist_name}-1.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "entry_points.txt").write_text(f"[pytest11]\n{plugin_name} = {module}\n")
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: 1.0\n"
    )
    return info


def _plant_module(root, module):
    (root / f"{module}.py").write_text("# plugin body\n")


def _unrelated_prefix(tmp_path):
    """A blocked prefix that cannot accidentally cover ``tmp_path``.

    On Linux ``tmp_path`` sits under ``/tmp``, so passing a real runtime prefix
    here would reject every planted tree for the wrong reason -- and quietly
    turn the "must be rejected" cases into passes that prove nothing.
    """
    return str(tmp_path / "no-such-agent-tree")


def _discover(script, *, path_entries, blocked):
    """Run the discovery script with ``path_entries`` visible on sys.path."""
    env = {
        k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL") if k in os.environ
    }
    env["PYTHONPATH"] = ":".join(str(p) for p in path_entries)
    argv = [sys.executable, "-c", script]
    if "sys.argv[1]" in script:
        argv.append(json.dumps(blocked))
    result = subprocess.run(argv, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout or "[]")


@pytest.fixture
def script():
    from benchflow.sandbox import lockdown

    return lockdown._DISCOVER_PYTEST_PLUGINS_SCRIPT


class TestWorkspaceRegistrations:
    """A registration the agent could have authored must not be trusted."""

    def test_old_form_trusted_a_workspace_registration(self, tmp_path):
        """The defect PR #1117 fixes: enumeration, nothing asked about source.

        The sandbox user owns its workspace, so creating a dist-info there is
        an ordinary file write -- no pip, no root, nothing to intercept.
        """
        _plant_registration(tmp_path, "forged", "forged", "forged")
        _plant_module(tmp_path, "forged")

        found = _discover(OLD_SCRIPT, path_entries=[tmp_path], blocked=[str(tmp_path)])

        assert "forged" in found

    def test_a_workspace_registration_is_rejected(self, tmp_path, script):
        """Guards PR #1117: the forged dist-info no longer clears discovery."""
        _plant_registration(tmp_path, "forged", "forged", "forged")
        _plant_module(tmp_path, "forged")

        found = _discover(script, path_entries=[tmp_path], blocked=[str(tmp_path)])

        assert "forged" not in found

    def test_the_rejection_covers_a_dotted_entry_point(self, tmp_path, script):
        """Guards PR #1117 for ``pkg.plugin``, the usual entry point shape."""
        pkg = tmp_path / "forgedpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "plugin.py").write_text("# plugin body\n")
        _plant_registration(tmp_path, "forgedpkg", "forged", "forgedpkg.plugin")

        found = _discover(script, path_entries=[tmp_path], blocked=[str(tmp_path)])

        assert "forged" not in found

    def test_a_home_directory_registration_is_rejected(self, tmp_path, script):
        """Guards PR #1117: ``pip install --user`` also writes where the
        sandbox user can reach."""
        home = tmp_path / "home" / "agent" / ".local"
        home.mkdir(parents=True)
        _plant_registration(home, "userpkg", "userplug", "userplug")
        _plant_module(home, "userplug")

        found = _discover(
            script, path_entries=[home], blocked=[str(tmp_path / "home" / "agent")]
        )

        assert "userplug" not in found


class TestEditableProjects:
    """The case a dist-info ownership check on its own would let through.

    An editable install puts a root-owned dist-info in site-packages while the
    project's *source* stays in the workspace, and adds that workspace path to
    ``sys.path``. The registration is genuinely root-authored; the code it
    resolves to is not.
    """

    @needs_root
    def test_an_editable_project_backed_by_workspace_source_is_rejected(
        self, tmp_path, script
    ):
        """Guards PR #1117 against the case a dist-info check alone misses."""
        site = tmp_path / "site-packages"
        workspace = tmp_path / "workspace"
        site.mkdir()
        workspace.mkdir()
        # Registration lives with the trusted system packages ...
        _plant_registration(site, "editable-plug", "editableplug", "editable_plug")
        # ... while the module it names resolves into the agent's workspace.
        _plant_module(workspace, "editable_plug")

        found = _discover(
            script, path_entries=[site, workspace], blocked=[str(workspace)]
        )

        assert "editableplug" not in found, (
            "a root-owned registration admitted agent-owned code"
        )

    @needs_root
    def test_a_wholly_root_owned_plugin_is_still_found(self, tmp_path, script):
        """Guards PR #1117 against silencing the plugins discovery exists for.

        Discovery replaced a hand-curated whitelist precisely so that
        image-installed plugins load without per-benchmark code changes;
        rejecting those would break the verifiers that need them.
        """
        site = tmp_path / "site-packages"
        site.mkdir()
        _plant_registration(site, "real-plug", "realplug", "realplug")
        _plant_module(site, "realplug")

        found = _discover(
            script, path_entries=[site], blocked=[_unrelated_prefix(tmp_path)]
        )

        assert "realplug" in found


class TestOtherWaysAgentCodeGetsReached:
    """Routes to agent code that a check on the named module alone would miss."""

    @needs_root
    def test_a_submodule_reached_through_a_widened_package_is_rejected(
        self, tmp_path, script
    ):
        """Guards PR #1117 against clearing ``pkg.evil`` on ``pkg``'s merits.

        A pkgutil-style namespace package runs ``extend_path`` in its
        ``__init__``, folding every same-named directory on ``sys.path`` into
        ``__path__`` -- the workspace included. That widening happens at import
        time, so ``find_spec`` never sees it: the top-level package looks
        wholly root-owned while the submodule pytest ends up importing exists
        only in the agent's tree.
        """
        site = tmp_path / "site-packages"
        workspace = tmp_path / "workspace"
        (site / "sharedns").mkdir(parents=True)
        (workspace / "sharedns").mkdir(parents=True)
        (site / "sharedns" / "__init__.py").write_text(
            '__path__ = __import__("pkgutil").extend_path(__path__, __name__)\n'
        )
        (workspace / "sharedns" / "evil.py").write_text("# plugin body\n")
        _plant_registration(site, "widened", "widenedplug", "sharedns.evil")

        found = _discover(
            script, path_entries=[site, workspace], blocked=[str(workspace)]
        )

        assert "widenedplug" not in found

    @needs_root
    def test_a_root_owned_dotted_plugin_is_still_found(self, tmp_path, script):
        """Guards PR #1117 against the walk refusing a legitimate ``pkg.plugin``.

        Walking the dotted path one component at a time must still clear the
        ordinary case, which is what most real plugins look like.
        """
        site = tmp_path / "site-packages"
        (site / "realpkg").mkdir(parents=True)
        (site / "realpkg" / "__init__.py").write_text("")
        (site / "realpkg" / "plugin.py").write_text("# plugin body\n")
        _plant_registration(site, "real-dotted", "realdotted", "realpkg.plugin")

        found = _discover(
            script, path_entries=[site], blocked=[_unrelated_prefix(tmp_path)]
        )

        assert "realdotted" in found

    @needs_root
    def test_a_system_plugin_shadowed_from_the_workspace_is_rejected(
        self, tmp_path, script
    ):
        """Guards PR #1117: judging the system copy is wrong when pytest
        imports the other one.

        The registration is root-authored and the system module is root-owned,
        but a workspace module of the same name comes first on sys.path, so
        that is what pytest loads. The name has to be refused on the strength
        of the file that will actually be imported.
        """
        site = tmp_path / "site-packages"
        workspace = tmp_path / "workspace"
        site.mkdir()
        workspace.mkdir()
        _plant_registration(site, "shadowed", "shadowplug", "shadowplug")
        _plant_module(site, "shadowplug")
        _plant_module(workspace, "shadowplug")

        found = _discover(
            script,
            path_entries=[workspace, site],  # workspace wins the lookup
            blocked=[str(workspace)],
        )

        assert "shadowplug" not in found

    @needs_root
    def test_a_root_owned_but_world_writable_plugin_is_rejected(self, tmp_path, script):
        """Guards PR #1117: root ownership means little if anyone can write."""
        loose = tmp_path / "loose"
        loose.mkdir()
        _plant_registration(loose, "loose-plug", "looseplug", "looseplug")
        _plant_module(loose, "looseplug")
        (loose / "looseplug.py").chmod(0o777)

        found = _discover(
            script, path_entries=[loose], blocked=[_unrelated_prefix(tmp_path)]
        )

        assert "looseplug" not in found


class TestDiscoveryWiring:
    """The prefixes the script is told to distrust have to reach it."""

    def test_the_command_carries_the_agent_writable_prefixes(self):
        """Guards PR #1117: the script is useless without the prefixes."""
        from benchflow.sandbox.lockdown import (
            _blocked_verifier_path_prefixes,
            _discover_pytest_plugins_cmd,
        )

        cmd = _discover_pytest_plugins_cmd(
            _blocked_verifier_path_prefixes("agent", "/app")
        )

        assert "/app" in cmd
        assert "/home/agent" in cmd

    @pytest.mark.asyncio
    async def test_the_workspace_reaches_the_container_script(self):
        """Guards PR #1117: ``_build_verifier_env`` knows the workspace, and
        discovery has to be told."""
        from benchflow.sandbox.lockdown import _discover_pytest_plugin_flags

        env = MagicMock()
        env.exec = AsyncMock(
            return_value=MagicMock(stdout="[]", stderr="", return_code=0)
        )
        task = MagicMock()
        task.config.verifier.pytest_plugins = []

        await _discover_pytest_plugin_flags(env, task, "agent", "/workspace")

        sent = env.exec.call_args.args[0]
        assert "/workspace" in sent
        assert "/home/agent" in sent

    @pytest.mark.asyncio
    async def test_task_declared_plugins_survive_discovery_returning_nothing(self):
        """Guards PR #1117: a task's own declarations are a separate channel."""
        from benchflow.sandbox.lockdown import _discover_pytest_plugin_flags

        env = MagicMock()
        env.exec = AsyncMock(
            return_value=MagicMock(stdout="[]", stderr="", return_code=0)
        )
        task = MagicMock()
        task.config.verifier.pytest_plugins = ["declared_plug"]

        flags = await _discover_pytest_plugin_flags(env, task, "agent", "/app")

        assert "-p declared_plug" in flags
