"""The pytest plugin guard's 0.8.0 fixes: #1170, #1175, #1176, #1180 and image conftest plugins.

Each fix trusts more, so each comes with the tampering it must still refuse:
a plugin the agent planted in its site-packages or on PYTHONPATH, and an
image plugin module the agent edited.

The scenarios run the real hardening code against a directory tree standing
in for the image and a real pytest with the armed guard, scored through
``_verify_rollout`` (the harness of test_verifier_uv_state.py, where owners
are modelled as root's and the blocked prefixes decide). #1176's owner and
group rule is checked with modelled ``os.stat`` results, and #1180 with a
real interpreter whose standard library directory the test owns.
"""

import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox import lockdown
from benchflow.sandbox.lockdown import (
    _install_pytest_plugin_guard,
    harden_before_verify,
)
from tests.test_sandbox_hardening import _guard_sources, _make_env, _make_task
from tests.test_verifier_guard_interpreters import HOSTILE_PLUGIN, ShellEnv
from tests.test_verifier_plugin_trust import _armed_markers
from tests.test_verifier_uv_state import (
    CTRF_PLUGIN,
    FAILING,
    GUARD_OWNERSHIP,
    PASSING,
    hostile_ctrf,
    install_plugin,
    markers,
    refused_scored_zero,
    score,
    uvx,
    workspace_venv,
)

# A plugin the image's Python has (anyio's role in Terminal-Bench Pro).
IMAGE_PLUGIN = "def pytest_configure(config):\n    config.image_plugin_loaded = True\n"
USES_IMAGE_PLUGIN = (
    "def test_solution(pytestconfig):\n"
    "    assert getattr(pytestconfig, 'image_plugin_loaded', False)\n"
)


def stdout_of(layout):
    return layout.paths.test_stdout_path.read_text()


# #1175: plugins found in the image or inferred from test.sh are optional.


async def test_image_plugin_a_fresh_environment_lacks_is_skipped(tmp_path, monkeypatch):
    """Guards #1175: Terminal-Bench Pro's oracles scored 0.

    The image's Python has anyio (a pytest11 plugin); test.sh runs pytest in
    a fresh ``uv init`` project or a ``uvx`` environment without it. The flag
    BenchFlow added reached that pytest, the guard refused the absent plugin
    and pytest never ran a test. Now the guard loads it only where it is.
    """
    layout, env, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        uvx,
        workspace_name="app",
        discovered=["image_plugin"],
    )

    assert "Verifier plugin trust rejected" not in stdout_of(layout), stdout_of(layout)
    assert error is None
    assert rewards == {"reward": 1.0}
    assert "-p image_plugin" not in env["PYTEST_ADDOPTS"]


async def test_image_plugin_the_running_pytest_has_is_loaded(tmp_path, monkeypatch):
    """Guards #1175's fix against dropping image plugins: where installed, it loads."""

    def image(layout):
        # The image's, so there before hardening arms the guard.
        install_plugin(
            layout.tmp / "image-site-packages",
            IMAGE_PLUGIN,
            name="image_plugin",
            module="image_plugin_module",
        )

    def image_python(env, layout):
        return [*uvx(env, layout), layout.tmp / "image-site-packages"]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        USES_IMAGE_PLUGIN,
        image_python,
        before=image,
        workspace_name="app",
        discovered=["image_plugin"],
    )

    assert error is None, error
    assert rewards == {"reward": 1.0}, stdout_of(layout)


@pytest.mark.parametrize(
    "planted", ["agent-plugins", ".local/lib/python3.12/site-packages"]
)
async def test_optional_plugin_the_agent_planted_is_still_refused(
    tmp_path, monkeypatch, planted, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
    proof = tmp_path / "hostile-imported"

    def plant(layout):
        install_plugin(layout.workspace / planted, hostile_ctrf(proof))

    def reach_it(env, layout):
        return [layout.workspace / planted]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        reach_it,
        before=plant,
        workspace_name="app",
        published=False,
    )

    assert "Verifier plugin trust rejected: ctrf" in stdout_of(layout)
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert markers(layout, "installed") == []
    assert not proof.exists()


@pytest.mark.parametrize("name", ["benchmark", "env"])
async def test_module_that_only_shares_an_absent_plugins_name_is_not_loaded(
    tmp_path, monkeypatch, name
):
    """Guards V2-R1-A-1: #1175 loads an optional plugin only where it is installed.

    The image has pytest-benchmark (``benchmark``) or pytest-env (``env``);
    DevEval's test.sh puts the workspace on PYTHONPATH and runs a fresh
    pytest without them. The repository's own ``benchmark.py`` or ``env.py``,
    which the solution edited, is not that plugin: the guard neither refuses
    nor imports it, and the solution decides the reward.
    """
    proof = tmp_path / "module-imported"

    def edit(layout):
        (layout.workspace / f"{name}.py").write_text(
            f"import pathlib\npathlib.Path({str(proof)!r}).touch()\n"
        )

    def workspace_on_pythonpath(env, layout):
        return [*uvx(env, layout), layout.workspace]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        workspace_on_pythonpath,
        before=edit,
        workspace_name="app",
        discovered=[name],
    )

    assert "Verifier plugin trust rejected" not in stdout_of(layout), stdout_of(layout)
    assert error is None
    assert rewards == {"reward": 1.0}
    assert not proof.exists()


def test_standard_library_module_is_not_an_absent_optional_plugin(monkeypatch):
    """pytest-asyncio's ``asyncio`` missing from this pytest is absent, stdlib or not."""
    monkeypatch.setattr(guard, "registrations", lambda: [])
    assert not guard._present("asyncio")
    registered = SimpleNamespace(name="asyncio")
    monkeypatch.setattr(guard, "registrations", lambda: [(registered, "/x")])
    assert guard._present("asyncio")


async def test_dependency_an_optional_plugin_declares_is_checked_too(
    tmp_path, monkeypatch, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
    proof = tmp_path / "dependency-imported"

    def plant(layout):
        (layout.workspace / "agent_dependency.py").write_text(
            f"import pathlib\npathlib.Path({str(proof)!r}).touch()\n"
        )
        # The image's plugin, there before hardening arms the guard.
        install_plugin(
            layout.tmp / "image-site-packages",
            "pytest_plugins = ['agent_dependency']\n" + IMAGE_PLUGIN,
            name="image_plugin",
            module="image_plugin_module",
        )

    def image_python(env, layout):
        site = layout.tmp / "image-site-packages"
        return [*uvx(env, layout), site, layout.workspace]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        USES_IMAGE_PLUGIN,
        image_python,
        before=plant,
        workspace_name="app",
        discovered=["image_plugin"],
    )

    assert "Verifier plugin trust rejected: agent_dependency" in stdout_of(layout)
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert not proof.exists()


def test_optional_plugin_the_command_line_disables_is_not_loaded(monkeypatch):
    """``-p no:<name>`` on test.sh's command line keeps an optional plugin out."""
    loaded = []
    manager = SimpleNamespace(
        is_blocked=lambda name: False,
        get_plugin=lambda name: None,
        import_plugin=lambda name, **kwargs: loaded.append(name),
    )
    monkeypatch.setattr(guard, "_BENCHFLOW_OPTIONAL", ["ctrf"])
    monkeypatch.setattr(guard, "_present", lambda name: True)
    monkeypatch.setattr(guard, "_validate", lambda names, **kwargs: None)

    guard._load_optional(manager, {"ctrf"})
    assert loaded == []
    guard._load_optional(manager, set())
    assert loaded == ["ctrf"]


def test_optional_plugin_that_fails_to_import_is_not_a_guard_crash(
    tmp_path, monkeypatch
):
    """A trusted optional plugin that fails to import fails pytest as ``-p`` did.

    That is the plugin's failure, scored as test.sh reports it, not the
    guard crashing (which would leave the run unscored).
    """
    markers = _armed_markers(monkeypatch, tmp_path)
    monkeypatch.setattr(guard, "_BENCHFLOW_OPTIONAL", ["broken"])
    monkeypatch.setattr(guard, "_present", lambda name: True)
    monkeypatch.setattr(guard, "_validate", lambda names, **kwargs: None)

    def import_plugin(name, consider_entry_points=False):
        raise ImportError("Error importing plugin 'broken'")

    manager = SimpleNamespace(
        is_blocked=lambda name: False,
        get_plugin=lambda name: None,
        import_plugin=import_plugin,
    )

    with pytest.raises(ImportError):
        guard.pytest_plugin_registered(sys.modules[guard.__name__], manager)
    kinds = sorted(path.name.rsplit(".", 1)[1] for path in markers.iterdir())
    assert kinds == ["registered"]


async def test_declared_plugins_stay_required_flags(tmp_path):
    """A plugin the task declares is still a ``-p`` flag every pytest loads."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("pytest --ctrf /logs/verifier/ctrf.json\n")
    env = _make_env()
    task = _make_task()
    task.task_dir = tmp_path
    task.config.verifier.pytest_plugins = ["declared_plugin"]

    await harden_before_verify(env, task, sandbox_user=None, workspace="/app")

    addopts = task.config.verifier.env["PYTEST_ADDOPTS"]
    assert "-p declared_plugin" in addopts
    assert "-p ctrf" not in addopts
    assert any(
        "_BENCHFLOW_OPTIONAL = ['ctrf']" in source
        and "_BENCHFLOW_REQUESTED = ['declared_plugin']" in source
        for source in _guard_sources(env)
    )


# #1170: a plugin test.sh installs into a workspace venv (in
# test_verifier_uv_state.py: test_plugin_the_verifier_installs_in_a_workspace_venv_is_scored,
# test_stale_plugin_module_under_a_fresh_registration_is_refused) and here the
# guard's own rule, which reads hardening's pre-arm record.


def recorded(monkeypatch, *trees):
    """Model hardening's record of the agent's trees: every entry of *trees* now.

    Each entry with its birth time where the filesystem tells one, as the
    record's writer takes it (V2-R1-A-2).
    """
    seen = {}
    for tree in map(Path, trees):
        for path in [tree, *tree.rglob("*")]:
            st = os.lstat(path)
            birth = guard._birth(str(path), st, False)
            seen.setdefault(guard._identity(st), set()).add(birth)
    monkeypatch.setattr(guard, "_RECORD", seen)


def new_venv(workspace: Path) -> Path:
    """What ``uv init && uv add`` makes after hardening: a venv and its project."""
    (workspace / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    (workspace / "uv.lock").write_text("version = 1\n")
    site = workspace / ".venv/lib/python3.13/site-packages"
    site.mkdir(parents=True)
    (workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
    return site


def test_fresh_file_counts_only_in_a_site_directory(tmp_path, monkeypatch):
    """A file test.sh installs into the agent's trees is never loaded.

    #1170's relief trusted a fresh file below a site directory of an
    environment new since the record. Plugins test.sh needs now come from
    BenchFlow's protected runtime, so such a file is refused like any other
    in those trees, root's or not, record or not.
    """
    workspace = tmp_path / "app"
    workspace.mkdir()
    recorded(monkeypatch, workspace)
    site = new_venv(workspace)
    (site / "plugin.py").write_text("# the verifier's install\n")
    monkeypatch.setattr(guard, "ownership_problem", lambda path, st: None)
    monkeypatch.setattr(guard, "_BENCHFLOW_ARMED", True)
    blocked = (str(workspace),)

    reason = guard.untrusted_reason(str(site / "plugin.py"), blocked)
    assert reason is not None and f"is under {workspace}" in reason


def test_fresh_link_to_an_old_file_is_not_vouched(tmp_path, monkeypatch):
    """A fresh link in a site directory to a file the agent left is still the agent's."""
    workspace = tmp_path / "app"
    workspace.mkdir()
    old = workspace / "agent.py"
    old.write_text("# agent\n")
    recorded(monkeypatch, workspace)
    site = new_venv(workspace)
    (site / "plugin.py").symlink_to(old)
    monkeypatch.setattr(guard, "ownership_problem", lambda path, st: None)
    monkeypatch.setattr(guard, "_BENCHFLOW_ARMED", True)

    assert not guard._installed_by_verifier(str(site / "plugin.py"))
    assert not guard.trusted(str(site / "plugin.py"), (str(workspace),))


@pytest.mark.parametrize("touched", ["chmod", "chown", "touch"])
def test_metadata_change_never_makes_a_planted_file_the_verifiers(
    tmp_path, monkeypatch, touched
):
    """The change time of a planted file is not the verifier's say.

    The agent leaves a ctrf registration and module in a site directory of
    its workspace; after the guard is written, test.sh runs ``chmod -R`` (or
    a chown or touch) over the workspace, which gives every file a new
    change time. The files are still the ones the record holds: refused.
    """
    workspace = tmp_path / "app"
    site = workspace / "plugins/site-packages"
    install_plugin(site, CTRF_PLUGIN)
    recorded(monkeypatch, workspace)
    monkeypatch.setattr(guard, "ownership_problem", lambda path, st: None)
    monkeypatch.setattr(guard, "_BENCHFLOW_ARMED", True)
    for path in [workspace, *workspace.rglob("*")]:
        if touched == "chmod":
            path.chmod(path.stat().st_mode)
        elif touched == "chown":
            os.chown(path, os.getuid(), os.getgid())
        else:
            os.utime(path)

    for path in site.rglob("*"):
        assert not guard._installed_by_verifier(str(path)), path
        assert "which the agent could write" in guard.untrusted_reason(
            str(path), (str(workspace),)
        )


def test_planted_file_moved_into_a_new_venv_is_not_the_verifiers(tmp_path, monkeypatch):
    """A new path does not make the agent's file new.

    test.sh makes a fresh venv and moves a module the agent left into its
    site-packages; the module keeps its identity, which the record holds.
    """
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "ctrf_model.py").write_text(CTRF_PLUGIN)
    recorded(monkeypatch, workspace)
    site = new_venv(workspace)
    (workspace / "ctrf_model.py").rename(site / "ctrf_model.py")

    assert not guard._installed_by_verifier(str(site / "ctrf_model.py"))


def test_new_file_in_an_environment_the_agent_left_is_not_the_verifiers(
    tmp_path, monkeypatch
):
    """test.sh installing into the agent's venv is not vouched for.

    Everything else in a venv the agent left (its pytest, its other
    libraries) is the agent's code, so a fresh file there proves nothing.
    """
    workspace = tmp_path / "app"
    workspace.mkdir()
    site = new_venv(workspace)
    recorded(monkeypatch, workspace)
    (site / "plugin.py").write_text("# the verifier's install\n")

    assert not guard._installed_by_verifier(str(site / "plugin.py"))


@pytest.mark.parametrize(
    "project_file", ["pyproject.toml", "uv.lock", "requirements-test.txt"]
)
def test_install_an_agent_project_file_chose_is_not_the_verifiers(
    tmp_path, monkeypatch, project_file
):
    """What chose the install must be the verifier's too.

    The agent leaves a project file in the workspace (``[tool.uv.sources]``
    naming its own wheel, a lock pinning it); test.sh's fresh venv beside it
    holds the agent's choice. One that still holds the image's bytes is the
    image's own.
    """
    copy = tmp_path / "copy"
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / project_file).write_text("# the image's\n")
    shutil.copytree(workspace, copy)
    monkeypatch.setattr(guard, "_BENCHFLOW_WORKSPACE", str(workspace))
    monkeypatch.setattr(guard, "_BENCHFLOW_SNAPSHOT", str(copy))
    monkeypatch.setattr(guard, "_private_to_root", lambda directory: True)
    (workspace / project_file).write_text("# the agent's\n")
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(workspace),))
    recorded(monkeypatch, workspace)
    site = workspace / ".venv/lib/python3.13/site-packages"
    site.mkdir(parents=True)
    (workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
    (site / "plugin.py").write_text("# installed from the agent's choice\n")

    assert not guard._installed_by_verifier(str(site / "plugin.py"))
    # The image's bytes: the image's own choice.
    (workspace / project_file).write_text("# the image's\n")
    assert guard._installed_by_verifier(str(site / "plugin.py"))


def test_record_counts_only_when_root_alone_wrote_all_of_it(tmp_path, monkeypatch):
    """A record anyone else could write, or a cut-off one, is none."""
    record = tmp_path / "guard" / ".record"
    record.parent.mkdir()
    record.write_text("1:2\nend\n")
    monkeypatch.setattr(guard, "_BENCHFLOW_RECORD", str(record))
    monkeypatch.setattr(guard, "ownership_problem", lambda path, st: None)

    monkeypatch.setattr(guard, "_RECORD", None)
    assert guard._record() == {"1:2": {None}}
    record.write_text("1:2\n")
    monkeypatch.setattr(guard, "_RECORD", None)
    assert guard._record() is None
    record.write_text("1:2\nend\n")
    monkeypatch.setattr(
        guard, "ownership_problem", lambda path, st: path + " is owned by uid 1000"
    )
    monkeypatch.setattr(guard, "_RECORD", None)
    assert guard._record() is None
    monkeypatch.setattr(guard, "_BENCHFLOW_RECORD", "")
    monkeypatch.setattr(guard, "_RECORD", None)
    assert guard._record() is None


def workspace_mode_ownership(workspace: Path) -> str:
    """The guard's owner and mode rule with real mode bits inside *workspace*.

    Nothing here runs as root, so owners are modelled as root's (the freeze's
    chown); a path in the workspace others can write is refused, and
    everything outside it is the image's own.
    """
    return (
        "def ownership_problem(path, st):\n"
        f"    if not under(path, {str(workspace)!r}):\n"
        "        return None\n"
        "    if st.st_mode & 0o022:\n"
        "        return path + ' is group- or world-writable'\n"
        "    return None\n"
    )


def world_writable_workspace(layout):
    """The agent ran ``chmod -R 777 /app`` (or the image ships it so)."""
    layout.workspace.chmod(0o777)


def verifier_runs_as(monkeypatch, uid):
    """Model ``id -u`` for test.sh, which hardening asks for the verifier's uid."""

    async def verifier_uid(env, user):
        return uid

    monkeypatch.setattr(lockdown, "_verifier_uid", verifier_uid)


def workspace_venv_under_umask_022(env, layout):
    """``uv add`` under root's usual mask: only the workspace itself is 0777."""
    previous = os.umask(0o022)
    try:
        return workspace_venv(env, layout)
    finally:
        os.umask(previous)


@pytest.mark.parametrize(
    ("solution", "reward"), [(PASSING, 1.0), (FAILING, 0.0)], ids=["oracle", "empty"]
)
async def test_root_verifier_install_in_a_world_writable_workspace_is_scored(
    tmp_path, monkeypatch, solution, reward
):
    """Guards V2-R1-A-2: #1170's relief held only while the agent left /app alone.

    The freeze chowns the workspace to root but keeps its modes, so a 0777
    /app stays 0777. USACO's test.sh, run as root, adds pytest-json-ctrf to a
    ``uv init`` project there; no agent process is left to use /app's write
    bits, so the guard loads ctrf and the solution decides the reward
    (before: refused at /app, an unscored verifier error).
    """
    verifier_runs_as(monkeypatch, "0")
    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        solution,
        workspace_venv_under_umask_022,
        before=world_writable_workspace,
        workspace_name="app",
        guard_model=workspace_mode_ownership(tmp_path / "app"),
    )

    assert stat.S_IMODE(layout.workspace.stat().st_mode) == 0o777
    assert "Verifier plugin trust rejected" not in stdout_of(layout), stdout_of(layout)
    assert error is None
    assert rewards == {"reward": reward}
    assert markers(layout, "installed") == []


async def test_world_writable_workspace_still_counts_for_a_verifier_not_root(
    tmp_path, monkeypatch, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
    # Any uid but root's; the guard's marker folder is handed to it.
    verifier_runs_as(monkeypatch, str(os.getuid() or 1000))
    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        workspace_venv_under_umask_022,
        before=world_writable_workspace,
        workspace_name="app",
        guard_model=workspace_mode_ownership(tmp_path / "app"),
        published=False,
    )

    assert "Verifier plugin trust rejected: ctrf" in stdout_of(layout)
    assert rewards == {"reward": 0.0}
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None


async def test_agent_module_in_a_world_writable_workspace_is_still_refused(
    tmp_path, monkeypatch, caplog
):
    """A root test.sh ignores write bits only for files it installed itself.

    The agent leaves its ctrf module in a 0777 workspace's venv; test.sh
    writes only the registration. The module predates the guard: refused,
    never imported, and scored.
    """
    verifier_runs_as(monkeypatch, "0")
    proof = tmp_path / "hostile-imported"
    site_path = Path(".venv/lib/python3.12/site-packages")

    def plant(layout):
        world_writable_workspace(layout)
        site = layout.workspace / site_path
        site.mkdir(parents=True)
        (site / "ctrf_model.py").write_text(hostile_ctrf(proof))

    def registration_only(env, layout):
        site = layout.workspace / site_path
        info = site / "ctrf_model-0.3.5.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: ctrf_model\n")
        (info / "entry_points.txt").write_text("[pytest11]\nctrf = ctrf_model\n")
        return [site]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        registration_only,
        before=plant,
        workspace_name="app",
        guard_model=workspace_mode_ownership(tmp_path / "app"),
        published=False,
    )

    assert "Verifier plugin trust rejected: ctrf" in stdout_of(layout)
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert not proof.exists()


def test_owners_still_count_where_write_bits_do_not(tmp_path, monkeypatch):
    """Outside the agent's trees a file is the image's only while it and every
    folder above are root's alone and it is unchanged since the arm."""
    image = tmp_path / "image-site"
    image.mkdir()
    plugin = image / "plugin.py"
    plugin.write_text("# the image's\n")
    key = tmp_path / "key"
    key.write_text("x")
    monkeypatch.setattr(guard, "_BENCHFLOW_ARMED", True)
    monkeypatch.setattr(guard, "_BENCHFLOW_KEY_FILE", str(key))
    blocked = (str(tmp_path / "app"),)

    def modelled(problem_at=None):
        def ownership_problem(path, st):
            if path == problem_at:
                return path + " is owned by uid 1000"
            return None

        return ownership_problem

    monkeypatch.setattr(guard, "ownership_problem", modelled())
    assert guard.untrusted_reason(str(plugin), blocked) is None
    monkeypatch.setattr(guard, "ownership_problem", modelled(str(image)))
    assert "owned by uid 1000" in guard.untrusted_reason(str(plugin), blocked)
    # Installed after the arm (V2-REDESIGN-R3-2): refused however root's.
    monkeypatch.setattr(guard, "ownership_problem", modelled())
    os.utime(key, (0, 0))
    key.chmod(0o600)  # the key's change time moves to now: the file predates it
    plugin.write_text("# test.sh's install, after the arm\n")
    assert "after the agent stopped" in guard.untrusted_reason(str(plugin), blocked)


def _debian_model(monkeypatch, staff, group):
    """Every path root's; the *staff* paths root:<group> mode 2775, the rest 0755."""
    original = os.stat
    staff = {str(p) for p in staff}

    def model(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        fields = list(result)
        fields[4] = 0
        if str(path) in staff:
            fields[5] = group
            fields[0] = (fields[0] & ~0o7777) | 0o2775
        else:
            fields[5] = 0
            fields[0] = (fields[0] & ~0o7777) | (
                0o755 if stat.S_ISDIR(fields[0]) else 0o644
            )
        return os.stat_result(fields, {"st_ctime_ns": result.st_ctime_ns})

    monkeypatch.setattr(os, "stat", model)


def _dist_packages(tmp_path):
    lib = tmp_path / "usr/local/lib/python3.8"
    site = lib / "dist-packages"
    (site / "pytestqt").mkdir(parents=True)
    (site / "pytestqt/__init__.py").write_text("")
    (site / "pytestqt/plugin.py").write_text("# the image's pytest-qt\n")
    return lib, site


def test_image_plugin_under_a_staff_writable_directory_is_trusted(
    tmp_path, monkeypatch
):
    """Guards #1176: qutebrowser's oracle ran none of its 182 tests.

    Ubuntu ships /usr/local/lib/python3.X and its dist-packages as root:staff
    mode 2775. The agent's user is not in staff, so that group's write bit
    does not make the image's pytest-qt the agent's.
    """
    lib, site = _dist_packages(tmp_path)
    _debian_model(monkeypatch, [lib, site], group=50)
    monkeypatch.setattr(guard, "_BENCHFLOW_AGENT_GIDS", (1000,), raising=False)
    monkeypatch.setattr(sys, "path", [str(site), *sys.path])

    assert guard.untrusted_reason(str(site / "pytestqt/plugin.py"), ()) is None
    assert guard.trusted_module("pytestqt.plugin", ())


@pytest.mark.parametrize("agent_gids", [(50, 1000), None], ids=["agents", "unknown"])
def test_group_writable_by_the_agents_group_or_unknown_stays_untrusted(
    tmp_path, monkeypatch, agent_gids
):
    """The group's write bit still counts when the agent is in it, or its groups are unknown."""
    lib, site = _dist_packages(tmp_path)
    _debian_model(monkeypatch, [lib, site], group=50)
    monkeypatch.setattr(guard, "_BENCHFLOW_AGENT_GIDS", agent_gids, raising=False)

    assert guard.untrusted_reason(str(site / "pytestqt/plugin.py"), ()) == (
        f"{site} is group- or world-writable (mode 2775)"
    )


async def test_hardening_gives_the_guard_and_discovery_the_agents_groups():
    """Guards #1176's wiring: the agent user's groups reach both trust checks."""

    def side_effect(cmd, **kwargs):
        if cmd == "id -G agent":
            return SimpleNamespace(stdout="1000 27\n", stderr="", exit_code=0)
        return SimpleNamespace(stdout="", stderr="", exit_code=0)

    env = _make_env(side_effect=side_effect)
    task = _make_task()

    await harden_before_verify(env, task, sandbox_user="agent", workspace="/app")

    assert any(
        "_BENCHFLOW_AGENT_GIDS = (27, 1000)" in source for source in _guard_sources(env)
    )
    # Discovery runs on root's protected Python (PT); its policy is data.
    policies = [
        json.loads(word)
        for call in env.exec.call_args_list
        if "from importlib.metadata import entry_points" in str(call.args[0])
        for word in shlex.split(str(call.args[0]))
        if word.startswith("{") and "agent_gids" in word
    ]
    assert policies and all(p["agent_gids"] == [27, 1000] for p in policies)


# #1180: a verifier that replaces PYTHONPATH and runs pytest in a fresh venv.


def _relocated_python(root: Path) -> Path:
    """A copy of this Python whose standard library directory the test owns.

    The executable is copied (Python finds its prefix from where it is) and
    every standard library entry is linked from the real one, so a guard
    written there stays in the test's folder (``__pycache__`` too, so no
    bytecode reaches the real one).
    """
    real = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    version = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    stdlib = Path(
        sysconfig.get_path("stdlib", vars={"installed_base": sys.base_prefix})
    )
    bindir = root / "bin"
    lib = root / "lib" / version
    bindir.mkdir(parents=True)
    lib.mkdir(parents=True)
    shutil.copy2(real, bindir / "python3")
    for entry in stdlib.iterdir():
        if entry.name not in ("site-packages", "__pycache__"):
            (lib / entry.name).symlink_to(entry)
    (lib / "site-packages").mkdir()
    (lib / "__pycache__").mkdir()
    for shared in (Path(sys.base_prefix) / "lib").glob("libpython*"):
        (root / "lib" / shared.name).symlink_to(shared)
    python = bindir / "python3"
    probe = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import sys, sysconfig; print(sysconfig.get_path('stdlib'))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if probe.returncode != 0 or Path(probe.stdout.strip()) != lib:
        pytest.skip(f"this Python cannot be relocated: {probe.stdout or probe.stderr}")
    return python


@pytest.mark.asyncio
async def test_guard_loads_in_a_new_venv_whose_test_sh_replaces_pythonpath(
    tmp_path, monkeypatch
):
    """Guards #1180: DevEval's oracle scored 0.

    DevEval's test.sh runs ``export PYTHONPATH=$(pwd)``, then ``uv init``,
    ``uv add pytest`` and ``uv run pytest``: a venv made after hardening from
    the image's Python, without its site-packages, and a PYTHONPATH without
    the guard. pytest stopped on ``Error importing plugin "<guard>"``. The
    guard now goes into that Python's standard library directory, which every
    venv made from it keeps on its path; a plugin planted on the replaced
    PYTHONPATH is still refused.
    """
    base = _relocated_python(tmp_path / "image-python")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_pass.py").write_text("def test_pass():\n    assert True\n")
    (workspace / "test_fail.py").write_text("def test_fail():\n    assert False\n")
    (workspace / "agent_plugin.py").write_text(HOSTILE_PLUGIN)
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))

    _, flags, copied, _ = await _install_pytest_plugin_guard(
        ShellEnv(), "agent", str(workspace), "", verifier_path=str(base.parent)
    )
    assert copied == (str(base),)

    # After hardening: test.sh's own venv, with this environment's pytest.
    venv = tmp_path / "project-venv"
    subprocess.run(
        [str(base), "-m", "venv", "--without-pip", str(venv)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    purelib = subprocess.run(
        [
            str(venv / "bin/python"),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    (Path(purelib) / "outer_pytest.pth").write_text(
        str(Path(pytest.__file__).resolve().parent.parent) + "\n"
    )

    def run(*args):
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("PYTEST_", "PYTHON"))
        }
        environment.update(
            PYTEST_ADDOPTS=f"-p no:cacheprovider {flags}",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(workspace),
        )
        return subprocess.run(
            [str(venv / "bin/python"), "-m", "pytest", "-q", *args],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )

    passed = run("test_pass.py")
    assert passed.returncode == 0, passed.stdout + passed.stderr
    hostile = run("-p", "agent_plugin", "test_fail.py")
    assert hostile.returncode != 0
    assert (
        "Verifier plugin trust rejected: agent_plugin"
        in hostile.stdout + hostile.stderr
    )


# Plugins the image's conftest.py declares, trusted while unchanged.

IMAGE_FIXTURES = (
    "import pytest\n\n\n@pytest.fixture\ndef image_fixture():\n    return 1\n"
)
EDITED_FIXTURES = IMAGE_FIXTURES + (
    "\n\ndef pytest_collection_modifyitems(items):\n"
    "    import pathlib\n"
    "    pathlib.Path(__PROOF__).touch()\n"
    "    for item in items:\n"
    "        item.obj = lambda *a, **k: None\n"
)
USES_IMAGE_FIXTURE = (
    "def test_solution(image_fixture):\n    assert image_fixture == 1\n"
)


@pytest.mark.parametrize("edited", [False, True], ids=["image-bytes", "agent-edited"])
async def test_plugin_the_image_conftest_declares_is_trusted_only_while_unchanged(
    tmp_path, monkeypatch, edited, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
    copy = tmp_path / "testbed_verify"
    proof = tmp_path / "edited-imported"
    monkeypatch.setattr(lockdown, "_VERIFIER_WORKSPACE_COPY", str(copy), raising=False)

    def image(layout):
        package = layout.workspace / "imgpkg" / "testing"
        package.mkdir(parents=True)
        (layout.workspace / "imgpkg/__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "fixtures.py").write_text(IMAGE_FIXTURES)
        shutil.copytree(layout.workspace, copy, symlinks=True)
        if edited:
            (package / "fixtures.py").write_text(
                EDITED_FIXTURES.replace("__PROOF__", repr(str(proof)))
            )

    def image_conftest(env, layout):
        (layout.tmp / "verifier-tests" / "conftest.py").write_text(
            "pytest_plugins = 'imgpkg.testing.fixtures'\n"
        )
        return [*uvx(env, layout), layout.workspace]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING if edited else USES_IMAGE_FIXTURE,
        image_conftest,
        before=image,
        workspace_name="testbed",
        # Nothing here is root: the copy stands for one only root can reach.
        guard_model=GUARD_OWNERSHIP
        + "def _private_to_root(directory):\n    return not os.path.islink(directory)\n",
    )

    if edited:
        assert "Verifier plugin trust rejected: imgpkg.testing.fixtures" in (
            stdout_of(layout)
        )
        assert refused_scored_zero(rewards, error), (rewards, error)
        assert "pytest plugin guard not confirmed" in caplog.text
        assert not proof.exists()
    else:
        assert "Verifier plugin trust rejected" not in stdout_of(layout), stdout_of(
            layout
        )
        assert error is None
        assert rewards == {"reward": 1.0}


def test_image_bytes_need_a_copy_that_is_not_a_link(tmp_path, monkeypatch):
    """A copy reached through a link (back into the live workspace) proves nothing."""
    workspace = tmp_path / "testbed"
    workspace.mkdir()
    (workspace / "plugin.py").write_text("# agent\n")
    link = tmp_path / "testbed_verify"
    link.symlink_to(workspace)
    monkeypatch.setattr(guard, "_BENCHFLOW_ARMED", True)
    monkeypatch.setattr(guard, "_BENCHFLOW_WORKSPACE", str(workspace))
    monkeypatch.setattr(guard, "_BENCHFLOW_SNAPSHOT", str(link))
    monkeypatch.setattr(guard, "_private_to_root", lambda directory: True)

    assert not guard._image_bytes(str(workspace / "plugin.py"))
    link.unlink()
    shutil.copytree(workspace, link)
    assert guard._image_bytes(str(workspace / "plugin.py"))
    (link / "plugin.py").write_text("# the image\n")
    assert not guard._image_bytes(str(workspace / "plugin.py"))


@pytest.mark.parametrize(
    ("uid", "mode", "private"),
    [(0, 0o700, True), (0, 0o750, False), (0, 0o755, False), (1000, 0o700, False)],
)
def test_only_a_copy_root_alone_can_reach_stands_for_the_image(
    tmp_path, monkeypatch, uid, mode, private
):
    """The pre-agent copy proves the image's bytes only while no other user can reach into it."""
    copy = tmp_path / "testbed_verify"
    copy.mkdir()
    original = os.lstat

    def model(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if str(path) != str(copy):
            return result
        fields = list(result)
        fields[4] = uid
        fields[0] = (fields[0] & ~0o7777) | mode
        return os.stat_result(fields)

    monkeypatch.setattr(os, "lstat", model)
    monkeypatch.setattr(guard, "_root_alone", lambda path: True)

    assert guard._private_to_root(str(copy)) is private


def test_image_bytes_count_in_a_world_writable_workspace(tmp_path, monkeypatch):
    """SWE-bench images ship /testbed and its folders 0777: the bytes decide there.

    Owners and modes in the workspace say nothing about who wrote a file the
    agent could reach anyway; the comparison with the root-only copy does.
    """
    workspace = tmp_path / "testbed"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "pkg/__init__.py").write_text("")
    (workspace / "pkg/fixtures.py").write_text("# the image's\n")
    copy = tmp_path / "testbed_verify"
    shutil.copytree(workspace, copy)
    monkeypatch.setattr(guard, "_BENCHFLOW_ARMED", True)
    monkeypatch.setattr(guard, "_BENCHFLOW_WORKSPACE", str(workspace))
    monkeypatch.setattr(guard, "_BENCHFLOW_SNAPSHOT", str(copy))
    monkeypatch.setattr(guard, "_private_to_root", lambda directory: True)
    monkeypatch.setattr(
        guard, "ownership_problem", lambda path, st: path + " is world-writable"
    )
    monkeypatch.setattr(sys, "path", [str(workspace), *sys.path])
    blocked = (str(workspace),)

    assert guard.trusted_module("pkg.fixtures", blocked)
    (workspace / "pkg/fixtures.py").write_text("# the agent's\n")
    assert not guard.trusted_module("pkg.fixtures", blocked)
    assert guard.untrusted_reason(
        str(workspace / "pkg/fixtures.py"), blocked
    ).startswith(
        f"{workspace}/pkg/fixtures.py is under {workspace}, which the agent could write"
    )
