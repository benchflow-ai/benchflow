"""The plugin guard's fixes from review round 2 of V2 (V2-R2-A-1, A-2, B-1, A-3).

A-1: with no complete record a refusal of test.sh's own install (USACO's
``uv add`` into /app/.venv) was a counted 0 blamed on the agent, and the
record failed without the agent's help when a service's file in /tmp
vanished during the walk. A-2: ``cp -r tests /app/`` named ``tests`` as a
requirements file, so a folder a correct solution made vetoed test.sh's
install. B-1: ``"$APP"/requirements.txt`` was read as ``/requirements.txt``,
so the agent-edited file the install read did not count. A-3 (low): a pip
``-c`` on a continued line was not named.

The end-to-end scenarios use test_verifier_uv_state.py's harness: the real
hardening, then a real pytest with the armed guard, scored through
``_verify_rollout``.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow.failures import cause_of
from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox import lockdown
from tests import test_verifier_uv_state as uv_state
from tests.test_verifier_plugin_guard_review1 import named, summary_of
from tests.test_verifier_uv_state import (
    CTRF_PLUGIN,
    FAILING,
    PASSING,
    hostile_ctrf,
    install_plugin,
    markers,
    refused_scored_zero,
    score,
    workspace_venv,
)

# A-1: entries that vanish while the record is taken.


def run_record(tmp_path: Path, tops: "list[Path]") -> "list[str]":
    """Run the record script in this process, as hardening runs it in the sandbox."""
    record = tmp_path / "record"
    argv = sys.argv
    sys.argv = ["-c", str(record), json.dumps([str(top) for top in tops])]
    try:
        exec(compile(lockdown._RECORD_AGENT_TREES_SCRIPT, "<record>", "exec"), {})
    finally:
        sys.argv = argv
    return record.read_text().splitlines()


def inode(path: Path) -> str:
    return str(os.lstat(path).st_ino)


def tmp_tree(tmp_path: Path) -> Path:
    top = tmp_path / "tmp"
    for folder in ("a", "doomed", "removed/inner", "z"):
        (top / folder).mkdir(parents=True)
    (top / "a/keep.txt").write_text("x\n")
    (top / "z/keep.txt").write_text("x\n")
    (top / "removed/inner/f.txt").write_text("x\n")
    (top / "removed/g.txt").write_text("x\n")
    return top


def test_entries_that_vanish_during_the_record_are_skipped(tmp_path, monkeypatch):
    """Guards V2-R2-A-1: a service deleting a temp file voided the record.

    Three races, made certain here: a name listed but gone before it is
    recorded, a folder gone between its record and its opening, and a
    folder removed while the walk is inside it. Each left the record without
    ``end``, so test.sh's own install in /app/.venv became a refusal.
    """
    top = tmp_tree(tmp_path)
    kept = [top / "a", top / "a/keep.txt", top / "z", top / "z/keep.txt"]
    removed = top / "removed"
    real_listdir, real_stat = os.listdir, os.stat

    def listdir(path):
        names = real_listdir(path)
        if isinstance(path, int) and os.fstat(path).st_ino == os.lstat(top).st_ino:
            names.append("ghost")
        if (
            isinstance(path, int)
            and removed.exists()
            and os.fstat(path).st_ino == os.lstat(removed).st_ino
        ):
            shutil.rmtree(removed)
        return names

    def stat_then_remove(name, *args, **kwargs):
        st = real_stat(name, *args, **kwargs)
        if name == "doomed" and kwargs.get("dir_fd") is not None:
            (top / "doomed").rmdir()
        return st

    monkeypatch.setattr(os, "listdir", listdir)
    monkeypatch.setattr(os, "stat", stat_then_remove)

    lines = run_record(tmp_path, [top])

    assert lines[-1] == "end"
    numbers = {line.split(":")[1] for line in lines[:-1]}
    assert {inode(path) for path in kept} <= numbers


def test_a_dangling_link_as_a_root_is_nothing_to_record(tmp_path):
    """A root that links to nothing holds no files; the record still ends."""
    (tmp_path / "work").mkdir()
    (tmp_path / "work/f.txt").write_text("x\n")
    (tmp_path / "app").symlink_to(tmp_path / "gone")

    lines = run_record(tmp_path, [tmp_path / "app", tmp_path / "work"])

    assert lines[-1] == "end"
    assert inode(tmp_path / "work/f.txt") in {line.split(":")[1] for line in lines[:-1]}


@pytest.mark.skipif(shutil.which("find") is None, reason="needs find and stat")
def test_the_shell_record_walks_a_linked_root_without_find_h(tmp_path):
    """The record without python3 needed ``find -H``, which old busybox lacks.

    It now walks ``<root>/.``, recording the linked folder itself too, as
    the python3 walk does.
    """
    real = tmp_path / "srv"
    real.mkdir()
    (real / "f.txt").write_text("x\n")
    (tmp_path / "app").symlink_to(real, target_is_directory=True)
    record = tmp_path / "record"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("find", "stat"):
        (bin_dir / tool).symlink_to(shutil.which(tool))
    command = lockdown._record_agent_trees_cmd(str(record), (str(tmp_path / "app"),))

    subprocess.run(
        ["/bin/sh", "-c", command],
        check=True,
        timeout=60,
        env={"PATH": str(bin_dir)},
    )

    lines = record.read_text().splitlines()
    assert lines[-1] == "end"
    numbers = {line.split(":")[1] for line in lines[:-1]}
    assert {inode(tmp_path / "app"), inode(real), inode(real / "f.txt")} <= numbers


def test_a_refusal_with_no_record_names_its_files_and_why(tmp_path, monkeypatch):
    """Guards V2-R2-A-1: no record left nothing to tell the agent's file from test.sh's."""
    site = tmp_path / "app/.venv/lib/python3.12/site-packages"
    install_plugin(site, CTRF_PLUGIN)
    (tmp_path / "app/.venv/pyvenv.cfg").write_text("home = /usr/bin\n")
    monkeypatch.setattr(guard.sys, "path", [str(site), *guard.sys.path])
    monkeypatch.setattr(guard, "_RECORD", None)
    blocked = (str(tmp_path / "app"),)

    monkeypatch.setattr(guard, "_BENCHFLOW_RECORD", "")
    why, files = guard._unrecorded(["ctrf"], True, blocked, ())
    assert why == "none was taken"
    assert str(site / "ctrf_model.py") in files

    monkeypatch.setattr(guard, "_RECORD", None)
    monkeypatch.setattr(guard, "_BENCHFLOW_RECORD", str(tmp_path / "missing"))
    assert guard._unrecorded(["ctrf"], True, blocked, ())[0] == (
        "the record did not finish"
    )
    # Files outside the agent's trees are refused for other reasons, which
    # no record would change.
    assert guard._unrecorded(["ctrf"], True, (str(tmp_path / "elsewhere"),), ()) is None
    # With a record, the record decides (_installed_during_verification).
    monkeypatch.setattr(
        guard, "_RECORD", {guard._identity(os.lstat(tmp_path / "app")): {None}}
    )
    assert guard._unrecorded(["ctrf"], True, blocked, ()) is None


def no_record(monkeypatch):
    """Hardening's record fails, as it does without python3 on a vanished file."""
    monkeypatch.setattr(lockdown, "_record_agent_trees_cmd", lambda *args: "true")


@pytest.mark.parametrize("solution", [PASSING, FAILING], ids=["oracle", "empty"])
async def test_without_a_record_test_shs_own_install_is_not_the_agents_fault(
    tmp_path, monkeypatch, solution, caplog
):
    """Guards V2-R2-A-1 end to end: the oracle's 0 was blamed on the agent.

    USACO's test.sh adds ctrf to a ``uv init`` project in /app; the record is
    missing, so the guard cannot trust the install and refuses it. The
    refusal still counts as a failure, but its cause is the task's, and its
    message says why no one can tell who put the plugin there.
    """
    no_record(monkeypatch)

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        solution,
        workspace_venv,
        workspace_name="app",
        published=False,
    )

    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None
    assert markers(layout, "installed") == []
    (marker,) = markers(layout, "unrecorded")
    assert marker.read_text().startswith("the record did not finish\n")
    cause = cause_of({"rewards": rewards, "verifier_error": error})
    assert cause is None
    # Counted as a failure, as every refusal, so a planted plugin gains nothing.
    summary = summary_of(error, rewards=rewards)
    assert summary["guard_refusals"] == 0
    assert summary["score_excl_errors_ratio"] == 0.5


async def test_without_a_record_a_planted_plugin_is_still_refused(
    tmp_path, monkeypatch, caplog
):
    """The agent's plugin in /app/.venv, no record: refused, never loaded, a 0."""
    no_record(monkeypatch)
    proof = tmp_path / "hostile-imported"

    def plant(layout):
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        install_plugin(site, hostile_ctrf(proof))
        (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")

    def reuse(env, layout):
        return [layout.workspace / ".venv/lib/python3.12/site-packages"]

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        reuse,
        before=plant,
        workspace_name="app",
        published=False,
    )

    assert not proof.exists()
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert summary_of(error, rewards=rewards)["score_excl_errors_ratio"] == 0.5


def test_agent_text_in_a_refusal_cannot_make_it_unrecorded():
    """Only the guard's own opening words choose the cause."""
    error = (
        "verifier crashed: PluginGuardLoadError: the pytest plugin guard "
        "_benchflow_guard_0123abcd refused a pytest plugin it could not trust, so "
        "pytest stopped and this run is not scored (Verifier plugin trust "
        "rejected: x (/app/x, with no complete record of the agent's files ))"
    )

    cause = cause_of({"rewards": None, "verifier_error": error})

    assert cause is not None and cause.fault == "agent"


# A-2, B-1 and A-3: the dependency files the verifier's scripts name.


def test_r_of_a_command_that_is_not_pip_names_no_dependency_file(tmp_path):
    """Guards V2-R2-A-2: ``cp -r tests /app/`` made ``tests`` a requirements file."""
    assert (
        named(
            tmp_path,
            "cp -r tests /app/\n"
            "grep -r TODO src\n"
            'cp -r "$TESTS_DIR" /app/tests\n'
            "cd /app && uv init && uv add pytest-json-ctrf\n",
        )
        == []
    )
    assert named(
        tmp_path,
        "uv add -r requirements-dev.txt\n"
        "python3 -m pip install -r deps.txt\n"
        "cp -r /tests/requirements-ci.txt /app/\n",
    ) == ["/tests/requirements-ci.txt", "deps.txt", "requirements-dev.txt"]


def test_a_path_after_a_closing_quote_is_named_by_its_base_name(tmp_path):
    """Guards V2-R2-B-1: ``"$APP"/requirements.txt`` was read as /requirements.txt."""
    assert named(
        tmp_path,
        'APP=/app\nuv venv "$APP"/.venv\n'
        'uv pip sync --python "$APP"/.venv/bin/python "$APP"/requirements.txt\n',
    ) == ["requirements.txt"]
    assert named(tmp_path, 'python3 "$APP"/setup.py develop\n') == ["setup.py"]
    assert named(tmp_path, "pip install -e `pwd`/setup.py\n") == ["setup.py"]
    # A quote that opens the word still leaves the path known.
    assert named(
        tmp_path,
        'uv pip sync "/tests/requirements.txt"\n'
        "pip install --requirement='/tests/requirements-ci.txt'\n",
    ) == ["/tests/requirements-ci.txt", "/tests/requirements.txt"]


def test_a_constraints_file_on_a_continued_line_is_named(tmp_path):
    """Guards V2-R2-A-3: a pip ``-c`` after a ``\\`` line break was dropped."""
    assert named(
        tmp_path,
        "uv pip install --system \\\n    -c /app/constraints.txt \\\n"
        "    pytest-json-ctrf\n",
    ) == ["/app/constraints.txt"]


async def test_copy_and_project_helpers_have_unknown_inputs(
    tmp_path, monkeypatch, caplog
):
    """Copy/project helpers cannot prove which workspace files an install reads.

    test.sh copies its tests with ``cp -r tests /app/``, then USACO's
    ``uv init && uv add pytest-json-ctrf`` in /app; a correct solution made
    /app/tests too.
    """
    test_sh = (
        "#!/bin/bash\ncp -r tests /app/\n"
        "cd /app && uv init && uv add pytest-json-ctrf\n"
        "uv run pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n"
    )
    make_task = uv_state.make_task
    monkeypatch.setattr(
        uv_state,
        "make_task",
        lambda layout, env=None: make_task(layout, test_sh=test_sh, env=env),
    )

    def solution_tests(layout):
        (layout.workspace / "tests").mkdir()
        (layout.workspace / "tests/test_solution.py").write_text(PASSING)

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        workspace_venv,
        before=solution_tests,
        workspace_name="app",
        published=False,
    )

    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None


async def test_an_agent_requirements_file_behind_a_quoted_variable_is_refused(
    tmp_path, monkeypatch, caplog
):
    """Guards V2-R2-B-1 end to end: the agent's requirements chose a trusted install.

    test.sh syncs ``"$APP"/requirements.txt`` into a fresh ``"$APP"/.venv``;
    the agent edited that file to install its own ctrf. The plugin must be
    refused and never imported.
    """
    proof = tmp_path / "hostile-imported"

    def test_sh(layout):
        return (
            f'#!/bin/bash\nAPP={layout.workspace}\nuv venv "$APP"/.venv\n'
            'uv pip sync --python "$APP"/.venv/bin/python "$APP"/requirements.txt\n'
            '"$APP"/.venv/bin/pytest --ctrf /logs/verifier/ctrf.json '
            "/tests/test_outputs.py\n"
        )

    make_task = uv_state.make_task
    monkeypatch.setattr(
        uv_state,
        "make_task",
        lambda layout, env=None: make_task(layout, test_sh=test_sh(layout), env=env),
    )

    def edit_requirements(layout):
        (layout.workspace / "requirements.txt").write_text("ctrf @ file:///evil\n")

    def uv_venv(env, layout):
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        install_plugin(site, hostile_ctrf(proof))
        (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
        return [site]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        uv_venv,
        before=edit_requirements,
        workspace_name="app",
        published=False,
    )

    assert not proof.exists()
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert markers(layout, "installed") == []
