"""The plugin guard's provenance fixes from the cross-model review of V2.

A file's change time was the guard's proof that test.sh installed
it, so a test.sh that ran ``chmod -R`` over the workspace made the agent's
planted plugin the verifier's own. The guard now reads hardening's record of
the agent's trees instead.

A refusal was scored as test.sh reported it (a 0); it is now a
verifier error (the scenarios in test_verifier_plugin_guard_issues.py and
test_verifier_uv_state.py), with its own cause here.

The scenarios use the harness of test_verifier_uv_state.py: the real
hardening (``_build_verifier_env``) against a directory tree standing in for
the image, then a real pytest with the armed guard, scored through
``_verify_rollout``.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow._utils.scoring import VERIFIER_FAILED, classify_verifier_error
from benchflow.evaluation import RetryConfig
from benchflow.failures import cause_of
from benchflow.rollout._verifier_recovery import (
    PRESERVED_SOLVER,
    needs_verifier_recovery,
)
from benchflow.sandbox import _pytest_plugin_guard as guard_module
from benchflow.sandbox import lockdown
from tests.test_verifier_uv_state import (
    FAILING,
    PASSING,
    hostile_ctrf,
    install_plugin,
    markers,
    refused_scored_zero,
    score,
    workspace_venv,
)


def chmod_r_755(tree: Path) -> None:
    """``chmod -R 755 <tree>``: every file and directory gets a new change time."""
    for path in [tree, *tree.rglob("*")]:
        path.chmod(0o755)


@pytest.mark.parametrize(
    "planted",
    ["plugins/site-packages", ".venv/lib/python3.12/site-packages"],
    ids=["plugins-folder", "agent-venv"],
)
async def test_chmod_after_hardening_does_not_make_a_planted_plugin_the_verifiers(
    tmp_path, monkeypatch, planted, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
    proof = tmp_path / "hostile-imported"

    def plant(layout):
        site = layout.workspace / planted
        install_plugin(site, hostile_ctrf(proof))
        if planted.startswith(".venv"):
            (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")

    def chmod_then_reach_it(env, layout):
        chmod_r_755(layout.workspace)
        return [layout.workspace / planted]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        chmod_then_reach_it,
        before=plant,
        workspace_name="app",
        published=False,
    )

    assert not proof.exists()
    assert "Verifier plugin trust rejected: ctrf" in (
        layout.paths.test_stdout_path.read_text()
    )
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert markers(layout, "installed") == []


async def test_fresh_install_chosen_by_an_agent_project_is_not_the_verifiers(
    tmp_path, monkeypatch, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
    proof = tmp_path / "agent-choice-imported"

    def agent_project(layout):
        (layout.workspace / "pyproject.toml").write_text(
            "[project]\nname = 'app'\n[tool.uv.sources]\n"
            "pytest-json-ctrf = { path = 'wheels/ctrf.whl' }\n"
        )

    def uv_add_through_the_agents_sources(env, layout):
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        install_plugin(site, hostile_ctrf(proof))
        (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
        return [site]

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        uv_add_through_the_agents_sources,
        before=agent_project,
        workspace_name="app",
        published=False,
    )

    assert not proof.exists()
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text


def test_record_command_lists_every_entry_without_following_links(tmp_path):
    """The record holds each entry of the agent's trees once, links as links."""
    tree = tmp_path / "app"
    (tree / "pkg").mkdir(parents=True)
    (tree / "pkg/mod.py").write_text("x = 1\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "far.py").write_text("y = 2\n")
    (tree / "link").symlink_to(outside, target_is_directory=True)
    record = tmp_path / "record"

    command = lockdown._record_agent_trees_cmd(
        str(record), (str(tree), str(tmp_path / "missing"))
    )
    subprocess.run(["/bin/sh", "-c", command], check=True)

    lines = record.read_text().splitlines()
    assert lines[-1] == "end"
    # Each line is what the guard reads for the entry: its number and, where
    # the filesystem tells one, its birth time (V2-R1-A-2).
    expected = set()
    for path in (tree, tree / "pkg", tree / "pkg/mod.py", tree / "link"):
        st = os.lstat(path)
        birth = guard_module._birth(str(path), st, False)
        expected.add(guard_module._identity(st) + (":" + birth if birth else ""))
    assert set(lines[:-1]) == expected
    assert oct(record.stat().st_mode & 0o777) == "0o600"


def test_refusal_is_its_own_cause():
    """`bench eval` names a refusal, not a generic verifier failure."""
    result = {
        "rewards": None,
        "verifier_error": (
            "verifier crashed: PluginGuardLoadError: the pytest plugin guard "
            "_benchflow_guard_x refused a pytest plugin it could not trust, so "
            "pytest stopped and this run is not scored (Verifier plugin trust "
            "rejected: ctrf (ctrf: /app/plugins/ctrf.py is under /app, which "
            "the agent could write))"
        ),
    }
    cause = cause_of(result, task_dir=Path("/tasks/usaco-101"))

    assert cause is not None
    assert cause.key == "verifier_plugin_refused"
    assert cause.fault == "agent"
    assert "bench tasks check /tasks/usaco-101" in cause.next_step


# V2-R1-A-1: a refusal's message quotes what the agent left (a path, the error
# from inspecting a planted file), so words in it must not decide its kind.
INFRA_WORDS = (
    "connection reset",
    "sandbox not found",
    "temporarily unavailable",
    "verifier recovery ",
    "dependency install failed",
)


def guard_refusal(quoted: str) -> str:
    """The trial's error for a refusal whose first line quotes *quoted*."""
    return (
        "verifier crashed: PluginGuardLoadError: the pytest plugin guard "
        "_benchflow_guard_x refused a pytest plugin it could not trust, so "
        "pytest stopped and this run is not scored (Verifier plugin trust "
        f"rejected: ctrf (cannot inspect: OSError: [Errno 40] Too many levels "
        f"of symbolic links: '/app/{quoted}-1.0.dist-info/entry_points.txt')); "
        "listed in verifier/_benchflow_guard_x.ab.refused"
    )


@pytest.mark.parametrize("words", INFRA_WORDS)
@pytest.mark.parametrize(
    "tag", ["", PRESERVED_SOLVER + " "], ids=["plain", "preserved"]
)
def test_refusal_quoting_infrastructure_words_is_a_failure_never_retried(words, tag):
    """Guards V2-R1-A-1: a planted dist-info named after an infra error re-rolled the trial.

    The error classified as verifier_infra, so the default retry ran the
    whole trial again with a fresh agent and kept the last attempt.
    """
    error = tag + guard_refusal(words)

    assert classify_verifier_error(error) == VERIFIER_FAILED
    assert not RetryConfig().should_retry_verifier_error(error)
    assert not needs_verifier_recovery(error)
    cause = cause_of({"rewards": None, "verifier_error": error})
    assert cause is not None
    assert (cause.key, cause.fault) == ("verifier_plugin_refused", "agent")


def test_refusal_quoting_the_installed_message_still_blames_the_agent():
    """Guards V2-R1-A-1: a planted path with the task message's words blamed the task."""
    error = guard_refusal("pytest plugin x was installed after the agent stopped")

    cause = cause_of({"rewards": None, "verifier_error": error})

    assert cause is not None
    assert (cause.key, cause.fault) == ("verifier_plugin_refused", "agent")


def test_guard_crash_quoting_the_refusal_words_is_not_a_refusal():
    """Guards V2-R1-A-1: only the guard's opening words name a refusal."""
    error = (
        "verifier crashed: PluginGuardLoadError: the pytest plugin guard "
        "_benchflow_guard_x crashed inside pytest (OSError: refused a pytest "
        "plugin; connection reset); traceback in verifier/x.crashed; its "
        "reward is not scored"
    )

    assert classify_verifier_error(error) == VERIFIER_FAILED
    assert not RetryConfig().should_retry_verifier_error(error)
    cause = cause_of({"rewards": None, "verifier_error": error})
    assert cause is not None and cause.key == VERIFIER_FAILED


def test_the_same_words_outside_a_guard_error_are_still_infrastructure():
    """The anchored match leaves every other verifier error's taxonomy alone."""
    error = "verifier crashed: RuntimeError: connection reset by peer"

    assert classify_verifier_error(error) != VERIFIER_FAILED
    assert RetryConfig().should_retry_verifier_error(error)


async def test_planted_path_with_infrastructure_words_is_not_retried(
    tmp_path, monkeypatch, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
    proof = tmp_path / "hostile-imported"
    planted = "connection reset/site-packages"

    def plant(layout):
        install_plugin(layout.workspace / planted, hostile_ctrf(proof))

    def reach_it(env, layout):
        return [layout.workspace / planted]

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        reach_it,
        before=plant,
        workspace_name="app",
        published=False,
    )

    assert not proof.exists()
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None
    cause = cause_of({"rewards": rewards, "verifier_error": error})
    assert cause is None


# V2-R1-A-2: the filesystem gives a deleted file's inode number to the next new
# file, so the number alone took test.sh's fresh install for the agent's.
def birth_or_skip(path: Path, follow: bool = False) -> str:
    st = os.stat(path) if follow else os.lstat(path)
    birth = guard_module._birth(str(path), st, follow)
    if birth is None:
        pytest.skip("this filesystem or C library reports no birth time")
    return birth


def test_a_reused_inode_number_is_not_the_recorded_entry(tmp_path):
    """Guards V2-R1-A-2: a recorded number with another birth time is a new file."""
    path = tmp_path / "ctrf_model.py"
    path.write_text("x = 1\n")
    birth = birth_or_skip(path)
    key = guard_module._identity(os.lstat(path))

    assert not guard_module._in_record(str(path), {key: {"1.000000001"}})
    assert guard_module._new_since_record(str(path), {key: {"1.000000001"}})
    assert guard_module._in_record(str(path), {key: {birth}})
    # Recorded without a birth time (the shell fallback): the number decides.
    assert guard_module._in_record(str(path), {key: {None}})


def test_chmod_touch_write_and_move_keep_an_entry_recorded(tmp_path):
    """Guards the fix under V2-R1-A-2: what test.sh does to the agent's file keeps its birth."""
    path = tmp_path / "ctrf_model.py"
    path.write_text("x = 1\n")
    birth = birth_or_skip(path)
    record = {guard_module._identity(os.lstat(path)): {birth}}

    path.chmod(0o777)
    os.utime(path, (1, 1))
    with path.open("a") as handle:
        handle.write("y = 2\n")
    moved = tmp_path / "elsewhere.py"
    path.rename(moved)

    assert guard_module._in_record(str(moved), record)
    assert not guard_module._new_since_record(str(moved), record)


def test_record_parses_lines_with_and_without_birth_times(tmp_path, monkeypatch):
    """Guards V2-R1-A-2: both writers' lines are read, and only a finished record counts."""
    record = tmp_path / "record"
    record.write_text("1:2:3.000000004\n1:2:5.000000006\n7:8\nend\n")
    monkeypatch.setattr(guard_module, "_BENCHFLOW_RECORD", str(record))
    monkeypatch.setattr(guard_module, "_RECORD", None)
    monkeypatch.setattr(guard_module, "ownership_problem", lambda path, st: None)
    monkeypatch.setattr(guard_module, "_root_alone", lambda path: True)

    assert guard_module._record() == {
        "1:2": {"3.000000004", "5.000000006"},
        "7:8": {None},
    }


@pytest.mark.parametrize(
    ("solution", "reward"), [(PASSING, 1.0), (FAILING, 0.0)], ids=["oracle", "empty"]
)
async def test_deleting_unknown_inputs_cannot_launder_install_provenance(
    tmp_path, monkeypatch, solution, reward, caplog
):
    """The prearm proof survives deletion, independently of inode reuse.

    The agent leaves output files and a project in /app. test.sh deletes them
    (``rm -rf out pyproject.toml uv.lock``), then runs USACO's ``uv init &&
    uv add pytest-json-ctrf``. The new venv's files and directories got the
    deleted files' inode numbers, which are in the record, so the guard took
    them for the agent's: ctrf refused, no ``installed`` marker. Their birth
    times are new, but the unknown inputs were tainted before test.sh ran.
    """
    recorded = set()

    def agent_left(layout):
        out = layout.workspace / "out"
        out.mkdir()
        for index in range(400):
            (out / f"{index}.txt").write_text("answer\n")
        (layout.workspace / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        (layout.workspace / "uv.lock").write_text("version = 1\n")
        for path in [out, *out.iterdir()]:
            recorded.add(guard_module._identity(os.lstat(path)))

    reused = []

    def delete_then_uv_init(env, layout):
        shutil.rmtree(layout.workspace / "out")
        (layout.workspace / "pyproject.toml").unlink()
        (layout.workspace / "uv.lock").unlink()
        sites = workspace_venv(env, layout)
        for path in [
            layout.workspace / ".venv",
            *(layout.workspace / ".venv").rglob("*"),
        ]:
            if guard_module._identity(os.lstat(path)) in recorded:
                reused.append(path)
        return sites

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        solution,
        delete_then_uv_init,
        before=agent_left,
        workspace_name="app",
        published=False,
    )

    stdout = layout.paths.test_stdout_path.read_text()
    assert "Verifier plugin trust rejected" in stdout, (len(reused), stdout)
    assert refused_scored_zero(rewards, error), (len(reused), rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None


def test_record_failure_is_ignored_but_an_earlier_failure_still_stops(tmp_path):
    """Guards V2-R1-A-3: the record's ``|| true`` swallowed a failed ``mkdir`` before it."""
    record = tmp_path / "missing-dir" / "record"
    command = lockdown._record_agent_trees_cmd(str(record), (str(tmp_path),))

    stopped = subprocess.run(
        ["/bin/sh", "-c", f"false && {command} && echo reached"],
        capture_output=True,
        text=True,
    )
    went_on = subprocess.run(
        ["/bin/sh", "-c", f"true && {command} && echo reached"],
        capture_output=True,
        text=True,
    )

    assert stopped.stdout == ""
    assert stopped.returncode != 0
    # The record itself failed (its folder is missing): hardening goes on.
    assert went_on.stdout == "reached\n"
    assert not record.exists()
