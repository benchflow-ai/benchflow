"""The plugin guard's fixes from review round 1 of V2 (V2-R1-A-1..5, V2-R1-B-1).

A-1: every pytest the guard loads in (pytest-xdist workers included) leaves a
``registered`` marker, and more than 256 of them made a clean run unscored.
A-2: a marker read the sandbox failed was the guard's verdict, never retried.
A-3 and B-1: a refusal that an agent-made ``.venv`` or ``pyproject.toml`` (or
a record the agent made fail) brought about got the task's ``installed``
message, which the error-excluding scores leave out. A-4: dependency files
the verifier's scripts name matched by base name in every directory, and any
``sh -c "$CMD"`` dropped the list. A-5: the two error-excluding scores
disagreed for a refusal on a trial whose agent errored.

The end-to-end scenarios use the harnesses of test_verifier_uv_state.py (the
real hardening, then a real pytest with the armed guard) and
test_verifier_guard_load_failure.py (a real pytest with the guard on its
path), scored through ``_verify_rollout``.
"""

import hashlib
import hmac
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchflow._utils import scoring
from benchflow._utils.scoring import (
    score_summary_fields,
)
from benchflow.failures import cause_of
from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from tests import test_verifier_guard_load_failure as load_failure
from tests import test_verifier_uv_state as uv_state
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


def signed_marker(guard_name: str, token: str, kind: str, detail: str = "") -> None:
    """Write a marker as the guard *guard_name* signs it."""
    directory = Path(lockdown.pytest_plugin_guard_markers_dir(guard_name))
    name = f"{guard_name}.{token}.{kind}"
    body = json.dumps(detail)
    key = lockdown.pytest_plugin_guard_key(guard_name).encode()
    mac = hmac.new(key, f"{name}\n{body}".encode(), hashlib.sha256).hexdigest()
    (directory / name).write_text(f"{mac}\t{body}\n")


def read_markers(guard_name: str) -> str:
    return subprocess.run(
        ["/bin/sh", "-c", lockdown.pytest_plugin_guard_markers_cmd(guard_name)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout


@pytest.fixture
def guard_parent(tmp_path, monkeypatch):
    """Guards and their markers live under this test's folder, as at / in the sandbox."""
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))


def guard_of(env) -> str:
    return re.search(r"-p (_benchflow_guard_[0-9a-f]+)", env["PYTEST_ADDOPTS"])[1]


# A-1: registered markers of many clean pytest runs.


class _ManyCleanRuns(load_failure.PytestSandbox):
    """test.sh runs pytest 300 times (or ``-n auto`` on a big host): 300 registrations."""

    async def exec(self, command, env=None, **kwargs):
        if "test-stdout.txt" in command:
            name = guard_of(env)
            for i in range(300):
                signed_marker(name, f"{i + 2}-{i:08x}", "registered")
        return await super().exec(command, env=env, **kwargs)


@pytest.mark.parametrize(
    ("solution", "reward"), [(PASSING, 1.0), (FAILING, 0.0)], ids=["oracle", "empty"]
)
async def test_many_clean_pytest_runs_stay_scored(
    tmp_path, monkeypatch, guard_parent, solution, reward
):
    """Guards V2-R1-A-1: 300 clean pytest processes made every run unscored.

    Each left a ``registered`` marker, the reader counted them against its
    cap of 256 and printed ``!truncated``, and the verifier refused the
    reward, the oracle's included. Only one registration is needed.
    """
    monkeypatch.setattr(load_failure, "PytestSandbox", _ManyCleanRuns)

    _, _, rewards, error = await load_failure._run(
        tmp_path, solution, guard_source=load_failure.armed()
    )

    assert error is None, error
    assert rewards == {"reward": reward}


def test_the_reader_caps_verdicts_not_registrations(guard_parent):
    """300 registrations read whole; 300 verdicts still stop at the cap."""
    name = "_benchflow_guard_" + "cd" * 16
    Path(lockdown.pytest_plugin_guard_markers_dir(name)).mkdir(parents=True)
    for i in range(300):
        signed_marker(name, f"1-{i:08x}", "registered")
    signed_marker(name, "9-ffffffff", "refused", "Verifier plugin trust rejected: x")

    out = read_markers(name)

    assert lockdown.pytest_plugin_guard_markers_problem(out) is None
    kinds = [m.kind for m in lockdown.parse_pytest_plugin_guard_markers(out, name)]
    assert kinds[0] == "refused"
    assert 1 <= kinds.count("registered") <= lockdown._GUARD_REGISTERED_READ

    for i in range(lockdown._GUARD_MARKER_MAX_COUNT + 5):
        signed_marker(name, f"2-{i:08x}", "refused", "x")
    out = read_markers(name)
    assert lockdown.pytest_plugin_guard_markers_problem(out) == (
        "more markers than the verifier reads"
    )


# A-2: a marker read the sandbox failed.


class _MarkerReadFails(load_failure.PytestSandbox):
    trouble = "timeout"

    async def exec(self, command, env=None, **kwargs):
        if "test-stdout.txt" in command and self.trouble == "many-refusals":
            name = guard_of(env)
            for i in range(lockdown._GUARD_MARKER_MAX_COUNT + 5):
                signed_marker(
                    name,
                    f"{i + 2}-{i:08x}",
                    "refused",
                    "Verifier plugin trust rejected: x",
                )
        if "/.markers" in command and "test-stdout.txt" not in command:
            if self.trouble == "timeout":
                raise TimeoutError("marker read timed out")
            if self.trouble == "exit":
                return ExecResult(stdout="", stderr="", return_code=1)
        return await super().exec(command, env=env, **kwargs)


@pytest.mark.parametrize("trouble", ["timeout", "exit"])
async def test_a_failed_marker_read_is_infrastructure(
    tmp_path, monkeypatch, guard_parent, trouble, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
    monkeypatch.setattr(_MarkerReadFails, "trouble", trouble)
    monkeypatch.setattr(load_failure, "PytestSandbox", _MarkerReadFails)

    _, paths, rewards, error = await load_failure._run(
        tmp_path, PASSING, guard_source=load_failure.armed()
    )

    assert paths.reward_text_path.read_text() == "1"
    assert rewards == {"reward": 0.0}
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None


async def test_a_verdict_read_before_the_limit_still_decides(
    tmp_path, monkeypatch, guard_parent, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
    monkeypatch.setattr(_MarkerReadFails, "trouble", "many-refusals")
    monkeypatch.setattr(load_failure, "PytestSandbox", _MarkerReadFails)

    _, _, rewards, error = await load_failure._run(
        tmp_path, PASSING, guard_source=load_failure.armed()
    )

    assert rewards == {"reward": 0.0}
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None


# A-3 and B-1: a refusal the agent's files brought about is a refusal.


def summary_of(error, *, rewards=None):
    return score_summary_fields(
        [{"rewards": {"reward": 1.0}}, {"rewards": rewards, "verifier_error": error}]
    )


def uv_add_into_the_workspace(env, layout):
    """``uv init; uv add pytest-json-ctrf`` in /app, reusing what is already there."""
    site = layout.workspace / ".venv/lib/python3.12/site-packages"
    install_plugin(site, CTRF_PLUGIN)
    cfg = layout.workspace / ".venv/pyvenv.cfg"
    if not cfg.exists():
        cfg.write_text("home = /usr/bin\n")
    return [site]


@pytest.mark.parametrize(
    "left",
    ["pyproject.toml", "uv.toml", ".python-version", ".venv"],
)
async def test_an_agent_file_the_verifiers_install_reused_is_a_refusal(
    tmp_path, monkeypatch, left, caplog
):
    """Guards V2-R1-A-3 and V2-R1-B-1 route 1: the agent chose the task's error.

    The agent leaves /app/pyproject.toml (or uv.toml, .python-version, or an
    empty /app/.venv); USACO's test.sh then adds ctrf to the project in /app,
    reusing it. The refusal is right, but every file of the plugin was new,
    so the guard wrote ``installed``: fault "task", "the reference
    solution's included" (false), and the trial left out of
    ``score_excl_errors`` and ``mean_reward``. Now it stays a refusal: a 0
    there. Since V2-R1-B-2 a reused project file makes it the task's problem,
    not the agent's: no reading of test.sh tells that file from the agent's
    answer, and either way the task's unpinned install is what the
    guard cannot check. A reused ``.venv`` (the environment itself was the
    agent's) is still the agent's. The score is the same.
    """

    def leave(layout):
        path = layout.workspace / left
        if left == ".venv":
            path.mkdir()
        else:
            path.write_text(
                "[project]\nname = 'app'\n" if left.endswith("toml") else "3.12\n"
            )

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        uv_add_into_the_workspace,
        before=leave,
        workspace_name="app",
        published=False,
    )

    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert markers(layout, "installed") == []
    assert error is None
    cause = cause_of({"rewards": rewards, "verifier_error": error})
    assert cause is None
    summary = summary_of(error, rewards=rewards)
    assert summary["guard_refusals"] == 0
    assert summary["score_excl_errors_ratio"] == 0.5
    assert summary["mean_reward"] == 0.5


def test_no_record_is_no_install_evidence(tmp_path, monkeypatch):
    """Guards V2-R1-B-1 route 2: without a record a refusal is never the task's.

    With no record, change times stood in, so test.sh's own fresh install
    (refused because nothing proves it the verifier's) got ``installed``.
    """
    site = tmp_path / "app/.venv/lib/python3.12/site-packages"
    install_plugin(site, CTRF_PLUGIN)
    (tmp_path / "app/.venv/pyvenv.cfg").write_text("home = /usr/bin\n")
    monkeypatch.setattr(guard.sys, "path", [str(site), *guard.sys.path])
    monkeypatch.setattr(guard, "_BENCHFLOW_RECORD", "")
    monkeypatch.setattr(guard, "_RECORD", None)
    blocked = (str(tmp_path / "app"),)

    assert guard._installed_during_verification(["ctrf"], True, blocked, ()) == []
    # With a record that holds only /app, the same files are the verifier's.
    monkeypatch.setattr(
        guard, "_RECORD", {guard._identity(os.lstat(tmp_path / "app")): {None}}
    )
    assert str(site / "ctrf_model.py") in guard._installed_during_verification(
        ["ctrf"], True, blocked, ()
    )


def deep_tree(top: Path, depth: int = 300, name: str = "d" * 20) -> int:
    """A chain of *depth* folders below *top*, longer than PATH_MAX; the deepest's inode."""
    fd = os.open(top, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for _ in range(depth):
            os.mkdir(name, dir_fd=fd)
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
        leaf = os.open("leaf.py", os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=fd)
        os.close(leaf)
        return os.stat("leaf.py", dir_fd=fd).st_ino
    finally:
        os.close(fd)


def test_a_record_over_a_path_longer_than_path_max_ends(tmp_path):
    """Guards V2-R1-B-1 route 2: a 6,000-byte path left the record without ``end``."""
    tree = tmp_path / "app"
    tree.mkdir()
    (tree / "solution.py").write_text("x = 1\n")
    leaf = deep_tree(tree)
    record = tmp_path / "record"

    subprocess.run(
        ["/bin/sh", "-c", lockdown._record_agent_trees_cmd(str(record), (str(tree),))],
        check=True,
        timeout=120,
    )

    lines = record.read_text().splitlines()
    assert lines[-1] == "end"
    numbers = {line.split(":")[1] for line in lines[:-1]}
    assert str(leaf) in numbers
    assert str(os.lstat(tree / "solution.py").st_ino) in numbers
    # 300 folders, the leaf, solution.py and the root itself.
    assert len(lines) - 1 == 303


async def test_an_unwalkable_unknown_input_tree_refuses_install_provenance(
    tmp_path, monkeypatch, caplog
):
    """A complete identity record alone cannot prove unknown inputs are clean.

    The separate input-proof walk fails at PATH_MAX and conservatively refuses;
    the descriptor-based identity record still has its own deep-tree tests.
    """

    def leave_deep_folders(layout):
        deep_tree(layout.workspace)

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        workspace_venv,
        before=leave_deep_folders,
        workspace_name="app",
        published=False,
    )

    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None


def test_a_pytest_stopped_while_checking_counts_as_a_refusal():
    """Guards V2-R1-B-1 route 3: the agent's FIFO under test.sh's timeout was left out."""
    error = (
        "verifier crashed: PluginGuardLoadError: pytest stopped while the plugin "
        "guard _benchflow_guard_0123abcd was still checking pytest plugins, so it "
        "never decided whether to trust them; listed in verifier/x.checking; this "
        "run is not scored"
    )

    assert scoring.is_guard_refusal(error)
    assert summary_of(error)["score_excl_errors_ratio"] == 0.5


# A-4: the dependency files the verifier's scripts name.


def named(tmp_path, script: str):
    tests = tmp_path / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "test.sh").write_text(script)
    task = MagicMock()
    task.paths = SimpleNamespace(tests_dir=tests)
    return lockdown._verifier_named_dependency_files(task)


def test_an_absolute_name_names_that_file_alone(tmp_path, monkeypatch):
    """Guards V2-R1-A-4's names, now hints that can only fail the proof sooner.

    /tests/requirements.txt names that file; relative names are not checked
    first. Neither makes /app/requirements.txt, which the agent changed, any
    less an input: the full walk counts every file.
    """
    names = named(
        tmp_path,
        "cd /app && uv venv && uv pip install -r /tests/requirements.txt\n"
        "pip install -r requirements-dev.txt\n"
        'pip install -r "$DIR/requirements-ci.txt"\n',
    )

    assert names == [
        "/tests/requirements.txt",
        "requirements-ci.txt",
        "requirements-dev.txt",
    ]
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "requirements.txt").write_text("DUMMY agent change\n")
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(workspace),))
    monkeypatch.setattr(guard, "_BENCHFLOW_NAMED_INPUTS", tuple(names))
    monkeypatch.setattr(guard, "_new_since_record", lambda path, record: False)
    monkeypatch.setattr(guard, "_image_bytes", lambda path: False)
    assert not guard._hinted_input_changed({})
    assert not guard._agent_files_unchanged({})


def test_an_absolute_name_matches_through_a_linked_folder(tmp_path, monkeypatch):
    """A hint reached through a linked folder still ends the proof when changed.

    Hints only order the proof: the agent's changed file behind the
    link is found first, and the full walk would find it anyway.
    """
    real = tmp_path / "srv"
    real.mkdir()
    (real / "deps.txt").write_text("x\n")
    (tmp_path / "app").symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(tmp_path / "app"),))
    monkeypatch.setattr(
        guard, "_BENCHFLOW_NAMED_INPUTS", (str(tmp_path / "app" / "deps.txt"),)
    )
    monkeypatch.setattr(guard, "_new_since_record", lambda path, record: False)
    monkeypatch.setattr(guard, "_image_bytes", lambda path: False)

    assert guard._hinted_input_changed({})
    assert not guard._agent_files_unchanged({})


def test_command_strings_are_not_dependency_files(tmp_path):
    """Guards V2-R1-A-4: ``sh -c "$CMD"`` brought every requirements file back."""
    assert named(
        tmp_path,
        'sh -c "$CMD"\n'
        "bash -c 'cd /app && pytest'\n"
        'python3 -c "$(cat /tests/gen.py)"\n'
        "pip install -c constraints.txt pytest\n",
    ) == ["constraints.txt"]
    assert named(tmp_path, 'pip install -r "$REQS"\n') is None
    assert named(tmp_path, 'uv pip install -c "$C" x\n') is None


async def test_an_edited_project_file_no_install_reads_keeps_the_pass(
    tmp_path, monkeypatch
):
    """Guards V2-R1-A-4 end to end: a correct solution was unscored as a task problem.

    test.sh installs ``-r /tests/requirements.txt`` into a fresh /app/.venv;
    the solution edited /app/requirements.txt, which no install read.
    """
    test_sh = (
        "#!/bin/bash\ncd /app && uv venv && uv pip install -r /tests/requirements.txt\n"
        ".venv/bin/pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n"
    )
    make_task = uv_state.make_task

    def task_with_requirements(layout, env=None):
        task = make_task(layout, test_sh=test_sh, env=env)
        (task.paths.tests_dir / "requirements.txt").write_text(
            "pytest-json-ctrf==0.3.5\n"
        )
        task.config.verifier.env = {}
        return task

    monkeypatch.setattr(uv_state, "make_task", task_with_requirements)

    def edit_requirements(layout):
        (layout.workspace / "requirements.txt").write_text("numpy\nscipy\n")

    def uv_venv(env, layout):
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        install_plugin(site, CTRF_PLUGIN)
        (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
        return [site]

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        uv_venv,
        before=edit_requirements,
        workspace_name="app",
    )

    assert error is None, error
    assert rewards == {"reward": 1.0}


async def test_an_agent_requirements_file_the_install_reads_is_still_refused(
    tmp_path, monkeypatch, caplog
):
    """The same test.sh with ``-r /app/requirements.txt``: the agent chose the install.

    The script names the workspace by its path in this model of the sandbox.
    """
    proof = tmp_path / "hostile-imported"

    def test_sh(layout):
        return (
            f"#!/bin/bash\ncd {layout.workspace} && uv venv && "
            f"uv pip install -r {layout.workspace}/requirements.txt\n"
            ".venv/bin/pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n"
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
    assert error is None
    assert markers(layout, "installed") == []


# A-5: one rule for refusals in both error-excluding scores.


def test_a_refusal_on_an_errored_agent_trial_is_left_out_of_both_scores():
    """Guards V2-R1-A-5: the job summary and the metrics disagreed."""
    refusal = (
        "verifier crashed: PluginGuardLoadError: the pytest plugin guard "
        "_benchflow_guard_0123abcd refused a pytest plugin it could not trust, so "
        "pytest stopped and this run is not scored"
    )
    rows = [
        {"task_name": "a", "rewards": {"reward": 1.0}},
        {
            "task_name": "b",
            "rewards": None,
            "error": "agent timed out",
            "verifier_error": refusal,
        },
    ]

    summary = score_summary_fields(rows)

    assert summary["guard_refusals"] == 0
    assert summary["score_excl_errors_ratio"] == 1.0
    assert scoring.count_guard_refusals(rows) == 0
