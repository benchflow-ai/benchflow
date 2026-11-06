"""The plugin guard's fixes from Codex's second review of V2.

R2-1: a blocked tree whose root is a link was recorded as the link alone, so
everything below it looked new. R2-2: a marker read that failed or stopped at
its limits was taken as "no marker". R2-3: a pytest killed while the guard
checked optional plugins or dependencies left only ``registered``. R2-4: a
guard refusal met during verifier recovery became infrastructure. R2-5: a
refusal left out of the error-excluding aggregates raised them. R2-6: a
requirements file no install read vetoed test.sh's install.
"""

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
    VERIFIER_FAILED,
    VERIFIER_INFRA,
    classify_verifier_error,
    mean_scored_reward,
    score_summary_fields,
)
from benchflow.evaluation import EvaluationResult
from benchflow.rollout import _verifier_recovery as recovery
from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from tests import test_verifier_guard_load_failure as load_failure
from tests.test_verifier_uv_state import (
    CTRF_PLUGIN,
    PASSING,
    install_plugin,
    refused_scored_zero,
    score,
    workspace_venv,
)

FAILING = "def test_solution():\n    assert False\n"
REFUSAL = (
    "verifier crashed: PluginGuardLoadError: the pytest plugin guard "
    "_benchflow_guard_0123abcd refused a pytest plugin it could not trust, so "
    "pytest stopped and this run is not scored (Verifier plugin trust "
    "rejected: ctrf (/app/connection reset/x.py is under /app, which the "
    "agent could write)); listed in verifier/x.refused"
)


@pytest.fixture
def guard_parent(tmp_path, monkeypatch):
    """Guards and their markers live under this test's folder."""
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))


def run_sh(command: str) -> None:
    subprocess.run(["/bin/sh", "-c", command], check=True, timeout=120)


def use_record(monkeypatch, record: Path) -> None:
    monkeypatch.setattr(guard, "_BENCHFLOW_RECORD", str(record))
    monkeypatch.setattr(guard, "_RECORD", None)
    monkeypatch.setattr(guard, "ownership_problem", lambda path, st: None)
    monkeypatch.setattr(guard, "_root_alone", lambda path: True)
    monkeypatch.setattr(guard, "_BENCHFLOW_ARMED", True)


# R2-1: the record of a tree whose root is a link.


def test_record_walks_the_directory_a_linked_root_resolves_to(tmp_path, monkeypatch):
    """/app -> /srv/app made every planted file look new.

    The writer recorded the link /app and skipped its target, then still
    wrote ``end``. A hostile ctrf the agent left in /srv/app/plugins counted
    as test.sh's fresh install, by its lexical and its resolved path alike.
    """
    real = tmp_path / "srv" / "app"
    site = real / "plugins" / "site-packages"
    install_plugin(site, CTRF_PLUGIN)
    app = tmp_path / "app"
    app.symlink_to(real, target_is_directory=True)
    record = tmp_path / "guard" / ".record"
    record.parent.mkdir()

    run_sh(lockdown._record_agent_trees_cmd(str(record), (str(app),)))

    assert record.read_text().splitlines()[-1] == "end"
    use_record(monkeypatch, record)
    planted = [p for p in site.rglob("*")]
    assert planted
    for path in planted:
        lexical = str(app) + str(path)[len(str(real)) :]
        assert not guard._installed_by_verifier(str(path)), path
        assert not guard._installed_by_verifier(lexical), lexical
    # What test.sh makes after the record is still new.
    fresh = real / ".venv" / "lib" / "python3.12" / "site-packages"
    fresh.mkdir(parents=True)
    (real / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (fresh / "ctrf_model.py").write_text(CTRF_PLUGIN)
    assert guard._installed_by_verifier(str(fresh / "ctrf_model.py"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads an unreadable folder")
def test_record_with_an_unreadable_folder_is_no_record(tmp_path):
    """A walk that could not read everything ends no record."""
    tree = tmp_path / "app"
    hidden = tree / "hidden"
    hidden.mkdir(parents=True)
    (hidden / "ctrf_model.py").write_text(CTRF_PLUGIN)
    hidden.chmod(0)
    record = tmp_path / "record"
    try:
        run_sh(lockdown._record_agent_trees_cmd(str(record), (str(tree),)))
    finally:
        hidden.chmod(0o755)

    lines = record.read_text().splitlines() if record.exists() else []
    assert "end" not in lines


# R2-2: a marker read that is incomplete.


class _MarkerTrouble(load_failure.PytestSandbox):
    """test.sh as before; the marker read fails, or many runs left markers."""

    trouble = "timeout"

    async def exec(self, command, env=None, **kwargs):
        if "test-stdout.txt" in command and self.trouble == "many":
            name = re.search(r"-p (_benchflow_guard_[0-9a-f]+)", env["PYTEST_ADDOPTS"])
            markers = Path(lockdown.pytest_plugin_guard_markers_dir(name[1]))
            for i in range(300):
                (markers / f"{name[1]}.1-{i:08x}.registered").write_text("x\n")
        if "/.markers" in command and "test-stdout.txt" not in command:
            if self.trouble == "timeout":
                raise TimeoutError("marker read timed out")
            if self.trouble == "exit":
                return ExecResult(stdout="", stderr="", return_code=1)
        return await super().exec(command, env=env, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("trouble", ["timeout", "exit", "many"])
async def test_a_marker_read_that_is_incomplete_is_not_scored(
    tmp_path, monkeypatch, guard_parent, trouble, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
    monkeypatch.setattr(_MarkerTrouble, "trouble", trouble)
    monkeypatch.setattr(load_failure, "PytestSandbox", _MarkerTrouble)

    _, paths, rewards, error = await load_failure._run(
        tmp_path, PASSING, guard_source=load_failure.armed(), plugins=("agent_plugin",)
    )

    assert paths.reward_text_path.read_text() == "0"
    assert rewards == {"reward": 0.0}
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None


def test_the_marker_read_puts_verdicts_before_registrations(tmp_path, guard_parent):
    """Registrations never push a refusal out of the read."""
    name = "_benchflow_guard_" + "ab" * 16
    markers = Path(lockdown.pytest_plugin_guard_markers_dir(name))
    markers.mkdir(parents=True)
    for i in range(lockdown._GUARD_MARKER_MAX_COUNT + 5):
        (markers / f"{name}.1-{i:08x}.registered").write_text("x\n")
    (markers / f"{name}.9-ffffffff.refused").write_text("y\n")

    out = subprocess.run(
        ["/bin/sh", "-c", lockdown.pytest_plugin_guard_markers_cmd(name)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert out.splitlines()[0].startswith(f"{name}.9-ffffffff.refused\t")
    # Registrations past the few the reader prints are skipped, not a
    # truncation (V2-R1-A-1).
    assert lockdown.pytest_plugin_guard_markers_problem(out) is None
    assert lockdown.pytest_plugin_guard_markers_problem("!end\n") is None
    assert lockdown.pytest_plugin_guard_markers_problem("") is not None


# R2-3: a pytest killed while the guard checks plugins.

KILLED_WHILE_CHECKING = """
_BENCHFLOW_OPTIONAL = ['ctrf']
def _load_optional(pluginmanager, disabled=()):
    os._exit(1)
"""
KILLED_IN_DEPENDENCY_CHECK = """
_real_validate = _validate
def _validate(names, *, entry_points=True):
    if os.path.exists(_marker('registered')):
        os._exit(1)
    return _real_validate(names, entry_points=entry_points)
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [KILLED_WHILE_CHECKING, KILLED_IN_DEPENDENCY_CHECK],
    ids=["optional-plugins", "dependencies"],
)
async def test_a_pytest_killed_while_the_guard_checks_is_not_scored(
    tmp_path, guard_parent, extra, caplog
):
    """A guard killed mid-check left only ``registered``.

    pytest dies (a FIFO the agent left as metadata, test.sh's own timeout)
    while the guard inspects optional plugins or a plugin's dependencies,
    after it wrote ``registered``: no ``refused`` and no ``crashed`` marker
    could follow. The leftover ``checking`` marker keeps test.sh's 0 out.
    """
    _, paths, rewards, error = await load_failure._run(
        tmp_path, PASSING, guard_source=load_failure.armed(extra)
    )

    assert paths.reward_text_path.read_text() == "0"
    assert rewards == {"reward": 0.0}
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None


def test_a_completed_check_leaves_no_checking_marker(tmp_path, monkeypatch):
    """A check that finishes clears its marker; a nested one does not clear the outer."""
    markers = tmp_path / "markers"
    markers.mkdir()
    key = tmp_path / "key"
    key.write_text(lockdown.pytest_plugin_guard_key("_benchflow_guard_test"))
    monkeypatch.setattr(
        guard, "_BENCHFLOW_MARKERS", str(markers / "_benchflow_guard_test")
    )
    monkeypatch.setattr(guard, "_BENCHFLOW_KEY_FILE", str(key))
    monkeypatch.setattr(guard, "_CHECKING", [0])

    guard._begin_check()
    guard._begin_check()
    guard._end_check()
    assert [p.name.rsplit(".", 1)[1] for p in markers.iterdir()] == ["checking"]
    guard._end_check()
    assert list(markers.iterdir()) == []


# R2-4: a refusal met during verifier recovery.


def test_a_guard_failure_during_recovery_stays_the_guards():
    """Recovery's wrapper turned a refusal into infrastructure."""
    wrapped = recovery._recovery_failure(REFUSAL)

    assert wrapped == f"{recovery.PRESERVED_SOLVER} {REFUSAL}"
    assert classify_verifier_error(wrapped) == VERIFIER_FAILED
    assert scoring.is_guard_refusal(wrapped)
    assert not recovery.needs_verifier_recovery(wrapped)
    other = recovery._recovery_failure("verifier crashed: OSError: boom")
    assert classify_verifier_error(other) == VERIFIER_INFRA


def test_only_the_rollouts_own_wrappers_are_infrastructure():
    """A phrase inside an error's detail decides nothing."""
    assert (
        classify_verifier_error(
            "verifier crashed: ValueError: test said verifier recovery failed"
        )
        == VERIFIER_FAILED
    )
    assert (
        classify_verifier_error("verifier crashed: RuntimeError: verifier_wedge: x")
        == VERIFIER_INFRA
    )
    assert classify_verifier_error("verifier_wedge: no receipt") == VERIFIER_INFRA
    assert (
        classify_verifier_error("[solver-preserved] verifier recovery failed: x")
        == VERIFIER_INFRA
    )


@pytest.mark.asyncio
async def test_scoring_does_not_recover_a_trial_the_guard_refused(tmp_path):
    """``bench eval score`` verified a refused trial again.

    The solver's first verification lost to infrastructure; its recovery
    met the guard's refusal. The persisted view keeps the refusal's kind, and
    scoring stops there instead of verifying again.
    """
    root = tmp_path / "trial"
    attempt = root / "verifier-recovery" / "a1"
    attempt.mkdir(parents=True)
    (root / "solver.json").write_text(
        json.dumps({"task_digest": "d", "verifier_error": "verifier_wedge: lost"})
    )
    (root / "verification.json").write_text(
        json.dumps({"attempt": "verifier-recovery/a1"})
    )
    (attempt / "recovery.json").write_text(
        json.dumps({"task_digest": "d", "status": "failed", "error": REFUSAL})
    )

    source = await recovery.resume_verification(root, tmp_path / "task")

    assert source["rewards"] is None
    assert source["verifier_error_category"] == VERIFIER_FAILED
    assert scoring.is_guard_refusal(source["verifier_error"])


# R2-5: refusals in the aggregates.


def test_a_refusal_counts_as_a_failure_in_the_error_excluding_scores():
    """Refusing only the failing trial doubled the scores.

    Two trials would score 1 and 0; the agent plants a refused plugin only in
    the failing one. Its reward stays None (the raw error is kept), but
    ``score_excl_errors`` and ``mean_reward`` stay at 50%.
    """
    rows = [
        {"rewards": {"reward": 1.0}},
        {"rewards": None, "verifier_error": REFUSAL},
    ]

    summary = score_summary_fields(rows)

    assert summary["guard_refusals"] == 1
    assert summary["score_excl_errors_ratio"] == 0.5
    assert summary["mean_reward"] == 0.5
    assert mean_scored_reward(rows) == 0.5
    result = EvaluationResult(
        job_name="j", config=MagicMock(), total=2, passed=1, guard_refused=1
    )
    assert result.score_excl_errors == 0.5
    # Not a refusal: infrastructure and the task's own install stay out.
    others = [
        {"rewards": {"reward": 1.0}},
        {"rewards": None, "verifier_error": "verifier crashed: OSError: x"},
        {
            "rewards": None,
            "verifier_error": "verifier crashed: PluginGuardLoadError: pytest "
            "plugin ctrf was installed after the agent stopped ...",
        },
    ]
    assert score_summary_fields(others)["guard_refusals"] == 0
    assert mean_scored_reward(others) == 1.0


def test_agent_text_cannot_pass_for_a_refusal():
    """Only the guard's own opening counts, not the same words later on."""
    assert not scoring.is_guard_refusal(
        "verifier crashed: ValueError: the pytest plugin guard _benchflow_guard_ab "
        "refused a pytest plugin it could not trust"
    )


# R2-6: dependency files no install read.


@pytest.mark.parametrize("published", [True, False])
async def test_an_unknown_install_treats_an_answer_file_as_a_possible_input(
    tmp_path, monkeypatch, published, caplog
):
    """An answer file is a possible input, and a published ctrf still loads.

    The solution leaves /app/requirements-answer.txt; USACO's test.sh makes
    a fresh project (``uv init && uv add pytest-json-ctrf``) that never
    reads it. No parser decides that: the file counts as an input,
    so the proof fails, and the oracle is scored only because the plugin
    holds a published release's bytes. A plugin that is not one is refused.
    """

    def answer(layout):
        (layout.workspace / "requirements-answer.txt").write_text("numpy\n")

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        workspace_venv,
        before=answer,
        workspace_name="app",
        published=published,
    )

    if published:
        assert rewards == {"reward": 1.0}, (rewards, error)
        assert error is None
    else:
        assert refused_scored_zero(rewards, error), (rewards, error)
        assert "pytest plugin guard not confirmed" in caplog.text
        assert error is None


def test_a_named_or_unknown_dependency_file_still_counts(tmp_path, monkeypatch):
    """Every file the agent changed counts, whatever the scripts name.

    The parser's names used to narrow the inputs; now no name, base name or
    project-file rule can leave a changed file out of the proof.
    """
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "requirements-dev.txt").write_text("evil\n")
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(workspace),))
    monkeypatch.setattr(guard, "_image_bytes", lambda path: False)
    changed = {str(workspace / "requirements-dev.txt")}
    monkeypatch.setattr(
        guard, "_new_since_record", lambda path, record: path not in changed
    )
    for hints in ((), ("requirements-dev.txt",), None, ("/tests/deps.txt",)):
        monkeypatch.setattr(guard, "_BENCHFLOW_NAMED_INPUTS", hints)
        assert not guard._agent_files_unchanged({})
    changed.clear()
    assert guard._agent_files_unchanged({})


def test_the_dependency_files_verifier_scripts_name(tmp_path):
    """The files hardening tells the guard about, from the verifier's scripts."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        "#!/bin/bash\n"
        "# requirements-old.txt is not used\n"
        "pip install -r /app/deps.txt -c constraints.txt\n"
        "uv pip install --requirement=requirements-test.txt\n"
        "uv pip sync pylock.toml\n"
        "uv add pytest-json-ctrf\n"
    )
    task = MagicMock()
    task.paths = SimpleNamespace(tests_dir=tests)

    assert lockdown._verifier_named_dependency_files(task) == [
        "/app/deps.txt",
        "constraints.txt",
        "pylock.toml",
        "requirements-test.txt",
    ]
    (tests / "test.sh").write_text('pip install -r "$REQS"\n')
    assert lockdown._verifier_named_dependency_files(task) is None
