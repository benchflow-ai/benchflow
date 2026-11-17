"""``bench check`` and the run share one CTRF rule (EN-719).

A run reads a verifier's CTRF report only from /logs/verifier/ctrf.json. In
the rc3 prep sweep ``bench check`` refused two terminal-bench-4.1 tasks whose
oracle passes a real run: wal-recovery-ordering copies its reports onto that
path, and ontology-kg-querying writes ``--ctrf /logs/verifier/ctrf.json; then``.
The run, for its part, took any test.sh and lost a report that never got
there. Both now call ``ctrf_report_issue``: a report a task.md criterion is
graded from is refused at every door; in a task that grades itself a lost
report is a warning, and the task runs (F2b, 2026-11-16 12:27Z).
"""

import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow._utils.task_authoring import (
    check_task,
    check_task_warnings,
    runnable_task_issues,
)
from benchflow.cli.main import app
from benchflow.evaluation import Evaluation
from benchflow.runtime import check_runnable_task

runner = CliRunner()

# The verifier shapes of the two refused tasks, written for this test.
WAL_RECOVERY_ORDERING = """\
#!/bin/bash
set -uo pipefail
rm -f /logs/verifier/reward.txt /logs/verifier/ctrf*.json
python3 -I -m pytest --ctrf /logs/verifier/ctrf_gates.json \\
    /tests/test_gates.py -x -rA
status=$?
cp -f /logs/verifier/ctrf_gates.json /logs/verifier/ctrf.json 2>/dev/null || true
for run in 1 2 3; do
    python3 -I -m pytest --ctrf "/logs/verifier/ctrf_${run}.json" /tests/test_outputs.py
    cp -f "/logs/verifier/ctrf_${run}.json" /logs/verifier/ctrf.json 2>/dev/null || true
done
python3 -I - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('/logs/verifier/ctrf_3.json').read_text())
PY
echo 1 > /logs/verifier/reward.txt
"""
ONTOLOGY_KG_QUERYING = """\
#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
if python3 -I -S /tests/test_outputs.py --ctrf /logs/verifier/ctrf.json; then
  RESULT=1
else
  RESULT=0
fi
printf '%s\\n' "$RESULT" > /logs/verifier/reward.txt
"""
STANDARD = "pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n"
MISSING = (
    "pytest --ctrf /logs/verifier/ctrf-report.json /tests/test_outputs.py\n"
    "echo 1 > /logs/verifier/reward.txt\n"
)
MISSING_ISSUE = (
    "test.sh uses non-standard CTRF path '/logs/verifier/ctrf-report.json' "
    "(expected '/logs/verifier/ctrf.json') and never copies it there; a run "
    "reads the report only from that path"
)


def _task(parent: Path, name: str, test_sh: str) -> Path:
    """A legacy task ``bench check`` passes apart from its test.sh."""
    task = parent / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "task.toml").write_text(
        "[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 120\n"
    )
    (task / "instruction.md").write_text("# Task\n")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests" / "test.sh").write_text(test_sh)
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\necho done\n")
    return task


def _ctrf_issues(issues: list[str]) -> list[str]:
    return [issue for issue in issues if "CTRF" in issue]


def _verdict(test_sh: str):
    # Imported here, so the check and run tests above it fail on a tree
    # without the rule by what they assert, not by an import error.
    from benchflow.task.ctrf_report import ctrf_report_verdict

    return ctrf_report_verdict(test_sh)


@pytest.mark.parametrize(
    "test_sh",
    [WAL_RECOVERY_ORDERING, ONTOLOGY_KG_QUERYING],
    ids=["wal-recovery-ordering", "ontology-kg-querying"],
)
def test_check_and_run_accept_the_two_refused_tasks(tmp_path, test_sh):
    task = _task(tmp_path, "t", test_sh)
    assert check_task(task) == []
    assert runnable_task_issues(task) == []
    check_runnable_task(task)
    checked = runner.invoke(app, ["check", str(task)])
    assert checked.exit_code == 0, checked.output
    dry = runner.invoke(app, ["run", str(task), "--agent", "oracle", "--dry-run"])
    assert dry.exit_code == 0, dry.output


def test_a_lost_report_in_a_task_that_grades_itself_is_a_warning(tmp_path, caplog):
    """F2b, 2026-11-16 12:27Z (RUSAGE's D1): nothing is graded from the report, so check
    and the run warn in the same words, and the task runs."""
    task = _task(tmp_path, "t", MISSING)
    assert _ctrf_issues(check_task(task)) == []
    assert MISSING_ISSUE in check_task_warnings(task)
    assert runnable_task_issues(task) == []
    with caplog.at_level(logging.WARNING):
        check_runnable_task(task)
    assert MISSING_ISSUE in caplog.text
    checked = runner.invoke(app, ["check", str(task)], terminal_width=400)
    assert checked.exit_code == 0, checked.output
    assert MISSING_ISSUE in " ".join(checked.output.split())


def test_the_dry_run_warns_of_a_lost_report_and_goes_ahead(tmp_path, caplog):
    task = _task(tmp_path, "t", MISSING)
    with caplog.at_level(logging.WARNING):
        dry = runner.invoke(
            app,
            ["run", str(task), "--agent", "oracle", "--dry-run"],
            terminal_width=400,
        )
    assert dry.exit_code == 0, dry.output
    assert MISSING_ISSUE in " ".join((dry.output + caplog.text).split())


def test_a_batch_warns_of_a_lost_report_by_name_and_runs_every_task(tmp_path, caplog):
    tasks = tmp_path / "tasks"
    _task(tasks, "good", STANDARD)
    _task(tasks, "lost", MISSING)
    evaluation = Evaluation(tasks_dir=tasks, jobs_dir=tmp_path / "jobs")
    with caplog.at_level(logging.WARNING):
        selected = evaluation._get_task_dirs()
    assert [d.name for d in selected] == ["good", "lost"]
    assert f"Task 'lost': {MISSING_ISSUE}" in caplog.text
    assert "Task 'good'" not in caplog.text


@pytest.mark.parametrize(
    "test_sh",
    [
        STANDARD,
        "pytest --ctrf=/logs/verifier/ctrf.json /tests\n",
        "pytest /tests/test_outputs.py\n",
        "# pytest --ctrf /bad/path.json\npytest /tests\n",
    ],
    ids=["standard", "equals-form", "no-ctrf", "commented"],
)
def test_the_standard_path_is_unchanged(tmp_path, test_sh):
    task = _task(tmp_path, "t", test_sh)
    assert check_task(task) == []
    assert runnable_task_issues(task) == []
    check_runnable_task(task)


@pytest.mark.parametrize(
    "test_sh",
    [
        "pytest --ctrf /tmp/r.json /tests\nmv /tmp/r.json /logs/verifier/ctrf.json\n",
        "pytest --ctrf /tmp/r.json /tests\ninstall -m 0644 /tmp/r.json "
        "/logs/verifier/ctrf.json\n",
        "pytest --ctrf /tmp/r.json /tests\ncat /tmp/r.json > /logs/verifier/ctrf.json\n",
        "pytest --ctrf /tmp/r.json /tests\ncat /tmp/r.json | tee /logs/verifier/ctrf.json\n",
        "pytest --ctrf /tmp/out/ctrf.json /tests\ncp /tmp/out/ctrf.json /logs/verifier/\n",
        "pytest --ctrf /tmp/out/ctrf.json /tests\ncp /tmp/out/*.json /logs/verifier/\n",
        "pytest --ctrf /tmp/out/ctrf.json /tests\ncp -t /logs/verifier /tmp/out/ctrf.json\n",
        "pytest --ctrf /logs//verifier/./ctrf.json /tests\n",
    ],
    ids=[
        "mv",
        "install",
        "redirect",
        "tee",
        "into-folder",
        "glob",
        "target-dir",
        "spelled",
    ],
)
def test_a_report_written_elsewhere_and_put_there_reaches_the_run(test_sh):
    assert _verdict(test_sh).verdict == "reaches"


@pytest.mark.parametrize(
    "test_sh",
    [
        'pytest --ctrf "$OUT/ctrf.json" /tests\n',
        'pytest --ctrf "/tmp/ctrf_$(date +%s).json" /tests\n',
        'pytest --ctrf /tmp/r.json /tests\neval "$FINISH"\n',
        "pytest --ctrf /tmp/r.json /tests\nsource /tests/finish.sh\n",
        "pytest --ctrf /tmp/r.json /tests\n. /tests/finish.sh\n",
        "pytest --ctrf /tmp/r.json /tests\nbash -c 'cp /tmp/r.json /logs/verifier/ctrf.json'\n",
        "pytest --ctrf /tmp/r.json /tests\nbash <<EOF\ncp /tmp/r.json x\nEOF\n",
        "cd /logs/verifier\npytest --ctrf ctrf.json /tests\n",
        "pytest --ctrf /tmp/r.json /tests\npython3 - <<'PY'\nimport shutil\n"
        "shutil.copy('/tmp/r.json', '/logs/verifier/ctrf.json')\nPY\n",
        'pytest --ctrf /tmp/r.json /tests\npython3 -c "import shutil; '
        "shutil.copy('/tmp/r.json', '/logs/verifier/ctrf.json')\"\n",
        'pytest --ctrf /tmp/r.json /tests\ncp /tmp/r.json "$DEST"\n',
        'pytest --ctrf /tmp/r.json /tests\ncp "$R" /logs/verifier/\n',
        'pytest --ctrf /tmp/r.json /tests\ncat /tmp/r.json > "$OUT"\n',
        "pytest --ctrf /tmp/r.json /tests\necho \"it's\n",
    ],
    ids=[
        "variable",
        "command-substitution",
        "eval",
        "source",
        "dot",
        "bash-c",
        "shell-heredoc",
        "relative-after-cd",
        "heredoc-code",
        "python-c",
        "copy-to-variable",
        "variable-into-folder",
        "redirect-to-variable",
        "open-quote",
    ],
)
def test_what_the_reader_cannot_follow_is_accepted(tmp_path, test_sh):
    assert _verdict(test_sh).verdict == "undecided"
    task = _task(tmp_path, "t", test_sh)
    assert _ctrf_issues(check_task(task)) == []
    assert runnable_task_issues(task) == []


@pytest.mark.parametrize(
    "test_sh",
    [
        MISSING,
        "pytest --ctrf /tmp/r.json /tests\n",
        "pytest --ctrf ctrf.json /tests\n",
        "pytest --ctrf /tmp/r.json /tests\ncp /tmp/r.json /tmp/ctrf.json\n",
        "pytest --ctrf /tmp/r.json /tests\n# cp /tmp/r.json /logs/verifier/ctrf.json\n",
        "pytest \\\n  --ctrf \\\n  /tmp/r.json /tests\n",
        # tests/fixtures/skillsbench_slice/r2r-mpc-control: the report is
        # written as ctrf-report.json; the glob copies /root's outputs, not it.
        "pytest /tests/test_outputs.py \\\n    --ctrf /logs/verifier/ctrf-report.json"
        " \\\n    -v\ncp /root/*.json /logs/verifier/ 2>/dev/null || true\n",
        "pytest --ctrf /tmp/r.json /tests\nrm -f /logs/verifier/ctrf.json\n",
    ],
    ids=[
        "other-name",
        "elsewhere",
        "relative",
        "copied-elsewhere",
        "commented-copy",
        "continued-lines",
        "r2r-mpc-control-fixture",
        "removed",
    ],
)
def test_a_report_that_never_reaches_the_run_is_a_warning(tmp_path, test_sh):
    verdict = _verdict(test_sh)
    assert verdict.verdict == "missing", verdict.reason
    task = _task(tmp_path, "t", test_sh)
    assert runnable_task_issues(task) == []
    assert _ctrf_issues(check_task(task)) == []
    assert verdict.reason in check_task_warnings(task)


@pytest.mark.parametrize(
    "test_sh",
    [WAL_RECOVERY_ORDERING, ONTOLOGY_KG_QUERYING],
    ids=["wal-recovery-ordering", "ontology-kg-querying"],
)
def test_the_rule_reads_both_tasks_reports_as_reaching_the_run(test_sh):
    assert _verdict(test_sh).verdict == "reaches"


# RUSAGE's review (2026-11-16 11:36Z): a `#` inside a word or in quotes is
# no comment, so a copy or a write of the report later on that line counts.
REPORT_LINE_HASHES = {
    "length of a variable": (
        "pytest --ctrf /tmp/r.json /tests || true\n"
        "if [ ${#REWARD} -ge 0 ]; then cp /tmp/r.json /logs/verifier/ctrf.json; fi\n"
    ),
    "argument count": (
        "pytest --ctrf /tmp/r.json /tests\n"
        "echo $# args; cp /tmp/r.json /logs/verifier/ctrf.json\n"
    ),
    "prefix removal": (
        "pytest --ctrf /tmp/r.json /tests\n"
        "x=${p##*/}; mv /tmp/r.json /logs/verifier/ctrf.json\n"
    ),
    "quoted": (
        "pytest --ctrf /tmp/r.json /tests\n"
        "echo 'step #2'; cp /tmp/r.json /logs/verifier/ctrf.json\n"
    ),
    "fragment": "curl -s http://h/x#y; pytest --ctrf /logs/verifier/ctrf.json /tests\n",
}


@pytest.mark.parametrize(
    "test_sh", list(REPORT_LINE_HASHES.values()), ids=list(REPORT_LINE_HASHES)
)
def test_a_hash_inside_a_word_or_quotes_is_no_comment(tmp_path, test_sh):
    assert _verdict(test_sh).verdict == "reaches", _verdict(test_sh).reason
    task = _task(tmp_path, "t", test_sh)
    assert _ctrf_issues(check_task(task)) == []
    assert runnable_task_issues(task) == []


@pytest.mark.parametrize(
    "test_sh",
    [
        "pytest --ctrf /tmp/r.json /tests  # cp /tmp/r.json /logs/verifier/ctrf.json\n",
        "pytest --ctrf /tmp/r.json /tests;# mv /tmp/r.json /logs/verifier/ctrf.json\n",
        "pytest --ctrf /tmp/r.json /tests\n\t# cp /tmp/r.json /logs/verifier/ctrf.json\n",
    ],
    ids=["after-a-space", "after-an-operator", "line-start"],
)
def test_a_hash_that_starts_a_word_is_still_a_comment(test_sh):
    assert _verdict(test_sh).verdict == "missing"


def test_a_task_md_report_written_after_a_length_test_is_graded(tmp_path):
    """RUSAGE's hello-world: the report is written on a line with ${#status}."""
    task = _draft2(
        tmp_path,
        "#!/bin/bash\n"
        "if printf 'Hello, world!' | cmp -s - /app/hello.txt; then status=passed; "
        "else status=failed; fi\n"
        'if [ ${#status} -gt 0 ]; then printf \'{"results": {"tool": {"name": '
        '"test.sh"}, "tests": [{"name": "test_hello", "status": "%s"}]}}\\n\' '
        '"$status" > /logs/verifier/ctrf.json; fi\n',
    )
    assert _plan_refusals(task) == []


def test_the_rule_names_the_path_the_run_downloads():
    from benchflow.task.ctrf_report import CTRF_REPORT_PATH
    from benchflow.task.paths import SandboxPaths

    assert str(SandboxPaths().verifier_dir / "ctrf.json") == CTRF_REPORT_PATH


# FXUSAGE's review of the two tasks (F2b, 2026-11-16 09:19Z).
# 1. `--ctrf` is read as a shell word: what ends it (`;`, `&&`, `|`, `)`) and
#    the quotes around it are no part of the path, so the standard path is
#    the standard path, not one the reader lets through.
@pytest.mark.parametrize(
    "test_sh",
    [
        "if python3 /tests/t.py --ctrf /logs/verifier/ctrf.json; then echo ok; fi\n",
        "pytest --ctrf /logs/verifier/ctrf.json&& echo ok\n",
        "pytest --ctrf /logs/verifier/ctrf.json || true\n",
        "pytest --ctrf /logs/verifier/ctrf.json| tee /tmp/log\n",
        '(pytest --ctrf "/logs/verifier/ctrf.json")\n',
        "pytest --ctrf '/logs/verifier/ctrf.json' /tests\n",
        "pytest --ctrf=/logs/verifier/ctrf.json;echo done\n",
    ],
    ids=[
        "semicolon-then",
        "and",
        "or",
        "pipe",
        "subshell-quoted",
        "single-quoted",
        "equals",
    ],
)
def test_the_standard_path_is_read_through_shell_punctuation(test_sh):
    verdict = _verdict(test_sh)
    assert (verdict.verdict, verdict.reason) == (
        "reaches",
        "a --ctrf path is the report path",
    )


# 2. A task.md package whose criteria are judged by tests is graded from
#    /logs/verifier/ctrf.json (taskmd verify). A report written anywhere else
#    failed every such criterion: a 0 the run never earned. The task.md plan
#    now refuses it, so bench check, the dry run and the run refuse it alike.
def _draft2(tmp_path: Path, test_sh: str | None, *, rubric: bool = True) -> Path:
    import shutil

    task = tmp_path / "hello"
    shutil.copytree(
        Path(__file__).parent / "fixtures" / "taskmd" / "examples" / "hello-world",
        task,
    )
    if test_sh is not None:
        (task / "verifier" / "test.sh").write_text(test_sh)
    if not rubric:
        (task / "verifier" / "rubric.json").unlink()
    return task


HELLO_ELSEWHERE = (
    "#!/bin/bash\n"
    "if printf 'Hello, world!' | cmp -s - /app/hello.txt; then status=passed; "
    "else status=failed; fi\n"
    'printf \'{"results": {"tool": {"name": "test.sh"}, "tests": '
    '[{"name": "test_hello", "status": "%s"}]}}\\n\' "$status" '
    "> /logs/verifier/report.json\n"
)
NEVER_WRITTEN = (
    "1 criteria are decided by tests, and test.sh never writes "
    "'/logs/verifier/ctrf.json', the report they are graded from"
)
OTHER_PATH = (
    "1 criteria are decided by tests, and test.sh uses non-standard CTRF path "
    "'/logs/verifier/report.json' (expected '/logs/verifier/ctrf.json') and never "
    "copies it there; a run reads the report only from that path"
)


def _plan_refusals(task: Path) -> list[str]:
    from benchflow.taskmd.materialize import parse_package
    from benchflow.taskmd.plan import plan_package

    return [f.detail for f in plan_package(parse_package(task), task).refused]


@pytest.mark.parametrize(
    "test_sh,issue",
    [
        (HELLO_ELSEWHERE, NEVER_WRITTEN),
        (
            "#!/bin/bash\npytest --ctrf /logs/verifier/report.json /verifier\n",
            OTHER_PATH,
        ),
    ],
    ids=["written-elsewhere", "other-ctrf-path"],
)
def test_test_judged_criteria_refuse_a_report_that_never_reaches_them(
    tmp_path, test_sh, issue
):
    task = _draft2(tmp_path, test_sh)
    assert _plan_refusals(task) == [issue]
    checked = runner.invoke(app, ["check", str(task)], terminal_width=400)
    assert checked.exit_code == 1
    assert issue in " ".join(checked.output.split())
    dry = runner.invoke(
        app, ["run", str(task), "--agent", "oracle", "--dry-run"], terminal_width=400
    )
    assert dry.exit_code != 0
    # Printed, or carried by the TaskMdError the run raises before any sandbox.
    said = dry.output + (str(dry.exception) if dry.exception else "")
    assert issue in " ".join(said.split())


@pytest.mark.parametrize(
    "test_sh",
    [
        None,  # the example as shipped: printf ... > /logs/verifier/ctrf.json
        "#!/bin/bash\npytest --ctrf /logs/verifier/ctrf.json /verifier\n",
        # A program the reader can't read may write the report: accepted.
        "#!/bin/bash\npython3 /verifier/grade.py\n",
        "#!/bin/bash\npytest /verifier\n",
    ],
    ids=["shipped", "standard-ctrf", "own-program", "pytest-config"],
)
def test_test_judged_criteria_accept_a_report_that_may_reach_them(tmp_path, test_sh):
    task = _draft2(tmp_path, test_sh)
    assert _plan_refusals(task) == []


def test_a_draft2_package_graded_by_its_script_warns_of_a_lost_ctrf_report(tmp_path):
    """No rubric: the script writes the reward and nothing is graded from the
    report, so it is a warning, as in every other such task (F2b, 2026-11-16 12:27Z)."""
    task = _draft2(
        tmp_path,
        "#!/bin/bash\npytest --ctrf /logs/verifier/report.json /verifier\n"
        "echo 1 > /logs/verifier/reward.txt\n",
        rubric=False,
    )
    assert _plan_refusals(task) == []
    lost = MISSING_ISSUE.replace("ctrf-report.json", "report.json")
    assert lost in check_task_warnings(task)
    checked = runner.invoke(app, ["check", str(task)], terminal_width=400)
    assert checked.exit_code == 0, checked.output
