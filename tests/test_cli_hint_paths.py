"""Guard D2b-CONF7-1/2 against the remaining hint siblings at ea579ca0.

Only dummy shell functions run: no agent, credentials or sandbox is needed.
The shell's argv, not merely the appearance of quotes, is the contract.
"""

from __future__ import annotations

import html
import io
import json
import re
import subprocess
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from benchflow import branch_run, failures
from benchflow.cli import _shared as shared
from benchflow.cli.main import app
from benchflow.evaluation import Evaluation, EvaluationConfig, EvaluationResult
from benchflow.sandbox.providers import SANDBOX_PROVIDERS

PATHS = [
    "plain",
    ":smile:",
    "[bold]with spaces[/bold]",
    "run; printf D2B_HINT;#",
    "$(printf D2B_HINT)",
    "`printf D2B_HINT`",
    "a'\"b",
    "a\\b",
    "*?[ab]",
    "~name",
    "-option",
    "{task}",
    "{job}",
    "a&b|c>d<e(f)",
    "é你好😀",
    "hf:local",
    "hf:",
    "hf:/org/dataset",
    "hf://org/dataset",
    "./hf:local",
    "http://local/path",
    "run  a",
    "  leading and trailing  ",
]
BREAKS = [
    "\n",
    "\r",
    "\v",
    "\f",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    "\u2028",
    "\u2029",
    "\t",
    "\x1b",
    "\x7f",
]
HELLO = Path(__file__).parent / "examples" / "hello-world-task"


@pytest.fixture(autouse=True)
def placeholder_credential(monkeypatch):
    """FF-test-env b29: guard the 393 hint-path failures seen at 7a2387ac."""
    monkeypatch.setenv("FF_TEST_ENV_TOKEN", "x")


def _argv(command: str) -> list[str]:
    result = subprocess.run(
        ["bash", "-c", 'bench() { printf "%s\\0" "$@"; }\n' + command],
        capture_output=True,
        text=True,
        check=True,
    )
    assert not result.stderr
    assert result.stdout.endswith("\0"), result.stdout
    return result.stdout[:-1].split("\0")


def _path(value: Path | str) -> str:
    text = str(value)
    return "./" + text if text.startswith("-") else text


def _assert_local_view(argv, expected, monkeypatch):
    """Guards CONF8-1 at e9bcbd89 through both CLI verbs and the real parser."""
    from benchflow.trajectories.viewer import server

    assert argv[:1] == ["view"] and len(argv) == 2
    served = []

    def single(path, *args, **kwargs):
        served.append(path)

    def remote(source):
        pytest.fail(f"local hint tried to fetch {source}")

    monkeypatch.setattr(server, "_serve_single", single)
    monkeypatch.setattr(server, "_serve_browse", single)
    monkeypatch.setattr(server, "resolve_hf_dataset", remote)
    for verb in (["eval", "view"], ["view"]):
        result = CliRunner().invoke(app, [*verb, argv[1]])
        assert result.exit_code == 0, result.output
    assert served == [Path(expected), Path(expected)]


@pytest.mark.parametrize("folder", PATHS)
@pytest.mark.parametrize("verb", [["tasks", "init"], ["new"]])
def test_authoring_hint_paths(tmp_path, monkeypatch, folder, verb):
    """Guards CONF7-1/2 in both task-authoring spellings and both commands."""
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, [*verb, "demo", "--dir", folder], terminal_width=40
    )
    assert result.exit_code == 0, result.output
    commands = [
        line for line in result.output.splitlines() if line.startswith("bench ")
    ]
    assert len(commands) == 2
    assert _argv(commands[0]) == ["check", _path(Path(folder) / "demo")]
    assert _argv(commands[1]) == [
        "run",
        _path(Path(folder) / "demo"),
        "--sandbox",
        "docker",
        "--jobs-dir",
        "jobs/demo-oracle",
        "--fresh",
    ]


@pytest.mark.parametrize("folder", PATHS)
def test_summary_view_recovery_and_usage_hint_paths(tmp_path, monkeypatch, folder):
    """Guards CONF7-1/2 through the shared run/resume/continue summary sink."""
    monkeypatch.chdir(tmp_path)
    job = Path(folder)
    job.mkdir(parents=True)
    (job / "summary.json").write_text(
        json.dumps({"usage_limit": {"not_started": ["x"]}})
    )
    result = EvaluationResult(
        job_name="one",
        config=EvaluationConfig(),
        total=1,
        errored=1,
        results={"task": {"error": "D2B_DUMMY", "error_category": "api_error"}},
    )
    output = io.StringIO()
    monkeypatch.setattr(shared, "console", Console(file=output, width=40))
    shared._report_eval_result(result, job)
    lines = output.getvalue().splitlines()
    view = next(
        line.split("bench view ", 1)[1] for line in lines if "bench view " in line
    )
    _assert_local_view(_argv("bench view " + view), job, monkeypatch)
    recovery = next(
        line.strip().removeprefix("next: ")
        for line in lines
        if "next: `bench eval resume " in line
    )
    command = recovery.removeprefix("`").removesuffix(
        "` reruns it; api_error_info in result.json has the status"
    )
    assert _argv(command) == ["eval", "resume", _path(job)]
    usage = next(
        line.split("`bench eval resume ", 1)[1]
        for line in lines
        if "not started:" in line
    )
    command = usage.removesuffix("` runs them on another login or after the reset")
    assert _argv("bench eval resume " + command) == ["eval", "resume", _path(job)]


@pytest.mark.parametrize("folder", PATHS)
@pytest.mark.parametrize("sandbox", SANDBOX_PROVIDERS)
@pytest.mark.parametrize("verb", [["eval", "branch"], ["branch"]])
def test_branch_view_hint_paths(tmp_path, monkeypatch, folder, sandbox, verb):
    """Guards CONF7-1/2 for every registered backend and both branch verbs."""
    monkeypatch.chdir(tmp_path)

    async def trial(plan, task, **kwargs):
        return branch_run.BranchTrialOutcome(task=task.name, fork_status="completed")

    monkeypatch.setattr(branch_run, "run_branch_trial", trial)
    result = CliRunner().invoke(
        app,
        [
            *verb,
            "--tasks-dir",
            str(HELLO),
            "--agent",
            "oracle",
            "--sandbox",
            sandbox,
            "--child",
            "label=a",
            "--child",
            "label=b",
            "--jobs-dir",
            folder,
            "--job-name",
            "one",
        ],
        terminal_width=40,
    )
    assert result.exit_code == 0, result.output
    command = next(
        line.removeprefix("Next: ")
        for line in result.output.splitlines()
        if line.startswith("Next: ")
    )
    _assert_local_view(_argv(command), Path(folder) / "one", monkeypatch)


TEMPLATES = [
    step
    for table in (
        failures._ERRORED,
        failures._UNSCORED,
        failures._SPECIAL,
        failures._INTEGRATION,
    )
    for _, _, step in table.values()
    if "{job}" in step or "{task}" in step
]


@pytest.mark.parametrize("folder", PATHS)
@pytest.mark.parametrize("template", TEMPLATES)
def test_every_recovery_template_quotes_once(folder, template):
    """Guards CONF7-2: substitutions are literal, even if they contain {task}."""
    filled = failures._fill(template, job=folder, task=folder)
    marker = "{job}" if "{job}" in template else "{task}"
    before, after = template.split(marker)
    assert filled.startswith(before) and filled.endswith(after)
    word = filled[len(before) : len(filled) - len(after) if after else None]
    assert _argv("bench " + word) == [_path(folder)]


@pytest.mark.parametrize("separator", BREAKS)
def test_non_displayable_hint_paths_are_placeholders(separator, tmp_path, monkeypatch):
    """Guards CONF7-2: no multiline command is silently changed to a new path."""
    folder = "run" + separator + "next: D2B_HINT"
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["tasks", "init", "demo", "--dir", folder])
    assert result.exit_code == 0, result.output
    assert "bench check <task folder>" in result.output
    for marker in ("job", "task"):
        placeholder = "<job dir>" if marker == "job" else "<task>"
        assert (
            failures._fill("{" + marker + "}", job=folder, task=folder) == placeholder
        )


@pytest.mark.parametrize("folder", PATHS)
def test_legacy_viewer_hint_is_shell_and_html_literal(tmp_path, monkeypatch, folder):
    """Guards CONF7-2's HTML sibling; the command names an actual rollout."""
    from benchflow.trajectories.viewer.legacy import render_rollout

    monkeypatch.chdir(tmp_path)
    job = Path(folder)
    rollout = job / "task__123"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "trajectory" / "acp_trajectory.jsonl").touch()
    output = render_rollout(job)
    command = html.unescape(
        re.search(r"<code(?: [^>]*)?>(bench view .*?)</code>", output).group(1)
    )
    _assert_local_view(_argv(command), rollout, monkeypatch)


@pytest.mark.parametrize("folder", PATHS)
def test_evaluation_diagnostic_hint_paths(tmp_path, monkeypatch, caplog, folder):
    """Guards CONF7-2 in empty-selection errors and auto-resume warnings."""
    from benchflow._utils import task_authoring

    monkeypatch.chdir(tmp_path)
    task = Path(folder) / "task"
    task.mkdir(parents=True)
    (task / "task.md").touch()
    monkeypatch.setattr(
        task_authoring, "runnable_task_issues", lambda path, **_: ["D2B_DUMMY"]
    )
    evaluation = Evaluation(tasks_dir=task, jobs_dir=Path("jobs"), job_name="one")
    assert "D2B_DUMMY" in evaluation._empty_selection_message()
    # One named task: the command is the next step (DF-p1-first-run-18).
    hint = evaluation._empty_selection_hint()
    command = hint.removeprefix("`").removesuffix("` explains each issue")
    # Guards 865d7a6: diagnostic paths retain the construction-time task root.
    assert _argv(command) == ["check", _path(task.absolute())]
    latest = Path(folder) / "latest"
    latest.mkdir()
    # The keys every job record has (a record without them marks no job).
    (latest / "evaluation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tasks_dir": str(task),
                "job_name": "latest",
                "config": {"agent": "oracle", "model": None},
            }
        )
    )
    Evaluation._resolve_job_name(Path(folder), EvaluationConfig(agent="nop"))
    message = next(
        record.message
        for record in caplog.records
        if "bench eval resume " in record.message
    )
    command = message.split("bench eval resume ", 1)[1].removesuffix(
        " finishes it with its own settings."
    )
    assert _argv("bench eval resume " + command) == ["eval", "resume", _path(latest)]


@pytest.mark.parametrize("folder", PATHS)
@pytest.mark.parametrize("stderr", [False, True])
def test_doctor_python_hint_path(tmp_path, monkeypatch, folder, stderr):
    """Guards CONF7-1/2 when doctor recommends this Python executable."""
    from benchflow.cli.doctor import _print_checks, render_report
    from benchflow.doctor import DoctorReport
    from tests.test_doctor_bench_path import Machine, _line

    machine = Machine(tmp_path)
    machine.add_rust()
    executable = tmp_path / folder / "python"
    check = _line(
        machine,
        [machine.rust_dir],
        entry=Path("/tmp/dummy/__main__.py"),
        python=executable,
    )
    output = io.StringIO()
    monkeypatch.setattr(shared, "console", Console(file=output, width=40))
    monkeypatch.setattr(shared, "err_console", Console(file=output, width=40))
    if stderr:
        _print_checks([check])
    else:
        render_report(DoctorReport(checks=[check], sandbox="docker"))
    line = next(line for line in output.getvalue().splitlines() if "run it as " in line)
    command = line.split("run it as ", 1)[1].split(": https://", 1)[0]
    assert _argv("bench " + command) == [str(executable), "-m", "benchflow"]


@pytest.mark.parametrize(
    "control", [*range(32), *range(127, 160), 0x200B, 0x2028, 0x2029, 0x202E]
)
def test_every_control_is_a_placeholder(control):
    """Guards CONF7-2 for every ASCII/C1 control and invisible format controls."""
    assert shared.command_arg("a" + chr(control) + "b", "<path>") == "<path>"


@pytest.mark.parametrize(
    "folder", [*PATHS, ".", "../hf:local", "/tmp/dummy/hf:local", "//tmp/dummy"]
)
def test_local_path_argument_keeps_source_namespace(folder):
    """Guards CONF8-1: string URI spellings must also stay local paths."""
    from benchflow._utils.command_hints import local_path_arg
    from benchflow.trajectories.viewer.sources import LocalPathSource, parse_source

    words = _argv("bench " + local_path_arg(folder, "<path>"))
    assert len(words) == 1
    assert parse_source(words[0]) == LocalPathSource(Path(folder))


@pytest.mark.parametrize("folder", [None, "", *("a" + c + "b" for c in BREAKS)])
def test_local_path_argument_keeps_placeholders(folder):
    """Guards CONF8-1: adding a local anchor must not hide invalid paths."""
    from benchflow._utils.command_hints import local_path_arg

    assert local_path_arg(folder, "<path>") == "<path>"


@pytest.mark.parametrize(
    "folder", [":smile: [bold]x[/bold]", "x; printf D2B_HINT;#", "x\u2028y"]
)
async def test_verifier_refusal_hint_path(tmp_path, monkeypatch, folder, caplog):
    """A verifier-writable refusal does not justify a task-blaming command."""
    from tests import test_verifier_uv_state as rig

    make_task = rig.make_task
    task_path = Path(folder)

    def task(layout, **kwargs):
        result = make_task(layout, **kwargs)
        result.paths.task_dir = task_path
        return result

    def installed(env, layout):
        # ``pip install --target``: not a site directory, so the guard refuses
        # the verifier's own install with the task diagnostic read below (a
        # fresh venv the agent had no hand in loads).
        site = layout.workspace / "verifier-plugins"
        rig.install_plugin(site, rig.CTRF_PLUGIN)
        return [site]

    monkeypatch.setattr(rig, "make_task", task)
    _, _, rewards, error = await rig.score(
        tmp_path, monkeypatch, rig.PASSING, installed, published=False
    )
    assert rewards == {"reward": 0.0} and error is None
    warning = next(
        message
        for message in caplog.messages
        if "plugin guard not confirmed" in message
    )
    assert folder not in warning and "bench check" not in warning
    assert "code run by test.sh can also print the diagnostic" in warning


@pytest.mark.parametrize("folder", PATHS)
@pytest.mark.parametrize("verb", ["run", "create", "inspect", "compare"])
def test_migration_hint_path(folder, verb):
    """Guards CONF7-2 in the runnable command suggested for each old spelling."""
    from benchflow.cli.verbs import note_for

    if verb in ("run", "create"):
        words = [
            "eval",
            verb,
            "--tasks-dir",
            folder,
            "--jobs-dir",
            folder,
            "--agent",
            "oracle",
        ]
        expected = [
            "run",
            _path(folder),
            "--jobs-dir",
            _path(folder),
            "--agent",
            "oracle",
        ]
    else:
        words = ["eval", verb, "--", folder]
        expected = ["results", "--", _path(folder)]
    note = note_for(words)
    assert note is not None
    assert _argv(note.split(": ", 1)[1]) == expected


@pytest.mark.parametrize("separator", BREAKS)
def test_migration_hint_does_not_rewrite_line_breaks(separator):
    """Guards CONF7-2 when a translated command names an undisplayable path."""
    from benchflow.cli.verbs import note_for

    folder = "x" + separator + "y"
    assert "bench run <task folder> --jobs-dir <path>" in note_for(
        ["eval", "run", "--tasks-dir", folder, "--jobs-dir", folder]
    )
    assert "bench results <job dir>" in note_for(["eval", "inspect", folder])
