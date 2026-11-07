"""Guards a3304fed and FF-sw-cli-run-min-C1 against user loss at 11d5ed5e."""

import asyncio
import inspect
import json

import click
import pytest
import typer.main
import yaml
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.eval_sharding import EvalShard, _config_payload
from benchflow.eval_worker import _evaluation_config, run_worker
from benchflow.evaluation import Evaluation, EvaluationConfig, EvaluationResult

OMITTED = "<omitted>"
CLI_ENTRY_POINTS = [
    (("eval", "run"), "--config"),
    (("eval", "run"), "--run-config"),
    (("eval", "create"), "--config"),
    (("eval", "create"), "--run-config"),
    (("run",), "--config"),
    (("run",), "--run-config"),
]
PYTHON_ENTRY_POINTS = [("eval_run", None), ("eval_create", None)]
ENTRY_POINTS = CLI_ENTRY_POINTS + PYTHON_ENTRY_POINTS
USER_CASES = [
    (None, "agent", "agent"),
    ("root", "grader-agent", "grader-agent"),
    ("custom", "agent", "agent"),
    (OMITTED, "grader-agent", "grader-agent"),
    *[("agent", value, None) for value in ("none", "NONE", "null", "NULL")],
    # root and uid 0 are the root agent, held as None from the config on
    # (one case for every later check); other spellings stay as given.
    *[("agent", value, None) for value in ("root", "0")],
    *[("agent", value, value) for value in ("0:0", "")],
    (OMITTED, OMITTED, "agent"),
    (None, OMITTED, None),
    ("none", OMITTED, None),
    ("NULL", OMITTED, None),
    ("root", OMITTED, None),
    ("", OMITTED, ""),
    ("custom", OMITTED, "custom"),
]


def test_every_public_file_entry_point_is_enumerated():
    """Guards FF-sw-cli-run-min-C1: new config aliases need the same coverage."""
    from benchflow.cli import main

    found = set()

    def visit(command, path=()):
        for param in command.params:
            if param.name == "config_file":
                found.update((path, option) for option in param.opts)
        if isinstance(command, click.Group):
            with click.Context(command) as ctx:
                for name in command.list_commands(ctx):
                    visit(command.get_command(ctx, name), (*path, name))

    visit(typer.main.get_command(app))
    assert found == set(CLI_ENTRY_POINTS)
    exported = {
        name
        for name in main.__all__
        if callable(value := getattr(main, name))
        and "config_file" in inspect.signature(value).parameters
    }
    assert exported == {name for name, _option in PYTHON_ENTRY_POINTS}


@pytest.mark.parametrize("command,option", ENTRY_POINTS)
@pytest.mark.parametrize("layout", ["native", "legacy"])
@pytest.mark.parametrize("saved,explicit,expected", USER_CASES)
def test_explicit_user_wins_before_execution(
    yaml_run, command, option, layout, saved, explicit, expected
):
    raw = {} if saved == OMITTED else {"sandbox_user": saved}
    args = [] if explicit == OMITTED else ["--sandbox-user", explicit]
    result, seen = yaml_run(raw, args, command, option, layout=layout)
    assert len(seen) == 1, result.output
    assert seen[0].sandbox_user == expected
    # The worker's JSON boundary must preserve the same effective user.
    shard = EvalShard(index=0, task_names=("task-a",), concurrency=1)
    payload = json.loads(json.dumps(_config_payload(seen[0], shard=shard)))
    assert _evaluation_config(payload).sandbox_user == expected


@pytest.mark.parametrize("command,option", PYTHON_ENTRY_POINTS)
@pytest.mark.parametrize("layout", ["native", "legacy"])
def test_python_explicit_none_overrides_yaml(yaml_run, command, option, layout):
    """Guards FF-sw-cli-run-min-C1: Python None explicitly opts into root."""
    result, seen = yaml_run(
        {"sandbox_user": "custom"},
        ["--sandbox-user", None],
        command,
        option,
        layout=layout,
    )
    assert len(seen) == 1, result.output
    assert seen[0].sandbox_user is None


@pytest.mark.parametrize("command,option", PYTHON_ENTRY_POINTS)
@pytest.mark.parametrize("explicit,expected", [(OMITTED, None), ("agent", "agent")])
def test_python_user_does_not_depend_on_ambient_click_context(
    yaml_run, command, option, explicit, expected
):
    """Guards FF-sw-cli-run-min-C1 even inside an unrelated Click command."""
    args = [] if explicit == OMITTED else ["--sandbox-user", explicit]
    with click.Context(click.Command("unrelated")) as ctx:
        ctx.params["sandbox_user"] = "unrelated-user"
        ctx.set_parameter_source("sandbox_user", click.core.ParameterSource.COMMANDLINE)
        result, seen = yaml_run({"sandbox_user": None}, args, command, option)
    assert len(seen) == 1, result.output
    assert seen[0].sandbox_user == expected


@pytest.mark.parametrize(
    "backend", ["docker", "remote-docker", "daytona", "modal", "apple-container"]
)
@pytest.mark.parametrize("command,option", ENTRY_POINTS)
@pytest.mark.parametrize("layout", ["native", "legacy"])
def test_non_root_override_on_every_backend(yaml_run, backend, command, option, layout):
    raw = {
        "sandbox_user": None,
        "environment": backend if layout == "native" else {"type": backend},
    }
    result, seen = yaml_run(
        raw,
        ["--sandbox-user", "agent"],
        command,
        option,
        layout=layout,
        model_agent=True,
    )
    assert len(seen) == 1, result.output
    assert seen[0].environment == backend
    assert seen[0].sandbox_user == "agent"


@pytest.mark.parametrize("command,option", ENTRY_POINTS)
@pytest.mark.parametrize("layout", ["native", "legacy"])
def test_override_precedes_resume_matching_and_recording(
    yaml_run, command, option, layout, tasks, tmp_path
):
    old = Evaluation(
        tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="old-root",
        config=EvaluationConfig(agent="nop", sandbox_user=None),
        preflight=False,
    )
    old._write_evaluation_record()

    def record(evaluation):
        evaluation._settle_auto_job_name()
        assert evaluation.job_dir != old.job_dir
        evaluation._write_evaluation_record()
        persisted = json.loads((evaluation.job_dir / "evaluation.json").read_text())
        assert persisted["config"]["sandbox_user"] == "agent"
        resumed = Evaluation.resume(evaluation.job_dir)
        assert resumed._config.sandbox_user == "agent"

    result, seen = yaml_run(
        {"sandbox_user": None},
        ["--sandbox-user", "agent"],
        command,
        option,
        layout=layout,
        on_run=record,
    )
    assert len(seen) == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert "fixture captured effective config" in result.output


@pytest.mark.parametrize("saved,explicit,expected", USER_CASES)
def test_resume_command_preserves_recorded_user(
    tasks, tmp_path, monkeypatch, saved, explicit, expected
):
    # Resume has no --sandbox-user flag: it loads the job's recorded config.
    raw = {} if saved == OMITTED else {"sandbox_user": saved}
    if explicit != OMITTED:
        raw["sandbox_user"] = explicit
    evaluation = Evaluation(
        tasks,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent="nop", **raw),
        preflight=False,
    )
    evaluation._write_evaluation_record()
    seen = []

    async def capture(self):
        seen.append(self._config.sandbox_user)
        raise ValueError("fixture captured resumed config")

    monkeypatch.setattr(Evaluation, "run", capture)
    result = CliRunner().invoke(app, ["eval", "resume", str(evaluation.job_dir)])
    assert seen == [expected], result.output


@pytest.mark.parametrize("command", [("eval", "run"), ("eval", "create")])
@pytest.mark.parametrize(
    "explicit,expected",
    [
        (OMITTED, "agent"),
        ("agent", "agent"),
        ("grader-agent", "grader-agent"),
        ("none", None),
        ("NULL", None),
        ("root", None),
        ("", ""),
    ],
)
def test_matrix_config_preserves_user(
    tasks, tmp_path, monkeypatch, command, explicit, expected
):
    from benchflow.cli import main

    path = tmp_path / "matrix.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "first": "dummy-a",
                    "second": {"model": "dummy-b", "agent": "nop"},
                }
            }
        )
    )
    seen = []

    def capture(plan, tasks_dir, config):
        seen.append(config.sandbox_user)
        shard = EvalShard(index=0, task_names=("task-a",), concurrency=1)
        assert (
            _evaluation_config(_config_payload(config, shard=shard)).sandbox_user
            == expected
        )
        return EvaluationResult(
            job_name="captured",
            config=config,
            total=1,
            passed=0,
            failed=1,
            errored=0,
            verifier_errored=0,
            results={},
            elapsed_sec=0,
        )

    monkeypatch.setattr(main, "run_batch_eval", capture)
    args = [] if explicit == OMITTED else ["--sandbox-user", explicit]
    result = CliRunner().invoke(
        app,
        [
            *command,
            "--tasks-dir",
            str(tasks),
            "--agent",
            "nop",
            "--matrix",
            str(path),
            "--trials",
            "2",
            "--jobs-dir",
            str(tmp_path / "matrix-jobs"),
            *args,
        ],
    )
    assert result.exit_code == 0, result.output or result.exception
    assert seen == [expected] * 4


@pytest.mark.parametrize("saved,explicit,expected", USER_CASES)
def test_worker_loads_user_from_json(
    tasks, tmp_path, monkeypatch, saved, explicit, expected
):
    raw = {} if saved == OMITTED else {"sandbox_user": saved}
    if explicit != OMITTED:
        raw["sandbox_user"] = explicit
    path = tmp_path / "worker.json"
    path.write_text(
        json.dumps(
            {
                "tasks_dir": str(tasks),
                "jobs_dir": str(tmp_path / "jobs"),
                "result_path": str(tmp_path / "result.json"),
                "config": {"agent": "nop", "environment": "daytona", **raw},
            }
        )
    )
    seen = []

    async def capture(self):
        seen.append(self._config.sandbox_user)
        raise ValueError("fixture captured worker config")

    monkeypatch.setattr(Evaluation, "run", capture)
    with pytest.raises(ValueError, match="fixture captured worker config"):
        asyncio.run(run_worker(path))
    assert seen == [expected]
