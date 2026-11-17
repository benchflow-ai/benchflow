"""The sandbox user setup is no longer fixed at the run's 120 s.

Guards the fix against 0.10.0rc3, where ``setup_sandbox_user`` got the
run's ``--sandbox-setup-timeout`` (120 s by default, a flag ``bench run
--help`` did not show): a task whose ``/app`` holds a large pnpm
``node_modules`` timed out handing the workspace to the agent's user, the
oracle included. The setup now takes the longer of the run's value and the
task's ``sandbox.setup_timeout_sec``, 600 s when the task names none.
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from benchflow.task.config import (
    DEFAULT_SANDBOX_USER_SETUP_SEC,
    SandboxConfig,
    TaskConfig,
    sandbox_user_setup_timeout_sec,
)
from benchflow.task.document import TaskDocument
from benchflow.task.imports import import_task_config_toml
from tests.test_dogfood_dx_b import _bench, _run_command, _task
from tests.test_trial_install_agent_timeout import _make_trial

TASK_MD = """---
schema_version: "1.3"
sandbox:
  setup_timeout_sec: 1800
---
Do it.
"""


def test_task_md_declares_the_setup_time(tmp_path):
    document = TaskDocument.from_text(TASK_MD, path=tmp_path / "task.md")
    assert document.config.sandbox.setup_timeout_sec == 1800


def test_task_toml_environment_declares_it_too():
    imported = import_task_config_toml(
        'version = "1.0"\n[environment]\nsetup_timeout_sec = 900\n', source="legacy"
    )
    assert imported.config.sandbox.setup_timeout_sec == 900


def test_undeclared_reads_as_none_and_is_not_written():
    config = TaskConfig.model_validate({})
    assert config.sandbox.setup_timeout_sec is None
    assert "setup_timeout_sec" not in config.model_dump(exclude_none=True)["sandbox"]


@pytest.mark.parametrize("value", [0, -5, math.inf, math.nan])
def test_a_setup_time_must_be_positive_and_finite(value):
    """Inf passed validation and raised OverflowError later (review)."""
    with pytest.raises(ValidationError):
        SandboxConfig.model_validate({"setup_timeout_sec": value})


@pytest.mark.parametrize(
    ("run", "declared", "expected"),
    [
        # The run named none: the task's, else 600.
        (None, None, DEFAULT_SANDBOX_USER_SETUP_SEC),
        (None, 1800, 1800),
        (None, 60, 60),
        (None, 900.5, 901),
        # The run's own value wins, lower or higher (review).
        (2400, 1800, 2400),
        (2400, None, 2400),
        (60, 1800, 60),
        (120, None, 120),
    ],
)
def test_the_runs_value_wins_then_the_tasks_then_600(run, declared, expected):
    sandbox = SandboxConfig.model_validate(
        {} if declared is None else {"setup_timeout_sec": declared}
    )
    assert sandbox_user_setup_timeout_sec(run, sandbox) == expected


def test_a_stand_in_without_the_field_reads_as_undeclared():
    assert sandbox_user_setup_timeout_sec(None, None) == DEFAULT_SANDBOX_USER_SETUP_SEC
    assert sandbox_user_setup_timeout_sec(None, object()) == 600


def _stub_setup(trial, monkeypatch) -> AsyncMock:
    setup = AsyncMock(return_value="/home/agent")
    for name in (
        "install_agent",
        "write_credential_files",
        "upload_subscription_auth",
        "snapshot_build_config",
        "seed_verifier_workspace",
        "deploy_skills",
        "lockdown_paths",
    ):
        monkeypatch.setattr(trial._planes, name, AsyncMock())
    monkeypatch.setattr(trial._planes, "setup_sandbox_user", setup)
    return setup


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["oracle", "claude-agent-acp"])
@pytest.mark.parametrize(
    ("run", "named", "declared", "expected"),
    [
        (120, False, 1800, 1800),
        (120, False, None, 600),
        # A named run value lowers the limit too (review).
        (60, True, 1800, 60),
        (120, True, None, 120),
        (2400, True, 1800, 2400),
    ],
)
async def test_the_rollout_gives_the_setup_the_runs_then_the_tasks_time(
    tmp_path, monkeypatch, agent, run, named, declared, expected
):
    trial = _make_trial(tmp_path, agent=agent, sandbox_setup_timeout=run)
    assert trial._config.sandbox_setup_timeout_given is (run != 120)
    trial._config.sandbox_setup_timeout_given = named
    trial._task.config.sandbox = SandboxConfig.model_validate(
        {} if declared is None else {"setup_timeout_sec": declared}
    )
    setup = _stub_setup(trial, monkeypatch)
    await trial.install_agent()
    setup.assert_awaited_once()
    assert setup.await_args.kwargs["timeout_sec"] == expected
    # The run's own setting is unchanged: the agent install still gets it.
    assert trial._config.sandbox_setup_timeout == run


def test_bench_run_help_shows_the_flag():
    out = _bench("run", "--help", env={"COLUMNS": "200"}).output
    assert "--sandbox-setup-timeout" in out
    (param,) = [p for p in _run_command().params if p.name == "sandbox_setup_timeout"]
    assert param.hidden is False
    assert "setup_timeout_sec" in param.help


def test_bench_run_passes_the_flag_to_the_run(monkeypatch, tmp_path):
    """``bench run --sandbox-setup-timeout`` reaches ``bench eval run``."""
    from benchflow.cli import main

    calls: list[dict] = []
    monkeypatch.setattr(main, "eval_run", lambda **kw: calls.append(kw))
    task = _task(tmp_path)
    result = _bench("run", str(task), "--sandbox-setup-timeout", "1800")
    assert result.exit_code == 0, result.output
    (call,) = calls
    assert call["sandbox_setup_timeout"] == 1800


def test_a_run_config_file_sets_it(tmp_path):
    from benchflow.evaluation import Evaluation

    (tmp_path / "tasks").mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("tasks_dir: tasks\nsandbox_setup_timeout: 1800\n")
    assert Evaluation.from_yaml(config)._config.sandbox_setup_timeout == 1800


# --- a saved config names the value only when the run did (re-review) --

_NAMED = [
    ({}, None),
    ({"sandbox_setup_timeout": 300}, 300),
    ({"sandbox_setup_timeout": 120, "sandbox_setup_timeout_given": True}, 120),
]


@pytest.mark.parametrize(("given", "named"), _NAMED)
def test_a_saved_rollout_config_names_only_what_the_run_named(tmp_path, given, named):
    """rollout_config_to_dict wrote the 120 s default; read back, it counted
    as named and won over every task's sandbox.setup_timeout_sec."""
    from pathlib import Path

    import benchflow as bf
    from benchflow.task.config import run_sandbox_setup_timeout

    task = Path(__file__).parent / "examples" / "hello-world-task"
    config = bf.RolloutConfig(task_path=task, agent="oracle", **given)
    saved = config.to_dict()
    assert ("sandbox_setup_timeout" in saved) is (named is not None)
    assert "verifier_setup_timeout" not in saved
    back = bf.RolloutConfig.from_yaml(config.to_yaml(tmp_path / "rollout.yaml"))
    assert run_sandbox_setup_timeout(back) == named


@pytest.mark.parametrize(("given", "named"), _NAMED)
def test_a_saved_job_config_names_only_what_the_job_named(tmp_path, given, named):
    """Evaluation.to_dict, read back by Evaluation.from_yaml: the same."""
    from pathlib import Path

    import benchflow as bf

    tasks = Path(__file__).parent / "examples"
    job = bf.Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        config=bf.EvaluationConfig(agent="oracle", **given),
    )
    saved = job.to_dict()
    assert ("sandbox_setup_timeout" in saved) is (named is not None)
    assert "verifier_setup_timeout" not in saved
    back = bf.Evaluation.from_yaml(job.to_yaml(tmp_path / "job.yaml"))
    assert back._config.sandbox_setup_timeout_given is (named is not None)
    assert back._config.sandbox_setup_timeout == (named or 120)


def test_the_field_says_the_runs_value_wins_either_way():
    """The field's text (and docs/reference/files.md made from it) said the
    run's value counted only when longer; a named run value wins either way."""
    from benchflow.task.config import SandboxConfig

    text = SandboxConfig.model_fields["setup_timeout_sec"].description
    assert "when longer" not in text
    assert "--sandbox-setup-timeout, when named, wins over it" in text
