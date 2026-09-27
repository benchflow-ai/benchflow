"""Issue #1107: explicit Apps policy survives public evaluation entry points."""

import inspect
import json
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from typer.testing import CliRunner

import benchflow.cli.main as main
from benchflow._utils.yaml_loader import rollout_config_from_dict
from benchflow.cli.eval_artifacts import _redacted_eval_config
from benchflow.eval_plan import EvalCreateRequest, EvalPlanError, build_eval_plan
from benchflow.eval_sharding import EvalShard, _config_payload
from benchflow.eval_worker import _evaluation_config
from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.models import RunResult
from benchflow.rollout import RolloutConfig
from benchflow.rollout._results import _write_config
from benchflow.rollout.task_runtime import TaskRuntimeConfig
from benchflow.runtime import RuntimeConfig
from benchflow.sdk import SDK
from benchflow.skill_policy import resolve_task_skill_policy


@pytest.fixture
def task(tmp_path):
    dest = tmp_path / "task"
    shutil.copytree(Path(__file__).parent / "examples/hello-world-task", dest)
    return dest


@pytest.mark.parametrize("policy", [None, "disabled", "inherit"])
def test_yaml_plan_and_worker_keep_policy(policy, task, tmp_path):
    """Issue #1107: a sharded or YAML job must not silently lose opt-in."""
    cfg = EvaluationConfig(codex_apps_policy=policy)
    payload = _config_payload(cfg, shard=EvalShard(0, (task.name,), 1))
    assert _evaluation_config(payload).codex_apps_policy == policy
    rollout = rollout_config_from_dict(
        {"agent": "codex", "codex_apps_policy": policy}, task_path=task
    )
    assert rollout.codex_apps_policy == policy
    plan = build_eval_plan(EvalCreateRequest(tasks_dir=task, codex_apps_policy=policy))
    assert plan.make_eval_config().codex_apps_policy == policy
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {"tasks_dir": str(task), "agent": "codex", "codex_apps_policy": policy}
        )
    )
    assert Evaluation.from_yaml(path)._config.codex_apps_policy == policy


@pytest.mark.asyncio
async def test_sdk_and_evaluation_forward_explicit_inherit(task, tmp_path, monkeypatch):
    """Issue #1107: test the actual Rollout.create boundary, without a sandbox."""
    seen = []

    async def create(config):
        seen.append(config)
        rollout = AsyncMock()
        rollout.run.return_value = RunResult(task_name=task.name, rewards={"reward": 1})
        return rollout

    monkeypatch.setattr("benchflow.rollout.Rollout.create", create)
    await SDK().run(task, agent="codex", codex_apps_policy="inherit")
    cfg = EvaluationConfig(agent="codex", codex_apps_policy="inherit")
    evaluation = Evaluation(tasks_dir=task, jobs_dir=tmp_path / "jobs", config=cfg)
    await evaluation._run_single_task(task, cfg)
    assert len(seen) == 2
    assert all(c.codex_apps_policy == "inherit" for c in seen)


def test_cli_forwards_explicit_policy_before_execution(task, monkeypatch):
    """Issue #1107: the supported CLI must expose the operator opt-in."""
    seen = []

    def capture(request):
        seen.append(request)
        raise EvalPlanError("fixture stops before execution")

    monkeypatch.setattr(main, "build_eval_plan", capture)
    result = CliRunner().invoke(
        main.app,
        ["eval", "run", "--tasks-dir", str(task), "--codex-apps-policy", "inherit"],
    )
    assert result.exit_code == 1
    assert len(seen) == 1
    assert seen[0].codex_apps_policy == "inherit"


def test_invalid_policy_fails_during_configuration(task):
    """Issue #1107: typos must fail before environments or agent credentials."""
    with pytest.raises(ValueError, match="codex_apps_policy"):
        EvaluationConfig(codex_apps_policy="enabled")
    with pytest.raises(EvalPlanError, match="codex-apps-policy"):
        build_eval_plan(EvalCreateRequest(tasks_dir=task, codex_apps_policy="enabled"))


@pytest.mark.parametrize(
    ("yaml_policy", "cli_policy", "expected"),
    [
        ("inherit", None, "inherit"),
        ("disabled", "inherit", "inherit"),
        ("inherit", "disabled", "disabled"),
    ],
)
def test_cli_override_precedence_over_yaml(
    yaml_policy, cli_policy, expected, task, tmp_path, monkeypatch
):
    """Issue #1107: absence preserves YAML; explicit CLI choice takes precedence."""
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tasks_dir": str(task),
                "jobs_dir": str(tmp_path / "jobs"),
                "agent": "codex",
                "codex_apps_policy": yaml_policy,
            }
        )
    )
    seen = []

    async def stop_before_sandbox(self):
        seen.append(self._config.codex_apps_policy)
        raise ValueError("fixture stops before execution")

    monkeypatch.setattr(Evaluation, "run", stop_before_sandbox)
    args = ["eval", "run", "--run-config", str(path)]
    if cli_policy:
        args.extend(["--codex-apps-policy", cli_policy])
    result = CliRunner().invoke(main.app, args)
    assert result.exit_code == 1
    assert seen == [expected], result.output


def test_hosted_runner_does_not_silently_ignore_explicit_policy():
    """Issue #1107: hosted environments cannot claim this local ACP control."""
    with pytest.raises(EvalPlanError, match="not supported for hosted"):
        build_eval_plan(
            EvalCreateRequest(source_env="fixture", codex_apps_policy="disabled")
        )


def test_existing_positional_runtime_constructors_keep_their_meaning(task):
    """Issue #1107: adding policy must not reinterpret existing positional callers."""
    config = RuntimeConfig("agent", 120)
    assert config.sandbox_user == "agent" and config.sandbox_setup_timeout == 120
    config = TaskRuntimeConfig(task, "docker", "agent")
    assert config.sandbox_user == "agent" and config.environment == "docker"
    for cls in (
        RuntimeConfig,
        TaskRuntimeConfig,
        RolloutConfig,
        EvaluationConfig,
        EvalCreateRequest,
    ):
        assert (
            inspect.signature(cls).parameters["codex_apps_policy"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )


def test_policy_survives_saved_configs(task, tmp_path):
    """Issue #1107: reusable run artifacts must retain the operator's explicit choice."""
    config = EvaluationConfig(codex_apps_policy="inherit")
    assert _redacted_eval_config(config)["codex_apps_policy"] == "inherit"
    _write_config(
        tmp_path,
        task_path=task,
        agent="codex-acp",
        model=None,
        environment="docker",
        skill_policy=resolve_task_skill_policy(
            task_path=task,
            skill_mode="no-skill",
            runtime_skills_dir=None,
            declared_sandbox_skills_dir=None,
        ),
        sandbox_user="agent",
        context_root=None,
        timeout=300,
        started_at=datetime.now(),
        agent_env={},
        codex_apps_policy="inherit",
    )
    assert (
        json.loads((tmp_path / "config.json").read_text())["codex_apps_policy"]
        == "inherit"
    )
