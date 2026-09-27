"""Job configs round-trip between YAML/JSON and Python.

``bench eval run --config job.yaml`` and ``Evaluation.from_yaml`` read the
native YAML, but nothing wrote it: a job built in Python could not be saved
for the CLI, and the native schema could not carry a full retry policy,
automatic checkpoints or an inline environment manifest. ``Evaluation`` gains
``to_dict``/``to_yaml``/``from_dict`` over that schema (extended additively),
and ``RolloutConfig`` gains the same over the rollout YAML schema. agent_env
values are written only when asked for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import benchflow as bf
from benchflow.evaluation import _config_to_record

TASK = Path(__file__).parent / "examples" / "hello-world-task"
SECRET = "sk-test-not-a-real-key-1234567890"


def _evaluation(tmp_path: Path) -> bf.Evaluation:
    manifest = bf.EnvironmentManifest.model_validate(
        {"name": "demo", "image": "python:3.12-slim"}
    )
    config = bf.EvaluationConfig(
        agent="claude-agent-acp",
        model="claude-haiku-4-5",
        reasoning_effort="high",
        environment="daytona",
        concurrency=7,
        build_concurrency=2,
        prompts=["one", None],
        agent_env={"OPENAI_API_KEY": SECRET},
        retry=bf.RetryConfig(
            max_retries=5, min_wait_sec=3.0, exclude_categories={"timeout"}
        ),
        reviewer=bf.ReviewerConfig(agent="opencode", model="m", concurrency=2),
        codex_apps_policy="inherit",
        sandbox_user=None,
        sandbox_locked_paths=["/x"],
        sandbox_setup_timeout=300,
        skip_agent_install=True,
        agent_idle_timeout=900,
        base_image_override="python:3.12",
        include_tasks={"a", "b"},
        exclude_tasks={"c"},
        self_gen_no_internet=True,
        environment_manifest=manifest,
        config_override={"agent": {"timeout_sec": 10}},
        loop_strategy="verify-retry:k=3",
        checkpoints="every-prompt",
        checkpoint_keep=2,
        freeze_workspace=True,
        retry_from_checkpoint="on-failure",
        retry_prompt="Try again from here.",
    )
    return bf.Evaluation(
        tasks_dir=TASK.parent, jobs_dir=tmp_path / "jobs", config=config
    )


def _same(a: bf.Evaluation, b: bf.Evaluation) -> None:
    ra, rb = _config_to_record(a._config), _config_to_record(b._config)
    assert ra == rb
    assert a._tasks_dir == b._tasks_dir and Path(a._jobs_dir) == Path(b._jobs_dir)


def test_yaml_round_trip_is_lossless_with_agent_env(tmp_path: Path) -> None:
    ev = _evaluation(tmp_path)
    path = ev.to_yaml(tmp_path / "job.yaml", include_agent_env=True)
    back = bf.Evaluation.from_yaml(path)
    _same(ev, back)
    assert back._config.agent_env == {"OPENAI_API_KEY": SECRET}


def test_dict_round_trip_through_json(tmp_path: Path) -> None:
    ev = _evaluation(tmp_path)
    raw = json.loads(json.dumps(ev.to_dict(include_agent_env=True)))
    _same(ev, bf.Evaluation.from_dict(raw))


def test_agent_env_values_are_left_out_by_default(tmp_path: Path) -> None:
    ev = _evaluation(tmp_path)
    text = ev.to_yaml(tmp_path / "job.yaml").read_text()
    assert SECRET not in text
    assert "OPENAI_API_KEY" in text  # named so the reader knows to pass it
    assert bf.Evaluation.from_yaml(tmp_path / "job.yaml")._config.agent_env == {}


def test_the_cli_reads_what_python_writes(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from benchflow.cli import main as cli_main

    seen = {}

    async def fake_run(self):
        seen["config"] = self._config
        raise SystemExit(0)

    monkeypatch.setattr(bf.Evaluation, "run", fake_run)
    path = _evaluation(tmp_path).to_yaml(tmp_path / "job.yaml", include_agent_env=True)
    CliRunner().invoke(cli_main.app, ["eval", "run", "--config", str(path)])
    assert seen["config"].concurrency == 7
    assert seen["config"].retry.max_retries == 5
    assert seen["config"].checkpoints == "every-prompt"
    assert seen["config"].freeze_workspace is True


def test_rollout_config_round_trip(tmp_path: Path) -> None:
    config = bf.RolloutConfig(
        task_path=TASK,
        agent="claude-agent-acp",
        model="claude-haiku-4-5",
        prompts=["first", None],
        environment="daytona",
        jobs_dir=tmp_path / "jobs",
        job_name="j",
        rollout_name="r",
        timeout=123,
        sandbox_setup_timeout=200,
        agent_env={"K": SECRET},
    )
    path = config.to_yaml(tmp_path / "rollout.yaml", include_agent_env=True)
    back = bf.RolloutConfig.from_yaml(path)
    for field in (
        "task_path",
        "agent",
        "model",
        "environment",
        "job_name",
        "rollout_name",
        "timeout",
        "sandbox_setup_timeout",
        "agent_env",
    ):
        assert getattr(back, field) == getattr(config, field), field
    assert Path(back.jobs_dir) == Path(config.jobs_dir)
    assert [t.prompt for t in back.scenes[0].turns] == ["first", None]
    assert SECRET not in yaml.safe_dump(config.to_dict())


def test_rollout_config_refuses_what_cannot_be_written(tmp_path: Path) -> None:
    config = bf.RolloutConfig(
        task_path=TASK, agent="oracle", user=bf.FunctionUser(lambda *a: None)
    )
    with pytest.raises(ValueError, match="user"):
        config.to_dict()


def test_agent_env_keys_survive_a_second_save(tmp_path: Path) -> None:
    """After from_yaml, a second save forgot MY_SECRET was needed."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    ev = bf.Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        config=bf.EvaluationConfig(agent="oracle", agent_env={"MY_SECRET": "s3cr3t"}),
    )
    first = ev.to_yaml(tmp_path / "a.yaml")
    assert "s3cr3t" not in first.read_text()
    loaded = bf.Evaluation.from_yaml(first)
    assert loaded.to_dict()["agent_env_keys"] == ["MY_SECRET"]
    second = bf.Evaluation.from_yaml(loaded.to_yaml(tmp_path / "b.yaml"))
    assert second.to_dict()["agent_env_keys"] == ["MY_SECRET"]
