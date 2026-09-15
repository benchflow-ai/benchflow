"""Automatic-review configuration boundaries and credential isolation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from benchflow._utils.yaml_loader import rollout_config_from_dict
from benchflow.cli.main import app
from benchflow.eval_sharding import EvalShard, _config_payload, _worker_payload_artifact
from benchflow.eval_worker import _evaluation_config
from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.review.options import ReviewerConfig
from benchflow.rollout import RolloutConfig


def test_reviewer_worker_credentials_round_trip_without_artifact_leak() -> None:
    """The automatic-review feature preserves private worker auth, not public secrets."""
    reviewer = ReviewerConfig(
        agent="opencode",
        model="azure/reviewer-deployment",
        environment="daytona",
        reasoning_effort="xhigh",
        concurrency=2,
        timeout_sec=900,
        agent_env={"AZURE_API_KEY": "reviewer-private-key"},
    )
    config = EvaluationConfig(
        reviewer=reviewer, agent_env={"AZURE_API_KEY": "solver-key"}
    )
    payload = _config_payload(
        config, shard=EvalShard(index=0, task_names=("a",), concurrency=1)
    )
    restored = _evaluation_config(json.loads(json.dumps(payload)))
    assert restored.reviewer == reviewer
    assert restored.agent_env["AZURE_API_KEY"] == "solver-key"
    public = _worker_payload_artifact({"config": payload})
    assert "reviewer-private-key" not in json.dumps(public)
    assert "solver-key" not in json.dumps(public)
    assert public["config"]["reviewer"]["agent_env_keys"] == ["AZURE_API_KEY"]
    assert payload["reviewer"]["agent_env"]["AZURE_API_KEY"] == "reviewer-private-key"


@pytest.mark.parametrize(
    "options",
    [
        {"concurrency": 0},
        {"timeout_sec": -1},
        {"timeout_sec": True},
        {"environment": "not-a-backend"},
        {"image": "bad image"},
        {"agent_env": {"API_KEY": 42}},
        {"disable": True},
    ],
)
def test_reviewer_rejects_invalid_execution_contract(options: dict) -> None:
    with pytest.raises(ValueError):
        ReviewerConfig.coerce(options)


def test_rollout_yaml_preserves_reviewer_and_purpose_defaults(tmp_path: Path) -> None:
    config = rollout_config_from_dict(
        {
            "agent": "oracle",
            "reviewer": {
                "agent": "opencode",
                "model": "azure/reviewer",
                "environment": "daytona",
            },
        },
        task_path=tmp_path,
    )
    assert config.reviewer.model == "azure/reviewer"
    assert config.reviewer.environment == "daytona"
    assert config.purpose == "task"
    child = RolloutConfig(
        task_path=tmp_path, purpose="reviewer", parent_rollout="parent"
    )
    assert child.purpose == "reviewer"
    assert child.parent_rollout == "parent"


def test_eval_cli_reviewer_flags_reach_evaluation(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    captured = []

    async def capture(self):
        captured.append(self._config.reviewer)
        return SimpleNamespace(
            passed=1, total=1, score=1.0, errored=0, verifier_errored=0
        )

    with patch.object(Evaluation, "run", new=capture):
        result = CliRunner().invoke(
            app,
            [
                "eval",
                "run",
                "--tasks-dir",
                str(task),
                "--agent",
                "oracle",
                "--reviewer-agent",
                "opencode",
                "--reviewer-model",
                "azure/reviewer",
                "--reviewer-sandbox",
                "daytona",
                "--reviewer-concurrency",
                "2",
                "--reviewer-timeout-sec",
                "700",
                "--reviewer-reasoning-effort",
                "xhigh",
                "--reviewer-agent-env",
                "AZURE_API_KEY=reviewer-private-key",
            ],
        )
    assert result.exit_code == 0, result.output
    assert captured == [
        ReviewerConfig(
            agent="opencode",
            model="azure/reviewer",
            environment="daytona",
            concurrency=2,
            timeout_sec=700,
            reasoning_effort="xhigh",
            agent_env={"AZURE_API_KEY": "reviewer-private-key"},
        )
    ]


def test_cli_partial_reviewer_override_preserves_yaml_fields(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "tasks_dir": str(task),
                "agent": "oracle",
                "reviewer": {
                    "agent": "opencode",
                    "model": "azure/original",
                    "environment": "daytona",
                    "timeout_sec": 600,
                    "agent_env": {"AZURE_ENDPOINT": "https://example.invalid"},
                },
            }
        )
    )
    captured = []

    async def capture(self):
        captured.append(self._config.reviewer)
        return SimpleNamespace(
            passed=1, total=1, score=1.0, errored=0, verifier_errored=0
        )

    with patch.object(Evaluation, "run", new=capture):
        result = CliRunner().invoke(
            app,
            [
                "eval",
                "run",
                "--config",
                str(config_path),
                "--reviewer-model",
                "azure/replacement",
                "--reviewer-agent-env",
                "AZURE_API_KEY=private-key",
            ],
        )
    assert result.exit_code == 0, result.output
    assert captured[0].model == "azure/replacement"
    assert captured[0].timeout_sec == 600
    assert captured[0].environment == "daytona"
    assert captured[0].agent_env == {
        "AZURE_ENDPOINT": "https://example.invalid",
        "AZURE_API_KEY": "private-key",
    }


async def test_evaluation_passes_reviewer_to_rollout(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    reviewer = ReviewerConfig(model="azure/reviewer", environment="daytona")
    evaluation = Evaluation(
        tasks_dir=task,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(reviewer=reviewer),
    )
    captured = []

    async def capture(config):
        captured.append(config.reviewer)

        async def run():
            from benchflow.models import RolloutResult

            return RolloutResult(task_name="task", rewards={"reward": 1.0})

        return SimpleNamespace(run=run)

    with patch("benchflow.rollout.Rollout.create", side_effect=capture):
        await evaluation._run_single_task(task, evaluation._config)
    assert captured == [reviewer]


async def test_scoring_failure_never_retries_the_solver(tmp_path: Path) -> None:
    """Automatic-review failures retry the committed scoring stage, not PR #902's solver."""
    from unittest.mock import AsyncMock

    from benchflow.evaluation import RetryConfig
    from benchflow.models import RolloutResult
    from benchflow.review.outcome import ScoringResult

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    config = EvaluationConfig(retry=RetryConfig(max_retries=2))
    evaluation = Evaluation(tasks_dir=task, jobs_dir=tmp_path / "jobs", config=config)
    result = RolloutResult(
        task_name="task",
        error="Connection reset by peer",
        scoring=ScoringResult(status="error", error="Reviewer provider unavailable"),
    )
    evaluation._sdk = AsyncMock()
    evaluation._sdk.run.return_value = result
    assert await evaluation._run_task(task) is result
    assert evaluation._sdk.run.await_count == 1


def test_runtime_result_pass_uses_explicit_gate_outcome() -> None:
    from benchflow.review.outcome import ScoringResult
    from benchflow.runtime import RuntimeResult

    scoring = ScoringResult(
        status="complete",
        passed=True,
        tests_pass=True,
        all_blockers_pass=True,
        verifier_reward=1.0,
        rubric_reward=0.8,
        reviewer_run="reviews/attempt-001/run",
    )
    result = RuntimeResult(
        task_name="physics",
        rollout_name="first",
        reward=0.8,
        rewards=scoring.numeric_rewards(),
        n_tool_calls=3,
        error=None,
        verifier_error=None,
        trajectory=[],
        scoring=scoring,
    )
    assert result.passed
    assert result.verified
    assert result.reward == 0.8


def test_hosted_environment_rejects_ignored_reviewer_flags() -> None:
    from benchflow.eval_plan import EvalCreateRequest, EvalPlanError, build_eval_plan

    with pytest.raises(EvalPlanError, match="source-env owns its scoring"):
        build_eval_plan(
            EvalCreateRequest(
                source_env="example/hosted",
                reviewer=ReviewerConfig(model="azure/reviewer"),
            )
        )


def test_reviewer_backend_preflight_reuses_docker_daemon_check() -> None:
    from benchflow.review.preflight import validate_reviewer_backend

    with (
        patch(
            "benchflow.sandbox.docker.DockerSandbox.preflight",
            side_effect=SystemExit("daemon down"),
        ),
        pytest.raises(ValueError, match=r"Reviewer sandbox.*daemon down"),
    ):
        validate_reviewer_backend(ReviewerConfig(environment="docker"))


def test_reviewer_daytona_preflight_does_not_allocate_vm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchflow.review.preflight import validate_reviewer_backend

    monkeypatch.setenv("DAYTONA_API_KEY", "configured-key")
    monkeypatch.setenv("DAYTONA_TARGET", "us")
    with (
        patch("benchflow.sandbox.daytona._load_daytona_sdk") as load_sdk,
        patch(
            "benchflow.sandbox.daytona.DaytonaSandbox.__init__",
            side_effect=AssertionError("No VM construction"),
        ),
    ):
        validate_reviewer_backend(ReviewerConfig(environment="daytona"))
    load_sdk.assert_called_once()
    assert os.environ["DAYTONA_TARGET"] == "us"


def test_reviewer_daytona_missing_key_fails_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchflow.review.preflight import validate_reviewer_backend

    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    with (
        patch("benchflow.sandbox.daytona._load_daytona_sdk"),
        pytest.raises(ValueError, match="DAYTONA_API_KEY"),
    ):
        validate_reviewer_backend(ReviewerConfig(environment="daytona"))


def test_reviewer_missing_optional_sdk_has_install_hint() -> None:
    from benchflow.review.preflight import validate_reviewer_backend

    with (
        patch(
            "benchflow.sandbox.daytona._load_daytona_sdk",
            side_effect=ImportError("missing SDK"),
        ),
        pytest.raises(ValueError, match=r"benchflow\[sandbox-daytona\]"),
    ):
        validate_reviewer_backend(ReviewerConfig(environment="daytona"))
