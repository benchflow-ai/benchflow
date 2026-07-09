"""Tests for source provenance threading through evaluation rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from benchflow._utils.hf_datasets import SOURCE_SIDECAR
from benchflow.cli.main import app
from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.models import RolloutResult


def _write_minimal_task_toml(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                'version = "1.0"',
                "",
                "[metadata]",
                'author_name = "benchflow"',
                "",
                "[agent]",
                "timeout_sec = 300",
                "",
                "[verifier]",
                "timeout_sec = 60",
                "",
                "[environment]",
                "cpus = 1",
                "memory_mb = 2048",
            ]
        )
    )


@pytest.mark.asyncio
async def test_evaluation_derives_per_task_source_provenance(monkeypatch, tmp_path):
    """Guards v0.5-integration@cb8759e against parent-only batch provenance."""
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "task-a"
    (task_dir / "tests").mkdir(parents=True)
    _write_minimal_task_toml(task_dir)
    (task_dir / "instruction.md").write_text("Solve it.\n")
    (task_dir / "tests" / "test.sh").write_text("exit 0\n")
    captured = {}

    async def fake_create(config):
        captured["config"] = config

        class FakeRollout:
            async def run(self):
                return RolloutResult(task_name="task-a", rewards={"reward": 1.0})

        return FakeRollout()

    monkeypatch.setattr("benchflow.rollout.Rollout.create", fake_create)

    parent_source = {
        "type": "github",
        "repo": "benchflow-ai/benchmarks",
        "requested_ref": "main",
        "resolved_sha": "c65af83ae2c76fda3f1fd4d2fcf56563975e283e",
        "path": "datasets/programbench/tasks",
        "local_path": str(tasks_root),
        "dirty": False,
        "file_hashes": {},
    }
    evaluation = Evaluation(
        tasks_dir=tasks_root,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent="gemini",
            model="gemini-3.1-flash-lite-preview",
            source_provenance=parent_source,
        ),
    )

    await evaluation._run_single_task(task_dir, evaluation._config)

    task_source = captured["config"].source_provenance
    assert task_source["path"] == "datasets/programbench/tasks/task-a"
    assert task_source["local_path"] == str(task_dir)
    assert set(task_source["file_hashes"]) == {
        "instruction.md",
        "task.toml",
        "tests/test.sh",
    }
    assert task_source["repo"] == "benchflow-ai/benchmarks"
    assert task_source["resolved_sha"] == parent_source["resolved_sha"]


@pytest.mark.asyncio
async def test_evaluation_summary_marks_resumed_source_mismatch(tmp_path):
    """Guards v0.5-integration@cb8759e against false resume provenance."""
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "task-a"
    _write_minimal_task_toml(task_dir)
    jobs_dir = tmp_path / "jobs"
    result_dir = jobs_dir / "old-run" / "task-a__old"
    result_dir.mkdir(parents=True)
    old_source = {
        "type": "github",
        "repo": "acme/benchmarks",
        "requested_ref": "old",
        "resolved_sha": "0" * 40,
        "path": "tasks/task-a",
        "local_path": str(task_dir),
        "dirty": False,
        "file_hashes": {"task.toml": "sha256:" + "0" * 64},
    }
    new_source = {
        **old_source,
        "requested_ref": "main",
        "resolved_sha": "1" * 40,
        "path": "tasks",
        "local_path": str(tasks_root),
        "file_hashes": {},
    }
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-a",
                "agent": "gemini",
                "rewards": {"reward": 1.0},
                "error": None,
                "verifier_error": None,
                "source": old_source,
            }
        )
    )

    evaluation = Evaluation(
        tasks_dir=tasks_root,
        jobs_dir=jobs_dir,
        config=EvaluationConfig(
            agent="gemini",
            model="gemini-3.1-flash-lite-preview",
            environment="daytona",
            source_provenance=new_source,
        ),
    )

    await evaluation.run()

    summary = json.loads((jobs_dir / "summary.json").read_text())
    assert "source" not in summary
    assert summary["source_mismatch_tasks"] == ["task-a"]


def test_eval_create_source_repo_single_task_passes_source_provenance(
    monkeypatch, tmp_path
):
    """Guards v0.5-integration@cb8759e against unaudited source single-task runs."""
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "task-a"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "task.toml").write_text("[task]\n")
    (task_dir / "instruction.md").write_text("Solve it.\n")
    (task_dir / "tests" / "test.sh").write_text("exit 0\n")
    captured = {}
    source = {
        "type": "github",
        "repo": "acme/benchmarks",
        "requested_ref": "main",
        "resolved_sha": "0123456789abcdef0123456789abcdef01234567",
        "path": "tasks/task-a",
        "local_path": str(task_dir),
        "dirty": False,
        "file_hashes": {},
    }

    monkeypatch.setattr(
        "benchflow._utils.benchmark_repos.resolve_source_with_metadata",
        lambda repo, path=None, ref=None: SimpleNamespace(
            path=task_dir, provenance=source
        ),
    )

    async def fake_eval_run(self):
        captured["tasks_dir"] = self._tasks_dir
        captured["source_provenance"] = self._config.source_provenance
        captured["include_tasks"] = self._config.include_tasks
        return SimpleNamespace(passed=1, total=1, score=1.0, errored=0)

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_eval_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--source-repo",
            "acme/benchmarks",
            "--source-path",
            "tasks/task-a",
            "--source-ref",
            "main",
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["tasks_dir"] == task_dir
    assert captured["include_tasks"] == set()
    assert captured["source_provenance"] == source


def test_evaluation_discovers_single_task_directory(tmp_path):
    """Guards v0.5-integration@cb8759e source single-task summary production."""
    task_dir = tmp_path / "task-a"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text("[task]\n")

    evaluation = Evaluation(tasks_dir=task_dir, jobs_dir=tmp_path / "jobs")

    assert evaluation._get_task_dirs() == [task_dir]


@pytest.mark.asyncio
async def test_evaluation_uses_hf_snapshot_tasks_subdir_and_sidecar_provenance(
    monkeypatch, tmp_path
):
    """Guards PR slice 1 so HF split-layout snapshots run from their root."""
    snapshot_root = tmp_path / "hf-snapshot"
    tasks_root = snapshot_root / "tasks"
    task_dir = tasks_root / "task-a"
    (task_dir / "tests").mkdir(parents=True)
    _write_minimal_task_toml(task_dir)
    (task_dir / "instruction.md").write_text("Solve it.\n")
    (task_dir / "tests" / "test.sh").write_text("exit 0\n")
    (snapshot_root / SOURCE_SIDECAR).write_text(
        json.dumps(
            {
                "type": "huggingface_dataset",
                "repo": "benchflow/sample-tasks",
                "repo_type": "dataset",
                "requested_revision": "main",
                "resolved_revision": "a" * 40,
                "path": "",
                "local_path": str(snapshot_root),
                "dirty": False,
                "file_hashes": {},
            }
        )
    )
    captured = {}

    async def fake_create(config):
        captured["source_provenance"] = config.source_provenance

        class FakeRollout:
            async def run(self):
                return RolloutResult(task_name="task-a", rewards={"reward": 1.0})

        return FakeRollout()

    monkeypatch.setattr("benchflow.rollout.Rollout.create", fake_create)

    evaluation = Evaluation(
        tasks_dir=snapshot_root,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent="gemini",
            model="gemini-3.1-flash-lite-preview",
        ),
    )

    assert evaluation._tasks_dir == tasks_root
    assert evaluation._get_task_dirs() == [task_dir]
    await evaluation._run_single_task(task_dir, evaluation._config)

    source = captured["source_provenance"]
    assert source["type"] == "huggingface_dataset"
    assert source["repo"] == "benchflow/sample-tasks"
    assert source["resolved_revision"] == "a" * 40
    assert source["path"] == "tasks/task-a"
    assert set(source["file_hashes"]) == {
        "instruction.md",
        "task.toml",
        "tests/test.sh",
    }


def test_eval_create_config_applies_concurrency_override(monkeypatch, tmp_path):
    """Guards v0.5 large validation from silently using YAML concurrency."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    config = tmp_path / "eval.yaml"
    config.write_text(f"tasks_dir: {tasks_root}\nagent: oracle\nconcurrency: 4\n")
    captured = {}

    async def fake_eval_run(self):
        captured["concurrency"] = self._config.concurrency
        return SimpleNamespace(passed=0, total=0, score=0.0, errored=0)

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_eval_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--config",
            str(config),
            "--concurrency",
            "100",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["concurrency"] == 100


def test_eval_create_single_task_applies_concurrency_override(monkeypatch, tmp_path):
    """Guards v0.5-integration@c30e130 against single-task CLI config drift."""
    task_dir = tmp_path / "task-a"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text("[task]\n")
    captured = {}

    async def fake_eval_run(self):
        captured["concurrency"] = self._config.concurrency
        return SimpleNamespace(
            passed=1, total=1, score=1.0, errored=0, verifier_errored=0
        )

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_eval_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(task_dir),
            "--agent",
            "oracle",
            "--concurrency",
            "64",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["concurrency"] == 64


def test_eval_create_single_task_passes_cli_prompts(monkeypatch, tmp_path):
    """Guards PR #608 so eval run keeps custom CLI prompts."""
    task_dir = tmp_path / "task-a"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text("[task]\n")
    captured = {}

    async def fake_eval_run(self):
        captured["prompts"] = self._config.prompts
        return SimpleNamespace(
            passed=1, total=1, score=1.0, errored=0, verifier_errored=0
        )

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_eval_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(task_dir),
            "--agent",
            "oracle",
            "--prompt",
            "first instruction",
            "--prompt",
            "second instruction",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["prompts"] == ["first instruction", "second instruction"]


def test_eval_create_single_task_applies_agent_idle_timeout(monkeypatch, tmp_path):
    """Guards v0.5-integration@219906c against unbounded active-dev ACP hangs."""
    task_dir = tmp_path / "task-a"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text("[task]\n")
    captured = {}

    async def fake_eval_run(self):
        captured["agent_idle_timeout"] = self._config.agent_idle_timeout
        return SimpleNamespace(
            passed=0, total=1, score=0.0, errored=1, verifier_errored=0
        )

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_eval_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(task_dir),
            "--agent",
            "gemini",
            "--agent-idle-timeout",
            "45",
        ],
    )

    assert result.exit_code == 1, result.stdout
    assert captured["agent_idle_timeout"] == 45


def test_eval_create_config_allows_literal_jobs_dir_override(monkeypatch, tmp_path):
    """Guards v0.5 config runs from treating --jobs-dir jobs as absent."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    config = tmp_path / "eval.yaml"
    config.write_text(
        f"tasks_dir: {tasks_root}\nagent: oracle\njobs_dir: {tmp_path / 'yaml-jobs'}\n"
    )
    captured = {}

    async def fake_eval_run(self):
        captured["jobs_dir"] = self._jobs_dir
        return SimpleNamespace(passed=0, total=0, score=0.0, errored=0)

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_eval_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--config",
            str(config),
            "--jobs-dir",
            "jobs",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["jobs_dir"] == Path("jobs")


def test_eval_create_config_prompt_overrides_yaml(monkeypatch, tmp_path):
    """Guards PR #608 so bench eval run --prompt overrides YAML prompts."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    config = tmp_path / "eval.yaml"
    config.write_text(
        "\n".join(
            [
                f"tasks_dir: {tasks_root}",
                "agent: oracle",
                "prompts:",
                "  - yaml instruction",
            ]
        )
    )
    captured = {}

    async def fake_eval_run(self):
        captured["prompts"] = self._config.prompts
        return SimpleNamespace(passed=0, total=0, score=0.0, errored=0)

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_eval_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--config",
            str(config),
            "--prompt",
            "cli instruction",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["prompts"] == ["cli instruction"]


def test_eval_create_config_applies_identity_overrides(monkeypatch, tmp_path):
    """Guards v0.5 config runs from silently ignoring expensive-run identity flags."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    config = tmp_path / "eval.yaml"
    config.write_text(
        "\n".join(
            [
                f"tasks_dir: {tasks_root}",
                "agent: oracle",
                "model: old-model",
                "environment: docker",
            ]
        )
    )
    captured = {}

    async def fake_eval_run(self):
        captured["agent"] = self._config.agent
        captured["model"] = self._config.model
        captured["environment"] = self._config.environment
        return SimpleNamespace(passed=0, total=0, score=0.0, errored=0)

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_eval_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--config",
            str(config),
            "--agent",
            "gemini",
            "--model",
            "gemini-3.1-flash-lite-preview",
            "--sandbox",
            "daytona",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured == {
        "agent": "gemini",
        "model": "gemini-3.1-flash-lite-preview",
        "environment": "daytona",
    }


def test_eval_create_source_repo_batch_preserves_parent_source_provenance(
    monkeypatch, tmp_path
):
    """Guards v0.5-integration@cb8759e against CLI batch provenance loss."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    captured = {}
    source = {
        "type": "github",
        "repo": "acme/benchmarks",
        "requested_ref": "main",
        "resolved_sha": "0123456789abcdef0123456789abcdef01234567",
        "path": "tasks",
        "local_path": str(tasks_root),
        "dirty": False,
        "file_hashes": {},
    }

    monkeypatch.setattr(
        "benchflow._utils.benchmark_repos.resolve_source_with_metadata",
        lambda repo, path=None, ref=None: SimpleNamespace(
            path=tasks_root, provenance=source
        ),
    )

    async def fake_eval_run(self):
        captured["source_provenance"] = self._config.source_provenance
        return SimpleNamespace(passed=1, total=1, score=1.0, errored=0)

    monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_eval_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--source-repo",
            "acme/benchmarks",
            "--source-path",
            "tasks",
            "--source-ref",
            "main",
            "--agent",
            "gemini",
            "--model",
            "gemini-3.1-flash-lite-preview",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["source_provenance"] == source


def test_eval_create_source_repo_writes_requested_concurrency_to_rollout_config(
    monkeypatch, tmp_path
):
    """Guards v0.5-integration@c30e130 against false per-rollout concurrency evidence."""
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "task-a"
    _write_minimal_task_toml(task_dir)
    (task_dir / "instruction.md").write_text("Solve it.\n")
    source = {
        "type": "github",
        "repo": "acme/benchmarks",
        "requested_ref": "main",
        "resolved_sha": "0123456789abcdef0123456789abcdef01234567",
        "path": "tasks",
        "local_path": str(tasks_root),
        "dirty": False,
        "file_hashes": {},
    }

    monkeypatch.setattr(
        "benchflow._utils.benchmark_repos.resolve_source_with_metadata",
        lambda repo, path=None, ref=None: SimpleNamespace(
            path=tasks_root, provenance=source
        ),
    )

    class FakePlanes:
        def install_docker_compat(self):
            return None

        def extract_usage(self, _runtime):
            return {
                "n_input_tokens": None,
                "n_output_tokens": None,
                "n_cache_read_tokens": None,
                "n_cache_creation_tokens": None,
                "total_tokens": None,
                "cost_usd": None,
                "usage_source": "unavailable",
                "price_source": None,
            }

        def resolve_locked_paths(self, _sandbox_user, _locked_paths):
            return []

        def resolve_agent_env(self, _agent, _model, agent_env):
            return agent_env or {}

        def agent_launch(self, agent, *, disallow_web_tools):
            return agent

        def stage_dockerfile_deps(self, *_args, **_kwargs):
            return None

        def inject_skills_into_dockerfile(self, *_args, **_kwargs):
            return None

        def create_environment(self, *_args, **_kwargs):
            return object()

    monkeypatch.setattr(
        "benchflow.rollout.default_rollout_planes",
        lambda: FakePlanes(),
    )

    async def setup_only_run(self):
        await self.setup()
        return RolloutResult(
            task_name=self._config.task_path.name,
            rollout_name=self._rollout_name or "",
            rewards={"reward": 1.0},
            source_provenance=self._config.source_provenance,
        )

    monkeypatch.setattr("benchflow.rollout.Rollout.run", setup_only_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--source-repo",
            "acme/benchmarks",
            "--source-path",
            "tasks",
            "--source-ref",
            "main",
            "--agent",
            "oracle",
            "--sandbox",
            "daytona",
            "--concurrency",
            "64",
            "--agent-idle-timeout",
            "45",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    config_paths = list((tmp_path / "jobs").rglob("config.json"))
    assert len(config_paths) == 1
    config = json.loads(config_paths[0].read_text())
    summary = json.loads((tmp_path / "jobs" / "summary.json").read_text())
    assert config["concurrency"] == 64
    assert config["agent_idle_timeout_sec"] == 45
    assert summary["concurrency"] == 64
    assert summary["agent_idle_timeout_sec"] == 45


def test_summary_source_omits_host_local_path() -> None:
    """Guards PR #779: summary.json source provenance is portable."""
    from benchflow._utils.source_provenance import summary_source_fields

    parent_source = {
        "type": "github",
        "repo": "acme/benchmarks",
        "requested_ref": "main",
        "resolved_sha": "0123456789abcdef0123456789abcdef01234567",
        "path": "tasks",
        "local_path": "/Users/dev/private/tasks",
        "dirty": False,
        "file_hashes": {},
    }
    result_source = {
        **parent_source,
        "path": "tasks/task-a",
        "local_path": "/Users/dev/private/tasks/task-a",
        "file_hashes": {"task.toml": "sha256:" + "0" * 64},
    }

    fields = summary_source_fields(parent_source, {"task-a": {"source": result_source}})

    assert fields["source"]["path"] == "tasks"
    assert "local_path" not in fields["source"]


def test_evaluation_yaml_source_records_source_provenance(monkeypatch, tmp_path):
    """Guards v0.5-integration@cb8759e against YAML source provenance loss."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    source = {
        "type": "github",
        "repo": "acme/benchmarks",
        "requested_ref": "main",
        "resolved_sha": "0123456789abcdef0123456789abcdef01234567",
        "path": "tasks",
        "local_path": str(tasks_root),
        "dirty": False,
        "file_hashes": {},
    }

    monkeypatch.setattr(
        "benchflow._utils.benchmark_repos.resolve_source_with_metadata",
        lambda repo, path=None, ref=None: SimpleNamespace(
            path=tasks_root, provenance=source
        ),
    )
    config = tmp_path / "eval.yaml"
    config.write_text(
        """
source:
  repo: acme/benchmarks
  path: tasks
  ref: main
agent: gemini
model: gemini-3.1-flash-lite-preview
jobs_dir: jobs
"""
    )

    evaluation = Evaluation.from_yaml(config)

    assert evaluation._tasks_dir == tasks_root
    assert evaluation._config.source_provenance == source
