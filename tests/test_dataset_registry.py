"""Tests for registry dataset resolution (`bench eval create -d name@version`).

Step 2 of the dataset-versioning plan (docs/dataset-versioning.md in
benchflow-ai/skillsbench, registry added in skillsbench PR #922): resolving a
registry entry clones the pinned snapshot, verifies every task's content
digest, and stamps results with dataset_name / dataset_version / task_digest.
"""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from benchflow._utils.dataset_registry import (
    DatasetDigestMismatchError,
    DatasetResolutionError,
    ResolvedDataset,
    bench_version_issue,
    parse_dataset_spec,
    resolve_dataset,
)
from benchflow._utils.task_authoring import task_digest
from benchflow.cli.main import app

COMMIT = "a" * 40


def _make_task(parent: Path, name: str) -> Path:
    task = parent / name
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.1"\n')
    (task / "instruction.md").write_text(f"Solve {name}.\n")
    (task / "tests" / "test.sh").write_text("exit 0\n")
    return task


def _write_registry(
    tmp_path: Path,
    tasks_parent: Path,
    *,
    digests: dict[str, str] | None = None,
    bench_version: str = ">=0.1,<999",
) -> Path:
    """Write a single-entry registry pinning the tasks under tasks_parent."""
    entry_tasks = []
    for task_dir in sorted(tasks_parent.iterdir()):
        entry_tasks.append(
            {
                "name": task_dir.name,
                "git_url": "https://github.com/acme/benchmarks.git",
                "git_commit_id": COMMIT,
                "path": f"tasks/{task_dir.name}",
                "digest": (digests or {}).get(task_dir.name) or task_digest(task_dir),
            }
        )
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(
        json.dumps(
            [
                {
                    "name": "skillsbench",
                    "version": "1.1",
                    "git_tag": "v1.1",
                    "bench_version": bench_version,
                    "tasks": entry_tasks,
                }
            ]
        )
    )
    return registry_file


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    """Two fake tasks plus a stubbed resolve_source_with_metadata."""
    tasks_parent = tmp_path / "snap" / "tasks"
    _make_task(tasks_parent, "task-a")
    _make_task(tasks_parent, "task-b")
    calls = {}

    def fake_resolve(repo, path=None, ref=None):
        calls.update(repo=repo, path=path, ref=ref)
        return SimpleNamespace(
            path=tasks_parent,
            provenance={
                "type": "github",
                "repo": repo,
                "requested_ref": ref,
                "resolved_sha": COMMIT,
                "path": path or "",
                "local_path": str(tasks_parent),
                "dirty": False,
                "file_hashes": {},
            },
        )

    monkeypatch.setattr(
        "benchflow._utils.benchmark_repos.resolve_source_with_metadata",
        fake_resolve,
    )
    return SimpleNamespace(tasks_parent=tasks_parent, calls=calls)


class TestParseDatasetSpec:
    def test_valid(self):
        assert parse_dataset_spec("skillsbench@1.1") == ("skillsbench", "1.1")

    @pytest.mark.parametrize("spec", ["skillsbench", "@1.1", "skillsbench@", ""])
    def test_invalid(self, spec):
        with pytest.raises(DatasetResolutionError, match="name>@<version"):
            parse_dataset_spec(spec)


class TestBenchVersionIssue:
    def test_no_declared_range(self):
        assert bench_version_issue(None) is None
        assert bench_version_issue("") is None

    def test_in_range(self):
        assert bench_version_issue(">=0,<999") is None

    def test_out_of_range_warns(self):
        issue = bench_version_issue("<0.0.1")
        assert issue is not None and "outside the range" in issue

    def test_unparseable_range_warns(self):
        issue = bench_version_issue("not-a-range")
        assert issue is not None and "unparseable" in issue


class TestResolveDataset:
    def test_resolves_and_verifies_digests(self, tmp_path, snapshot):
        registry = _write_registry(tmp_path, snapshot.tasks_parent)
        resolved = resolve_dataset("skillsbench@1.1", registry=str(registry))
        assert isinstance(resolved, ResolvedDataset)
        assert resolved.spec == "skillsbench@1.1"
        assert resolved.tasks_dir == snapshot.tasks_parent
        assert resolved.task_names == {"task-a", "task-b"}
        assert set(resolved.task_digests) == {"task-a", "task-b"}
        assert resolved.bench_version == ">=0.1,<999"
        assert resolved.provenance["resolved_sha"] == COMMIT
        # snapshot is cloned by pinned commit, not by (movable) tag
        assert snapshot.calls == {
            "repo": "acme/benchmarks",
            "path": "tasks",
            "ref": COMMIT,
        }

    def test_digest_mismatch_fails_hard(self, tmp_path, snapshot):
        registry = _write_registry(tmp_path, snapshot.tasks_parent)
        (snapshot.tasks_parent / "task-b" / "instruction.md").write_text("tampered\n")
        with pytest.raises(DatasetDigestMismatchError, match="task-b"):
            resolve_dataset("skillsbench@1.1", registry=str(registry))

    def test_missing_task_dir_fails_hard(self, tmp_path, snapshot):
        registry = _write_registry(
            tmp_path,
            snapshot.tasks_parent,
            digests={"task-c": "sha256:" + "0" * 64},
        )
        # registry pins task-c which the snapshot does not contain
        registry_data = json.loads(registry.read_text())
        registry_data[0]["tasks"].append(
            {
                "name": "task-c",
                "git_url": "https://github.com/acme/benchmarks.git",
                "git_commit_id": COMMIT,
                "path": "tasks/task-c",
                "digest": "sha256:" + "0" * 64,
            }
        )
        registry.write_text(json.dumps(registry_data))
        with pytest.raises(DatasetDigestMismatchError, match="task-c: missing"):
            resolve_dataset("skillsbench@1.1", registry=str(registry))

    def test_unknown_version_lists_available(self, tmp_path, snapshot):
        registry = _write_registry(tmp_path, snapshot.tasks_parent)
        with pytest.raises(
            DatasetResolutionError, match=r"available: skillsbench@1\.1"
        ):
            resolve_dataset("skillsbench@9.9", registry=str(registry))

    def test_resolved_sha_must_match_registry_commit(
        self, tmp_path, snapshot, monkeypatch
    ):
        registry = _write_registry(tmp_path, snapshot.tasks_parent)
        monkeypatch.setattr(
            "benchflow._utils.benchmark_repos.resolve_source_with_metadata",
            lambda repo, path=None, ref=None: SimpleNamespace(
                path=snapshot.tasks_parent,
                provenance={"resolved_sha": "b" * 40},
            ),
        )
        with pytest.raises(DatasetResolutionError, match="registry pins"):
            resolve_dataset("skillsbench@1.1", registry=str(registry))

    def test_multi_snapshot_dataset_rejected(self, tmp_path, snapshot):
        registry = _write_registry(tmp_path, snapshot.tasks_parent)
        registry_data = json.loads(registry.read_text())
        registry_data[0]["tasks"][0]["git_commit_id"] = "c" * 40
        registry.write_text(json.dumps(registry_data))
        with pytest.raises(DatasetResolutionError, match="snapshots"):
            resolve_dataset("skillsbench@1.1", registry=str(registry))

    def test_non_github_git_url_rejected(self, tmp_path, snapshot):
        registry = _write_registry(tmp_path, snapshot.tasks_parent)
        registry_data = json.loads(registry.read_text())
        for t in registry_data[0]["tasks"]:
            t["git_url"] = "https://gitlab.com/acme/benchmarks.git"
        registry.write_text(json.dumps(registry_data))
        with pytest.raises(DatasetResolutionError, match=r"github\.com"):
            resolve_dataset("skillsbench@1.1", registry=str(registry))


class TestEvalCreateDatasetCli:
    def test_dataset_threads_config_and_filter(self, tmp_path, snapshot, monkeypatch):
        registry = _write_registry(tmp_path, snapshot.tasks_parent)
        captured = {}

        async def fake_eval_run(self):
            captured["tasks_dir"] = self._tasks_dir
            captured["include_tasks"] = self._config.include_tasks
            captured["source"] = self._config.source_provenance
            captured["dataset_name"] = self._config.dataset_name
            captured["dataset_version"] = self._config.dataset_version
            captured["dataset_task_digests"] = self._config.dataset_task_digests
            return SimpleNamespace(passed=2, total=2, score=1.0, errored=0)

        monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_eval_run)

        result = CliRunner().invoke(
            app,
            [
                "eval",
                "create",
                "--dataset",
                "skillsbench@1.1",
                "--registry",
                str(registry),
                "--agent",
                "oracle",
                "--jobs-dir",
                str(tmp_path / "jobs"),
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert captured["tasks_dir"] == snapshot.tasks_parent
        assert captured["include_tasks"] == {"task-a", "task-b"}
        assert captured["source"]["resolved_sha"] == COMMIT
        assert captured["dataset_name"] == "skillsbench"
        assert captured["dataset_version"] == "1.1"
        assert set(captured["dataset_task_digests"]) == {"task-a", "task-b"}
        assert "digests verified" in result.stdout

    def test_user_include_intersects_dataset(self, tmp_path, snapshot, monkeypatch):
        registry = _write_registry(tmp_path, snapshot.tasks_parent)
        captured = {}

        async def fake_eval_run(self):
            captured["include_tasks"] = self._config.include_tasks
            return SimpleNamespace(passed=1, total=1, score=1.0, errored=0)

        monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_eval_run)

        result = CliRunner().invoke(
            app,
            [
                "eval",
                "create",
                "--dataset",
                "skillsbench@1.1",
                "--registry",
                str(registry),
                "--include",
                "task-a",
                "--agent",
                "oracle",
                "--jobs-dir",
                str(tmp_path / "jobs"),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert captured["include_tasks"] == {"task-a"}

    def test_dataset_is_exclusive_with_tasks_dir(self, tmp_path):
        result = CliRunner().invoke(
            app,
            [
                "eval",
                "create",
                "--dataset",
                "skillsbench@1.1",
                "--tasks-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1
        assert "only one source" in result.stdout

    def test_registry_requires_dataset(self, tmp_path):
        result = CliRunner().invoke(
            app,
            ["eval", "create", "--registry", str(tmp_path / "r.json")],
        )
        assert result.exit_code == 1
        assert "--registry requires --dataset" in result.stdout

    def test_out_of_range_bench_version_blocks(self, tmp_path, snapshot, monkeypatch):
        """The bench_version range is a hard gate for dataset runs; the
        registry declares what the version was validated against."""
        registry = _write_registry(
            tmp_path, snapshot.tasks_parent, bench_version="<0.0.1"
        )
        ran = {}

        async def fake_eval_run(self):
            ran["ran"] = True
            return SimpleNamespace(passed=2, total=2, score=1.0, errored=0)

        monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_eval_run)

        result = CliRunner().invoke(
            app,
            [
                "eval",
                "create",
                "--dataset",
                "skillsbench@1.1",
                "--registry",
                str(registry),
                "--agent",
                "oracle",
                "--jobs-dir",
                str(tmp_path / "jobs"),
            ],
        )
        assert result.exit_code == 1
        assert "outside the range" in result.stdout
        assert "--ignore-bench-version" in result.stdout
        assert not ran

    def test_ignore_bench_version_overrides_gate(self, tmp_path, snapshot, monkeypatch):
        registry = _write_registry(
            tmp_path, snapshot.tasks_parent, bench_version="<0.0.1"
        )
        ran = {}

        async def fake_eval_run(self):
            ran["ran"] = True
            return SimpleNamespace(passed=2, total=2, score=1.0, errored=0)

        monkeypatch.setattr("benchflow.evaluation.Evaluation.run", fake_eval_run)

        result = CliRunner().invoke(
            app,
            [
                "eval",
                "create",
                "--dataset",
                "skillsbench@1.1",
                "--registry",
                str(registry),
                "--ignore-bench-version",
                "--agent",
                "oracle",
                "--jobs-dir",
                str(tmp_path / "jobs"),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert ran == {"ran": True}
        assert "Warning:" in result.stdout

    def test_ignore_bench_version_requires_dataset(self, tmp_path):
        result = CliRunner().invoke(
            app,
            ["eval", "create", "--ignore-bench-version", "--tasks-dir", str(tmp_path)],
        )
        assert result.exit_code == 1
        assert "--ignore-bench-version requires --dataset" in result.stdout

    def test_resolution_error_exits_cleanly(self, tmp_path, snapshot):
        registry = _write_registry(tmp_path, snapshot.tasks_parent)
        result = CliRunner().invoke(
            app,
            [
                "eval",
                "create",
                "--dataset",
                "skillsbench@9.9",
                "--registry",
                str(registry),
            ],
        )
        assert result.exit_code == 1
        assert "not found in registry" in result.stdout


class TestDatasetStamping:
    DATASET: ClassVar[dict] = {"name": "skillsbench", "version": "1.1"}
    TASK_DIGEST = "sha256:" + "f" * 64

    def test_rollout_config_threads_dataset_through_from_legacy(self, tmp_path):
        from benchflow.rollout import RolloutConfig

        cfg = RolloutConfig.from_legacy(
            task_path=tmp_path,
            agent="oracle",
            dataset=self.DATASET,
            task_digest=self.TASK_DIGEST,
        )
        assert cfg.dataset == self.DATASET
        assert cfg.task_digest == self.TASK_DIGEST

    def test_config_json_stamped(self, tmp_path):
        from benchflow.rollout import _write_config
        from benchflow.skill_policy import resolve_task_skill_policy

        policy = resolve_task_skill_policy(
            task_path=tmp_path,
            skill_mode="no-skill",
            runtime_skills_dir=None,
            declared_sandbox_skills_dir=None,
        )
        _write_config(
            tmp_path,
            task_path=tmp_path / "task",
            agent="oracle",
            model=None,
            environment="docker",
            skill_policy=policy,
            sandbox_user="agent",
            context_root=None,
            timeout=60,
            started_at=datetime(2026, 6, 12),
            agent_env={},
            dataset=self.DATASET,
            task_digest=self.TASK_DIGEST,
        )
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["dataset_name"] == "skillsbench"
        assert config["dataset_version"] == "1.1"
        assert config["task_digest"] == self.TASK_DIGEST

    def test_config_json_unstamped_without_dataset(self, tmp_path):
        from benchflow.rollout import _write_config
        from benchflow.skill_policy import resolve_task_skill_policy

        policy = resolve_task_skill_policy(
            task_path=tmp_path,
            skill_mode="no-skill",
            runtime_skills_dir=None,
            declared_sandbox_skills_dir=None,
        )
        _write_config(
            tmp_path,
            task_path=tmp_path / "task",
            agent="oracle",
            model=None,
            environment="docker",
            skill_policy=policy,
            sandbox_user="agent",
            context_root=None,
            timeout=60,
            started_at=datetime(2026, 6, 12),
            agent_env={},
        )
        config = json.loads((tmp_path / "config.json").read_text())
        assert "dataset_name" not in config
        assert "task_digest" not in config

    def test_result_json_stamped(self, tmp_path):
        from benchflow.rollout import _build_rollout_result

        _build_rollout_result(
            tmp_path,
            task_name="task-a",
            rollout_name="task-a__r1",
            agent="oracle",
            agent_name="oracle",
            model=None,
            n_tool_calls=0,
            prompts=[],
            error=None,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            rewards={"reward": 1.0},
            started_at=datetime(2026, 6, 12),
            timing={},
            dataset=self.DATASET,
            task_digest=self.TASK_DIGEST,
        )
        result = json.loads((tmp_path / "result.json").read_text())
        assert result["dataset_name"] == "skillsbench"
        assert result["dataset_version"] == "1.1"
        assert result["task_digest"] == self.TASK_DIGEST

    def test_result_json_dev_run_stamps_digest_without_dataset(self, tmp_path):
        """Dev runs carry task_digest but no dataset identity fields."""
        from benchflow.rollout import _build_rollout_result

        _build_rollout_result(
            tmp_path,
            task_name="task-a",
            rollout_name="task-a__r1",
            agent="oracle",
            agent_name="oracle",
            model=None,
            n_tool_calls=0,
            prompts=[],
            error=None,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            rewards={"reward": 1.0},
            started_at=datetime(2026, 6, 12),
            timing={},
            task_digest=self.TASK_DIGEST,
        )
        result = json.loads((tmp_path / "result.json").read_text())
        assert result["task_digest"] == self.TASK_DIGEST
        assert "dataset_name" not in result
        assert "dataset_version" not in result

    @staticmethod
    def _capture_rollout_create(monkeypatch):
        captured = {}

        async def _fake_run():
            return SimpleNamespace(rewards={"reward": 1.0})

        async def fake_create(rollout_config):
            captured["config"] = rollout_config
            return SimpleNamespace(run=_fake_run)

        monkeypatch.setattr("benchflow.rollout.Rollout.create", fake_create)
        return captured

    async def test_run_single_task_uses_registry_digest_for_dataset_runs(
        self, tmp_path, monkeypatch
    ):
        from benchflow.evaluation import Evaluation, EvaluationConfig

        task_dir = _make_task(tmp_path / "tasks", "task-a")
        digest = task_digest(task_dir)
        cfg = EvaluationConfig(
            agent="oracle",
            dataset_name="skillsbench",
            dataset_version="1.1",
            dataset_task_digests={"task-a": digest},
        )
        captured = self._capture_rollout_create(monkeypatch)
        evaluation = Evaluation(
            tasks_dir=tmp_path / "tasks", jobs_dir=tmp_path / "jobs", config=cfg
        )
        await evaluation._run_single_task(task_dir, cfg)
        assert captured["config"].dataset == {"name": "skillsbench", "version": "1.1"}
        assert captured["config"].task_digest == digest

    async def test_run_single_task_computes_digest_for_dev_runs(
        self, tmp_path, monkeypatch
    ):
        """Dev runs (no --dataset) stamp a live-computed content digest."""
        from benchflow.evaluation import Evaluation, EvaluationConfig

        task_dir = _make_task(tmp_path / "tasks", "task-a")
        cfg = EvaluationConfig(agent="oracle")
        captured = self._capture_rollout_create(monkeypatch)
        evaluation = Evaluation(
            tasks_dir=tmp_path / "tasks", jobs_dir=tmp_path / "jobs", config=cfg
        )
        await evaluation._run_single_task(task_dir, cfg)
        assert captured["config"].dataset is None
        assert captured["config"].task_digest == task_digest(task_dir)
