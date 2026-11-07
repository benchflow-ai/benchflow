"""Source and resume regressions present in candidate ac3bedaa1."""

import json
import shutil
from pathlib import Path

import pytest

from benchflow._utils import benchmark_repos as repos
from benchflow._utils.task_authoring import task_digest
from benchflow.evaluation import Evaluation, EvaluationConfig, _task_digest_for
from benchflow.task.formats import materialize_task_dir
from tests.test_source_sparse_clone import _IDENTITY, _git, _write

EXAMPLE = Path(__file__).parent / "fixtures/taskmd/examples/hello-world"


def test_compose_contexts_include_merged_service_maps_and_stop_alias_cycles():
    """Review of 865d7a6f2: whole-service merges retain source contexts."""
    text = """
x-services: &shared
  simulator:
    build: {context: ../../../runtimes/base}
services: &services
  <<: [*shared, *services]
  main:
    image: python:3.12-slim
"""
    assert repos._compose_build_contexts(text) == ["../../../runtimes/base"]
    assert (
        repos._compose_build_contexts(
            text + "  simulator:\n    image: python:3.12-slim\n"
        )
        == []
    )


@pytest.fixture(autouse=True)
def format_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "format-cache"))


def test_resume_keeps_relative_task_directory_after_chdir(tmp_path, monkeypatch):
    """Guards ac3bedaa1: resume uses the original task directory, not the new cwd."""
    original = tmp_path / "original"
    task = original / "tasks" / "sample"
    _write(task / "task.md", '---\nschema_version: "1.3"\n---\nDo the task.\n')
    _write(task / "environment/Dockerfile", "FROM python:3.12-slim\n")
    _write(task / "verifier/test.sh", "true\n")
    _write(task / "oracle/solve.sh", "true\n")
    monkeypatch.chdir(original)
    evaluation = Evaluation(
        "tasks", tmp_path / "jobs", EvaluationConfig(agent="oracle"), job_name="job"
    )
    evaluation._write_evaluation_record()
    record = json.loads((evaluation.job_dir / "evaluation.json").read_text())
    assert Path(record["tasks_dir"]) == original / "tasks"
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    resumed = Evaluation.resume(evaluation.job_dir)
    assert resumed._get_task_dirs() == [task]


@pytest.mark.parametrize("batch", [False, True])
def test_converted_task_provenance_uses_the_selected_source(tmp_path, batch):
    """Guards ac3bedaa1: conversion may leave source root without changing its origin."""
    source = tmp_path / "source" / "tasks" / "hello-world"
    shutil.copytree(EXAMPLE, source)
    root = source.parent if batch else source
    config = EvaluationConfig(
        agent="oracle",
        source_provenance={
            "repo": "acme/tasks",
            "path": "tasks" if batch else "tasks/hello-world",
            "local_path": str(root),
        },
    )
    evaluation = Evaluation(root, tmp_path / "jobs", config, job_name="job")
    (native,) = evaluation._get_task_dirs()
    assert not native.is_relative_to(root)
    rollout = evaluation._rollout_config(native, config)
    assert rollout.task_path == native
    assert rollout.source_provenance["path"] == "tasks/hello-world"
    assert rollout.source_provenance["local_path"] == str(source)
    assert rollout.source_provenance["file_hashes"] == repos.task_file_hashes(source)


def test_converted_package_cannot_claim_an_unselected_origin(tmp_path):
    """Guards ac3bedaa1: package metadata cannot bypass source containment."""
    source = tmp_path / "inside"
    source.mkdir()
    outside = tmp_path / "outside"
    shutil.copytree(EXAMPLE, outside)
    native = materialize_task_dir(outside)
    config = EvaluationConfig(
        agent="oracle", source_provenance={"local_path": str(source), "path": "tasks"}
    )
    evaluation = Evaluation(source, tmp_path / "jobs", config, job_name="job")
    with pytest.raises(ValueError, match="outside source local_path"):
        evaluation._rollout_config(native, config)


def test_copied_draft2_packages_have_identical_materialized_digests(
    tmp_path, monkeypatch
):
    """Guards ac3bedaa1: source/cache locations do not change task content identity."""
    copies = []
    for name in ("one", "two"):
        source = tmp_path / name / "hello-world"
        shutil.copytree(EXAMPLE, source)
        monkeypatch.setenv(
            "BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / f"cache-{name}")
        )
        native = materialize_task_dir(source)
        copies.append((source, native))
    assert task_digest(copies[0][0]) == task_digest(copies[1][0])
    config = EvaluationConfig(agent="oracle")
    assert _task_digest_for(copies[0][1], config) == _task_digest_for(
        copies[1][1], config
    )
    for path in copies[0][1].rglob("*"):
        if path.is_file():
            assert (
                path.read_bytes()
                == (copies[1][1] / path.relative_to(copies[0][1])).read_bytes()
            )
    changed = copies[1][0] / "oracle" / "solve.sh"
    changed.write_text(changed.read_text() + "\n# changed solution\n")
    assert _task_digest_for(
        materialize_task_dir(copies[1][0]), config
    ) != _task_digest_for(copies[0][1], config)


@pytest.mark.parametrize(
    "build",
    [
        "    build:\n      context: ../../../runtimes/base\n",
        "    build: ../../../runtimes/base\n",
        "    build: {context: ../../../runtimes/base}\n",
        "    <<: &shared\n      build:\n        context: ../../../runtimes/base\n",
    ],
    ids=["long", "short", "flow", "merge"],
)
def test_sparse_checkout_retains_each_compose_build_form(tmp_path, monkeypatch, build):
    """Guards ac3bedaa1: short/flow build contexts survive sparse source resolution."""
    source = tmp_path / "source"
    task = source / "tasks" / "sample"
    _write(task / "task.md", '---\nschema_version: "1.3"\n---\nDo the task.\n')
    _write(task / "environment/Dockerfile", "FROM python:3.12-slim\n")
    _write(
        task / "environment/compose.yaml",
        "services:\n  main:\n    volumes: !override []\n  simulator:\n" + build,
    )
    _write(source / "runtimes/base/Dockerfile", "FROM python:3.12-slim\n")
    _write(source / "runtimes/other/Dockerfile", "FROM python:3.12-slim\n")
    _git(tmp_path, "init", "-q", "-b", "main", str(source))
    _git(source, "add", "-A")
    _git(source, *_IDENTITY, "commit", "-q", "-m", "fixture")
    bare = tmp_path / "remote.git"
    _git(tmp_path, "clone", "-q", "--bare", str(source), str(bare))
    _git(bare, "config", "uploadpack.allowFilter", "true")
    _git(bare, "config", "uploadpack.allowAnySHA1InWant", "true")
    monkeypatch.setattr(repos, "_repo_url", lambda org, repo: f"file://{bare}")
    monkeypatch.setattr(repos, "_cache_dir", lambda: tmp_path / "cache")
    resolved = repos.resolve_source_with_metadata("acme/tasks", path="tasks/sample")
    snapshot = resolved.path.parents[1]
    assert (snapshot / "runtimes/base/Dockerfile").is_file()
    assert not (snapshot / "runtimes/other").exists()
    assert not resolved.provenance["dirty"]
