"""A task with symlinks must not abort a batch through provenance inference.

Regression test: ``bench eval run --tasks-dir
docs/examples/task-md/real-skillsbench`` ran the other tasks, then
``citation-check-network`` (whose ``environment``, ``oracle`` and
``verifier`` are symlinks to ``citation-check``) raised ``ValueError: Task
path ... contains symlink`` from the best-effort source-provenance inference.
The task was recorded as an unexpected exception, and building its result
payload raised the same error again after every task had finished, so the
job exited 1 without ``summary.json`` or ``results.jsonl`` for the
finished trials.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow._utils import benchmark_repos
from benchflow._utils.benchmark_repos import infer_task_source_provenance


@pytest.fixture
def symlinked_task(tmp_path: Path, monkeypatch) -> Path:
    """A task inside a (faked) git checkout whose verifier/ is a symlink."""
    base = tmp_path / "tasks" / "base"
    (base / "verifier").mkdir(parents=True)
    (base / "verifier" / "test.sh").write_text("#!/bin/bash\n")
    task = tmp_path / "tasks" / "variant"
    task.mkdir()
    (task / "task.md").write_text("---\n---\n")
    (task / "verifier").symlink_to(Path("..") / "base" / "verifier")
    monkeypatch.setattr(benchmark_repos, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        benchmark_repos, "_repo_slug_from_git_root", lambda root: "org/repo"
    )
    monkeypatch.setattr(benchmark_repos, "_git_stdout", lambda root, *args: "0" * 40)
    return task


def test_inference_skips_hashing_a_task_with_symlinks(symlinked_task: Path) -> None:
    provenance = infer_task_source_provenance(symlinked_task)
    assert provenance is None or "file_hashes" not in provenance


def test_result_payload_of_an_errored_symlinked_task_builds(
    symlinked_task: Path,
) -> None:
    from benchflow._utils.evaluation_results import rollout_result_payload
    from benchflow.models import RolloutResult

    result = RolloutResult(task_name="variant", error="Unexpected: something failed")
    payload = rollout_result_payload(
        result,
        source_provenance=None,
        tasks_dir=symlinked_task.parent,
        task_name="variant",
    )
    assert payload["error"] == "Unexpected: something failed"
