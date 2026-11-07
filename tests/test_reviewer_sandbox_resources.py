"""Guards the in-run reviewer's sandbox size and timeouts
(benchflow-ai/benchflow#1205) against review/wrapper.py:196-200 at e118a8d3:
every reviewer got 1 CPU, 2048 MB of memory and 4096 MB of disk, its evidence
install kept install_review_evidence's 600 s (review/runner.py:581 passed no
timeout), and its rollout never took the run's sandbox_setup_timeout, so the
in-sandbox model proxy install had 600 s. Workspaces of 3.3 and 5.9 GB failed
every review retry. A benchmark's own infrastructure works around it with
patches.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from typer.testing import CliRunner

import benchflow
from benchflow.cli.main import app
from benchflow.review import automatic, evidence, persistence
from benchflow.review.config import load_rubric
from benchflow.review.options import ReviewerConfig
from benchflow.review.resume import resume_review
from benchflow.review.runner import run_review
from benchflow.review.wrapper import assemble_review_task
from benchflow.rollout import RolloutConfig
from benchflow.rollout._review import prepare_terminal_review
from tests.test_review_resume import _complete, _parent, saved_trial  # noqa: F401
from tests.test_review_runtime import (
    WEIGHTED_RUBRIC,
    FakeRun,
    good_weighted_review,
    make_rollout,
    make_task,
)
from tests.test_review_runtime_automatic import make_bundle

GIB = 1024**3


def _sandbox(task_md: str) -> tuple[int, int]:
    memory = re.search(r"^  memory_mb: (\d+)$", task_md, re.MULTILINE)
    storage = re.search(r"^  storage_mb: (\d+)$", task_md, re.MULTILINE)
    assert memory and storage, task_md
    return int(memory.group(1)), int(storage.group(1))


@pytest.fixture
def evidence_of(monkeypatch):
    """Report a bundle's files as *size* bytes, as its manifests would."""

    def report(size: int) -> None:
        monkeypatch.setattr(
            evidence, "evidence_bytes", lambda bundle: size, raising=False
        )

    return report


@pytest.fixture
def review_inputs(tmp_path):
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    rollout = make_rollout(tmp_path / "jobs", "rollout-a", reward=1.0)
    rubric_path = task / "verifier/rubric.json"
    return task, rollout, load_rubric(rubric_path), rubric_path


@pytest.fixture
def auth(monkeypatch):
    monkeypatch.setattr(
        "benchflow.agents.env.resolve_agent_env",
        Mock(return_value={"AZURE_API_KEY": "private-test-secret"}),
    )
    monkeypatch.setattr("benchflow.review.preflight.validate_reviewer_backend", Mock())


def test_a_6_gb_bundle_gets_a_reviewer_disk_above_4096_mb(
    tmp_path, review_inputs, evidence_of
):
    """Red at e118a8d3: the wrapper declared 4096 MB whatever it carried."""
    _, rollout, rubric, _ = review_inputs
    evidence_of(6 * GIB)

    dest, _ = assemble_review_task(
        rollout, None, rubric, tmp_path / "w", workspace_bundle=make_bundle(tmp_path)
    )

    _, storage = _sandbox((dest / "task.md").read_text())
    # The workspace unpacked beside its archive, the uploads, and the margin.
    assert storage > 6 * 1024 + 2048


def test_the_reviewer_disk_stops_at_the_daytona_limit(
    tmp_path, review_inputs, evidence_of
):
    _, rollout, rubric, _ = review_inputs
    evidence_of(12 * GIB)

    daytona, _ = assemble_review_task(
        rollout,
        None,
        rubric,
        tmp_path / "daytona",
        workspace_bundle=make_bundle(tmp_path / "a"),
        environment="daytona",
    )
    docker, _ = assemble_review_task(
        rollout,
        None,
        rubric,
        tmp_path / "docker",
        workspace_bundle=make_bundle(tmp_path / "b"),
        environment="docker",
    )

    assert _sandbox((daytona / "task.md").read_text())[1] == 10240
    assert _sandbox((docker / "task.md").read_text())[1] > 12 * 1024 + 2048


def test_the_reviewer_disk_rule():
    from benchflow.review.wrapper import reviewer_storage_mb

    assert reviewer_storage_mb(0, "docker") == 4096
    assert reviewer_storage_mb(GIB, "docker") == 4096
    assert reviewer_storage_mb(6 * GIB, "docker") == 8192
    assert reviewer_storage_mb(6 * GIB, "daytona") == 8192
    assert reviewer_storage_mb(20 * GIB, "daytona") == 10240
    assert reviewer_storage_mb(20 * GIB, "docker") == 20 * 1024 + 2048


async def test_reviewer_settings_reach_the_reviewer_rollout(
    tmp_path, monkeypatch, review_inputs
):
    """Red at e118a8d3: ReviewerConfig had none of these settings."""
    task, rollout, rubric, rubric_path = review_inputs
    fake = FakeRun(review_payload=good_weighted_review())
    monkeypatch.setattr(benchflow, "run", fake)

    await run_review(
        rollout,
        task,
        rubric,
        rubric_path,
        ReviewerConfig(
            agent="codex",
            model="azure/gpt5.6terra",
            storage_mb=9000,
            memory_mb=4096,
            setup_timeout_sec=1800,
            evidence_timeout_sec=2400,
        ),
        tmp_path / "review-output",
        deterministic_pass=True,
        workspace_bundle=make_bundle(tmp_path),
    )

    (config,) = fake.configs
    # The model proxy's install takes max(600, sandbox_setup_timeout).
    assert config.sandbox_setup_timeout == 1800
    assert config.pre_agent_hooks[0].keywords["timeout_sec"] == 2400
    assert _sandbox(fake.task_docs[0]) == (4096, 9000)
    (record,) = (tmp_path / "review-output").rglob("reviewer-environment.json")
    assert {
        key: value
        for key, value in json.loads(record.read_text()).items()
        if key.endswith(("_mb", "_sec"))
    } == {
        "memory_mb": 4096,
        "storage_mb": 9000,
        "setup_timeout_sec": 1800,
        "evidence_timeout_sec": 2400,
    }


async def test_by_default_the_reviewer_follows_the_run_and_the_evidence(
    tmp_path, monkeypatch, review_inputs, evidence_of, auth
):
    """Red at e118a8d3: the run's 1800 s setup budget never reached the
    reviewer (its proxy install got 600 s) and a 6 GB evidence install got
    600 s per step."""
    task, rollout, _, _ = review_inputs
    run = SimpleNamespace(
        _config=RolloutConfig(
            task_path=task,
            sandbox_setup_timeout=1800,
            reviewer=ReviewerConfig(agent="codex", model="azure/gpt5.6terra"),
        )
    )
    prepare_terminal_review(run)
    plan = run._review_plan
    assert plan.config.setup_timeout_sec == 1800
    fake = FakeRun(review_payload=good_weighted_review())
    monkeypatch.setattr(benchflow, "run", fake)
    evidence_of(6 * GIB)

    await run_review(
        rollout,
        task,
        plan.rubric,
        plan.rubric_path,
        plan.config,
        tmp_path / "review-output",
        deterministic_pass=True,
        workspace_bundle=make_bundle(tmp_path),
    )

    (config,) = fake.configs
    assert config.sandbox_setup_timeout == 1800
    assert config.pre_agent_hooks[0].keywords["timeout_sec"] == 6 * 1024
    assert _sandbox(fake.task_docs[0])[1] > 6 * 1024 + 2048


async def test_a_small_bundle_keeps_the_old_sizes(tmp_path, monkeypatch, review_inputs):
    """A control: 2048 MB, 4096 MB and 600 s per evidence step, as before."""
    task, rollout, rubric, rubric_path = review_inputs
    fake = FakeRun(review_payload=good_weighted_review())
    monkeypatch.setattr(benchflow, "run", fake)

    await run_review(
        rollout,
        task,
        rubric,
        rubric_path,
        ReviewerConfig(agent="codex", model="azure/gpt5.6terra"),
        tmp_path / "review-output",
        deterministic_pass=True,
        workspace_bundle=make_bundle(tmp_path),
    )

    (config,) = fake.configs
    assert _sandbox(fake.task_docs[0]) == (2048, 4096)
    assert config.pre_agent_hooks[0].keywords.get("timeout_sec", 600) == 600


async def test_a_resumed_review_takes_the_recorded_runs_setup_timeout(
    saved_trial,  # noqa: F811
    monkeypatch,
):
    """Red at e118a8d3: bench eval score gave the reviewer no setup timeout."""
    rollout, task = saved_trial
    config = json.loads((rollout / "config.json").read_text())
    (rollout / "config.json").write_text(
        json.dumps({**config, "sandbox_setup_timeout": 1800})
    )
    prepare = Mock(return_value=SimpleNamespace(config=ReviewerConfig()))
    monkeypatch.setattr(automatic, "prepare_review", prepare)
    monkeypatch.setattr(automatic, "finish_review", AsyncMock(return_value=_complete()))
    monkeypatch.setattr(
        persistence, "commit_scoring_result", lambda path, score: _parent(path, score)
    )

    await resume_review(rollout, tasks_root=task.parent)

    assert prepare.call_args.kwargs.get("sandbox_setup_timeout") == 1800


def test_eval_run_and_eval_score_take_the_reviewer_sandbox_flags(
    tmp_path,
    saved_trial,  # noqa: F811
    monkeypatch,
):
    """Red at e118a8d3: neither command had these flags."""
    from benchflow.cli import rescore
    from benchflow.evaluation import Evaluation

    flags = [
        "--reviewer-storage-mb",
        "9000",
        "--reviewer-memory-mb",
        "4096",
        "--reviewer-setup-timeout",
        "1800",
        "--reviewer-evidence-timeout",
        "2400",
    ]
    expected = {
        "storage_mb": 9000,
        "memory_mb": 4096,
        "setup_timeout_sec": 1800,
        "evidence_timeout_sec": 2400,
    }
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    captured = []

    async def capture(self):
        captured.append(self._config.reviewer)
        return SimpleNamespace(
            passed=1, total=1, score=1.0, errored=0, verifier_errored=0
        )

    monkeypatch.setattr(Evaluation, "run", capture)
    ran = CliRunner().invoke(
        app,
        ["eval", "run", "--tasks-dir", str(task), "--agent", "oracle", *flags],
    )
    assert ran.exit_code == 0, ran.output
    assert captured == [ReviewerConfig(**expected)]

    rollout, task_dir = saved_trial
    retry = AsyncMock(return_value=_parent(rollout, _complete()))
    monkeypatch.setattr(rescore, "resume_review", retry)
    monkeypatch.setattr(rescore, "_apply_dotenv_to_process_env", lambda: None)
    monkeypatch.setattr(rescore, "write_job_results_jsonl", lambda _: None)
    scored = CliRunner().invoke(
        app,
        ["eval", "score", str(rollout), "--tasks-root", str(task_dir.parent), *flags],
    )
    assert scored.exit_code == 0, scored.output
    reviewer = retry.call_args.kwargs["reviewer"]
    assert reviewer.model_dump(include=set(expected)) == expected
