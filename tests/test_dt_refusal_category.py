"""A runtime refusal is recorded as ``unsupported_task_feature`` with code R401.

Round PY8, target DT1. Before this change a task the runtime capability
check refuses (``UnsupportedTaskFeatureError``,
src/benchflow/task/runtime_capabilities.py) was recorded with
``error_category`` "other": the engine bench's py-0.8 ``snapshot-hook``
trial (tests/fixtures/dt/py-0.8__snapshot-hook.json) says so, while Rust
0.8.2 records the same refusal as ``unsupported_task_feature`` with code
R401 (integ/p1:crates/benchflow-run/src/record.rs:743). The recorded error
is ``describe_exception(e)`` (src/benchflow/rollout/__init__.py:2984-2993),
"UnsupportedTaskFeatureError: Task uses ...", so the classifier matches the
header as a substring, first, before any marker the free-text reasons could
contain.
"""

from __future__ import annotations

import json
import typing
from pathlib import Path
from unittest.mock import patch

import pytest

from benchflow._utils.scoring import ERROR_CATEGORIES, classify_error
from benchflow.diagnostics import RolloutDiagnostics
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.task.runtime_capabilities import (
    UnsupportedTaskFeature,
    UnsupportedTaskFeatureError,
    validate_task_runtime_support,
)

FIXTURES = Path(__file__).parent / "fixtures" / "dt"
CATEGORY = "unsupported_task_feature"


def _archived_error() -> str:
    doc = json.loads((FIXTURES / "py-0.8__snapshot-hook.json").read_text())
    (result,) = [v for k, v in doc["files"].items() if k.endswith("/result.json")]
    assert result["error_category"] == "other"  # what py-0.8 recorded
    return result["error"]


def test_the_archived_snapshot_hook_error_classifies_as_unsupported_task_feature():
    error = _archived_error()
    assert error.startswith(
        "UnsupportedTaskFeatureError: Task uses parsed runtime features that "
        "BenchFlow cannot execute on the selected sandbox:\n- verifier.collect: "
    )

    assert classify_error(error) == CATEGORY


@pytest.mark.parametrize(
    "reason",
    [
        "the hook timed out on the reference run",
        "its image's install failed on arm64",
        "agent idle for 600s in the hook",
        "sandbox startup needs a second service",
        "ACP error -32603 when the hook ran",
    ],
)
def test_reason_text_after_the_header_cannot_claim_the_refusal(reason):
    """The feature reasons are free text: "timed out" (TIMED_OUT) or
    "install failed" (INSTALL_FAILED, which is retried) after the header
    must not win."""
    error = "UnsupportedTaskFeatureError: " + str(
        UnsupportedTaskFeatureError(
            [
                UnsupportedTaskFeature(
                    path="verifier.collect", reason=reason, sandbox="docker"
                )
            ]
        )
    )

    assert classify_error(error) == CATEGORY


def test_the_bare_message_and_other_cases_still_classify():
    bare = str(
        UnsupportedTaskFeatureError(
            [UnsupportedTaskFeature(path="steps", reason="x", sandbox="daytona")]
        )
    )
    assert classify_error(bare) == CATEGORY
    assert classify_error("Agent oracle timed out after 10s") == "timeout"
    assert classify_error("npm install failed: E404") == "install_failure"
    assert classify_error("something else broke") == "other"


def test_every_refused_feature_carries_code_r401(tmp_path: Path):
    feature = UnsupportedTaskFeature(
        path="verifier.collect", reason="r", sandbox="docker"
    )
    assert feature.code == "R401"
    # The format line is unchanged (tools/dt/normalise.py pins it).
    assert feature.format() == "verifier.collect: r (sandbox=docker)"

    from benchflow.task import Task

    task_dir = _collect_task(tmp_path / "tasks")
    issues = validate_task_runtime_support(
        Task(task_dir).config, sandbox="docker", task_dir=task_dir
    )
    assert issues and {issue.code for issue in issues} == {"R401"}


def test_the_fixed_schema_catalog_accepts_the_category():
    """diagnostics.py:771-781 keeps a category only when it is in
    ERROR_CATEGORIES; an unknown one is dropped from results.jsonl."""
    assert CATEGORY in ERROR_CATEGORIES

    block = RolloutDiagnostics().to_results_jsonl_block(
        error_category=CATEGORY, verifier_error_category=None
    )

    assert block == {"schema_version": 1, "error_category": CATEGORY}


COLLECT_TOML = """\
version = "1.0"

[agent]
timeout_sec = 60.0

[environment]
build_timeout_sec = 60.0
workdir = "/app"

[verifier]
timeout_sec = 60.0

[[steps]]
name = "one"

[[steps.verifier.collect]]
command = "echo collected > /tmp/collected.txt"
timeout_sec = 30.0
"""


def _collect_task(root: Path) -> Path:
    task = root / "collect-hook"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /app\n"
    )
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    (task / "task.toml").write_text(COLLECT_TOML)
    (task / "instruction.md").write_text("Do nothing.\n")
    return task


class _FakeSandbox:
    """Stands in for DockerSandbox: a launch that got past the refusal would
    build one, and the test would see it."""

    built: typing.ClassVar[list[dict]] = []

    def __init__(self, **kwargs):
        type(self).built.append(kwargs)

    def __getattr__(self, name):
        raise AssertionError(f"the refused task reached the sandbox ({name})")


@pytest.mark.asyncio
async def test_a_refused_split_task_records_the_category_through_the_real_except_path(
    tmp_path: Path,
):
    """The real path: Evaluation -> Rollout.setup -> sandbox/setup.py
    _create_sandbox_environment (:715) -> _validate_task_runtime_for_launch
    (:892) -> raise_for_task_runtime_support (runtime_capabilities.py:151),
    caught at rollout/__init__.py:2984-2993. Base: error_category "other"."""
    tasks_dir = tmp_path / "tasks"
    _collect_task(tasks_dir)
    jobs_dir = tmp_path / "jobs"
    _FakeSandbox.built = []
    job = Evaluation(
        tasks_dir=tasks_dir,
        jobs_dir=jobs_dir,
        config=EvaluationConfig(
            agent="oracle", environment="docker", retry=RetryConfig(max_retries=0)
        ),
        job_name="refused",
        preflight=False,
    )

    with patch("benchflow.sandbox.docker.DockerSandbox", _FakeSandbox):
        result = await job.run()

    assert _FakeSandbox.built == []
    (result_path,) = (jobs_dir / "refused").glob("*/result.json")
    record = json.loads(result_path.read_text())
    assert record["error"].startswith(
        "UnsupportedTaskFeatureError: Task uses parsed runtime features"
    )
    assert "- steps[0].verifier.collect: " in record["error"]
    assert record["error_category"] == CATEGORY
    assert record["rewards"] is None
    summary = json.loads((jobs_dir / "refused" / "summary.json").read_text())
    assert summary["error_categories"] == {CATEGORY: 1}
    assert summary["errored"] == 1 and summary["passed"] == 0
    assert result.errored == 1
    # Never retried: another attempt refuses the same way.
    assert not RetryConfig().should_retry(record["error"], category=CATEGORY)


def test_old_job_folders_with_other_still_read(tmp_path: Path):
    """A py-0.8 job folder written before this change (category "other")."""
    import benchflow as bf

    doc = json.loads((FIXTURES / "py-0.8__snapshot-hook.json").read_text())
    job_dir = tmp_path / doc["job_name"]
    for rel, content in doc["files"].items():
        path = job_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content) if rel.endswith(".json") else content)

    job = bf.load_job(job_dir)

    (trial,) = job.trials
    assert trial.result.error_category == "other"
    assert trial.execution == "errored"
    assert trial.assessment == "unscored"
    exported = trial.to_json_dict(include_trajectory=False)
    assert exported["error_category"] == "other"
    assert exported["score_outcome"] == "errored"
