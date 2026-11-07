"""A newcomer's first task runs the same through the CLI and the Python API.

DF-p1-first-run-1/-2 (PY8 dogfood, Oct 18): the README's Python quickstart
wrote a task.md with no ``agent.timeout_sec``. Through the Python API every
oracle rollout ended ``Command timed out after 0 seconds`` and was filed as an
infrastructure failure, because a missing timeout was read as 0. The CLI
refused the same folder as "not a runnable task" (no ``environment/``)
although it names a prebuilt image.

Round 2 (review D-1, D-2, L-1, L-2): a missing
timeout gives the agent no limit of its own, bounded by the rollout's whole
deadline as in 0.7.11; and both front doors refuse a task only for the issues
that stop a run, while ``bench check``'s authoring issues let it run.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from benchflow import RolloutConfig
from benchflow._utils.task_authoring import (
    check_task,
    check_task_warnings,
    runnable_task_issues,
)
from benchflow.evaluation import Evaluation, EvaluationConfig, _budget_overrides
from benchflow.rollout import Rollout
from benchflow.rollout._deadline import _computed_deadline_sec
from benchflow.runtime import check_rollout_config
from benchflow.task import Task

PREBUILT = "sandbox: {docker_image: 'python:3.12-slim', workdir: /app}\n"
REWARD = "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"


def _task(
    root: Path,
    name: str,
    *,
    frontmatter: str = PREBUILT,
    dockerfile: bool = False,
    compose: bool = False,
    verifier: bool = True,
    test_sh: str = REWARD,
    oracle: str = "#!/bin/bash\necho hi > /app/hello.txt\n",
) -> Path:
    """The DF-2 repro folder, with the parts each case changes."""
    task = root / name
    (task / "solution").mkdir(parents=True)
    (task / "task.md").write_text(
        f"---\nschema_version: '1.3'\n{frontmatter}---\n\n## prompt\n\nWrite hello.\n"
    )
    (task / "solution" / "solve.sh").write_text(oracle)
    if verifier:
        (task / "tests").mkdir()
        (task / "tests" / "test.sh").write_text(test_sh)
    if dockerfile:
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    if compose:
        (task / "environment").mkdir(exist_ok=True)
        (task / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    image: python:3.12-slim\n"
        )
    return task


class _FakeInner:
    async def start(self, *a: Any, **kw: Any) -> None: ...

    async def stop(self, *a: Any, **kw: Any) -> None: ...

    async def exec(self, cmd: str, **_: Any) -> Any:
        return type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()

    async def upload_file(self, *a: Any, **kw: Any) -> None: ...

    async def upload_dir(self, *a: Any, **kw: Any) -> None: ...


async def _budget(cfg: RolloutConfig) -> int:
    rollout = Rollout(cfg)
    rollout.use_prebuilt_env(_FakeInner())
    await rollout.setup()
    return rollout._timeout


# --- a missing agent timeout: no limit of its own, the deadline bounds it ----


@pytest.mark.asyncio
async def test_a_task_without_agent_timeout_is_bounded_by_the_rollout_deadline(
    tmp_path: Path,
) -> None:
    """0.7.11's rule: never 0 (integ), never a 900 s cap
    (9994ea93). The agent's budget is the rollout's whole deadline."""
    cfg = RolloutConfig(task_path=_task(tmp_path, "hello"), environment="docker")

    budget = await _budget(cfg)

    assert budget == math.ceil(_computed_deadline_sec(cfg))
    assert budget > 1200, "a 1,200 s oracle must fit"


@pytest.mark.asyncio
async def test_a_timeout_of_0_means_the_tasks_own(tmp_path: Path) -> None:
    """0 is "no limit of its own", as 0.7.11 read it."""
    cfg = RolloutConfig(
        task_path=_task(tmp_path, "hello"), environment="docker", timeout=0
    )
    assert await _budget(cfg) == math.ceil(_computed_deadline_sec(cfg))


@pytest.mark.asyncio
async def test_a_task_timeout_still_wins(tmp_path: Path) -> None:
    task = _task(
        tmp_path, "hello", frontmatter="agent: {timeout_sec: 120}\n" + PREBUILT
    )
    assert await _budget(RolloutConfig(task_path=task, environment="docker")) == 120


def test_the_deadline_and_the_multiplier_read_a_missing_timeout_alike(
    tmp_path: Path,
) -> None:
    """The deadline counts the unset agent's share as 900 s, as 0.7.11's did;
    --timeout-multiplier scales the deadline-bounded budget, so a multiplier
    above 1 never shortens it."""
    unset = _task(tmp_path, "unset")
    explicit = _task(
        tmp_path, "explicit", frontmatter="agent: {timeout_sec: 900}\n" + PREBUILT
    )
    assert Task(unset).config.agent.timeout_sec is None
    assert _computed_deadline_sec(
        RolloutConfig(task_path=unset)
    ) == _computed_deadline_sec(RolloutConfig(task_path=explicit))
    cfg = EvaluationConfig(agent="oracle", timeout_multiplier=2.0)
    unset_budget = math.ceil(_computed_deadline_sec(RolloutConfig(task_path=unset)))
    assert _budget_overrides(cfg, unset)["timeout"] == 2 * unset_budget
    assert _budget_overrides(cfg, explicit)["timeout"] == 1800


@pytest.mark.parametrize("bad", [-5, 0.5, 0.999, float("inf"), float("nan")])
def test_a_timeout_under_1_s_or_not_finite_is_refused(bad: float) -> None:
    """L-1: 0.5 passed the old check and was truncated to an instant timeout."""
    with pytest.raises(ValueError) as excinfo:
        RolloutConfig(task_path=Path("t"), timeout=bad)
    msg = str(excinfo.value)
    assert msg.startswith("timeout must be at least 1 second, got ")
    assert "pass None (or 0) to use the task's agent.timeout_sec" in msg


def test_bench_check_reports_the_effective_bound(tmp_path: Path) -> None:
    task = _task(tmp_path, "hello")
    bound = round(_computed_deadline_sec(RolloutConfig(task_path=task)))
    (warning,) = [w for w in check_task_warnings(task) if "timeout_sec" in w]
    assert warning.startswith(
        "agent.timeout_sec is not set: the agent has no time limit of its own"
    )
    assert f"about {bound} s for this task" in warning
    timed = _task(
        tmp_path, "timed", frontmatter="agent: {timeout_sec: 120}\n" + PREBUILT
    )
    assert not [w for w in check_task_warnings(timed) if "timeout_sec" in w]


# --- one list of the issues that stop a run ----------------------------------


def _with_extra_link(task: Path) -> Path:
    (task / "notes.md").write_text("x")
    (task / "link.md").symlink_to(task / "notes.md")
    return task


def _with_linked_test_sh(task: Path) -> Path:
    real = task / "real-test.sh"
    real.write_text(REWARD)
    (task / "tests" / "test.sh").unlink()
    (task / "tests" / "test.sh").symlink_to(real)
    return task


def _with_linked_verifier_folder(task: Path) -> Path:
    """#411: a linked verifier/ reached the sandbox as an empty /tests."""
    shared = task.parent / f"{task.name}-shared-tests"
    (task / "tests").rename(shared)
    (task / "tests").symlink_to(shared)
    return task


# (name, folder builder kwargs, post-build step, runnable, bench check passes)
CASES = [
    (
        "prebuilt-image-no-environment",
        {"frontmatter": "agent: {timeout_sec: 120}\n" + PREBUILT},
        None,
        True,
        True,
    ),
    ("prebuilt-image-no-timeout", {}, None, True, True),
    ("dockerfile-task", {"frontmatter": "", "dockerfile": True}, None, True, True),
    # Authoring issues: bench check lists them, the task runs (review D-1).
    (
        "ctrf-path",
        {
            "test_sh": "#!/bin/bash\npytest --ctrf /logs/verifier/report.json "
            "/tests/test_x.py || true\n" + REWARD
        },
        None,
        True,
        False,
    ),
    (
        "placeholder-oracle",
        {"oracle": "#!/bin/bash\n# [REPLACE: the solution]\n"},
        None,
        True,
        False,
    ),
    ("unused-symlink", {}, _with_extra_link, True, False),
    # Issues that stop a run: both front doors refuse.
    ("no-image-no-dockerfile", {"frontmatter": ""}, None, False, False),
    ("prebuilt-image-no-verifier", {"verifier": False}, None, False, False),
    ("linked-test-sh", {}, _with_linked_test_sh, False, False),
    # The sandbox builds main from the Dockerfile even when the Compose file
    # names an image ("open Dockerfile: no such file" on Docker).
    (
        "compose-image-no-dockerfile",
        {"frontmatter": "", "compose": True},
        None,
        False,
        False,
    ),
    ("linked-verifier-folder", {}, _with_linked_verifier_folder, False, False),
]


@pytest.mark.parametrize(
    ("name", "kwargs", "post", "runnable", "check_passes"),
    CASES,
    ids=[case[0] for case in CASES],
)
def test_cli_and_python_api_agree_on_what_a_task_is(
    tmp_path: Path,
    name: str,
    kwargs: dict[str, Any],
    post: Any,
    runnable: bool,
    check_passes: bool,
) -> None:
    """The same folder through both front doors: ``bench run`` / ``bench eval
    run`` discovery (``Evaluation``) and the Python API's request check
    (``bf.run``, ``bf.run_sync``, ``bf.run_batch``, ``bf.branch``)."""
    task = _task(tmp_path, name, **kwargs)
    if post is not None:
        post(task)

    cli_selects = Evaluation(
        tasks_dir=task, jobs_dir=tmp_path / "jobs"
    )._get_task_dirs()
    try:
        check_rollout_config(RolloutConfig(task_path=task, agent="oracle"))
        python_runs, python_error = True, ""
    except ValueError as exc:
        python_runs, python_error = False, str(exc)

    assert bool(cli_selects) is runnable
    assert python_runs is runnable, python_error
    assert (runnable_task_issues(task) == []) is runnable
    assert (check_task(task) == []) is check_passes
    if not runnable:
        # Both name the same issues, and the command that explains them.
        issues = runnable_task_issues(task)
        assert set(issues) <= set(check_task(task))
        assert f"{task} is not a runnable task" in python_error
        assert all(issue in python_error for issue in issues)
        assert f"`bench check {task}` explains each issue" in python_error
        message = Evaluation(
            tasks_dir=task, jobs_dir=tmp_path / "jobs"
        )._empty_selection_message()
        assert all(issue in message for issue in issues)


def test_a_manifest_supplies_the_image(tmp_path: Path) -> None:
    """A run whose environment manifest names the image needs no Dockerfile."""
    task = _task(tmp_path, "bare", frontmatter="")
    assert runnable_task_issues(task) != []
    assert runnable_task_issues(task, image_supplied=True) == []


def test_bench_check_explains_the_prebuilt_image_alternative(tmp_path: Path) -> None:
    task = _task(tmp_path, "bare", frontmatter="")
    issues = check_task(task)
    assert (
        "Missing environment/Dockerfile (or name a prebuilt image in "
        "sandbox.docker_image)"
    ) in issues


def test_a_prebuilt_image_task_passes_bench_check(tmp_path: Path) -> None:
    """DF-2's folder: ``bench check`` listed environment/ and the Dockerfile."""
    assert check_task(_task(tmp_path, "hello")) == []


# --- a batch refuses a task, not the batch (review D-1) ----------------------


def test_a_batch_records_a_refused_task_and_runs_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchflow as bf
    from benchflow import batch
    from benchflow.models import RolloutResult

    good = _task(tmp_path, "good")
    broken = _task(tmp_path, "broken", verifier=False)
    ran: list[str] = []

    async def fake_arun(config: RolloutConfig) -> RolloutResult:
        ran.append(Path(config.task_path).name)
        return RolloutResult(
            task_name=Path(config.task_path).name, rewards={"reward": 1.0}
        )

    monkeypatch.setattr(batch, "arun", fake_arun)
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    results = bf.run_batch(
        [
            RolloutConfig(task_path=broken, agent="oracle", jobs_dir=tmp_path / "j"),
            RolloutConfig(task_path=good, agent="oracle", jobs_dir=tmp_path / "j"),
        ],
        concurrency=2,
    )

    assert ran == ["good"]
    refused, done = list(results)
    assert refused.task_name == "broken"
    assert refused.error and "is not a runnable task" in refused.error
    assert "`bench check" in refused.error
    assert done.task_name == "good" and done.error is None
