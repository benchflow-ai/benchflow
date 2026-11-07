"""Results and run correctness, from the 0.8.3 release-candidate dogfood
(results/DOGFOOD-RC p1, p3 and p5: DFR-p5-5, -6, -13, -18, -19; DFR-p3-10,
-11; DFR-p1-2, -7, -11, -17).

Each test is the dogfooder's repro on files (the job folders a run writes)
or on a run with the rollout faked, and checks the words a user reads.
DFR-p5-4 (a run scores what it selected) is in test_resume_settings_match.py;
the stop on real Docker is in test_stop_summary_docker.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.evaluation import (
    EmptyTaskSelectionError,
    Evaluation,
    EvaluationConfig,
    MalformedTaskError,
    RetryConfig,
)
from benchflow.models import RolloutResult
from tests.test_python_sdk_load_job import _trial

WIDE = {"COLUMNS": "240", "NO_COLOR": "1"}


def _invoke(*args: str):
    return CliRunner().invoke(app, list(args), env=WIDE)


def _config(**overrides) -> EvaluationConfig:
    return EvaluationConfig(
        **{"agent": "oracle", "retry": RetryConfig(max_retries=0), **overrides}
    )


def _write_task(task_dir: Path) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    return task_dir


# ── DFR-p5-5: JSON denominators of a job of only control runs ─────────────


def test_json_denominators_count_a_control_only_job(tmp_path: Path):
    """`bench results --json` on 4 passing oracle trials gave
    denominators.passed 0 while the text said 4/4."""
    job = tmp_path / "jobs" / "tty4"
    for task in ("a", "b", "c", "d"):
        _trial(job, task, reward=1.0, agent="oracle", model=None)
    text = _invoke("results", str(job))
    assert "4/4 attempted passed" in text.output, text.output
    out = _invoke("results", "--json", str(job))
    assert out.exit_code == 0, out.output
    doc = json.loads(out.stdout)
    d = doc["denominators"]
    assert (d["attempted"], d["scored"], d["passed"]) == (4, 4, 4)
    assert d["controls_excluded"] == 0
    assert doc["solve_rates"]["trials"] == 4
    assert doc["denominators_with_controls"]["passed"] == 4


def test_json_denominators_still_leave_controls_out_of_a_mixed_job(tmp_path: Path):
    job = tmp_path / "mixed"
    _trial(job, "a", reward=1.0)
    _trial(job, "a", reward=1.0, agent="oracle", model=None, suffix="0000oracl")
    d = json.loads(_invoke("results", "--json", str(job)).stdout)["denominators"]
    assert (d["attempted"], d["controls_excluded"]) == (1, 1)


# ── DFR-p3-10: running, interrupted and never started ─────────────────────


def _unfinished_job(job: Path) -> Path:
    """3 planned tasks: one finished, one started with no result, one not started."""
    job.mkdir(parents=True)
    (job / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "planned_tasks": ["t0", "t1", "t2"]})
    )
    _trial(job, "t0", reward=1.0, agent="oracle", model=None)
    cut = job / "t1__deadbeef"
    cut.mkdir()
    (cut / "config.json").write_text(
        json.dumps({"agent": "oracle", "task_path": "tasks/t1"})
    )
    return job


def _lock(job: Path, pid: int) -> None:
    (job / ".evaluation.lock").write_text(
        json.dumps({"pid": pid, "host": socket.gethostname(), "started_at": "x"})
    )


def _gone_pid() -> int:
    """The pid of a process that has exited."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_a_running_job_shows_running_trials_without_a_resume_hint(tmp_path: Path):
    job = _unfinished_job(tmp_path / "jobs" / "live")
    _lock(job, os.getpid())
    out = _invoke("results", str(job))
    assert out.exit_code == 0, out.output
    assert (
        f"Running (process {os.getpid()}): 1 of 3 planned tasks finished, "
        "1 running, 1 not started yet" in out.output
    ), out.output
    assert "bench eval resume" not in out.output
    assert "interrupted" not in out.output
    metrics = _invoke("eval", "metrics", str(job))
    assert "3 tasks, 1 ran, 1 running, 1 not started yet" in metrics.output
    listed = _invoke("eval", "list", str(tmp_path / "jobs"))
    assert "running: 1 of 3 tasks have a result" in listed.output, listed.output
    assert "bench eval resume" not in listed.output


def test_a_stopped_job_tells_started_from_never_started(tmp_path: Path):
    """After a stop, `eval metrics` said "89 tasks, 50 ran, 39 not run" though
    all 39 had started."""
    job = _unfinished_job(tmp_path / "jobs" / "stopped")
    _lock(job, _gone_pid())  # left by a process that is gone
    out = _invoke("results", str(job))
    assert (
        "Not finished: 1 of 3 planned tasks finished, 1 interrupted, 1 not started"
        in out.output
    ), out.output
    assert "bench eval resume" in out.output
    metrics = _invoke("eval", "metrics", str(job))
    assert "3 tasks, 1 ran, 1 interrupted, 1 not started" in metrics.output, (
        metrics.output
    )
    assert "not run" not in metrics.output


def test_a_running_job_with_no_result_yet_is_not_called_stopped(tmp_path: Path):
    job = tmp_path / "jobs" / "first"
    job.mkdir(parents=True)
    (job / "evaluation.json").write_text(json.dumps({"schema_version": 1}))
    cut = job / "t1__deadbeef"
    cut.mkdir()
    (cut / "config.json").write_text(json.dumps({"agent": "oracle"}))
    _lock(job, os.getpid())
    out = _invoke("results", str(job))
    assert "is a job still running in process" in out.output, out.output
    assert "stopped" not in out.output and "bench eval resume" not in out.output


# ── DFR-p3-11: an orderly stop writes a partial summary ───────────────────


def _stopping_job(tmp_path: Path) -> tuple[Evaluation, asyncio.Event]:
    tasks = tmp_path / "tasks"
    for name in ("t0", "t1", "t2"):
        _write_task(tasks / name)
    job = Evaluation(
        tasks_dir=tasks, jobs_dir=tmp_path / "jobs", config=_config(concurrency=2)
    )
    finished = asyncio.Event()

    async def fake_run(task_path, _cfg):
        if task_path.name == "t2":
            await asyncio.Event().wait()  # waits for a slot: never starts
        trial = job.job_dir / f"{task_path.name}__abcd1234"
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "config.json").write_text(
            json.dumps({"agent": "oracle", "task_path": str(task_path)})
        )
        if task_path.name != "t0":
            await asyncio.Event().wait()  # runs until the job is stopped
        payload = {
            "task_name": task_path.name,
            "rollout_name": trial.name,
            "rewards": {"reward": 1.0},
            "agent": "oracle",
        }
        (trial / "result.json").write_text(json.dumps(payload))
        finished.set()
        return RolloutResult(
            task_name=task_path.name, rewards={"reward": 1.0}, rollout_dir=trial
        )

    job._run_single_task = AsyncMock(side_effect=fake_run)
    return job, finished


def test_a_stopped_job_writes_a_partial_summary(tmp_path: Path):
    job, finished = _stopping_job(tmp_path)

    async def main():
        runner = asyncio.create_task(job.run())
        await finished.wait()
        await asyncio.sleep(0.05)
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner

    asyncio.run(main())
    summary = json.loads((job.job_dir / "summary.json").read_text())
    assert summary["stopped"] is True
    assert (summary["total"], summary["passed"]) == (1, 1)
    assert (summary["planned"], summary["ran"], summary["not_run"]) == (3, 1, 2)
    assert (summary["interrupted"], summary["not_started"]) == (1, 1)
    assert summary["interrupted_tasks"] == ["t1"]
    assert summary["not_run_tasks"] == ["t1", "t2"]
    result = job.result
    assert result is not None and result.stopped and result.interrupted == 1
    # The partial summary lists the job as stopped, with its resume line.
    listed = _invoke("eval", "list", str(tmp_path / "jobs"))
    assert "stopped: 1 interrupted, 1 not started" in listed.output, listed.output
    assert "bench eval resume" in listed.output


def test_summary_out_is_written_for_a_stopped_run(tmp_path: Path):
    from types import SimpleNamespace

    from benchflow.cli._termination import RunStopped
    from benchflow.cli.main import _write_stopped_summary_out

    job, finished = _stopping_job(tmp_path)

    async def main():
        runner = asyncio.create_task(job.run())
        await finished.wait()
        await asyncio.sleep(0.05)
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner

    asyncio.run(main())
    out = tmp_path / "ci" / "run.json"
    req = SimpleNamespace(summary_out=out, fail_under=0.9, fail_on=None)
    _write_stopped_summary_out(req, job, RunStopped("Terminated (SIGTERM)", 143))
    doc = json.loads(out.read_text())
    assert doc["kind"] == "benchflow.run-summary"
    assert doc["stopped"] is True and doc["exit_code"] == 143
    assert (doc["total"], doc["passed"], doc["not_run"], doc["interrupted"]) == (
        1,
        1,
        2,
        1,
    )
    assert doc["gate"]["failed"] == []


# ── DFR-p1-2: every selected task malformed ───────────────────────────────


def test_all_selected_tasks_malformed_is_not_a_filter_mismatch(tmp_path: Path):
    tasks = tmp_path / "tasks"
    _write_task(tasks / "healthy")
    broken = tasks / "citation-check"
    broken.mkdir(parents=True)
    (broken / "task.toml").write_text("[verifier\ntimeout_sec = 60\n")
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        config=_config(include_tasks={"citation-check"}),
    )
    job._run_single_task = AsyncMock()
    with pytest.raises(EmptyTaskSelectionError) as caught:
        asyncio.run(job.run())
    message, hint = str(caught.value), caught.value.hint
    assert "1 selected task is malformed, so none can run" in message, message
    assert "first: citation-check:" in message
    assert "1 other task(s) did not match --include/--exclude" in message
    assert "include/exclude filtering" not in message
    assert hint is not None and hint.startswith("fix citation-check's task file (")
    assert "bench check" in hint and "citation-check" in hint
    job._run_single_task.assert_not_called()


def test_the_parse_error_is_named_without_the_task_files_path(tmp_path: Path):
    """skillsbench@1.1: the error named the snapshot path first, and the
    `next:` line cut it to "task.md parse error:…"."""
    tasks = tmp_path / "tasks"
    broken = tasks / "citation-check"
    broken.mkdir(parents=True)
    (broken / "task.md").write_text(
        "---\nschema_version: '1.3'\nenvironment:\n  cpus: 1\n---\n# x\n\n"
        "## prompt\n\nDo it.\n"
    )
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        config=_config(include_tasks={"citation-check"}),
    )
    with pytest.raises(EmptyTaskSelectionError) as caught:
        asyncio.run(job.run())
    hint = caught.value.hint
    assert "'environment' key was renamed" in hint, hint
    assert str(broken / "task.md") not in hint
    assert str(broken / "task.md") not in str(caught.value)


def test_the_cli_names_the_parse_error_and_bench_check(tmp_path: Path, monkeypatch):
    tasks = tmp_path / "tasks"
    broken = tasks / "citation-check"
    broken.mkdir(parents=True)
    (broken / "task.toml").write_text("[verifier\n")
    _write_task(tasks / "other")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    out = _invoke("run", "tasks", "-i", "citation-check", "--jobs-dir", "jobs")
    assert out.exit_code == 1, out.output
    assert "malformed" in out.output
    assert "next: fix citation-check's task file" in out.output, out.output
    assert "bench check" in out.output
    assert "check --include and --exclude" not in out.output


# ── DFR-p1-7: a scaffold with its placeholders is refused before the build ─


def test_a_scaffold_with_placeholders_is_refused_before_any_trial(tmp_path: Path):
    from benchflow._utils.task_authoring import check_task
    from benchflow._utils.task_authoring.scaffolding import init_task

    task = init_task("my-task", parent_dir=tmp_path / "tasks")
    found = [i for i in check_task(task) if "unreplaced placeholder" in i]
    assert found  # what bench check lists
    job = Evaluation(tasks_dir=task, jobs_dir=tmp_path / "jobs", config=_config())
    job._run_single_task = AsyncMock()
    with pytest.raises(MalformedTaskError) as caught:
        asyncio.run(job.run())
    message = str(caught.value)
    assert "1 task(s) still hold [REPLACE: ...] placeholders" in message, message
    assert "would build and then score 0: my-task" in message
    assert found[0] in message
    assert "bench check" in caught.value.hint and "my-task" in caught.value.hint
    job._run_single_task.assert_not_called()
    # No job folder is left behind.
    assert not any((tmp_path / "jobs").glob("*/*__*"))


def test_a_filled_task_still_runs(tmp_path: Path):
    task = _write_task(tmp_path / "tasks" / "ok")
    job = Evaluation(tasks_dir=task, jobs_dir=tmp_path / "jobs", config=_config())
    job._run_single_task = AsyncMock(
        return_value=RolloutResult(task_name="ok", rewards={"reward": 1.0})
    )
    assert asyncio.run(job.run()).passed == 1


# ── DFR-p5-6 (a): the sandbox-start error points at the cause ─────────────

_OFFLINE = (
    "RuntimeUnavailable: sandbox startup failed: the protected verifier runtime "
    "could not be prepared before the agent's turn: no network to PyPI "
    "(ConnectError), and this host's cache has no checked copy of "
    "pytest==8.4.2's file list: run a task once where PyPI is reachable"
)


def test_the_sandbox_start_cause_is_on_the_summary_line():
    from benchflow.failures import cause_of

    cause = cause_of(
        {"task_name": "fix-git", "error": _OFFLINE, "error_category": "sandbox_setup"}
    )
    assert cause is not None
    assert cause.label.startswith("the sandbox did not start: the protected")
    assert "no network to PyPI (ConnectError)" in cause.label
    assert "sandbox_startup_info" not in cause.next_step
    assert "error in result.json" in cause.next_step


def test_the_summary_warning_and_explain_point_at_error():
    from benchflow.diagnostics import SandboxStartupDiagnostic, summary_warning

    line = summary_warning(SandboxStartupDiagnostic, 1, 1)
    assert "error in result.json has the cause" in line
    assert "check sandbox_startup_info in result.json for diagnostics" not in line
    out = _invoke("explain", "sandbox_setup")
    assert "sandbox_startup_info in result.json has the error" not in out.output
    assert "error in" in out.output


# ── DFR-p5-6 (b): offline, the verifier runtime uses its cache ────────────


def _no_network(url: str):
    raise httpx.ConnectError("no route", request=httpx.Request("GET", url))


def _fake_wheel() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as wheel:
        wheel.writestr("fakeplug/__init__.py", "X = 1\n")
        wheel.writestr(
            "fakeplug-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: fakeplug\nVersion: 1.0\n",
        )
        wheel.writestr(
            "fakeplug-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        wheel.writestr("fakeplug-1.0.dist-info/RECORD", "")
    return buf.getvalue()


def _warm_cache(home: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    cache = home / ".cache" / "benchflow" / "verifier-runtime"
    (cache / "wheels").mkdir(parents=True)
    (cache / "index").mkdir(parents=True)
    (cache / "index" / "fakeplug-1.0.json").write_text(
        json.dumps(
            [
                {
                    "filename": "fakeplug-1.0-py3-none-any.whl",
                    "packagetype": "bdist_wheel",
                    "digests": {"sha256": digest},
                    "url": "https://files.pythonhosted.org/x/fakeplug-1.0-py3-none-any.whl",
                }
            ]
        )
    )
    (cache / "wheels" / f"{digest}.whl").write_bytes(data)
    return digest


def test_offline_with_a_warm_cache_needs_no_network(tmp_path: Path, monkeypatch):
    from benchflow.sandbox import _verifier_runtime as vr

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("BENCHFLOW_PLUGIN_RELEASES", raising=False)
    digest = _warm_cache(tmp_path, _fake_wheel())
    monkeypatch.setattr(vr, "_http_get", _no_network)
    resolution = vr.resolve(["fakeplug==1.0"], ctrf=False, interpreter=None)
    assert [(w.name, w.sha256) for w in resolution.wheels] == [("fakeplug", digest)]


def test_offline_with_a_cold_cache_says_the_network_is_needed(
    tmp_path: Path, monkeypatch
):
    from benchflow.sandbox import _verifier_runtime as vr

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("BENCHFLOW_PLUGIN_RELEASES", raising=False)
    monkeypatch.setattr(vr, "_http_get", _no_network)
    with pytest.raises(vr.RuntimeUnavailable) as caught:
        vr.resolve(["fakeplug==1.0"], ctrf=False, interpreter=None)
    message = str(caught.value)
    assert "no network to PyPI (ConnectError)" in message, message
    assert "fakeplug==1.0's file list" in message
    assert "run a task once where PyPI is reachable" in message
    assert "~/.cache/benchflow/verifier-runtime" in message


def test_offline_a_cached_wheel_with_another_hash_is_never_used(
    tmp_path: Path, monkeypatch
):
    from benchflow.sandbox import _verifier_runtime as vr

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("BENCHFLOW_PLUGIN_RELEASES", raising=False)
    digest = _warm_cache(tmp_path, _fake_wheel())
    cached = tmp_path / ".cache/benchflow/verifier-runtime/wheels" / f"{digest}.whl"
    cached.write_bytes(_fake_wheel() + b"tampered")
    monkeypatch.setattr(vr, "_http_get", _no_network)
    with pytest.raises(vr.RuntimeUnavailable) as caught:
        vr.resolve(["fakeplug==1.0"], ctrf=False, interpreter=None)
    assert "no network to PyPI" in str(caught.value)
    assert "fakeplug-1.0-py3-none-any.whl" in str(caught.value)


# ── DFR-p5-13: which figures are "scored-only" ─────────────────────────────


def test_help_and_summary_agree_on_the_scored_only_figures(tmp_path: Path):
    from benchflow._utils.scoring import MEAN_REWARD_RULE

    assert "BenchFlow 0.7's mean_reward" in MEAN_REWARD_RULE
    help_text = " ".join(_invoke("eval", "metrics", "--help").output.split())
    assert "(BenchFlow 0.8's)" not in help_text
    assert "as BenchFlow 0.7's mean_reward was" in help_text
    job = tmp_path / "j"
    _trial(job, "a", reward=1.0)
    _trial(job, "b", reward=None, error="boom", error_category="other")
    out = " ".join(_invoke("eval", "metrics", str(job)).output.split())
    assert "(BenchFlow 0.8's)" not in out
    assert "as BenchFlow 0.7's mean_reward did" in out


# ── DFR-p1-11: oracle against nop is no settings mismatch ─────────────────


def test_comparing_oracle_and_nop_does_not_warn_about_the_harness(tmp_path: Path):
    a, b = tmp_path / "tb2-oracle", tmp_path / "tb2-nop"
    for task in ("fix-git", "regex-log"):
        _trial(a, task, reward=1.0, agent="oracle", model=None)
        _trial(b, task, reward=0.0, agent="nop", model=None)
    out = _invoke("results", str(a), str(b))
    assert out.exit_code == 0, out.output
    assert "Settings differ" not in out.output, out.output
    assert "2 tasks paired" in out.output


def test_two_agents_still_warn_about_the_harness(tmp_path: Path):
    import benchflow as bf

    a, b = tmp_path / "a", tmp_path / "b"
    _trial(a, "fix-git", reward=1.0, agent="claude-agent-acp")
    _trial(b, "fix-git", reward=1.0, agent="codex-acp")
    with pytest.warns(UserWarning, match="harness"):
        bf.compare(a, b)


# ── DFR-p1-17: the jobs table ──────────────────────────────────────────────


def _summary_job(job: Path, memory: float | None = None) -> None:
    _trial(job, "a", reward=1.0, agent="oracle", model=None)
    summary = {"total": 1, "passed": 1, "score": "100.0%"}
    if memory is not None:
        summary["memory_score"] = memory
    (job / "summary.json").write_text(json.dumps(summary))


def test_the_jobs_table_is_titled_jobs_without_an_empty_memory_column(
    tmp_path: Path,
):
    _summary_job(tmp_path / "jobs" / "oracle")
    out = _invoke("eval", "list", str(tmp_path / "jobs"))
    assert out.exit_code == 0, out.output
    assert "Jobs" in out.output and "Evaluations" not in out.output
    assert "Memory" not in out.output and "—" not in out.output


def test_the_memory_column_shows_when_a_job_has_a_memory_score(tmp_path: Path):
    _summary_job(tmp_path / "jobs" / "learn", memory=0.5)
    _summary_job(tmp_path / "jobs" / "plain")
    out = _invoke("eval", "list", str(tmp_path / "jobs"))
    assert "Memory" in out.output and "50.0%" in out.output and "—" in out.output


# ── DFR-p5-18: train convert on a job of only control runs ────────────────


def test_train_convert_skips_control_runs(tmp_path: Path):
    job = tmp_path / "oracle-job"
    for task in ("a", "b"):
        _trial(job, task, reward=1.0, agent="oracle", model=None)
    out_file = tmp_path / "x.jsonl"
    out = _invoke("train", "convert", str(job), "-o", str(out_file))
    assert out.exit_code == 0, out.output
    assert "0 rows: the job holds only control runs" in out.output, out.output
    assert "cannot read LLM trajectory JSONL" not in out.output
    assert out_file.read_text() == ""


# ── DFR-p5-19: an unknown cost stays null ──────────────────────────────────


def test_metrics_json_keeps_an_unknown_cost_null(tmp_path: Path):
    job = tmp_path / "j"
    _trial(job, "a", reward=1.0, agent="oracle", model=None, cost=None)
    doc = json.loads(_invoke("eval", "metrics", str(job), "--json").stdout)
    assert doc["total_cost_usd"] is None
