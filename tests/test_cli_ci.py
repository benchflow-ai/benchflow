"""CI-facing fixes for bench eval run.

`--agent notanagent` created a job, defaulted the model and failed later
with a traceback about ANTHROPIC_API_KEY. An unknown bare agent name is now
refused before any job directory exists, with the registry's suggestion, the
same way (exit 1) as the other invalid-flag errors of `bench eval run`.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from benchflow.cli.main import app


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "t1"
    task.mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n[agent]\ntimeout_sec = 60\n[environment]\n'
    )
    (task / "instruction.md").write_text("x\n")
    return task


def _run(tmp_path: Path, *args: str):
    return CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(_task(tmp_path)),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--sandbox",
            "daytona",
            *args,
        ],
        terminal_width=200,
    )


def test_unknown_agent_is_a_usage_error_before_any_job(
    tmp_path: Path, monkeypatch
) -> None:
    # No network: the remote agent-manifest lookup is off.
    monkeypatch.setenv("BENCHFLOW_AGENTS_SOURCE", "off")
    result = _run(tmp_path, "--agent", "claud-agent-acp")
    assert result.exit_code == 1, result.output
    assert "claude-agent-acp" in result.output  # the suggestion
    assert "Traceback" not in result.output
    assert not (tmp_path / "jobs").exists()


def test_raw_commands_are_still_accepted_by_the_plan(tmp_path: Path) -> None:
    from benchflow.eval_plan import _normalize_eval_agent

    assert _normalize_eval_agent("myagent --acp") == "myagent --acp"
    assert _normalize_eval_agent("nop") == "nop"


# Resume, --fresh, --job-name, --config.


def _fake_rollouts(monkeypatch) -> list[str]:
    """Evaluation runs without sandboxes: each task writes a scored result."""
    import json

    from benchflow.evaluation import Evaluation
    from benchflow.models import RolloutResult

    ran: list[str] = []

    async def fake(self, task_path, _cfg):
        ran.append(task_path.name)
        trial = self._jobs_dir / self._job_name / f"{task_path.name}__{len(ran):08d}"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_path.name,
                    "rollout_name": trial.name,
                    "rewards": {"reward": 1.0},
                    "agent": "oracle",
                }
            )
        )
        return RolloutResult(
            task_path.name,
            rollout_name=trial.name,
            rewards={"reward": 1.0},
            agent="oracle",
            rollout_dir=trial,
        )

    monkeypatch.setattr(Evaluation, "_run_single_task", fake)
    return ran


def _eval(tmp_path: Path, *args: str):
    return CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(tmp_path / "t1"),
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--sandbox",
            "daytona",
            *args,
        ],
        terminal_width=200,
    )


def test_a_rerun_that_resumes_says_so_and_names_fresh(
    tmp_path: Path, monkeypatch
) -> None:
    _task(tmp_path)
    ran = _fake_rollouts(monkeypatch)
    first = _eval(tmp_path)
    assert first.exit_code == 0, first.output
    second = _eval(tmp_path)
    assert second.exit_code == 0, second.output
    assert ran == ["t1"]  # the rerun resumed; nothing new ran
    assert "--fresh" in second.output and "nothing ran" in second.output


def test_fresh_and_job_name_start_new_jobs(tmp_path: Path, monkeypatch) -> None:
    _task(tmp_path)
    ran = _fake_rollouts(monkeypatch)
    assert _eval(tmp_path).exit_code == 0
    fresh = _eval(tmp_path, "--fresh")
    assert fresh.exit_code == 0, fresh.output
    named = _eval(tmp_path, "--job-name", "ci-run-42")
    assert named.exit_code == 0, named.output
    assert ran == ["t1", "t1", "t1"]
    assert (tmp_path / "jobs" / "ci-run-42").is_dir()
    assert len([d for d in (tmp_path / "jobs").iterdir() if d.is_dir()]) == 3
    both = _eval(tmp_path, "--fresh", "--job-name", "x")
    assert both.exit_code == 1 and "--fresh" in both.output


def test_config_honours_jobs_dir_and_artifact_flags(
    tmp_path: Path, monkeypatch
) -> None:
    import json

    task = _task(tmp_path)
    ran = _fake_rollouts(monkeypatch)
    # An unrelated older job under the YAML's own jobs dir must not be reused.
    (tmp_path / "yaml-jobs" / "2026-01-01__00-00-00").mkdir(parents=True)
    cfg = tmp_path / "job.yaml"
    cfg.write_text(
        f"tasks_dir: {task.parent}\njobs_dir: {tmp_path / 'yaml-jobs'}\nagent: oracle\n"
        "include: [t1]\n"
    )
    out = tmp_path / "rc.json"
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--config",
            str(cfg),
            "--jobs-dir",
            str(tmp_path / "cli-jobs"),
            "--run-config-out",
            str(out),
            "--health-summary-out",
            str(tmp_path / "h.json"),
        ],
        terminal_width=200,
    )
    assert result.exit_code == 0, result.output
    assert ran == ["t1"]
    jobs = [d.name for d in (tmp_path / "cli-jobs").iterdir() if d.is_dir()]
    assert jobs and jobs[0] != "2026-01-01__00-00-00"
    assert json.loads(out.read_text())["schema_version"] == 1
    assert (tmp_path / "h.json").is_file()


def test_eval_resume_finishes_a_job_from_its_folder(
    tmp_path: Path, monkeypatch
) -> None:
    _task(tmp_path)
    ran = _fake_rollouts(monkeypatch)
    assert _eval(tmp_path, "--job-name", "j1").exit_code == 0
    result = CliRunner().invoke(
        app, ["eval", "resume", str(tmp_path / "jobs" / "j1")], terminal_width=200
    )
    assert result.exit_code == 0, result.output
    assert ran == ["t1"] and "nothing ran" in result.output
    missing = CliRunner().invoke(app, ["eval", "resume", str(tmp_path / "nope")])
    assert missing.exit_code == 2


# Gates and a machine-readable result.


def _fake_rewards(
    monkeypatch, reward: float, error_category: str | None = None
) -> None:
    import json

    from benchflow.evaluation import Evaluation
    from benchflow.models import RolloutResult

    async def fake(self, task_path, _cfg):
        trial = self._jobs_dir / self._job_name / f"{task_path.name}__00000001"
        trial.mkdir(parents=True, exist_ok=True)
        error = "agent timed out" if error_category == "timeout" else None
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_path.name,
                    "rollout_name": trial.name,
                    "rewards": {"reward": reward},
                    "agent": "oracle",
                    "error": error,
                    "error_category": error_category,
                }
            )
        )
        return RolloutResult(
            task_path.name,
            rollout_name=trial.name,
            rewards={"reward": reward},
            agent="oracle",
            error=error,
            error_category=error_category,
            rollout_dir=trial,
        )

    monkeypatch.setattr(Evaluation, "_run_single_task", fake)


def test_fail_under_gates_the_pass_rate(tmp_path: Path, monkeypatch) -> None:
    _task(tmp_path)
    _fake_rewards(monkeypatch, 0.5)
    low = _eval(tmp_path, "--fresh", "--fail-under", "0.8")
    assert low.exit_code == 1, low.output
    assert "--fail-under 0.8" in low.output
    _fake_rewards(monkeypatch, 1.0)
    ok = _eval(tmp_path, "--fresh", "--fail-under", "0.8")
    assert ok.exit_code == 0, ok.output


def test_fail_on_timeout_catches_a_scored_timeout(tmp_path: Path, monkeypatch) -> None:
    _task(tmp_path)
    _fake_rewards(monkeypatch, 0.0, error_category="timeout")
    plain = _eval(tmp_path, "--fresh")
    assert plain.exit_code == 0  # a scored timeout is not an error by default
    gated = _eval(tmp_path, "--fresh", "--fail-on", "timeout")
    assert gated.exit_code == 1 and "timeout" in gated.output
    bad = _eval(tmp_path, "--fresh", "--fail-on", "timeouts")
    assert bad.exit_code == 1 and "--fail-on" in bad.output


def test_summary_out_is_a_versioned_document(tmp_path: Path, monkeypatch) -> None:
    import json

    from benchflow import job_export

    _task(tmp_path)
    _fake_rewards(monkeypatch, 0.5)
    out = tmp_path / "run.json"
    result = _eval(
        tmp_path, "--fresh", "--fail-under", "0.8", "--summary-out", str(out)
    )
    assert result.exit_code == 1
    doc = json.loads(out.read_text())
    assert doc["kind"] == "benchflow.run-summary" and doc["schema_version"] == 1
    assert doc["total"] == 1 and doc["passed"] == 0 and doc["exit_code"] == 1
    assert doc["gate"]["failed"] == ["pass rate 0.00 < --fail-under 0.8"]
    assert Path(doc["job_dir"]).is_dir() and doc["ran"] == 1
    job_export.RunSummaryExport.model_validate(doc)


# A bad Daytona key failed only after the job existed.


def _daytona_fails(monkeypatch) -> None:
    from benchflow import doctor as doctor_mod
    from benchflow.doctor import Check

    monkeypatch.delenv("BENCHFLOW_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(
        doctor_mod,
        "check_daytona",
        lambda probes, *, required, offline: Check(
            "daytona",
            "sandbox",
            "daytona",
            "fail",
            "SDK 0.1, but the API call with DAYTONA_API_KEY failed: Invalid credentials",
            fix="Check the key at https://app.daytona.io/dashboard/keys",
        ),
    )


def test_a_bad_daytona_key_fails_before_the_job(tmp_path: Path, monkeypatch) -> None:
    _task(tmp_path)
    _daytona_fails(monkeypatch)
    result = _eval(tmp_path)
    assert result.exit_code == 1, result.output
    assert "Invalid credentials" in result.output
    assert not (tmp_path / "jobs").exists()


def test_the_sdk_host_check_refuses_a_bad_daytona_key(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    import benchflow as bf
    from benchflow.runtime import check_host

    _daytona_fails(monkeypatch)
    with pytest.raises(RuntimeError, match="Invalid credentials"):
        check_host(
            [
                bf.RolloutConfig(
                    task_path=_task(tmp_path), agent="oracle", environment="daytona"
                )
            ]
        )


def test_prompt_help_names_the_task_prompt() -> None:
    """Native tasks have no instruction.md."""
    import re

    result = CliRunner().invoke(app, ["eval", "run", "--help"])
    text = " ".join(re.sub(r"[│─╭╮╰╯]", " ", result.output).split())
    assert "(default: the task prompt: the task.md body" in text


def test_log_level_and_format_come_from_the_environment() -> None:
    """No way to change the INFO, message-only log for CI."""
    import logging

    from benchflow.cli.main import _log_settings

    assert _log_settings({}) == (logging.INFO, "%(message)s")
    level, fmt = _log_settings(
        {"BENCHFLOW_LOG_LEVEL": "warning", "BENCHFLOW_LOG_FORMAT": "time"}
    )
    assert level == logging.WARNING
    assert fmt == "%(asctime)s %(levelname)s %(name)s: %(message)s"
    assert _log_settings({"BENCHFLOW_LOG_LEVEL": "loud"})[0] == logging.INFO


def test_one_task_gets_the_heartbeat_at_any_concurrency(
    tmp_path: Path, monkeypatch
) -> None:
    """A job with fewer running tasks than --concurrency still gets a heartbeat."""
    import os

    _task(tmp_path)
    _fake_rollouts(monkeypatch)
    monkeypatch.setenv("BENCHFLOW_PROGRESS_AUTO", "unset")
    assert _eval(tmp_path, "--concurrency", "4").exit_code == 0
    assert os.environ["BENCHFLOW_PROGRESS_AUTO"] == "1"


def test_report_paths_are_not_wrapped(tmp_path: Path, monkeypatch) -> None:
    """At 80 columns the Artifacts/Summary paths split mid-path."""
    deep = tmp_path / ("a-rather-long-folder-name-" * 3)
    deep.mkdir()
    _task(deep)
    _fake_rollouts(monkeypatch)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(deep / "t1"),
            "--agent",
            "oracle",
            "--jobs-dir",
            str(deep / "jobs"),
            "--sandbox",
            "daytona",
        ],
        terminal_width=80,
    )
    assert result.exit_code == 0, result.output
    job = next(p for p in (deep / "jobs").iterdir() if p.is_dir())
    lines = result.output.splitlines()
    assert any(line.endswith(f"{job}/summary.json") for line in lines), result.output
    artifacts = next(line for line in lines if line.startswith("Artifacts:"))
    assert artifacts.rstrip().endswith(f"{deep.name}/jobs/{job.name}"), artifacts


def test_summary_says_whether_a_reviewer_ran(tmp_path: Path, monkeypatch) -> None:
    """An oracle run's summary.json showed a reviewer block
    (opencode, docker, 1800 s) although no reviewer ran."""
    import json

    _task(tmp_path)
    _fake_rollouts(monkeypatch)
    assert _eval(tmp_path).exit_code == 0
    job = next(p for p in (tmp_path / "jobs").iterdir() if p.is_dir())
    summary = json.loads((job / "summary.json").read_text())
    assert summary["reviewer"]["ran"] is False
