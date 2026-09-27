"""``benchflow.doctor_smoke`` orchestration, with fake runners and fake bench.

The real ``subprocess_runner`` is exercised against a stand-in ``bench`` (a
tiny Python script that writes a result.json), so the process handling,
timeouts and result parsing are covered without Docker or a model.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from benchflow import doctor_smoke
from benchflow.doctor import AgentAuth, CredentialSource, DoctorReport
from benchflow.doctor_smoke import (
    SmokeJob,
    SmokeOutcome,
    SmokeTarget,
    eval_command,
    parse_agent_request,
    plan_smoke,
    read_outcome,
    run_smoke,
    stage_task,
    subprocess_runner,
)


def _auth(agent: str, name: str, *, usable: bool = True, note: str = "", origin="env"):
    src = CredentialSource(name, "oauth-token", origin, usable=usable, note=note)
    endpoints = {
        "claude-agent-acp": ("https://api.anthropic.com/",),
        "codex-acp": ("https://chatgpt.com/", "https://auth.openai.com/"),
        "gemini": ("https://generativelanguage.googleapis.com/",),
    }[agent]
    return AgentAuth(agent, agent, (src,), src, endpoints)


def _report(
    *auths: AgentAuth, unreachable: frozenset[str] = frozenset()
) -> DoctorReport:
    return DoctorReport([], "docker", {a.agent: a for a in auths}, unreachable)


# ── Planning ────────────────────────────────────────────────────────────


def test_parse_agent_request_resolves_aliases_and_models():
    assert parse_agent_request("claude") == ("claude-agent-acp", None)
    assert parse_agent_request("codex=gpt-5.6-sol") == ("codex-acp", "gpt-5.6-sol")
    assert parse_agent_request(" gemini = gemini-3.8-flash ") == (
        "gemini",
        "gemini-3.8-flash",
    )
    with pytest.raises(ValueError, match="missing agent name"):
        parse_agent_request("=gpt-5.5")
    with pytest.raises(ValueError, match="empty model"):
        parse_agent_request("codex=")


def test_default_plan_runs_ready_agents_and_explains_the_rest():
    report = _report(
        _auth("claude-agent-acp", "CLAUDE_CODE_OAUTH_TOKEN"),
        _auth("codex-acp", "~/.codex/auth.json", origin="file"),
    )
    targets, skipped = plan_smoke(report)
    assert [(t.agent, t.model, t.auth) for t in targets] == [
        (
            "claude-agent-acp",
            "claude-haiku-4-5-20251001",
            "CLAUDE_CODE_OAUTH_TOKEN (env)",
        ),
        ("codex-acp", "gpt-5.5", "~/.codex/auth.json"),
    ]
    assert not any(t.explicit for t in targets)
    assert [(s.agent, s.reason) for s in skipped] == [("gemini", "no credential found")]


def test_default_plan_skips_expired_login_with_a_way_to_force_it():
    report = _report(
        _auth(
            "claude-agent-acp",
            "~/.claude/.credentials.json",
            usable=False,
            note="usable only if its refresh token still works",
            origin="file",
        ),
        _auth("codex-acp", "~/.codex/auth.json", origin="file"),
    )
    targets, skipped = plan_smoke(report)
    assert [t.agent for t in targets] == ["codex-acp"]
    claude_skip = next(s for s in skipped if s.agent == "claude-agent-acp")
    assert "refresh token" in claude_skip.reason
    assert "--agent claude " in claude_skip.reason


def test_default_plan_skips_agents_whose_endpoint_is_unreachable():
    report = _report(
        _auth("claude-agent-acp", "CLAUDE_CODE_OAUTH_TOKEN"),
        unreachable=frozenset({"https://api.anthropic.com/"}),
    )
    targets, skipped = plan_smoke(report)
    assert targets == []
    assert "unreachable: https://api.anthropic.com/" in skipped[0].reason


def test_explicit_agents_always_run_in_order_without_duplicates():
    report = _report(_auth("codex-acp", "~/.codex/auth.json", origin="file"))
    targets, skipped = plan_smoke(report, ["codex=gpt-5.6-sol", "claude", "codex-acp"])
    assert skipped == []
    assert [(t.agent, t.model, t.auth, t.explicit) for t in targets] == [
        ("codex-acp", "gpt-5.6-sol", "~/.codex/auth.json", True),
        ("claude-agent-acp", "claude-haiku-4-5-20251001", "none found", True),
    ]


def test_explicit_agent_errors_are_clear():
    report = _report()
    with pytest.raises(ValueError, match="unknown agent 'nope'"):
        plan_smoke(report, ["nope"])
    with pytest.raises(ValueError, match="no default model"):
        plan_smoke(report, ["openhands"])
    targets, _ = plan_smoke(report, ["openhands=deepseek/deepseek-chat"])
    assert targets[0].model == "deepseek/deepseek-chat"


# ── Task + command ──────────────────────────────────────────────────────


def test_bundled_task_is_a_complete_task_and_is_copied(tmp_path):
    bundled = doctor_smoke.BUNDLED_TASK_DIR
    for rel in (
        "task.toml",
        "instruction.md",
        "tests/test.sh",
        "environment/Dockerfile",
    ):
        assert (bundled / rel).is_file(), rel
    staged = stage_task(tmp_path)
    assert staged == tmp_path / "hello-world"
    assert (staged / "tests" / "test.sh").read_text() == (
        bundled / "tests" / "test.sh"
    ).read_text()
    # Re-staging replaces the copy rather than failing on an existing dir.
    (staged / "stray.txt").write_text("x")
    assert not (stage_task(tmp_path) / "stray.txt").exists()


def test_eval_command_is_one_quiet_run_with_no_credentials(tmp_path):
    job = SmokeJob(
        SmokeTarget("codex-acp", "gpt-5.5", "~/.codex/auth.json"),
        tmp_path / "task",
        tmp_path / "jobs",
        tmp_path / "log",
        "daytona",
        60,
    )
    argv = eval_command(job)
    assert argv[:2] == ["eval", "run"]
    pairs = dict(zip(argv[2::2], argv[3::2], strict=False))
    assert pairs["--agent"] == "codex-acp"
    assert pairs["--model"] == "gpt-5.5"
    assert pairs["--sandbox"] == "daytona"
    assert pairs["--concurrency"] == "1"
    assert pairs["--tasks-dir"] == str(tmp_path / "task")
    assert pairs["--jobs-dir"] == str(tmp_path / "jobs")
    assert argv[-1] == "--quiet"


# ── Reading results ─────────────────────────────────────────────────────


def _job(
    tmp_path: Path, agent: str = "claude-agent-acp", timeout: float = 60
) -> SmokeJob:
    return SmokeJob(
        SmokeTarget(agent, "m", "CLAUDE_CODE_OAUTH_TOKEN (env)"),
        tmp_path / "task" / "hello-world",
        tmp_path / agent,
        tmp_path / "logs" / f"{agent}.log",
        "docker",
        timeout,
    )


def _write_result(
    job: SmokeJob, *, trajectory: str = '{"type":"tool_call"}\n', **fields
):
    rollout = job.jobs_dir / "2026-01-01__00-00-00" / "hello-world__abcd1234"
    (rollout / "trajectory").mkdir(parents=True)
    data = {"rewards": {"reward": 1.0}, "error": None, "verifier_error": None}
    data.update(fields)
    (rollout / "result.json").write_text(json.dumps(data))
    (rollout / "trajectory" / "acp_trajectory.jsonl").write_text(trajectory)
    return rollout


def test_read_outcome_pass(tmp_path):
    job = _job(tmp_path)
    rollout = _write_result(job)
    outcome = read_outcome(job, seconds=42.0, exit_code=0)
    assert outcome.status == "pass"
    assert outcome.reward == 1.0
    assert outcome.rollout_dir == rollout
    assert outcome.trajectory == rollout / "trajectory" / "acp_trajectory.jsonl"


def test_read_outcome_zero_reward_is_a_fail(tmp_path):
    job = _job(tmp_path)
    _write_result(job, rewards={"reward": 0.0})
    outcome = read_outcome(job, seconds=1.0, exit_code=0)
    assert outcome.status == "fail"
    assert "reward 0" in outcome.reason


def test_read_outcome_agent_error_is_an_error_with_redacted_reason(tmp_path):
    job = _job(tmp_path)
    token = "sk-ant-oat01-SECRETtokenvalue-xyz"
    _write_result(
        job,
        rewards=None,
        error=f"ACP error: invalid bearer {token}\nsecond line",
        error_category="agent_error",
    )
    outcome = read_outcome(
        job, seconds=1.0, exit_code=1, environ={"CLAUDE_CODE_OAUTH_TOKEN": token}
    )
    assert outcome.status == "error"
    assert outcome.reason.startswith("agent_error: ACP error: invalid bearer ***")
    assert token not in outcome.reason
    assert "second line" not in outcome.reason


def test_read_outcome_verifier_error(tmp_path):
    job = _job(tmp_path)
    _write_result(
        job, verifier_error="test.sh exited 2", verifier_error_category="verifier"
    )
    outcome = read_outcome(job, seconds=1.0, exit_code=1)
    assert outcome.status == "error"
    assert outcome.reason == "verifier: test.sh exited 2"


def test_read_outcome_reward_without_trajectory_is_not_a_pass(tmp_path):
    job = _job(tmp_path)
    _write_result(job, trajectory="")
    outcome = read_outcome(job, seconds=1.0, exit_code=0)
    assert outcome.status == "fail"
    assert outcome.trajectory is None


def test_read_outcome_without_result_uses_the_log(tmp_path):
    job = _job(tmp_path)
    job.log_path.parent.mkdir(parents=True)
    job.log_path.write_text(
        "starting\nError: Docker daemon is not running. Please start Docker\ncleanup\n"
    )
    outcome = read_outcome(job, seconds=3.0, exit_code=1)
    assert outcome.status == "error"
    assert outcome.reason == (
        "bench exited 1 without writing result.json: "
        "Error: Docker daemon is not running. Please start Docker"
    )
    timed_out = read_outcome(job, seconds=60.0, exit_code=-15, timed_out=True)
    assert timed_out.reason.startswith("timed out after 60s")


# ── Orchestration ───────────────────────────────────────────────────────


def test_run_smoke_runs_targets_one_at_a_time_and_writes_a_summary(tmp_path):
    active = threading.Lock()
    seen: list[SmokeJob] = []
    events: list[str] = []

    def fake_runner(job: SmokeJob) -> SmokeOutcome:
        # Never two runs at once: the lock would already be held.
        assert active.acquire(blocking=False), "runs overlapped"
        try:
            seen.append(job)
            assert (job.task_dir / "task.toml").is_file()
            status = "pass" if job.target.agent == "claude-agent-acp" else "fail"
            return SmokeOutcome(
                job.target,
                status,
                1.0 if status == "pass" else 0.0,
                12.5,
                job.log_path,
                reason=""
                if status == "pass"
                else "agent did not solve the task (reward 0)",
            )
        finally:
            active.release()

    targets = [
        SmokeTarget(
            "claude-agent-acp",
            "claude-haiku-4-5-20251001",
            "CLAUDE_CODE_OAUTH_TOKEN (env)",
        ),
        SmokeTarget("codex-acp", "gpt-5.5", "~/.codex/auth.json"),
    ]
    root = tmp_path / "smoke" / "run1"
    outcomes = run_smoke(
        targets,
        root=root,
        sandbox="docker",
        timeout_sec=120,
        runner=fake_runner,
        on_start=lambda i, t: events.append(f"start {i} {t.agent}"),
        on_done=lambda i, o: events.append(f"done {i} {o.status}"),
    )
    assert [o.status for o in outcomes] == ["pass", "fail"]
    assert events == [
        "start 0 claude-agent-acp",
        "done 0 pass",
        "start 1 codex-acp",
        "done 1 fail",
    ]
    assert [j.jobs_dir for j in seen] == [root / "claude-agent-acp", root / "codex-acp"]
    assert [j.log_path for j in seen] == [
        root / "logs" / "claude-agent-acp.log",
        root / "logs" / "codex-acp.log",
    ]
    assert {j.task_dir for j in seen} == {root / "task" / "hello-world"}
    assert all(j.timeout_sec == 120 and j.sandbox == "docker" for j in seen)
    summary = json.loads((root / "smoke-summary.json").read_text())
    assert summary["ok"] is False
    assert summary["sandbox"] == "docker"
    assert [(r["agent"], r["status"], r["reward"]) for r in summary["runs"]] == [
        ("claude-agent-acp", "pass", 1.0),
        ("codex-acp", "fail", 0.0),
    ]


# ── The real subprocess runner against a stand-in bench ────────────────

_FAKE_BENCH = r"""
import json, os, sys, time
args = sys.argv[1:]
opts = dict(zip(args[2::2], args[3::2]))
mode = os.environ["FAKE_BENCH_MODE"]
print("fake bench", " ".join(args), flush=True)
if mode == "sleep":
    time.sleep(60)
if mode == "crash":
    print("Error: Docker daemon is not running.", flush=True)
    sys.exit(1)
rollout = os.path.join(opts["--jobs-dir"], "job", "hello-world__0001")
os.makedirs(os.path.join(rollout, "trajectory"))
with open(os.path.join(rollout, "result.json"), "w") as f:
    json.dump({"rewards": {"reward": 1.0}, "error": None, "verifier_error": None}, f)
with open(os.path.join(rollout, "trajectory", "acp_trajectory.jsonl"), "w") as f:
    f.write('{"type": "tool_call"}\n')
"""


@pytest.fixture
def fake_bench(monkeypatch):
    monkeypatch.setattr(
        doctor_smoke,
        "_bench_argv",
        lambda args: [sys.executable, "-c", _FAKE_BENCH, *args],
    )

    def _mode(mode: str) -> None:
        monkeypatch.setenv("FAKE_BENCH_MODE", mode)

    return _mode


def test_subprocess_runner_reads_the_result_and_logs_the_command(tmp_path, fake_bench):
    fake_bench("ok")
    job = _job(tmp_path, agent="codex-acp")
    outcome = subprocess_runner(job)
    assert outcome.status == "pass", outcome.reason
    assert outcome.exit_code == 0
    log = job.log_path.read_text()
    assert log.startswith("$ bench eval run --tasks-dir ")
    assert "--agent codex-acp" in log
    assert "fake bench eval run" in log


def test_subprocess_runner_reports_a_crash_from_the_log(tmp_path, fake_bench):
    fake_bench("crash")
    outcome = subprocess_runner(_job(tmp_path))
    assert outcome.status == "error"
    assert outcome.exit_code == 1
    assert "Docker daemon is not running" in outcome.reason


def test_subprocess_runner_kills_a_hung_run(tmp_path, fake_bench, monkeypatch):
    fake_bench("sleep")
    monkeypatch.setattr(doctor_smoke, "_TERMINATE_GRACE_SEC", 5)
    outcome = subprocess_runner(_job(tmp_path, timeout=1))
    assert outcome.status == "error"
    assert outcome.reason.startswith("timed out after 1s")
    assert outcome.seconds < 30
