"""CLI wiring for ``bench doctor`` and ``bench eval smoke``.

``run_doctor`` and the smoke runner are monkeypatched, so these tests check
rendering, exit codes and the gate between doctor and smoke without touching
Docker, the network or a model.
"""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from benchflow import doctor, doctor_smoke
from benchflow.cli.main import app
from benchflow.doctor import AgentAuth, Check, CredentialSource, DoctorReport
from benchflow.doctor_smoke import SmokeJob, SmokeOutcome

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _out(result) -> str:
    return _ANSI.sub("", result.output)


def _claude_auth(usable: bool = True) -> AgentAuth:
    src = CredentialSource(
        "CLAUDE_CODE_OAUTH_TOKEN", "oauth-token", "env", usable=usable
    )
    return AgentAuth(
        "claude-agent-acp", "Claude", (src,), src, ("https://api.anthropic.com/",)
    )


def _report(
    *checks: Check, agents: dict | None = None, sandbox="docker"
) -> DoctorReport:
    return DoctorReport(list(checks), sandbox, agents or {})


OK_CHECKS = (
    Check("python", "runtime", "python", "pass", "Python 3.12.9"),
    Check("docker", "sandbox", "docker", "pass", "Docker 29.5.2 via colima"),
    Check(
        "auth.claude-agent-acp",
        "agents",
        "claude-agent-acp",
        "pass",
        "CLAUDE_CODE_OAUTH_TOKEN (env)",
    ),
)
DOCKER_DOWN = Check(
    "docker",
    "sandbox",
    "docker",
    "fail",
    "daemon unreachable (context colima)",
    "colima start default",
)


@pytest.fixture
def fake_doctor(monkeypatch):
    calls: list[dict] = []

    def install(report: DoctorReport):
        def _run_doctor(**kwargs):
            calls.append(kwargs)
            return report

        monkeypatch.setattr(doctor, "run_doctor", _run_doctor)
        return calls

    return install


# ── bench doctor ────────────────────────────────────────────────────────


def test_doctor_all_pass_exits_zero_and_points_at_smoke(fake_doctor):
    calls = fake_doctor(_report(*OK_CHECKS))
    result = runner.invoke(app, ["doctor"])
    out = _out(result)
    assert result.exit_code == 0, out
    assert "Runtime" in out and "Sandbox" in out and "Agent credentials" in out
    assert "PASS" in out
    assert "Next: bench eval smoke" in out
    assert calls == [{"sandbox": "docker", "offline": False}]


def test_doctor_failure_exits_one_with_fix_line(fake_doctor):
    fake_doctor(_report(OK_CHECKS[0], DOCKER_DOWN))
    result = runner.invoke(app, ["doctor"])
    out = _out(result)
    assert result.exit_code == 1
    assert "FAIL" in out
    assert "fix: colima start default" in out
    assert "1 failed" in out


def test_doctor_columns_stay_aligned_with_long_host_names(fake_doctor):
    """Regression test: names longer than 24 characters (network
    hosts such as cloudcode-pa.googleapis.com) pushed their summary right of
    every other row's summary column."""
    fake_doctor(
        _report(
            *OK_CHECKS,
            Check(
                "net.cloudcode-pa.googleapis.com",
                "network",
                "cloudcode-pa.googleapis.com",
                "pass",
                "reachable — Gemini model API",
            ),
        )
    )
    lines = _out(runner.invoke(app, ["doctor"])).splitlines()
    rows = {
        name: line
        for line in lines
        for name in ("python", "docker", "cloudcode-pa.googleapis.com")
        if line.strip().startswith("PASS") and f" {name} " in line
    }
    starts = {
        name: rows[name].index(summary)
        for name, summary in (
            ("python", "Python 3.12.9"),
            ("docker", "Docker 29.5.2"),
            ("cloudcode-pa.googleapis.com", "reachable"),
        )
    }
    assert len(set(starts.values())) == 1, starts


def test_doctor_json_is_parseable_and_carries_exit_code(fake_doctor):
    fake_doctor(_report(OK_CHECKS[0], DOCKER_DOWN))
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["ok"] is False
    assert data["checks"][1]["fix"] == "colima start default"

    fake_doctor(_report(*OK_CHECKS))
    result = runner.invoke(
        app, ["doctor", "--json", "--offline", "--sandbox", "daytona"]
    )
    assert result.exit_code == 0
    assert json.loads(result.output)["ok"] is True


def test_doctor_passes_flags_through(fake_doctor):
    calls = fake_doctor(_report(*OK_CHECKS))
    runner.invoke(app, ["doctor", "--offline", "--sandbox", "daytona"])
    assert calls == [{"sandbox": "daytona", "offline": True}]


def test_doctor_rejects_unknown_sandbox(fake_doctor):
    calls = fake_doctor(_report(*OK_CHECKS))
    result = runner.invoke(app, ["doctor", "--sandbox", "nope"])
    assert result.exit_code == 2
    assert calls == []


# ── bench eval smoke ────────────────────────────────────────────────────


@pytest.fixture
def fake_runner(monkeypatch):
    jobs: list[SmokeJob] = []
    results: dict[str, tuple[str, float | None, str]] = {}

    def _runner(job: SmokeJob) -> SmokeOutcome:
        jobs.append(job)
        status, reward, reason = results.get(job.target.agent, ("pass", 1.0, ""))
        rollout = job.jobs_dir / "job" / "hello-world__0001"
        traj = rollout / "trajectory" / "acp_trajectory.jsonl"
        traj.parent.mkdir(parents=True, exist_ok=True)
        traj.write_text("{}\n")
        return SmokeOutcome(
            job.target,
            status,  # type: ignore[arg-type]
            reward,
            61.0,
            job.log_path,
            rollout_dir=rollout,
            trajectory=traj if status == "pass" else None,
            reason=reason,
        )

    monkeypatch.setattr(doctor_smoke, "subprocess_runner", _runner)
    return jobs, results


def test_smoke_blocks_on_sandbox_failure_without_running(
    fake_doctor, fake_runner, tmp_path
):
    jobs, _ = fake_runner
    fake_doctor(_report(DOCKER_DOWN, agents={"claude-agent-acp": _claude_auth()}))
    result = runner.invoke(app, ["eval", "smoke", "--jobs-dir", str(tmp_path)])
    out = _out(result)
    assert result.exit_code == 1
    assert "fix: colima start default" in out
    assert jobs == []


def test_smoke_runs_credentialed_agents_and_prints_the_table(
    fake_doctor, fake_runner, tmp_path
):
    jobs, _ = fake_runner
    calls = fake_doctor(
        _report(*OK_CHECKS, agents={"claude-agent-acp": _claude_auth()})
    )
    result = runner.invoke(
        app,
        ["eval", "smoke", "--jobs-dir", str(tmp_path), "--timeout-sec", "300"],
        terminal_width=200,
    )
    out = _out(result)
    assert result.exit_code == 0, out
    assert calls == [{"sandbox": "docker"}]
    assert [j.target.agent for j in jobs] == ["claude-agent-acp"]
    assert jobs[0].timeout_sec == 300
    assert "skip codex-acp: no credential found" in out
    assert "skip gemini: no credential found" in out
    assert "Smoke results" in out
    assert "claude-haiku-4-5-20251001" in out
    assert "Trajectory" in out
    assert "acp_trajectory.jsonl" in out  # printed unwrapped, stays copyable
    summary = json.loads(next(tmp_path.glob("*/smoke-summary.json")).read_text())
    assert summary["ok"] is True
    assert summary["runs"][0]["trajectory"].endswith("acp_trajectory.jsonl")


def test_smoke_next_steps_stay_on_one_line_and_name_the_viewer(
    fake_doctor, fake_runner, tmp_path
):
    """The Next: hint wrapped mid-path at 80 columns and pointed at a docs file
    that a `uv tool` install does not have."""
    fake_doctor(_report(*OK_CHECKS, agents={"claude-agent-acp": _claude_auth()}))
    result = runner.invoke(
        app, ["eval", "smoke", "--jobs-dir", str(tmp_path)], terminal_width=80
    )
    out = _out(result)
    assert result.exit_code == 0, out
    next_lines = [line for line in out.splitlines() if line.startswith("Next:")]
    assert next_lines and all("getting-started" not in n for n in next_lines)
    assert any("bench eval view " in line for line in out.splitlines())
    assert any(
        "https://github.com/benchflow-ai/benchflow/blob/main/docs/getting-started.md"
        in line
        for line in out.splitlines()
    )


def test_smoke_failure_exits_one_and_explains(fake_doctor, fake_runner, tmp_path):
    jobs, results = fake_runner
    results["codex-acp"] = ("error", None, "agent_error: model not available")
    fake_doctor(_report(*OK_CHECKS, agents={"claude-agent-acp": _claude_auth()}))
    result = runner.invoke(
        app,
        [
            "eval",
            "smoke",
            "--jobs-dir",
            str(tmp_path),
            "--agent",
            "claude",
            "--agent",
            "codex=gpt-5.6-sol",
        ],
        terminal_width=200,
    )
    out = _out(result)
    assert result.exit_code == 1
    assert [(j.target.agent, j.target.model) for j in jobs] == [
        ("claude-agent-acp", "claude-haiku-4-5-20251001"),
        ("codex-acp", "gpt-5.6-sol"),
    ]
    assert "agent_error: model not available" in out
    assert "log:" in out


def test_smoke_with_no_working_credential_exits_one(fake_doctor, fake_runner, tmp_path):
    jobs, _ = fake_runner
    fake_doctor(
        _report(*OK_CHECKS, agents={"claude-agent-acp": _claude_auth(usable=False)})
    )
    result = runner.invoke(app, ["eval", "smoke", "--jobs-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "No agent to smoke" in _out(result)
    assert jobs == []


def test_smoke_rejects_bad_agent_spec(fake_doctor, fake_runner, tmp_path):
    fake_doctor(_report(*OK_CHECKS))
    result = runner.invoke(
        app, ["eval", "smoke", "--jobs-dir", str(tmp_path), "--agent", "nope"]
    )
    assert result.exit_code == 2
    assert "unknown agent 'nope'" in _out(result)


# ── docs/reference/cli.md stays in sync ────────────────────────────────


@pytest.mark.parametrize(
    ("path", "header"),
    [(["doctor"], "## bench doctor"), (["eval", "smoke"], "### bench eval smoke")],
)
def test_cli_md_documents_exactly_the_live_flags(path, header):
    from tests.test_cli_docs_drift import _cli_long_flags, _doc_flags

    assert _cli_long_flags(path) == _doc_flags(header)
