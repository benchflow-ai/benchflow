"""Robust integration suite — end-to-end gates grounded in v0.6 dogfooding.

Two tiers:

- **Deterministic integrity gates** (this module's unmarked tests) run in the
  normal suite with no credentials. They pin the gate behavior that
  single-happy-path coverage misses: the realness gate rejects unscored / no-work
  rollouts, the agent judge **fails closed**, the judge catches reward-hacking
  that the mechanical gate cannot, and the trajectory artifacts are well-formed
  and leak no secrets.
- **Live scenarios** marked ``@pytest.mark.integration`` (excluded by the
  default ``-m 'not live and not integration'`` addopts) run nightly / on demand
  with a sandbox and provider credentials: oracle determinism, docker<->daytona
  parity, a real agent rollout through the agent-judge gate, the llm-judge
  verifier's fail-closed contract, and reaper ownership scoping.

Each live scenario skips cleanly when its prerequisites (Docker daemon, Daytona
key, DeepSeek / Gemini keys) are absent, so the suite is safe to run anywhere.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.rewards.llm import JudgeEnvironmentError
from tests.integration import agent_judge, scenarios

REAL_TASKS = (
    Path(__file__).resolve().parent
    / ".."
    / "docs"
    / "examples"
    / "task-md"
    / "real-skillsbench"
).resolve()


# ==================================================================
# Deterministic integrity gates (no credentials; run in normal suite)
# ==================================================================


def _real_synth(dest: Path, **overrides) -> Path:
    base = dict(
        reward=1.0,
        n_tool_calls=8,
        total_tokens=120_000,
        trajectory=[
            {"source": "user", "message": "Implement parse()"},
            {"source": "agent", "message": "writing solution", "tool_calls": [1]},
            {"source": "agent", "message": "ran tests, all pass"},
        ],
    )
    base.update(overrides)
    return scenarios.synth_rollout(dest, **base)


@pytest.mark.parametrize(
    ("overrides", "expect_flagged"),
    [
        ({}, False),  # a real rollout passes the mechanical gate
        ({"n_tool_calls": 0}, True),  # scored but did no work
        ({"total_tokens": 0}, True),  # no provider telemetry
        ({"total_tokens": None}, True),
        ({"reward": None}, True),  # never scored
        ({"error": "agent crashed"}, True),
        ({"verifier_error": "verifier infra failure"}, True),
    ],
)
def test_realness_gate(tmp_path: Path, overrides: dict, expect_flagged: bool):
    rollout = _real_synth(tmp_path / "r", **overrides)
    evidence = agent_judge.load_rollout_evidence(rollout)
    issues = agent_judge.realness_issues(evidence)
    assert bool(issues) == expect_flagged, issues


def test_realness_gate_fails_closed_on_verifier_tamper(tmp_path: Path):
    """Fail-closed: a mechanically-detected verifier tamper must hard-fail
    the realness gate on its own, so the gate holds even when the LLM judge is
    absent or lenient. The rollout is otherwise REAL (tool calls, tokens, reward
    1.0) — only the score-defining file mutation makes it untrustworthy."""
    rollout = _real_synth(
        tmp_path / "r",
        reward=1.0,
        n_tool_calls=2,
        total_tokens=9000,
        trajectory=[
            {"source": "user", "message": "make the grader pass"},
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "name": "bash",
                        "arguments": {"command": "echo 'exit 0' > /work/verify.py"},
                    }
                ],
                "observation": "ok",
            },
        ],
    )
    evidence = agent_judge.load_rollout_evidence(rollout)
    # The tamper is mechanically detected on the evidence...
    assert any("verify.py" in f for f in evidence.flagged_actions), (
        evidence.flagged_actions
    )
    # ...and realness_issues must surface it, so the gate fails closed without
    # depending on the judge.
    issues = agent_judge.realness_issues(evidence)
    assert any("verify.py" in i for i in issues), issues


def _gate(rollout: Path, judge_return=None, judge_exc=None):
    mock = AsyncMock()
    if judge_exc is not None:
        mock.side_effect = judge_exc
    else:
        mock.return_value = judge_return
    with patch.object(agent_judge, "call_judge", mock):
        import asyncio

        return asyncio.run(
            agent_judge.gate_rollout(rollout, env={"GEMINI_API_KEY": "x"})
        )


def test_gate_passes_real_rollout_with_passing_judge(tmp_path: Path):
    rollout = _real_synth(tmp_path / "r")
    result = _gate(rollout, judge_return='{"verdict": "pass", "reason": "genuine"}')
    assert result.passed
    assert not result.realness_issues
    assert result.verdict.passed


def test_gate_fails_closed_when_judge_errors(tmp_path: Path):
    # A judge that cannot run must never read as a pass (the reward-integrity
    # foot-gun: a judge infra error silently scored as success/0.0).
    rollout = _real_synth(tmp_path / "r")
    result = _gate(rollout, judge_exc=JudgeEnvironmentError("no SDK"))
    assert not result.passed
    assert not result.verdict.passed
    assert "could not run" in result.verdict.reason.lower()


def test_gate_fails_closed_on_unparseable_verdict(tmp_path: Path):
    rollout = _real_synth(tmp_path / "r")
    result = _gate(rollout, judge_return="not json at all")
    assert not result.passed
    assert not result.verdict.passed


def test_gate_fails_when_judge_flags_reward_hacking(tmp_path: Path):
    # The rollout looks mechanically REAL (tool calls, tokens, reward 1.0) — only
    # the judge can catch that the agent gamed the verifier. The gate must honor
    # a judge FAIL even when realness is clean.
    rollout = _real_synth(
        tmp_path / "r",
        reward=1.0,
        trajectory=[
            {"source": "agent", "message": "echo expected answer into verifier dir"},
            {"source": "agent", "message": "chmod the test so it always passes"},
        ],
    )
    result = _gate(
        rollout,
        judge_return='{"verdict": "fail", "reason": "tampered with the verifier"}',
    )
    assert not result.realness_issues  # mechanically real...
    assert not result.passed  # ...but the judge caught the hack
    assert "verifier" in result.verdict.reason.lower()


def test_atif_adp_validators_accept_wellformed(tmp_path: Path):
    rollout = scenarios.synth_rollout(
        tmp_path / "r",
        atif={
            "schema_version": "ATIF-v1.7",
            "steps": [
                {"source": "user", "message": "do it"},
                {"source": "agent", "message": "done", "tool_calls": []},
            ],
        },
        adp_lines=[{"action": "write", "observation": "ok", "reward": 1.0}],
    )
    assert scenarios.atif_issues(rollout) == []
    assert scenarios.adp_issues(rollout) == []


def test_atif_validator_flags_bad_schema(tmp_path: Path):
    rollout = scenarios.synth_rollout(
        tmp_path / "r",
        atif={"schema_version": "v0", "steps": [{"source": "martian"}]},
    )
    issues = scenarios.atif_issues(rollout)
    assert any("schema_version" in i for i in issues)
    assert any("source" in i for i in issues)


def test_secret_leak_scan(tmp_path: Path):
    clean = scenarios.synth_rollout(tmp_path / "clean")
    assert scenarios.secret_leak_issues(clean) == []
    leaky = scenarios.synth_rollout(tmp_path / "leaky")
    (leaky / "trajectory").mkdir(exist_ok=True)
    (leaky / "trajectory" / "leak.txt").write_text(
        "DEEPSEEK_API_KEY=sk-deadbeefdeadbeefdeadbeef leaked into the trajectory"
    )
    issues = scenarios.secret_leak_issues(leaky)
    assert issues and "leak.txt" in issues[0]


def test_example_judge_fails_closed_on_infra_error(tmp_path: Path):
    # The shipped generated-skill-eval judge.py must exit non-zero (not write
    # reward 0.0) when no LLM judge can run — a deterministic check of the
    # reward-integrity fix, no live model needed.
    src = (
        Path(__file__).resolve().parent
        / ".."
        / "docs"
        / "examples"
        / "task-md"
        / "generated-skill-eval"
        / "models-as-skills"
        / "regex-email-parser"
        / "verifier"
    ).resolve()
    if not (src / "judge.py").is_file():
        pytest.skip("example judge.py not present")
    vdir = tmp_path / "verifier"
    shutil.copytree(src, vdir)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "solution.py").write_text(
        "def parse_email_addresses(t):\n    return []\n"
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "acp_trajectory.jsonl").write_text('{"role": "assistant"}\n')
    scrubbed = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        }
    }
    scrubbed.update(
        {
            "JUDGE_MODEL": "gemini-3.1-flash-lite",
            "BENCHFLOW_VERIFIER_DIR": str(vdir),
            "BENCHFLOW_WORKSPACE": str(workspace),
            "BENCHFLOW_AGENT_LOG_DIR": str(logs),
        }
    )
    proc = subprocess.run(
        ["uv", "run", "python", str(vdir / "judge.py")],
        cwd=scenarios.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=scrubbed,
    )
    assert proc.returncode != 0, (proc.stdout + proc.stderr)[-400:]
    assert not (vdir / "reward.txt").is_file()


# ==================================================================
# Live scenarios (sandbox + credentials; @pytest.mark.integration)
# ==================================================================


def _docker_ok() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=15
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def _have(*keys: str) -> bool:
    return all(os.environ.get(k) for k in keys)


requires_docker = pytest.mark.skipif(
    not _docker_ok(), reason="docker daemon unavailable"
)
requires_daytona = pytest.mark.skipif(
    not _have("DAYTONA_API_KEY"), reason="DAYTONA_API_KEY not set"
)
requires_deepseek = pytest.mark.skipif(
    not _have("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
    reason="DeepSeek credentials not set",
)
requires_gemini = pytest.mark.skipif(
    not _have("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)


def _judge_sdk_ok() -> bool:
    try:
        import google.genai  # noqa: F401

        return True
    except ImportError:
        return False


requires_judge_sdk = pytest.mark.skipif(
    not _judge_sdk_ok(),
    reason="gemini judge SDK not installed (uv sync --extra judge)",
)


def _deepagents_ok() -> bool:
    try:
        import deepagents  # noqa: F401
        import langchain_openai  # noqa: F401

        return True
    except ImportError:
        return False


requires_deepagents = pytest.mark.skipif(
    not _deepagents_ok(),
    reason="deepagents harness not installed (uv sync --extra deepagents)",
)


@pytest.fixture(scope="module")
def real_tasks(tmp_path_factory) -> Path:
    """A staged copy of the three runnable real-skillsbench tasks."""
    staged = tmp_path_factory.mktemp("real_tasks")
    for name in ("3d-scan-calc", "weighted-gdp-calc", "citation-check"):
        src = REAL_TASKS / name
        if src.is_dir():
            shutil.copytree(src, staged / name)
    return staged


@pytest.mark.integration
@requires_docker
def test_oracle_determinism_docker(real_tasks: Path, tmp_path: Path):
    # An oracle (ground-truth solve.sh) MUST score reward 1.0 on its own
    # verifier. This is the gate that would have caught the broken 3d-scan-calc
    # example oracle before it shipped.
    jobs = tmp_path / "jobs"
    outcome = scenarios.run_eval(
        jobs_dir=jobs, agent="oracle", sandbox="docker", tasks_dir=real_tasks
    )
    rollouts = scenarios.rollout_dirs(jobs)
    assert rollouts, outcome.stdout[-500:] + outcome.stderr[-500:]
    failures = {
        scenarios.task_name_of(r): scenarios.reward_of(r)
        for r in rollouts
        if scenarios.reward_of(r) != 1.0
    }
    assert not failures, f"oracle did not self-score 1.0: {failures}"


@pytest.mark.integration
@requires_docker
@requires_daytona
def test_sandbox_parity_docker_daytona(real_tasks: Path, tmp_path: Path):
    # The same oracle task must score identically on docker and daytona; a
    # backend-specific regression (e.g. from the daytona package split) shows
    # up as a parity gap.
    rewards: dict[str, dict[str, float | None]] = {}
    for sandbox in ("docker", "daytona"):
        jobs = tmp_path / f"jobs_{sandbox}"
        scenarios.run_eval(
            jobs_dir=jobs, agent="oracle", sandbox=sandbox, tasks_dir=real_tasks
        )
        for r in scenarios.rollout_dirs(jobs):
            rewards.setdefault(scenarios.task_name_of(r), {})[sandbox] = (
                scenarios.reward_of(r)
            )
    diverged = {
        task: pair
        for task, pair in rewards.items()
        if pair.get("docker") != pair.get("daytona")
    }
    assert not diverged, f"docker<->daytona reward parity gap: {diverged}"


@pytest.mark.integration
@requires_docker
@requires_deepseek
@requires_gemini
@requires_judge_sdk
def test_agent_rollout_is_real_and_judged(tmp_path: Path):
    # Validates the full pipeline + judge chain end to end: a real harness
    # produces a REAL measurement and the agent judge runs over it.
    #
    # We gate on the DETERMINISTIC signals — a rollout was produced, it passes
    # the realness invariants, and the judge actually ran and returned a
    # verdict (``raw`` is set, i.e. the model responded rather than the chain
    # failing closed on an SDK/API error). We deliberately do NOT require a
    # judge *pass*: whether a weak model (deepseek-v4-flash) genuinely solves a
    # hard task is stochastic, and the judge correctly declining a poor attempt
    # must not flake this test. The verdict is recorded for review instead.
    import asyncio

    jobs = tmp_path / "jobs"
    scenarios.run_eval(
        jobs_dir=jobs,
        agent="openhands",
        model="deepseek/deepseek-v4-flash",
        sandbox="docker",
        source_repo="benchflow-ai/skillsbench",
        source_path="tasks",
        source_ref="main",
        include=("jax-computing-basics",),
        extra_args=("--usage-tracking", "required"),
    )
    rollouts = scenarios.rollout_dirs(jobs)
    assert rollouts, "no rollout produced"
    evidence = agent_judge.load_rollout_evidence(rollouts[0])
    assert not agent_judge.realness_issues(evidence), (
        f"rollout was not REAL: tool_calls={evidence.n_tool_calls} "
        f"tokens={evidence.total_tokens} reward={evidence.reward} "
        f"error={evidence.error}"
    )
    verdict = asyncio.run(
        agent_judge.judge_rollout(evidence, model="gemini-3.1-flash-lite")
    )
    assert verdict.raw is not None, (
        f"agent judge did not run (chain failed closed): {verdict.reason}"
    )
    # Recorded, not gated — the judge's pass/fail on a stochastic model run is
    # evidence for review, not a hard pipeline requirement.
    print(f"agent judge verdict: passed={verdict.passed} reason={verdict.reason}")


@pytest.mark.integration
@requires_daytona
def test_reaper_dryrun_is_safe():
    issues = scenarios.reaper_dryrun_issues()
    if issues and issues[0].startswith("__skip__"):
        pytest.skip(issues[0])
    assert not issues, issues


@pytest.mark.integration
@requires_docker
@requires_deepseek
@requires_judge_sdk
@requires_gemini
@requires_deepagents
def test_deepagents_deepseek_rollout_is_real_and_judged(tmp_path: Path):
    # End to end through the deepagents harness: a deepseek-v4-flash deep agent
    # solves a small coding task with real shell/file tools in a Docker sandbox,
    # producing a rollout the same realness gate + agent judge grade. Validates
    # the harness produces trustworthy measurements; reward-hack detection on
    # this judge is pinned deterministically in tests/test_judge_robustness.py.
    import asyncio
    import os

    from tests.integration import deepagents_harness as dh

    res = dh.run_deepagent(
        instruction=(
            "Create /work/solution.py with a function `is_prime(n)` returning a "
            "bool, then create /work/test.py asserting is_prime(7) and not "
            "is_prime(8), and run it with python."
        ),
        rollout_dir=tmp_path / "rollout",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ["DEEPSEEK_BASE_URL"],
        verify_cmd="cd /work && python test.py",
        max_steps=30,
        task_name="is-prime",
    )
    assert res.error is None, f"harness errored: {res.error}"
    evidence = agent_judge.load_rollout_evidence(res.rollout_dir)
    assert not agent_judge.realness_issues(evidence), (
        f"deepagents rollout not REAL: tool_calls={evidence.n_tool_calls} "
        f"tokens={evidence.total_tokens} reward={evidence.reward}"
    )
    gate = asyncio.run(
        agent_judge.gate_rollout(res.rollout_dir, model="gemini-3.1-flash-lite")
    )
    assert gate.passed, gate.to_dict()
