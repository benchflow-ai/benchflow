"""Scenarios that need a real model: smoke, Codex Apps policy, native Claude
OAuth on a no-web task, rubric review, and trials of the public example tasks.

Off unless ``BENCHFLOW_E2E_MODELS=1``. Credentials are subscription logins
only, handed to the child process in its environment (never on a command
line): ``BENCHFLOW_E2E_CLAUDE_OAUTH_TOKEN`` becomes ``CLAUDE_CODE_OAUTH_TOKEN``
and ``BENCHFLOW_E2E_CODEX_AUTH_JSON_PATH`` (a ChatGPT-login ``auth.json``)
becomes ``CODEX_AUTH_JSON``. ``HOME`` is a fresh empty folder, so no login
file or API key of the host is picked up. The Codex model is
``BENCHFLOW_E2E_CODEX_MODEL``; the Claude model ``BENCHFLOW_E2E_CLAUDE_MODEL``
(default ``claude-haiku-4-5``).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from tests.e2e import harness as h

CLAUDE = "claude-agent-acp"
CODEX = "codex-acp"
PUBLIC_TASKS = h.REPO_ROOT / "docs" / "examples" / "task-md" / "real-skillsbench"
HELLO = h.REPO_ROOT / "tests" / "examples" / "hello-world-task"


def _claude_model() -> str:
    return os.environ.get("BENCHFLOW_E2E_CLAUDE_MODEL", "claude-haiku-4-5")


def _codex_model() -> str:
    model = os.environ.get("BENCHFLOW_E2E_CODEX_MODEL")
    if not model:
        pytest.skip("BENCHFLOW_E2E_CODEX_MODEL is not set")
    return model


@pytest.fixture(scope="session")
def model_env(sandbox: str) -> dict[str, str]:
    if os.environ.get("BENCHFLOW_E2E_MODELS") != "1":
        pytest.skip("model scenarios are off: set BENCHFLOW_E2E_MODELS=1")
    home = tempfile.mkdtemp(prefix="bf-e2e-home-")
    env = {"HOME": home}
    token = os.environ.get("BENCHFLOW_E2E_CLAUDE_OAUTH_TOKEN")
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    auth = os.environ.get("BENCHFLOW_E2E_CODEX_AUTH_JSON_PATH")
    if auth:
        data = json.loads(Path(auth).read_text())
        assert data.get("auth_mode") == "chatgpt" and not data.get("OPENAI_API_KEY"), (
            "the Codex auth.json must be a ChatGPT login, not an API key"
        )
        env["CODEX_AUTH_JSON"] = json.dumps(data)
    return env


def _need(env: dict[str, str], key: str) -> None:
    if key not in env:
        pytest.skip(f"{key} is not available for model scenarios")


def test_smoke_claude(model_env, sandbox: str, jobs_root: Path, ledger: h.Ledger):
    _need(model_env, "CLAUDE_CODE_OAUTH_TOKEN")
    out = jobs_root / "smoke"
    h.clear_job(out)
    run = h.bench(
        "eval", "smoke", "--sandbox", sandbox, "--jobs-dir", out,
        "--agent", f"{CLAUDE}={_claude_model()}",
        env=model_env, log=jobs_root / "smoke.log", timeout=1800,
    )  # fmt: skip
    ledger.record("eval smoke (claude OAuth)", surface="CLI", seconds=run.seconds)
    h.assert_exit(run, 0)
    (summary_path,) = out.glob("*/smoke-summary.json")
    assert h.read_json(summary_path)


def test_codex_apps_policy_receipt_and_unoffered_model(
    model_env, sandbox: str, jobs_root: Path, ledger: h.Ledger
):
    """A scored codex run records the Apps policy as disabled before any prompt.

    The requested model is ``BENCHFLOW_E2E_CODEX_MODEL``. When the pinned
    adapter does not offer it, the run must stop once (no retries) naming
    the offered models, which is itself a regression scenario.
    """
    _need(model_env, "CODEX_AUTH_JSON")
    model = _codex_model()
    job = jobs_root / "codex-policy"
    h.clear_job(job)
    run = h.bench(
        "eval", "run", "--tasks-dir", HELLO, "--agent", CODEX, "--model", model,
        "--sandbox", sandbox, "--jobs-dir", jobs_root, "--job-name", job.name,
        "--max-sandbox-seconds", "600", env=model_env, log=jobs_root / "codex-policy.log",
    )  # fmt: skip
    ledger.record(f"codex-acp scored run ({model}): Apps policy receipt", surface="CLI",
                  seconds=run.seconds, job_dir=job)  # fmt: skip
    (trial,) = h.trial_dirs(job)
    receipt = h.read_json(trial / "codex_apps_policy.json")
    assert receipt["applied"] == "disabled" and receipt["effective_apps"] is False, (
        receipt
    )
    result = h.read_json(trial / "result.json")
    if "does not offer model" in (result.get("error") or ""):
        assert "it offers:" in result["error"]
        assert "Retrying" not in run.output, "an unoffered model must not be retried"
    else:
        assert result["rewards"] == {"reward": 1.0}, result.get("error")


def test_codex_scored_run_as_root_is_refused(model_env, sandbox: str, jobs_root: Path):
    _need(model_env, "CODEX_AUTH_JSON")
    job = jobs_root / "codex-root"
    h.clear_job(job)
    run = h.bench(
        "eval", "run", "--tasks-dir", HELLO, "--agent", CODEX, "--model", _codex_model(),
        "--sandbox", sandbox, "--sandbox-user", "null", "--jobs-dir", jobs_root,
        "--job-name", job.name, "--max-sandbox-seconds", "300",
        env=model_env, log=jobs_root / "codex-root.log",
    )  # fmt: skip
    assert run.returncode != 0 or not any(
        (h.read_json(t / "result.json").get("rewards") or {}).get("reward")
        for t in h.trial_dirs(job)
    ), run.tail()
    assert "root" in run.output.lower() or "sandbox-user" in run.output.lower(), (
        run.tail()
    )


NO_WEB_DOCKERFILE = (
    "FROM ubuntu:24.04\n"
    "RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends "
    "python3 openssl ca-certificates iptables && rm -rf /var/lib/apt/lists/*\n"
    "WORKDIR /app\nRUN mkdir -p /logs/verifier /logs/agent /logs/artifacts\n"
)


def test_native_claude_oauth_on_no_web_task(
    model_env, sandbox: str, tasks_root: Path, jobs_root: Path, ledger: h.Ledger
):
    _need(model_env, "CLAUDE_CODE_OAUTH_TOKEN")
    task = h.write_task(
        tasks_root / "no-web", "e2e-no-web", dockerfile=NO_WEB_DOCKERFILE,
        frontmatter={"sandbox": {"network_mode": "no-network"}},
    )  # fmt: skip
    job = jobs_root / "no-web"
    h.clear_job(job)
    run = h.bench(
        "eval", "run", "--tasks-dir", task, "--agent", CLAUDE, "--model", _claude_model(),
        "--sandbox", sandbox, "--jobs-dir", jobs_root, "--job-name", job.name,
        "--max-sandbox-seconds", "600", env=model_env, log=jobs_root / "no-web.log",
    )  # fmt: skip
    ledger.record("native Claude OAuth on a no-network task", surface="CLI",
                  seconds=run.seconds, job_dir=job)  # fmt: skip
    trial = h.trial_of(job, "e2e-no-web")
    result = h.read_json(trial / "result.json")
    assert result["rewards"] == {"reward": 1.0}, result.get("error")
    receipts = list(trial.rglob("native-oauth-network.json"))
    assert receipts, "no native-oauth-network.json admission receipt"


RUBRIC = {
    "criteria": [
        {
            "name": "file_present",
            "blocker": 1,
            "weight": 10,
            "description": "/app/hello.txt exists.",
            "guidance": "PASS when the evidence contains hello.txt. Otherwise FAIL.",
        },
        {
            "name": "exact_greeting",
            "blocker": 0,
            "weight": 5,
            "description": "hello.txt holds exactly the requested greeting.",
            "guidance": "Score 2 for exactly 'Hello, world!', 1 for a close variant, 0 otherwise.",
        },
    ]
}


LENIENT_HELLO_TEST = """#!/bin/bash
# Accepts any capitalisation; the rubric judges exactness.
if [ "$(tr -d '\\n' < /app/hello.txt 2>/dev/null | tr 'A-Z' 'a-z')" = "hello, world!" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""
CLOSE_SOLVE = "#!/bin/bash\nprintf 'hello, world!\\n' > /app/hello.txt\n"


@pytest.mark.parametrize(
    "reviewer,case",
    [(CLAUDE, "exact"), (CLAUDE, "close"), (CODEX, "exact")],
    ids=["claude-exact", "claude-close", "codex-exact"],
)
def test_rubric_review_verdicts(
    model_env, sandbox: str, tasks_root: Path, jobs_root: Path, ledger: h.Ledger,
    reviewer, case,
):  # fmt: skip
    """A real reviewer judges an oracle run; the host computes the reward.

    ``exact``: the greeting is exact, so quality 2/2 and reward 1.
    ``close``: the verifier passes a lower-case greeting, the rubric gives
    the close variant 1 of 2, so reward = 5 / 10 = 0.5 and the trial passes.
    """
    reviewer_model = _claude_model() if reviewer == CLAUDE else _codex_model()
    _need(
        model_env,
        "CLAUDE_CODE_OAUTH_TOKEN" if reviewer == CLAUDE else "CODEX_AUTH_JSON",
    )
    name = f"e2e-rubric-{case}"
    task = h.write_task(
        tasks_root / "rubric", name,
        solve=h.HELLO_SOLVE if case == "exact" else CLOSE_SOLVE,
        test=h.HELLO_TEST if case == "exact" else LENIENT_HELLO_TEST,
        extra_files={"verifier/rubric.json": json.dumps(RUBRIC, indent=2)},
    )  # fmt: skip
    job = jobs_root / f"rubric-{reviewer}-{case}"
    h.clear_job(job)
    run = h.bench(
        "eval", "run", "--tasks-dir", task, "--agent", "oracle", "--sandbox", sandbox,
        "--reviewer-agent", reviewer, "--reviewer-model", reviewer_model,
        "--reviewer-sandbox", sandbox, "--reviewer-timeout-sec", "900",
        "--jobs-dir", jobs_root, "--job-name", job.name, "--max-sandbox-seconds", "1800",
        env=model_env, log=jobs_root / f"rubric-{reviewer}-{case}.log", timeout=2400,
    )  # fmt: skip
    ledger.record(f"rubric review by {reviewer} ({reviewer_model}), {case} greeting",
                  surface="CLI", seconds=run.seconds, job_dir=job)  # fmt: skip
    trial = h.trial_of(job, name)
    result = h.read_json(trial / "result.json")
    scoring = result.get("scoring") or {}
    assert scoring.get("passed") is True, (scoring, result.get("error"), run.tail())
    _, doc = h.bench_json("eval", "inspect", trial, "--json")
    h.validate(doc, "benchflow-trial.v1.schema.json")
    reviews = doc.get("rubric_reviews") or []
    assert reviews, "no rubric review in the trial document"
    review = reviews[-1]
    assert review["review_valid"] is True and review["status"] == "complete"
    verdicts = {v["name"]: v for v in review["verdicts"]}
    assert set(verdicts) == {"file_present", "exact_greeting"}
    assert verdicts["file_present"]["outcome"] == "pass"
    assert all(v["explanation"] for v in verdicts.values())
    expected_score = 2 if case == "exact" else 1
    assert verdicts["exact_greeting"]["score"] == expected_score, verdicts
    # The host arithmetic: points / max points, gated by tests and blockers.
    assert review["reward"]["rubric_reward"] == pytest.approx(expected_score / 2)
    assert result["rewards"]["reward"] == pytest.approx(expected_score / 2), (
        result["rewards"], review["reward"],
    )  # fmt: skip


@pytest.mark.parametrize("agent", [CLAUDE, CODEX])
def test_public_example_tasks(
    model_env, sandbox: str, jobs_root: Path, ledger: h.Ledger, agent
):
    _need(
        model_env, "CLAUDE_CODE_OAUTH_TOKEN" if agent == CLAUDE else "CODEX_AUTH_JSON"
    )
    model = _claude_model() if agent == CLAUDE else _codex_model()
    job = jobs_root / f"public-{agent}"
    h.clear_job(job)
    run = h.bench(
        "eval", "run", "--tasks-dir", PUBLIC_TASKS, "--agent", agent, "--model", model,
        "--sandbox", sandbox, "--concurrency", "4", "--jobs-dir", jobs_root,
        "--job-name", job.name, "--max-sandbox-seconds", "7200", "--freeze-workspace",
        env=model_env, log=jobs_root / f"public-{agent}.log", timeout=3600,
    )  # fmt: skip
    ledger.record(f"public example tasks with {agent} ({model})", surface="CLI",
                  seconds=run.seconds, job_dir=job)  # fmt: skip
    summary = h.read_json(job / "summary.json")
    # Every trial ran to a verdict or an honest error; nothing silently lost.
    assert summary["total"] == len([p for p in PUBLIC_TASKS.iterdir() if p.is_dir()])
    for trial in h.trial_dirs(job):
        result = h.read_json(trial / "result.json")
        assert (trial / "trajectory" / "acp_trajectory.jsonl").is_file(), trial
        if result.get("rewards") is None:
            assert result.get("error") or result.get("verifier_error"), trial
