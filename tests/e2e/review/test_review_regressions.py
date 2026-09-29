"""Regression scenarios that read the recorded trial files.

Each scenario reproduces a defect on real Daytona sandboxes with synthetic
tasks, and checks the recorded files the way a reviewer reads them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.e2e.review import support as s

UBUNTU = "FROM ubuntu:24.04\nWORKDIR /app\n"
ALPINE = "FROM alpine:3.22\nRUN apk add --no-cache bash\nWORKDIR /app\n"

MULTI_STEP_TOML = """\
version = "1.0"

[environment]
cpus = 1
memory_mb = 2048

[[steps]]
name = "create-file"

[[steps]]
name = "append-content"
"""


def _multi_step_task(root: Path) -> Path:
    task = root / "multi-step"
    (task / "environment").mkdir(parents=True, exist_ok=True)
    (task / "environment" / "Dockerfile").write_text(UBUNTU)
    (task / "task.toml").write_text(MULTI_STEP_TOML)
    for name in ("create-file", "append-content"):
        step = task / "steps" / name
        (step / "tests").mkdir(parents=True, exist_ok=True)
        (step / "instruction.md").write_text(f"Step {name}.\n")
        (step / "tests" / "test.sh").write_text(
            "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
        )
    return task


def test_harbor_multi_step_task_is_refused_on_steps(
    review_sandbox: str, review_out: Path
) -> None:
    """A Harbor multi-step task is refused on ``steps``; the run and the check
    used to name a missing instruction.md instead."""
    tasks = review_out / "tasks-multistep"
    task = _multi_step_task(tasks)

    check = s.bench(
        "tasks",
        "check",
        task,
        "--sandbox",
        review_sandbox,
        log=review_out / "logs" / "multistep-check.log",
    )
    assert check.returncode == 1, check.tail()
    flat = " ".join(check.output.split())
    assert "Unsupported runtime feature: steps" in flat
    assert "Missing required file: instruction.md" not in flat

    job = s.fresh_job(review_out / "jobs" / "multistep")
    run = s.bench(
        "eval",
        "run",
        "--tasks-dir",
        tasks,
        "--agent",
        "nop",
        "--sandbox",
        review_sandbox,
        "--jobs-dir",
        job.parent,
        "--job-name",
        job.name,
        "--quiet",
        log=review_out / "logs" / "multistep-run.log",
    )
    trial = s.trials(job)["multi-step"]
    result = s.read_json(trial / "result.json")
    assert "steps: Harbor multi-step tasks are not supported" in result["error"]
    assert "instruction.md" not in result["error"]
    # Refused before launch: no sandbox was created for it.
    assert not (trial / "sandbox.json").exists(), run.tail()


def test_oracle_batch_records_what_the_tasks_declare(
    review_sandbox: str, review_out: Path
) -> None:
    """The oracle runs as the task's agent user and in its workdir, the sandbox
    user is created with busybox adduser, and the oracle's solve.sh time is
    recorded as agent_execution.

    Three synthetic tasks whose reference solutions are correct: one checks
    the declared ``[agent].user``, one the declared ``[environment].workdir``,
    one runs on Alpine (no useradd). The oracle must pass all three, and each
    trial's record must agree with its own files.
    """
    tasks = review_out / "tasks-oracle"
    s.write_task(
        tasks,
        "declared-user",
        dockerfile=UBUNTU,
        toml_extra='user = "agent"\n',  # lands in the [agent] table
        solve="sleep 2\nwhoami > /app/whoami.txt\n",
        test=s.reward_if("grep -qx agent /app/whoami.txt"),
    )
    s.write_task(
        tasks,
        "declared-workdir",
        dockerfile=UBUNTU,
        toml_extra='[environment]\nworkdir = "/custom-workdir"\n',
        solve="pwd > where.txt\n",
        test=s.reward_if("grep -qx /custom-workdir /custom-workdir/where.txt"),
    )
    s.write_task(
        tasks,
        "alpine-image",
        dockerfile=ALPINE,
        solve="echo done > /app/out.txt\n",
        test=s.reward_if("grep -qx done /app/out.txt"),
    )

    job = s.fresh_job(review_out / "jobs" / "oracle-declared")
    run = s.bench(
        "eval",
        "run",
        "--tasks-dir",
        tasks,
        "--agent",
        "oracle",
        "--sandbox",
        review_sandbox,
        "--jobs-dir",
        job.parent,
        "--job-name",
        job.name,
        "--concurrency",
        "3",
        "--max-sandbox-seconds",
        "1800",
        "--quiet",
        log=review_out / "logs" / "oracle-declared.log",
    )
    assert run.returncode == 0, run.tail()

    trials = s.trials(job)
    assert sorted(trials) == ["alpine-image", "declared-user", "declared-workdir"]
    summary = s.read_json(job / "summary.json")
    assert (summary["total"], summary["passed"]) == (3, 3), summary
    for name, trial in trials.items():
        result = s.read_json(trial / "result.json")
        assert result["rewards"]["reward"] == 1.0, (name, result.get("verifier_error"))
        reward_txt = float((trial / "verifier" / "reward.txt").read_text().strip())
        assert reward_txt == result["rewards"]["reward"], name
        timing = s.read_json(trial / "timing.json")
        assert timing == result["timing"], name
        assert timing.get("agent_execution", 0) > 0, (name, timing)
        oracle = json.loads(
            (trial / "agent" / "acp_trajectory.jsonl").read_text().splitlines()[0]
        )
        assert oracle["type"] == "oracle" and oracle["return_code"] == 0, oracle
    assert s.read_json(trials["declared-user"] / "timing.json")["agent_execution"] >= 2


@pytest.mark.skipif(
    not os.environ.get("CODEX_AUTH_JSON"),
    reason="needs a ChatGPT-login CODEX_AUTH_JSON (no API key is used)",
)
def test_codex_model_the_agent_does_not_offer_fails_once(
    review_sandbox: str, review_out: Path
) -> None:
    """codex-acp asked for a model it does not offer fails fast instead of
    retrying an opaque -32603. No model call is made: the refusal comes from
    session/new, before the prompt."""
    tasks = review_out / "tasks-codex"
    s.write_task(
        tasks,
        "any-task",
        dockerfile=UBUNTU,
        solve="true\n",
        test=s.reward_if("true"),
    )
    job = s.fresh_job(review_out / "jobs" / "codex-unoffered")
    run = s.bench(
        "eval",
        "run",
        "--tasks-dir",
        tasks,
        "--agent",
        "codex-acp",
        "--model",
        "gpt-0-not-a-model",
        "--sandbox",
        review_sandbox,
        "--jobs-dir",
        job.parent,
        "--job-name",
        job.name,
        "--max-sandbox-seconds",
        "900",
        "--quiet",
        log=review_out / "logs" / "codex-unoffered.log",
        keep_logins=True,
    )
    assert run.returncode == 1, run.tail()
    results = list(job.glob("*/result.json"))
    assert len(results) == 1, "a deterministic model refusal must not be retried"
    result = s.read_json(results[0])
    assert result["error_category"] == "agent_integration"
    assert result["error"].startswith("agent integration failure [agent_model]")
    assert "for this login; it offers:" in result["error"]
    assert result["integration_failure_info"]["cause"] == "agent_model"
    summary = s.read_json(job / "summary.json")
    assert summary["integration_failures"]["by_cause"] == {"agent_model": 1}
