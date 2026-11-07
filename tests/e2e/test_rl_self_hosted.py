"""End to end: online-RL data from ``bench eval run`` on a self-hosted policy route.

``bench eval run --agent claude-agent-acp --model vllm/fake-policy`` (and
``sglang/fake-policy``) in a real sandbox, with gateway token capture on. The
policy server is the deterministic fake provider running inside the sandbox,
answering OpenAI chat completions with token ids and logprobs shaped like
vLLM or SGLang. Checked: rewards, per-call token capture in
``llm_trajectory.jsonl``, and ``bench train token-coverage`` reporting every
rollout as training-grade on its route.

Marked ``e2e`` (not in the default suite). Needs a sandbox: Docker, or
``BENCHFLOW_DETERMINISTIC_SANDBOX=daytona`` with ``DAYTONA_API_KEY``. No model
key and no model cost.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.e2e

SANDBOX, SKIP_REASON = h.select_sandbox()
REQUESTED = os.environ.get(h.SANDBOX_ENV, "").strip().lower() in {"docker", "daytona"}
JOBS_ENV = "BENCHFLOW_E2E_JOBS_DIR"


def _need_sandbox() -> str:
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    assert SANDBOX is not None
    return SANDBOX


@pytest.fixture(scope="module")
def e2e_root(tmp_path_factory) -> Path:
    kept = os.environ.get(JOBS_ENV)
    if kept:
        root = Path(kept)
        root.mkdir(parents=True, exist_ok=True)
        return root
    return tmp_path_factory.mktemp("rl-e2e")


@pytest.fixture(scope="module")
def host_fake(e2e_root):
    sandbox = _need_sandbox()
    if h.proxy_runs_in_sandbox(sandbox):
        yield None
        return
    with h.host_fake_llm(e2e_root / "host-fake-llm.jsonl") as url:
        yield url


def _tasks(root: Path, variants: list[h.TaskVariant]) -> Path:
    tasks = root / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        if not (tasks / variant.name).exists():
            h.materialize_task(variant, tasks)
    return tasks


def token_coverage(path: Path) -> dict:
    proc = subprocess.run(
        [h.bench_executable(), "train", "token-coverage", str(path), "--json"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return json.loads(proc.stdout)


@pytest.fixture(scope="module", params=["vllm", "sglang"])
def policy_job(request, e2e_root, host_fake):
    route = request.param
    root = e2e_root / route
    tasks = _tasks(root, [h.TaskVariant("hello-pass", "hello-pass")])
    run = h.run_bench(
        ["eval", "run"],
        tasks_dir=tasks,
        jobs_dir=root / "jobs",
        sandbox=_need_sandbox(),
        host_fake_url=host_fake,
        route=route,
        extra=["--concurrency", "1"],
    )
    return route, run


def test_policy_route_run_scores_and_captures_token_ids(policy_job):
    route, run = policy_job
    assert run.returncode == 0, run.output[-4000:]
    trial = run.trial("hello-pass")
    result = h.read_json(trial / "result.json")
    assert result["rewards"]["reward"] == 1.0, result
    assert result["model"] == h.route_model(route)

    captures = [
        row["metadata"]["token_capture"]
        for row in h.read_jsonl(trial / "trajectory" / "llm_trajectory.jsonl")
    ]
    assert captures, "the gateway recorded no calls"
    for capture in captures:
        assert capture["provider"] == route
        assert capture["requested"]["token_ids"] is True
        assert capture["unavailable"] == {}, capture["unavailable"]
        assert capture["prompt_token_ids"]
        assert capture["completions"][0]["token_ids"]
        assert len(capture["completions"][0]["logprobs"]) == len(
            capture["completions"][0]["token_ids"]
        )


def test_token_coverage_reports_the_policy_route_training_grade(policy_job):
    route, run = policy_job
    report = token_coverage(run.job_dirs()[0])
    assert report["rollouts"] == 1
    [rollout] = report["per_rollout"]
    assert rollout["status"] == "captured"
    assert rollout["path"] == route
    assert rollout["training_grade"] is True, rollout
    assert report["training_grade_rollouts"] == 1


# ---------------------------------------------------------------------------
# Streaming: a trainer reads rollouts while the job runs
# ---------------------------------------------------------------------------

STREAM_VARIANTS = [
    h.TaskVariant("hello-pass", "hello-pass"),
    h.TaskVariant("hello-pass-again", "hello-pass"),
    h.TaskVariant("wrong-answer", "hello-wrong"),
]


@pytest.fixture(scope="module")
def streamed_job(e2e_root, host_fake):
    """``bench eval run`` (3 tasks, one at a time) with two live consumers:
    ``bench train stream --format jsonl`` and ``bf.stream_rollouts`` grouping
    all three rollouts (``group_by=agent,model``, ``group_size=3``)."""
    import threading
    import time

    import benchflow as bf

    root = e2e_root / "stream"
    tasks = _tasks(root, STREAM_VARIANTS)
    jobs = root / "jobs"
    args, env = h.bench_command(
        ["eval", "run"],
        tasks_dir=tasks,
        jobs_dir=jobs,
        sandbox=_need_sandbox(),
        host_fake_url=host_fake,
        route="vllm",
        extra=["--concurrency", "1"],
    )
    started = time.monotonic()
    job = subprocess.Popen(
        args,
        cwd=h.REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    cli = subprocess.Popen(
        [
            h.bench_executable(),
            "train",
            "stream",
            str(jobs),
            "--format",
            "jsonl",
            "--poll-interval",
            "0.5",
            "--timeout",
            "1500",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    python_records: list = []
    python_error: list = []

    def consume() -> None:
        try:
            python_records.extend(
                bf.stream_rollouts(
                    jobs,
                    poll_interval=0.5,
                    timeout=1500,
                    group_by="agent,model",
                    group_size=3,
                )
            )
        except Exception as exc:  # reported by the test
            python_error.append(exc)

    consumer = threading.Thread(target=consume)
    consumer.start()
    cli_lines: list[tuple[float, dict]] = []
    assert cli.stdout is not None
    for line in cli.stdout:
        cli_lines.append((time.monotonic() - started, json.loads(line)))
    cli_code = cli.wait(timeout=60)
    job_output = job.communicate(timeout=1500)[0]
    job_seconds = time.monotonic() - started
    consumer.join(timeout=120)
    (root / "cli-output.txt").write_text(job_output)
    (root / "stream-times.txt").write_text(
        "".join(f"{t:.1f} {r['rollout']}\n" for t, r in cli_lines)
        + f"{job_seconds:.1f} job exited\n"
    )
    (root / "stream.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for _, r in cli_lines)
    )
    return {
        "job_code": job.returncode,
        "job_output": job_output,
        "job_seconds": job_seconds,
        "cli_code": cli_code,
        "cli_stderr": cli.stderr.read() if cli.stderr else "",
        "cli_lines": cli_lines,
        "python_records": python_records,
        "python_error": python_error,
        "jobs": jobs,
    }


def test_stream_cli_emits_each_rollout_before_the_job_ends(streamed_job):
    import jsonschema

    s = streamed_job
    assert s["job_code"] == 0, s["job_output"][-4000:]
    assert s["cli_code"] == 0, s["cli_stderr"]
    records = [r for _, r in s["cli_lines"]]
    assert sorted(r["task"] for r in records) == sorted(v.name for v in STREAM_VARIANTS)
    # The first rollout reached the trainer well before the job finished.
    assert s["cli_lines"][0][0] < s["job_seconds"] - 20, (
        [t for t, _ in s["cli_lines"]],
        s["job_seconds"],
    )
    schema = json.loads(
        (
            h.REPO_ROOT
            / "docs/reference/schemas/benchflow-rollout-stream.v1.schema.json"
        ).read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    for record in records:
        assert list(validator.iter_errors(record)) == []
        assert record["token_capture"]["training_grade"] is True, record[
            "token_capture"
        ]
        assert record["token_capture"]["path"] == "vllm"
        agent = [q for q in record["sequences"] if q["kind"] == "agent"]
        assert len(agent) == 1 and len(agent[0]["calls"]) >= 2
        assert len(agent[0]["completion_ids"]) == len(agent[0]["completion_mask"])
    rewards = {r["task"]: r["reward"] for r in records}
    assert rewards == {"hello-pass": 1.0, "hello-pass-again": 1.0, "wrong-answer": 0.0}


def test_python_stream_groups_the_rollouts_with_advantages(streamed_job):
    s = streamed_job
    assert s["python_error"] == []
    records = {r.task: r for r in s["python_records"]}
    assert set(records) == {v.name for v in STREAM_VARIANTS}
    assert all(r.group_complete for r in records.values())
    assert len({r.group_id for r in records.values()}) == 1
    assert records["wrong-answer"].advantage < 0 < records["hello-pass"].advantage
