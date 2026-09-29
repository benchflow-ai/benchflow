"""Deterministic integration tier: real CLI, real sandbox, scripted model.

Each scenario runs ``bench eval run`` / ``bench eval branch`` with
``claude-agent-acp`` in a real sandbox (Docker, or Daytona on request). The
model is the scripted fake provider under
``tests/integration/deterministic/task/environment/fake_llm``, reached through
BenchFlow's own LiteLLM proxy, so no model key is needed and every run is
reproducible. Each scenario checks ``result.json``, the outcome labels
(``bf.load_trial``), rewards, token usage and cost against the fake's fixed
usage, timing keys, the ATIF export's structure, the ``benchflow.trial`` JSON
schema, and a normalised golden file (ids and timestamps stripped).

Marked ``deterministic``. The tier skips with a reason when no sandbox is
available; see ``tests/integration/README.md`` for backends, adding a
scenario and regenerating golden files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
# An explicitly requested backend that is unavailable fails instead of
# skipping, so a CI job cannot go green by skipping every scenario.
REQUESTED = os.environ.get(h.SANDBOX_ENV, "").strip().lower() in {"docker", "daytona"}


@pytest.fixture
def require_sandbox() -> None:
    if SANDBOX is None:
        if REQUESTED:
            pytest.fail(SKIP_REASON)
        pytest.skip(SKIP_REASON)


needs_sandbox = pytest.mark.usefixtures("require_sandbox")

# One `bench eval run` job runs these five tasks together (one sandbox each).
EVAL_VARIANTS = {
    "hello-pass": h.TaskVariant("hello-pass", "hello-pass"),
    "wrong-answer": h.TaskVariant("wrong-answer", "hello-wrong"),
    "timeout-pass": h.TaskVariant(
        "timeout-pass", "pass-then-hang", agent_timeout_sec=45.0
    ),
    "agent-crash": h.TaskVariant("agent-crash", "crash"),
    "verifier-error": h.TaskVariant(
        "verifier-error", "hello-pass", broken_verifier=True
    ),
}

DRAFT_PROMPT = f"Write a draft first. {h.marker('draft')}"
PASS_CHILD = f"label=pass,prompt=Write the final answer. {h.marker('hello-pass')}"
WRONG_CHILD = f"label=wrong,prompt=Write a different answer. {h.marker('hello-wrong')}"
RETRY_PROMPT = f"Your answer was wrong; fix hello.txt. {h.marker('hello-pass')}"


# ---------------------------------------------------------------------------
# Session fixtures: each CLI job runs once, on first use
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def det_root(tmp_path_factory) -> Path:
    kept = os.environ.get(h.KEEP_JOBS_ENV)
    if kept:
        root = Path(kept)
        root.mkdir(parents=True, exist_ok=True)
        return root
    return tmp_path_factory.mktemp("deterministic")


@pytest.fixture(scope="session")
def fake_url(det_root) -> Any:
    """Host fake provider for backends whose LiteLLM proxy runs on the host."""
    if SANDBOX is None or h.proxy_runs_in_sandbox(SANDBOX):
        yield None
        return
    with h.host_fake_llm(det_root / "host-fake-llm.jsonl") as url:
        yield url


def _run(
    det_root: Path, name: str, subcommand, variants, fake_url, extra=(), route="proxy"
):
    # Session fixtures are set up before the function-scoped require_sandbox.
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    tasks = det_root / name / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        h.materialize_task(variant, tasks)
    tasks_dir = tasks / variants[0].name if len(variants) == 1 else tasks
    run = h.run_bench(
        subcommand,
        tasks_dir=tasks_dir,
        jobs_dir=det_root / name / "jobs",
        sandbox=SANDBOX,
        host_fake_url=fake_url,
        route=route,
        extra=extra,
    )
    (det_root / name / "seconds.txt").write_text(f"{run.seconds:.1f}\n")
    return run


@pytest.fixture(scope="session")
def eval_job(det_root, fake_url) -> h.CliRun:
    return _run(
        det_root,
        "eval",
        ["eval", "run"],
        list(EVAL_VARIANTS.values()),
        fake_url,
        extra=[
            "--concurrency",
            os.environ.get("BENCHFLOW_DETERMINISTIC_CONCURRENCY", "4"),
            # The default policy re-runs the crashed task twice; one trial per
            # task keeps the job short and every assertion about one trial.
            "--retry-attempts",
            "0",
        ],
    )


def _branch(det_root, fake_url, name: str, extra=()) -> h.CliRun:
    return _run(
        det_root,
        name,
        ["eval", "branch"],
        [h.TaskVariant(name, "hello-pass")],
        fake_url,
        # Branching refuses rollouts with a provider runtime (the LiteLLM
        # proxy), so branches take the native route straight to the fake.
        route="native",
        extra=[
            "--prompt",
            DRAFT_PROMPT,
            "--prompt",
            "@instruction",
            "--checkpoint-after-prompt",
            "1",
            "--child",
            PASS_CHILD,
            "--child",
            WRONG_CHILD,
            *extra,
        ],
    )


@pytest.fixture(scope="session")
def branch_in_place(det_root, fake_url) -> h.CliRun:
    return _branch(det_root, fake_url, "branch-in-place")


@pytest.fixture(scope="session")
def branch_parallel(det_root, fake_url) -> h.CliRun:
    return _branch(det_root, fake_url, "branch-parallel", extra=["--concurrency", "2"])


@pytest.fixture(scope="session")
def checkpoint_retry(det_root, fake_url):
    run = _run(
        det_root,
        "checkpoint-retry",
        ["eval", "run"],
        [h.TaskVariant("checkpoint-retry", "hello-wrong")],
        fake_url,
        extra=[
            "--checkpoints",
            "prompt:1",
            "--retry-from-checkpoint",
            "on-failure",
            "--retry-prompt",
            RETRY_PROMPT,
        ],
    )
    yield run
    # Kept checkpoints outlive a run on purpose; delete this tier's own.
    for trial in run.trial_dirs():
        assert h.release_kept_checkpoints(trial, SANDBOX) == []


# ---------------------------------------------------------------------------
# Shared checks
# ---------------------------------------------------------------------------


def _trial_facts(trial_dir: Path) -> dict[str, Any]:
    """The run-independent facts of one trial, as stored in its golden file."""
    import benchflow as bf

    result = h.read_json(trial_dir / "result.json")
    trial = bf.load_trial(trial_dir)
    timeout = result.get("agent_timeout_info") or None
    return {
        "outcome": {
            "execution": trial.execution,
            "assessment": trial.assessment,
            "reward": trial.reward,
        },
        "result": {
            "rewards": result.get("rewards"),
            "n_tool_calls": result.get("n_tool_calls"),
            "n_prompts": result.get("n_prompts"),
            "error_category": result.get("error_category"),
            "verifier_error_category": result.get("verifier_error_category"),
            "agent_timeout": {
                k: timeout.get(k)
                for k in (
                    "reason",
                    "timeout_sec",
                    "n_tool_calls",
                    "pending_tool_call_ids",
                )
            }
            if timeout
            else None,
            "agent_result": result.get("agent_result"),
            "final_metrics": result.get("final_metrics"),
            "trajectory_summary": result.get("trajectory_summary"),
            # environment and endpoint_kind name the backend; golden files are
            # shared by every backend, so _check_trial asserts those two.
            "usage_tracking": {
                k: v
                for k, v in (result.get("usage_tracking") or {}).items()
                if k not in {"environment", "endpoint_kind"}
            },
            "timing_keys": sorted((result.get("timing") or {}).keys()),
        },
        "llm_calls": h.llm_calls(trial_dir),
        "acp_trajectory": h.normalized_acp(trial_dir),
        "atif": h.normalized_atif(trial_dir),
    }


_USAGE_KEYS = {
    # field -> which side-call quantity it counts
    "n_input_tokens": "input",
    "n_output_tokens": "output",
    "total_tokens": "total",
    "cost_usd": "cost",
    "total_prompt_tokens": "input",
    "total_completion_tokens": "output",
    "total_cost_usd": "cost",
}


def _golden_view(facts: dict[str, Any]) -> dict[str, Any]:
    """``facts`` without Claude Code's side calls, for the golden file.

    After its last reply Claude Code asks the model for a session title (a
    request without tools). Whether that request reaches the proxy before
    BenchFlow tears the agent down is a race (seen on Docker and on
    Daytona), so the golden file keeps agent-loop calls only and reports
    usage net of side calls. _check_trial still checks every captured call,
    side calls included, against the recorded totals.
    """
    view = json.loads(json.dumps(facts))
    n_side = sum(c["kind"] == "side" for c in view["llm_calls"])
    view["llm_calls"] = [c for c in view["llm_calls"] if c["kind"] == "main"]
    side = h.SIDE_CALL_USAGE
    per_call = {
        "input": side["input"],
        "output": side["output"],
        "total": side["input"] + side["output"],
        "cost": side["input"] * h.PRICE_PER_TOKEN["input"]
        + side["output"] * h.PRICE_PER_TOKEN["output"],
    }
    blocks = [view["result"]["agent_result"], view["result"]["final_metrics"]]
    if view.get("atif"):
        blocks.append(view["atif"].get("final_metrics"))
    for block in blocks:
        for key, quantity in _USAGE_KEYS.items():
            if isinstance(block, dict) and isinstance(block.get(key), int | float):
                value = block[key] - n_side * per_call[quantity]
                block[key] = round(value, 9) if quantity == "cost" else value
    return view


def _check_trial(
    trial_dir: Path, golden: str | None, route: str = "proxy"
) -> dict[str, Any]:
    facts = _trial_facts(trial_dir)
    result = h.read_json(trial_dir / "result.json")
    agent_result = result["agent_result"]
    metrics = result["final_metrics"]

    # Every scripted model reply starts with text, so each model call shows up
    # as exactly one agent message in the ACP trajectory.
    n_agent_messages = sum(
        e.get("type") == "agent_message" for e in facts["acp_trajectory"]
    )
    calls = facts["llm_calls"]
    if route == "proxy":
        # Usage and cost follow exactly from the fake's fixed per-call usage,
        # as captured by the LiteLLM proxy.
        assert calls, "the LiteLLM proxy captured no provider calls"
        for call in calls:
            expected = (
                h.MAIN_CALL_USAGE if call["kind"] == "main" else h.SIDE_CALL_USAGE
            )
            assert (call["input_tokens"], call["output_tokens"]) == (
                expected["input"],
                expected["output"],
            ), call
        assert sum(c["kind"] == "main" for c in calls) == n_agent_messages
        usage = h.expected_usage(calls)
        assert agent_result["cost_usd"] == pytest.approx(usage["cost_usd"], abs=1e-9)
        assert metrics["total_cost_usd"] == pytest.approx(usage["cost_usd"], abs=1e-9)
        assert agent_result["usage_source"] == "provider_response"
    else:
        # Native route: no proxy; Claude Code reports the fake's usage over
        # ACP, and a subscription run has no price.
        assert calls == []
        main = [{"kind": "main"}] * n_agent_messages
        usage = h.expected_usage(main)
        assert agent_result["cost_usd"] is None and metrics["total_cost_usd"] is None
        assert agent_result["usage_source"] == "agent_native_acp"
    tracking = result["usage_tracking"]
    assert tracking["usage_source"] == agent_result["usage_source"]
    assert tracking["environment"] == SANDBOX
    proxy_kind = "sandbox" if h.proxy_runs_in_sandbox(SANDBOX) else "host"
    assert tracking["endpoint_kind"] == (
        "agent_native" if route == "native" else proxy_kind
    )
    for key in ("n_input_tokens", "n_output_tokens", "total_tokens"):
        assert agent_result[key] == usage[key], (key, agent_result, usage)
    assert metrics["total_prompt_tokens"] == usage["n_input_tokens"]
    assert metrics["total_completion_tokens"] == usage["n_output_tokens"]

    # Timing: every phase that ran is timed.
    timing = result.get("timing") or {}
    assert {"environment_setup", "agent_setup", "total"} <= set(timing)
    assert all(isinstance(v, int | float) and v >= 0 for v in timing.values()), timing

    # Trajectory exports: ATIF structure and the public trial document schema.
    atif = h.read_json(trial_dir / "trainer" / "atif.json")
    assert h.atif_problems(atif) == []
    assert h.trial_document_problems(trial_dir) == []

    # Tool call ids come from the fake provider (toolu_fake_<script>_<step>).
    acp_ids = [
        e.get("tool_call_id")
        for e in facts["acp_trajectory"]
        if e.get("type") == "tool_call"
    ]
    assert acp_ids and all(i.startswith("toolu_fake_") for i in acp_ids), acp_ids
    if route == "proxy":
        scripted = [
            tc["id"] for c in calls if c["kind"] == "main" for tc in c["tool_calls"]
        ]
        assert acp_ids == scripted

    if golden is not None:
        h.Golden(golden).check(_golden_view(facts))
    return facts


def _verifier_ran(trial_dir: Path) -> bool:
    return (trial_dir / "verifier" / "test-stdout.txt").is_file() or (
        trial_dir / "verifier" / "reward.txt"
    ).is_file()


# ---------------------------------------------------------------------------
# Scenarios: bench eval run
# ---------------------------------------------------------------------------


@needs_sandbox
def test_eval_job_summary(eval_job):
    """The summary counts each outcome once; errored trials make the CLI exit 1."""
    # _exit_if_evaluation_had_errors: an errored or verifier-errored trial.
    assert eval_job.returncode == 1, eval_job.output[-4000:]
    (job,) = eval_job.job_dirs()
    summary = h.read_json(job / "summary.json")
    assert summary["total"] == 5
    counts = {
        k: summary[k]
        for k in ("passed", "failed", "errored", "verifier_errored", "unscored")
    }
    # timeout-pass counts as passed (it has a reward); agent-crash is the error.
    assert counts == {
        "passed": 2,
        "failed": 1,
        "errored": 1,
        "verifier_errored": 1,
        "unscored": 0,
    }
    assert summary["error_categories"] == {"acp_error": 1, "timeout": 1}
    assert summary["verifier_error_categories"] == {"verifier_failure": 1}
    # usage_summary totals only completed trials (a reward, no agent or
    # verifier error): here hello-pass and wrong-answer, not the other three.
    results = [h.read_json(t / "result.json") for t in eval_job.trial_dirs()]
    completed = [
        r
        for r in results
        if r["rewards"] is not None and not r["error"] and not r["verifier_error"]
    ]
    assert len(completed) == 2
    for field in ("input", "output"):
        assert summary[f"total_{field}_tokens"] == sum(
            r["agent_result"][f"n_{field}_tokens"] for r in completed
        )


@needs_sandbox
def test_hello_world_passes(eval_job):
    trial = eval_job.trial("hello-pass")
    facts = _check_trial(trial, "eval-hello-pass")
    assert facts["outcome"] == {
        "execution": "completed",
        "assessment": "scored",
        "reward": 1.0,
    }
    assert _verifier_ran(trial)
    assert set(facts["result"]["timing_keys"]) >= h.TIMING_KEYS


@needs_sandbox
def test_wrong_answer_scores_zero(eval_job):
    trial = eval_job.trial("wrong-answer")
    facts = _check_trial(trial, "eval-wrong-answer")
    assert facts["outcome"] == {
        "execution": "completed",
        "assessment": "scored",
        "reward": 0.0,
    }
    assert (trial / "verifier" / "reward.txt").read_text().strip() == "0"


@needs_sandbox
def test_agent_timeout_with_passing_file_is_timed_out_and_passed(eval_job):
    """The agent wrote the right file, then hung: the verifier still runs."""
    trial = eval_job.trial("timeout-pass")
    facts = _check_trial(trial, "eval-timeout-pass")
    assert facts["outcome"] == {
        "execution": "timed_out",
        "assessment": "scored",
        "reward": 1.0,
    }
    assert facts["result"]["error_category"] == "timeout"
    timeout = facts["result"]["agent_timeout"]
    assert timeout["reason"] == "wall_clock_timeout"
    assert timeout["pending_tool_call_ids"] == ["toolu_fake_pass-then-hang_1"]
    assert _verifier_ran(trial)
    assert "verifier" in facts["result"]["timing_keys"]


@needs_sandbox
def test_agent_crash_is_unscored_and_not_verified(eval_job):
    """The agent process dies mid-run: no reward, and the verifier is not run."""
    trial = eval_job.trial("agent-crash")
    facts = _check_trial(trial, "eval-agent-crash")
    assert facts["outcome"]["execution"] == "errored"
    assert facts["outcome"]["assessment"] == "unscored"
    assert facts["outcome"]["reward"] is None
    assert facts["result"]["rewards"] is None
    assert not _verifier_ran(trial)


@needs_sandbox
def test_verifier_error_is_an_assessment_error_never_zero(eval_job):
    trial = eval_job.trial("verifier-error")
    facts = _check_trial(trial, "eval-verifier-error")
    assert facts["outcome"] == {
        "execution": "completed",
        "assessment": "error",
        "reward": None,
    }
    assert facts["result"]["rewards"] is None
    assert facts["result"]["verifier_error_category"] == "verifier_failure"
    result = h.read_json(trial / "result.json")
    assert "rc=3" in result["verifier_error"]


# ---------------------------------------------------------------------------
# Scenarios: bench eval branch
# ---------------------------------------------------------------------------


def _check_branch(run: h.CliRun, golden: str, *, isolated: bool) -> None:
    assert run.returncode == 0, run.output[-4000:]
    (trial,) = run.trial_dirs()
    tree = h.read_json(trial / "tree.json")
    (fork,) = tree["forks"]
    children = fork["children"]
    labels = [(c.get("intervention") or {}).get("label") for c in children]
    assert labels == ["pass", "wrong"]
    assert [c["reward"] for c in children] == [1.0, 0.0]
    assert [c["status"] for c in children] == ["scored", "scored"]
    assert {c["reward_source"] for c in children} == {"verifier"}
    assert fork["status"] == "completed"
    assert fork["value"] == 0.5
    assert fork["value_stderr"] == 0.5
    assert fork["parent_restore"] == "restored"
    mode = fork["children_mode"]
    assert mode["isolated"] is isolated
    assert mode["concurrency"] == (2 if isolated else 1)

    # The parent was restored to the checkpoint and finished its own prompts.
    result = h.read_json(trial / "result.json")
    assert result["rewards"] == {"reward": 1.0}
    assert result["n_prompts"] == 2
    assert result["branches"]["forks"]
    child_root = trial / "branches" / fork["id"] / "children"
    child_dirs = sorted(p for p in child_root.iterdir() if p.is_dir())
    assert [p.name for p in child_dirs] == sorted(c["node_id"] for c in children)
    for child_dir, child in zip(
        [child_root / c["node_id"] for c in children], children, strict=True
    ):
        assert (child_dir / "observation.json").is_file()
        if isolated:
            child_result = h.read_json(child_dir / "result.json")
            assert child_result["rewards"] == {"reward": child["reward"]}
            _check_trial(child_dir, None, route="native")

    parent = _check_trial(trial, None, route="native")
    h.Golden(golden).check(
        {
            "fork": {
                "status": fork["status"],
                "value": fork["value"],
                "value_stderr": fork["value_stderr"],
                "parent_restore": fork["parent_restore"],
                "children_mode": mode,
                "children": [
                    {
                        "label": label,
                        **{k: c.get(k) for k in ("status", "reward", "reward_source")},
                    }
                    for label, c in zip(labels, children, strict=True)
                ],
            },
            "parent": _golden_view(parent),
        }
    )


@needs_sandbox
def test_branch_two_children_in_place(branch_in_place):
    _check_branch(branch_in_place, "branch-in-place", isolated=False)


@needs_sandbox
def test_branch_two_children_parallel(branch_parallel):
    _check_branch(branch_parallel, "branch-parallel", isolated=True)


# ---------------------------------------------------------------------------
# Scenario: checkpoints + retry from checkpoint
# ---------------------------------------------------------------------------


@needs_sandbox
def test_retry_from_checkpoint_keeps_both_scores(checkpoint_retry):
    assert checkpoint_retry.returncode == 0, checkpoint_retry.output[-4000:]
    (trial,) = [p for p in checkpoint_retry.trial_dirs() if "/branches/" not in str(p)]
    result = h.read_json(trial / "result.json")
    assert result["rewards"] == {"reward": 0.0}, "the trial keeps its own score"
    retry = result["retry"]
    assert (retry["status"], retry["reason"], retry["checkpoint"]) == (
        "completed",
        "failure",
        "prompt:1",
    ), retry
    assert (retry["original_reward"], retry["reward"]) == (0.0, 1.0), retry
    retry_dir = trial / retry["path"]
    assert h.read_json(retry_dir / "result.json")["rewards"] == {"reward": 1.0}
    _check_trial(retry_dir, None)
    checkpoints = h.read_json(trial / "checkpoints.json")
    assert checkpoints
    h.Golden("checkpoint-retry").check(
        {
            "retry": {
                k: retry.get(k)
                for k in ("status", "reason", "checkpoint", "reward", "original_reward")
            },
            "trial": _golden_view(_trial_facts(trial)),
        }
    )


def test_tier_skips_with_a_reason_or_selects_a_backend():
    """The marker never fails for lack of a sandbox; it says why it skipped."""
    if SANDBOX is None:
        assert SKIP_REASON and h.SANDBOX_ENV in SKIP_REASON
    else:
        assert SANDBOX in {"docker", "daytona"}


def test_fixture_task_is_valid_and_marks_one_script():
    from benchflow._utils.task_authoring import check_task

    assert check_task(h.TEMPLATE_TASK) == []
    scripts = json.loads((h.FAKE_LLM_DIR / "scripts.json").read_text())
    for variant in EVAL_VARIANTS.values():
        assert variant.script in scripts
    for prompt in (DRAFT_PROMPT, PASS_CHILD, WRONG_CHILD, RETRY_PROMPT):
        (name,) = __import__("re").findall(r"\[\[fake-llm:([^\]]+)\]\]", prompt)
        assert name in scripts
