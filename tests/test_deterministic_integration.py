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
import re
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
def native_eval(det_root, fake_url) -> h.CliRun:
    """One unbranched trial on the subscription (native) route."""
    return _run(
        det_root,
        "native-eval",
        ["eval", "run"],
        [h.TaskVariant("native-hello", "hello-pass")],
        fake_url,
        route="native",
        extra=["--retry-attempts", "0"],
    )


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
        # ACP, and the gateway has no price for it.
        assert calls == []
        main = [{"kind": "main"}] * n_agent_messages
        usage = h.expected_usage(main)
        assert agent_result["usage_source"] == "agent_native_acp"
        if agent_result.get("price_source") == "agent_session_log":
            # An unbranched run is priced from Claude Code's own session log,
            # an estimate; Claude Code's session-title call may be in it.
            estimate = agent_result["usage_details"]["cost_estimate"]
            assert estimate["source"] == "claude-code-session-log", estimate
            assert (trial_dir / estimate["path"]).is_dir()
            assert agent_result["cost_usd"] == pytest.approx(
                usage["cost_usd"], abs=1e-4
            )
            assert metrics["total_cost_usd"] == agent_result["cost_usd"]
        else:
            # A branched run is not: its children share the session log.
            assert agent_result["cost_usd"] is None
            assert metrics["total_cost_usd"] is None
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
    """Claude Code dies mid-run under a live adapter: no reward, no verifier."""
    trial = eval_job.trial("agent-crash")
    facts = _check_trial(trial, "eval-agent-crash")
    assert facts["outcome"]["execution"] == "errored"
    assert facts["outcome"]["assessment"] == "unscored"
    assert facts["outcome"]["reward"] is None
    assert facts["result"]["rewards"] is None
    assert not _verifier_ran(trial)


@needs_sandbox
def test_a_subscription_run_is_priced_from_claude_codes_session_log(native_eval):
    """No gateway price: the USD is Claude Code's own, from its session log.

    The native route is a subscription run (Claude Code calls the model
    itself); BenchFlow used to record its tokens with cost_usd null.
    """
    trial = native_eval.trial("native-hello")
    facts = _check_trial(trial, None, route="native")
    result = h.read_json(trial / "result.json")
    assert result["agent_result"]["price_source"] == "agent_session_log"
    assert facts["outcome"]["reward"] == 1.0
    assert list((trial / "agent" / "claude-sessions").rglob("*.jsonl"))


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


def _check_branch(
    run: h.CliRun, golden: str, *, isolated: bool, native: bool = False
) -> None:
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
    parent_view = _golden_view(parent)
    if native:
        # Claude Code's own CLI: the ACP run's golden file, bar the ATIF name.
        _check_native_contract(trial, "claude-code", proxied=False)
        parent_view = _as_acp_golden(parent_view)
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
            "parent": parent_view,
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


# ---------------------------------------------------------------------------
# Native harness (benchflow.native_harness): the same scenarios through the
# agents' own CLIs (`--harness native`). Claude Code native must reproduce
# the ACP golden files above; Codex runs both harnesses here against the
# fake's Responses API and compares them. Every native trial also passes the
# harness contract: the pinned CLI ran with its headless flags, its JSON
# events are kinds the recorded samples of that pin contain, and every model
# call it made reached BenchFlow's proxy.
# ---------------------------------------------------------------------------

NATIVE = ("--harness", "native")
NATIVE_SAMPLES = Path(__file__).parent / "fixtures" / "native_harness"
# The one field of the golden view a native Claude Code run legitimately
# changes: the ATIF agent name is the CLI, not the ACP adapter.
ACP_AGENT_NAME = "@agentclientprotocol/claude-agent-acp"

CODEX_VARIANTS = {
    "codex-hello-pass": h.TaskVariant("codex-hello-pass", "hello-pass"),
    "codex-wrong-answer": h.TaskVariant("codex-wrong-answer", "hello-wrong"),
    # codex-acp does not notice its Codex process dying: this runs to the
    # agent timeout on the ACP side (see _CODEX_CRASH).
    "codex-agent-crash": h.TaskVariant(
        "codex-agent-crash", "codex-crash", agent_timeout_sec=60.0
    ),
    "codex-slow-model": h.TaskVariant(
        "codex-slow-model", "slow-model", agent_timeout_sec=45.0
    ),
}
# One allowlisted task (agent.network_mode: allowlist): only example.com and
# the model gateway are reachable for the agent.
ALLOWLIST_VARIANT = h.TaskVariant(
    "codex-allowlist", "hello-pass", allowed_hosts=("example.com",)
)


def _concurrency() -> list[str]:
    return [
        "--concurrency",
        os.environ.get("BENCHFLOW_DETERMINISTIC_CONCURRENCY", "4"),
        "--retry-attempts",
        "0",
    ]


@pytest.fixture(scope="session")
def native_eval_job(det_root, fake_url) -> h.CliRun:
    return _run(
        det_root,
        "eval-native",
        ["eval", "run"],
        list(EVAL_VARIANTS.values()),
        fake_url,
        extra=[*_concurrency(), *NATIVE],
    )


@pytest.fixture(scope="session")
def native_branch_in_place(det_root, fake_url) -> h.CliRun:
    return _branch(det_root, fake_url, "branch-in-place-native", extra=NATIVE)


@pytest.fixture(scope="session")
def native_branch_resume(det_root, fake_url) -> h.CliRun:
    return _branch(
        det_root, fake_url, "branch-resume-native", extra=[*NATIVE, "--resume-session"]
    )


@pytest.fixture(scope="session")
def codex_acp_job(det_root, fake_url) -> h.CliRun:
    return _run(
        det_root,
        "codex-acp",
        ["eval", "run"],
        list(CODEX_VARIANTS.values()),
        fake_url,
        route="codex",
        extra=_concurrency(),
    )


@pytest.fixture(scope="session")
def codex_native_job(det_root, fake_url) -> h.CliRun:
    return _run(
        det_root,
        "codex-native",
        ["eval", "run"],
        list(CODEX_VARIANTS.values()),
        fake_url,
        route="codex",
        extra=[*_concurrency(), *NATIVE],
    )


@pytest.fixture(scope="session")
def codex_allowlist_jobs(det_root, fake_url) -> dict[str, h.CliRun]:
    """The allowlisted task on both harnesses (proxy in the sandbox)."""
    return {
        harness: _run(
            det_root,
            f"codex-allowlist-{harness}",
            ["eval", "run"],
            [ALLOWLIST_VARIANT],
            fake_url,
            route="codex-in-sandbox",
            extra=["--retry-attempts", "0", "--harness", harness],
        )
        for harness in ("acp", "native")
    }


def _stream(trial: Path, cli: str) -> list[dict[str, Any]]:
    path = trial / "agent" / f"{cli}.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    return [e for e in events if "benchflow_native_turn" not in e]


def _event_kind(event: dict[str, Any]) -> tuple[str, ...]:
    kind = str(event.get("type"))
    if kind == "stream_event":
        return kind, str((event.get("event") or {}).get("type"))
    if kind in ("system", "result"):
        return kind, str(event.get("subtype"))
    if kind.startswith("item."):
        return kind, str((event.get("item") or {}).get("type"))
    return (kind,)


def _sample_kinds(cli: str) -> set[tuple[str, ...]]:
    from benchflow.agents.registry import pinned_npm_package

    version = pinned_npm_package("claude-code" if cli == "claude-code" else "codex")[1]
    kinds: set[tuple[str, ...]] = set()
    for path in (NATIVE_SAMPLES / f"{cli}-{version}").glob("*.jsonl"):
        kinds |= {
            _event_kind(json.loads(line)) for line in path.read_text().splitlines()
        }
    return kinds


def _check_native_contract(trial: Path, cli: str, *, proxied: bool = True) -> None:
    """The harness contract, on one native trial.

    ``proxied=False`` is the subscription route (the CLI calls the model
    itself, as over ACP), where there is no proxy to account for the calls.
    """
    from benchflow.native_harness.harnesses import CLAUDE_CODE, CODEX

    harness = CLAUDE_CODE if cli == "claude-code" else CODEX
    config = h.read_json(trial / "config.json")
    assert config["harness_mode"] == "native"
    assert config["native_harness"] == {"cli": cli, "package": harness.package}
    turns = h.read_json(trial / "agent" / "native-turns.json")
    assert turns, "no native turn ran"
    for turn in turns:
        assert turn["version"] == harness.version
        argv = turn["argv"]
        # The headless flags the harness launches the pinned CLI with.
        if cli == "claude-code":
            assert argv[:3] == ["-p", "--output-format", "stream-json"]
            assert {"--verbose", "--include-partial-messages"} <= set(argv)
        else:
            assert argv[0] == "exec" and "--json" in argv and argv[-1] == "-"
    # The JSON output parses, and every event is a kind the pin's recorded
    # samples contain (a new kind is a format change to look at).
    stream = _stream(trial, cli)
    assert stream, "the CLI wrote no JSON events"
    unknown = {_event_kind(e) for e in stream} - _sample_kinds(cli)
    assert not unknown, f"event kinds the {cli} samples do not have: {unknown}"
    # Every model call the CLI made reached BenchFlow's proxy: the model
    # responses the CLI reports equal the calls the proxy captured.
    if cli == "claude-code":
        responses = {
            (e.get("message") or {}).get("id")
            for e in stream
            if e.get("type") == "assistant" and e.get("parent_tool_use_id") is None
        }
    else:
        responses = {
            (e.get("item") or {}).get("id")
            for e in stream
            if e.get("type") == "item.completed"
            and (e.get("item") or {}).get("type") == "agent_message"
        }
    main_calls = [c for c in h.llm_calls(trial) if c["kind"] == "main"]
    if proxied:
        assert len(main_calls) == len(responses), (main_calls, responses)
    else:
        assert main_calls == [] and responses


def _as_acp_golden(view: dict[str, Any]) -> dict[str, Any]:
    atif = view.get("atif")
    if isinstance(atif, dict):
        atif["agent"]["name"] = ACP_AGENT_NAME
    return view


@needs_sandbox
def test_native_eval_job_summary_matches_acp(native_eval_job):
    """The same five tasks end the same way through Claude Code's own CLI."""
    assert native_eval_job.returncode == 1, native_eval_job.output[-4000:]
    (job,) = native_eval_job.job_dirs()
    summary = h.read_json(job / "summary.json")
    counts = {
        k: summary[k]
        for k in ("passed", "failed", "errored", "verifier_errored", "unscored")
    }
    assert counts == {
        "passed": 2,
        "failed": 1,
        "errored": 1,
        "verifier_errored": 1,
        "unscored": 0,
    }
    assert summary["error_categories"] == {"acp_error": 1, "timeout": 1}
    assert summary["verifier_error_categories"] == {"verifier_failure": 1}


@needs_sandbox
@pytest.mark.parametrize(
    ("variant", "golden"),
    [
        ("hello-pass", "eval-hello-pass"),
        ("wrong-answer", "eval-wrong-answer"),
        ("timeout-pass", "eval-timeout-pass"),
        ("agent-crash", "eval-agent-crash"),
        ("verifier-error", "eval-verifier-error"),
    ],
)
def test_native_claude_code_reproduces_the_acp_golden(native_eval_job, variant, golden):
    """Outcome parity, and more: the ACP run's golden file, field for field.

    The golden view holds the outcome labels, the result.json subset
    (rewards, counts, error categories, usage, metrics, trajectory summary,
    timing keys), the provider calls, the ACP trajectory and the ATIF export.
    Only the ATIF agent name may differ.
    """
    trial = native_eval_job.trial(variant)
    facts = _check_trial(trial, None)
    _check_native_contract(trial, "claude-code")
    h.Golden(golden).check(_as_acp_golden(_golden_view(facts)))


@needs_sandbox
def test_native_timeout_kills_the_cli_and_keeps_the_pending_call(native_eval_job):
    trial = native_eval_job.trial("timeout-pass")
    (turn,) = h.read_json(trial / "agent" / "native-turns.json")
    assert turn["cancelled"] is True and turn["stop_reason"] == "cancelled"


@needs_sandbox
def test_native_crash_is_an_agent_error_with_the_clis_exit_status(native_eval_job):
    trial = native_eval_job.trial("agent-crash")
    result = h.read_json(trial / "result.json")
    assert result["error"].startswith("Native harness error (claude-code)")
    assert "exit code 137" in result["error"]


@needs_sandbox
def test_native_branch_reproduces_the_acp_golden(native_branch_in_place):
    """Two prompts on one CLI session, a checkpoint, two children, a restore."""
    _check_branch(
        native_branch_in_place, "branch-in-place", isolated=False, native=True
    )


@needs_sandbox
def test_native_branch_children_resume_the_cli_session(native_branch_resume):
    """--resume-session: each child continues the parent's CLI session.

    The session lives in the sandbox (Claude Code's session log under the
    sandbox user's home), so the child's ``claude -p --resume`` finds it in
    the restored checkpoint; a missing session fails the child's turn.
    """
    run = native_branch_resume
    assert run.returncode == 0, run.output[-4000:]
    (trial,) = run.trial_dirs()
    (fork,) = h.read_json(trial / "tree.json")["forks"]
    assert [c["reward"] for c in fork["children"]] == [1.0, 0.0]
    draft = h.read_json(trial / "agent" / "native-turns.json")[0]
    assert draft["resumed"] is None and draft["stop_reason"] == "end_turn"
    children = trial / "branches" / fork["id"] / "children"
    for child in fork["children"]:
        # Each child writes its own evidence (the rollout follows it there).
        (turn,) = h.read_json(
            children / child["node_id"] / "agent" / "native-turns.json"
        )
        assert turn["resumed"] == draft["session_id"]
        assert turn["argv"][turn["argv"].index("--resume") + 1] == draft["session_id"]
        # The CLI continued that session rather than starting a new one.
        assert turn["session_id"] == draft["session_id"]
        assert turn["stop_reason"] == "end_turn"


# ----- wire parity ---------------------------------------------------------

# The written list of expected request differences: where the two harnesses'
# requests may differ, and why. Each one is removed by exactly one
# normalization below (_normalize_claude, _normalize_codex); after them, and
# after ids are normalized, every request pair must be identical: model,
# system prompt, messages, tool definitions, thinking, limits and identifying
# headers.
WIRE_EXPECTED_DIFFERENCES: dict[str, list[tuple[str, str]]] = {
    "claude-code": [
        (
            "headers.user-agent",
            "Claude Code names its entrypoint: sdk-ts and the Agent SDK version "
            "(the SDK inside claude-agent-acp), or sdk-cli (print mode)",
        ),
        (
            "body.system[0].text",
            "the same entrypoint in the billing line (cc_entrypoint)",
        ),
        (
            "body.tools",
            "the ACP session starts in the default permission mode, whose "
            "EnterPlanMode and ExitPlanMode tools bypassPermissions does not "
            "offer; the other tool definitions are identical, in order",
        ),
        (
            "body.messages[*].content[*].content",
            "the adapter hands the SDK the session directory as an additional "
            "working directory, which adds an 'Environment update' reminder "
            "(Additional working directories added: /app) to the first tool "
            "result",
        ),
    ],
    "codex": [
        (
            "headers.originator, headers.user-agent",
            "codex-acp names the ACP client (benchflow, with its version); exec "
            "names itself (codex_exec)",
        ),
    ],
}
_CLAUDE_UA_ENTRYPOINT = re.compile(
    r"\(external, sdk-(?:ts|cli)(?:, agent-sdk/[\w.]+)?\)"
)
_CLAUDE_BILLING_ENTRYPOINT = re.compile(r"cc_entrypoint=sdk-(?:ts|cli);")
_ACP_ONLY_TOOLS = ["EnterPlanMode", "ExitPlanMode"]
_ACP_ENVIRONMENT_UPDATE = re.compile(
    r"\n\n<system-reminder>\n# Environment update\n"
    r" - Additional working directories added:\n(?:  - [^\n]*\n)+</system-reminder>"
)
_CODEX_CLIENT_NAME = re.compile(
    r"^(?:benchflow|codex_exec)/|\((?:benchflow|codex_exec); [\w.]+\)$"
)


def _normalize_claude(request: dict[str, Any], *, acp: bool) -> dict[str, Any]:
    """Remove the written Claude Code differences (and only those)."""
    headers = request["headers"]
    headers["user-agent"] = _CLAUDE_UA_ENTRYPOINT.sub(
        "(external, <entrypoint>)", headers.get("user-agent", "")
    )
    body = request["body"]
    system = body.get("system")
    if isinstance(system, list) and system and isinstance(system[0], dict):
        system[0]["text"] = _CLAUDE_BILLING_ENTRYPOINT.sub(
            "cc_entrypoint=<entrypoint>;", system[0].get("text", "")
        )
    if acp:
        tools = body.get("tools") or []
        assert [t["name"] for t in tools if t["name"] in _ACP_ONLY_TOOLS] == (
            _ACP_ONLY_TOOLS
        )
        body["tools"] = [t for t in tools if t["name"] not in _ACP_ONLY_TOOLS]
        for message in body.get("messages") or []:
            content = message.get("content")
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "tool_result" and isinstance(
                    block.get("content"), str
                ):
                    block["content"] = _ACP_ENVIRONMENT_UPDATE.sub("", block["content"])
    return request


def _tool_results(request: dict[str, Any]) -> list[str]:
    return [
        block["content"]
        for message in request["body"].get("messages") or []
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "tool_result" and isinstance(block.get("content"), str)
    ]


def _normalize_codex(request: dict[str, Any]) -> dict[str, Any]:
    """Remove the written Codex difference (the client's name and version)."""
    headers = request["headers"]
    if headers.get("originator") in ("benchflow", "codex_exec"):
        headers["originator"] = "<client>"
    headers["user-agent"] = _CODEX_CLIENT_NAME.sub(
        lambda m: "<client>/" if m.group(0).endswith("/") else "(<client>)",
        headers.get("user-agent", ""),
    )
    return request


_WIRE_VOLATILE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|"
    r"[0-9a-f]{64}|Chunk ID: [0-9a-f]+"
)


def _wire(job: h.CliRun, marker: str) -> list[dict[str, Any]]:
    """The proxy-entry requests of the task with ``marker``, ids normalized."""
    rows = []
    for path in sorted(job.wire_dir.glob("wire-*.jsonl")):
        rows += h.read_jsonl(path)
    picked = [r for r in rows if marker in json.dumps(r["body"])]
    return [json.loads(_WIRE_VOLATILE.sub("<id>", json.dumps(r))) for r in picked]


def _json_diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in a or key not in b:
                out.append(sub)
            else:
                out += _json_diff(a[key], b[key], sub)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in _json_diff(x, y, f"{path}[{i}]")
        ]
    return [] if a == b else [path]


def _drop_volatile_keys(request: dict[str, Any]) -> dict[str, Any]:
    body = request["body"]
    body.pop("metadata", None)  # Claude: user_id carries device and session ids
    body.pop("client_metadata", None)  # Codex: thread, turn and installation ids
    body.pop("prompt_cache_key", None)  # Codex: the thread id
    return request


@needs_sandbox
@pytest.mark.parametrize("variant", ["hello-pass", "wrong-answer"])
def test_claude_code_wire_parity(eval_job, native_eval_job, variant):
    """Both harnesses send the proxy the same requests, bar the written list."""
    marker = h.marker(EVAL_VARIANTS[variant].script)
    acp = [_drop_volatile_keys(r) for r in _wire(eval_job, marker)]
    native = [_drop_volatile_keys(r) for r in _wire(native_eval_job, marker)]
    # The session-title side call is a race on the ACP side; compare the loop.
    acp = [r for r in acp if r["body"].get("tools")]
    native = [r for r in native if r["body"].get("tools")]
    assert len(acp) == len(native) > 0
    # The adapter's reminder is really there, in a tool result.
    assert any(
        _ACP_ENVIRONMENT_UPDATE.search(text) for r in acp for text in _tool_results(r)
    )
    for a, n in zip(acp, native, strict=True):
        assert (
            _json_diff(_normalize_claude(a, acp=True), _normalize_claude(n, acp=False))
            == []
        )


@needs_sandbox
@pytest.mark.parametrize("variant", ["codex-hello-pass", "codex-wrong-answer"])
def test_codex_wire_parity(codex_acp_job, codex_native_job, variant):
    marker = h.marker(CODEX_VARIANTS[variant].script)
    acp = [_drop_volatile_keys(r) for r in _wire(codex_acp_job, marker)]
    native = [_drop_volatile_keys(r) for r in _wire(codex_native_job, marker)]
    assert len(acp) == len(native) > 0
    for a, n in zip(acp, native, strict=True):
        assert _json_diff(_normalize_codex(a), _normalize_codex(n)) == []


# ----- Codex: both harnesses on the same scenarios -------------------------

_CODEX_OUTCOMES = {
    "codex-hello-pass": ("completed", "scored", 1.0),
    "codex-wrong-answer": ("completed", "scored", 0.0),
    # No model response before the agent timeout (no tool call, no usage):
    # an integration failure, never a score.
    "codex-slow-model": ("integration_failed", "unscored", None),
}
# The one written outcome difference. When the Codex process dies mid-turn
# (its own tool call kills it here), codex-acp (1.13.1, 2.0.1) does not report
# it: the prompt runs to the agent timeout and the verifier scores the
# untouched workspace. The native harness reports the CLI's death at once as
# an agent error, as both harnesses do for Claude Code (the agent-crash golden).
_CODEX_CRASH = {
    "acp": ("timed_out", "scored", 0.0),
    "native": ("errored", "unscored", None),
}


def _codex_facts(trial: Path) -> dict[str, Any]:
    """The facts both Codex harnesses must share, ids of tool calls aside."""
    facts = _trial_facts(trial)
    result = facts["result"]
    trajectory = [
        {k: v for k, v in e.items() if k not in ("tool_call_id", "content")}
        for e in facts["acp_trajectory"]
    ]
    for event in trajectory:
        if event.get("type") == "agent_timeout":
            event["pending_tool_call_ids"] = len(event["pending_tool_call_ids"])
    timeout = result["agent_timeout"]
    return {
        "outcome": facts["outcome"],
        "rewards": result["rewards"],
        "n_tool_calls": result["n_tool_calls"],
        "n_prompts": result["n_prompts"],
        "error_category": result["error_category"],
        "agent_timeout": {
            **timeout,
            "pending_tool_call_ids": len(timeout["pending_tool_call_ids"]),
        }
        if timeout
        else None,
        "usage": {
            k: result["agent_result"][k]
            for k in (
                "n_input_tokens",
                "n_output_tokens",
                "total_tokens",
                "cost_usd",
                "usage_source",
            )
        },
        "trajectory_summary": result["trajectory_summary"],
        "llm_calls": [
            {**c, "tool_calls": [{**t, "id": None} for t in c["tool_calls"]]}
            for c in facts["llm_calls"]
            if c["kind"] == "main"
        ],
        "trajectory": trajectory,
    }


@needs_sandbox
@pytest.mark.parametrize("variant", sorted(CODEX_VARIANTS))
def test_codex_native_matches_codex_acp(codex_acp_job, codex_native_job, variant):
    """Outcome and trajectory parity for Codex (tool call ids aside: codex-acp
    exposes the model's call id, exec only its own item id)."""
    acp_trial = codex_acp_job.trial(variant)
    native_trial = codex_native_job.trial(variant)
    acp, native = _codex_facts(acp_trial), _codex_facts(native_trial)
    labels = ("execution", "assessment", "reward")
    if variant == "codex-agent-crash":
        assert acp["outcome"] == dict(zip(labels, _CODEX_CRASH["acp"], strict=True))
        assert native["outcome"] == dict(
            zip(labels, _CODEX_CRASH["native"], strict=True)
        )
        error = h.read_json(native_trial / "result.json")["error"]
        assert error.startswith("Native harness error (codex)"), error
        assert "exit code 137" in error
        # Up to the crash the two runs are the same: the one model call, its
        # usage, the trajectory before ACP's timeout event.
        assert native["usage"] == acp["usage"]
        assert native["trajectory"] == [
            e for e in acp["trajectory"] if e["type"] != "agent_timeout"
        ]
    else:
        assert acp["outcome"] == dict(
            zip(labels, _CODEX_OUTCOMES[variant], strict=True)
        )
        assert native == acp
    _check_native_contract(native_trial, "codex")
    # Usage and cost come from the proxy on both harnesses (the slow model
    # never answered, so there is none).
    source = "unavailable" if variant == "codex-slow-model" else "provider_response"
    assert native["usage"]["usage_source"] == source
    assert h.trial_document_problems(native_trial) == []


# ----- Codex under the egress allowlist ------------------------------------


def _egress_blocked(trial: Path) -> list[dict[str, Any]]:
    path = trial / "trajectory" / "egress_denylist.jsonl"
    return h.read_jsonl(path) if path.is_file() else []


@needs_sandbox
def test_native_codex_reaches_nothing_but_the_gateway(codex_allowlist_jobs):
    """With the agent allowlisted to example.com, native Codex still passes:
    every model call went through the proxy, and the egress proxy saw no
    attempt to reach api.openai.com, chatgpt.com or any other host."""
    native = codex_allowlist_jobs["native"].trial("codex-allowlist")
    result = h.read_json(native / "result.json")
    assert result["rewards"] == {"reward": 1.0}, result.get("error")
    assert result["agent_result"]["usage_source"] == "provider_response"
    _check_native_contract(native, "codex")
    assert _egress_blocked(native) == []


@needs_sandbox
def test_codex_acp_attempts_are_blocked_under_the_allowlist(codex_allowlist_jobs):
    """The same task on codex-acp: whatever it tries outside the gateway (its
    plugin marketplace sync opens chatgpt.com) is refused and logged."""
    acp = codex_allowlist_jobs["acp"].trial("codex-allowlist")
    result = h.read_json(acp / "result.json")
    assert result["rewards"] == {"reward": 1.0}, result.get("error")
    blocked = {str(r.get("url")) for r in _egress_blocked(acp)}
    assert not any("api.openai.com" in url for url in blocked), blocked
