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
import subprocess
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
# Every request the host fake provider received (after the proxy), when the
# proxy runs on the host.
UPSTREAM_WIRE = "host-fake-wire.jsonl"


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
    with h.host_fake_llm(
        det_root / "host-fake-llm.jsonl", det_root / UPSTREAM_WIRE
    ) as url:
        yield url


def _run(
    det_root: Path,
    name: str,
    subcommand,
    variants,
    fake_url,
    extra=(),
    route="proxy",
    model=None,
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
        model=model,
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


# Timing spans that say where the proxy ran (installed in the sandbox: on
# Daytona or under a network policy) or which harness connected (ACP only),
# and the cloud metadata block (run under open networks only). Backends and
# harnesses that share a golden file differ in them, so the golden leaves them
# out and _check_trial checks them. firewall_power runs wherever a rule of
# BenchFlow's holds the agent user (the metadata block, a network policy,
# Daytona's daemon guard). The warm pool's two spans belong to whichever trial
# of a job took a prepared sandbox, which the job's scheduling decides.
_UNSHARED_TIMING_KEYS = frozenset(
    {
        "litellm_install",
        "acp_connect",
        "cloud_metadata_block",
        "firewall_power",
        "warm_pool_wait",
        "warm_pool_idle",
    }
)


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
    view["result"]["timing_keys"] = [
        k for k in view["result"]["timing_keys"] if k not in _UNSHARED_TIMING_KEYS
    ]
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
    # A span with sub-keys (harden_before_verify) holds one number per step.
    seconds = [
        s for v in timing.values() for s in (v.values() if isinstance(v, dict) else [v])
    ]
    assert all(isinstance(s, int | float) and s >= 0 for s in seconds), timing
    # The spans the golden leaves out: present exactly when their step ran.
    config_path = trial_dir / "config.json"
    config = h.read_json(config_path) if config_path.is_file() else {}
    native_harness = config.get("harness_mode") == "native"
    if native_harness:
        assert "acp_connect" not in timing, timing
    in_sandbox_proxy = route == "proxy" and h.proxy_runs_in_sandbox(SANDBOX)
    assert ("litellm_install" in timing) is in_sandbox_proxy, timing
    assert ("proxy_health_wait" in timing) is (route == "proxy"), timing
    # The cloud metadata block ran exactly when the trial records it armed.
    record_path = trial_dir / "cloud-metadata.json"
    record = h.read_json(record_path) if record_path.is_file() else {}
    armed = record.get("state") in {"on", "unproven", "not_installed"}
    assert ("cloud_metadata_block" in timing) is armed, (timing, record)
    # The firewall-power check (PY8 S3) ran exactly when the trial records it.
    checked = (trial_dir / "firewall-power.json").is_file()
    assert ("firewall_power" in timing) is checked, timing
    # A prepared sandbox's idle time is recorded only with the wait for it.
    if "warm_pool_idle" in timing:
        assert "warm_pool_wait" in timing, timing

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
    # timeout-pass counts as passed (it has a reward); agent-crash is the error;
    # a verifier that wrote no reward is a verifier error, never a silent 0.
    assert counts == {
        "passed": 2,
        "failed": 1,
        "errored": 1,
        "verifier_errored": 1,
        "unscored": 0,
    }
    assert summary["error_categories"] == {"acp_error": 1, "timeout": 1}
    assert summary["verifier_error_categories"] == {"verifier_failure": 1}
    assert summary["unusable_rewards"] == 1
    # usage_summary totals the scored trials (a reward, no verifier error):
    # hello-pass, wrong-answer and the scored timeout; agent-crash and the
    # missing reward have no score.
    results = [h.read_json(t / "result.json") for t in eval_job.trial_dirs()]
    completed = [
        r for r in results if r["rewards"] is not None and not r["verifier_error"]
    ]
    assert len(completed) == 3
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
    # P6: config.json records the lenient mode and the model the provider's
    # responses named (the fake answers with the model it was asked for).
    config = h.read_json(trial / "config.json")
    assert config["reward_lenient"] is False
    assert config["reported_models_source"] == "provider_response"
    assert config["reported_models"] == [config["model"]]
    # Over ACP the connect is timed (the golden leaves it out: see above).
    assert "acp_connect" in facts["result"]["timing_keys"]
    # #1182: the scripted final reply is where reply-grading verifiers read it
    # (on Docker this folder is the sandbox's /logs/agent, by bind mount).
    assert (trial / "agent" / "output.txt").read_text() == "Created /app/hello.txt.\n"


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
    # P6, G-28: with no gateway, the model that answered is Claude Code's own
    # report, from the same session log.
    config = h.read_json(trial / "config.json")
    assert config["reported_models_source"] == "agent_session_log"
    assert config["reported_models"], config


@needs_sandbox
def test_a_verifier_exit_without_rewards_is_a_verifier_error(eval_job):
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
    assert result["verifier_error"].startswith("verifier wrote no usable reward: ")


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


# ---------------------------------------------------------------------------
# Scenario: the cloud metadata endpoints under the open network
# ---------------------------------------------------------------------------

# The template task's network is open. No test may connect to a real metadata
# server (on a cloud VM it is the VM's own), so the job runs under
# tests/_metadata_mock.py's watcher: as soon as a trial's main runs, each IPv4
# endpoint becomes a local address of main's namespace with a listener on
# port 80. The agent's one Bash call connects only to an endpoint that is
# local there (it waits up to 30 s for the mock), opens a TCP connection to
# port 80 and closes it (never a request), and prints ``probe <address>
# <refused|reached|timeout|unreachable|failed>``; an endpoint that is not
# local prints ``probe <address> not-mocked`` and is never connected to. The
# IPv6 endpoints cannot be mocked here (no IPv6 in these containers), so the
# probe leaves them out.
METADATA_VARIANTS = {
    "metadata-probe": h.TaskVariant("metadata-probe", "metadata-probe"),
    "metadata-opt-out": h.TaskVariant(
        "metadata-opt-out", "metadata-probe", allow_cloud_metadata=True
    ),
}


# The IPv4 endpoints, as the Rust bench 0.8.2 lists them (the lists in
# benchflow.sandbox.cloud_metadata must agree; test_cloud_metadata_block checks).
METADATA_V4 = (
    "169.254.169.254",
    "169.254.170.2",
    "169.254.170.23",
    "100.100.100.200",
    "168.63.129.16",
    "169.254.0.23",
)


def _metadata_trial_main(_cid: str, labels: dict[str, str]) -> bool:
    """A metadata variant's main: its Compose project is the trial's name."""
    project = labels.get("com.docker.compose.project", "")
    return labels.get("com.docker.compose.service") == "main" and any(
        project.startswith(f"{name}__") for name in METADATA_VARIANTS
    )


@pytest.fixture(scope="session")
def metadata_job(det_root, fake_url) -> h.CliRun:
    from tests._metadata_mock import MockWatcher

    if SANDBOX != "docker":
        # The mock joins main's network namespace on a local daemon only.
        pytest.skip("the metadata scenario mocks the endpoints on Docker only")
    with MockWatcher(_metadata_trial_main, "bf.test.det-metadata-mock=1") as mock:
        run = _run(
            det_root,
            "metadata",
            ["eval", "run"],
            list(METADATA_VARIANTS.values()),
            fake_url,
            extra=["--concurrency", "2", "--retry-attempts", "0"],
        )
    assert not mock.errors, mock.errors
    assert len(mock.mains) == len(METADATA_VARIANTS), mock.mains
    return run


def _probes(trial: Path) -> dict[str, str]:
    """What the agent's probe printed, per address."""
    for event in h.normalized_acp(trial):
        if event.get("type") == "tool_call" and event.get("tool_name") == "Bash":
            output = str(event.get("raw_output") or "")
            return dict(re.findall(r"(?m)^probe (\S+) (\S+)$", output))
    return {}


@needs_sandbox
def test_open_network_refuses_the_cloud_metadata_endpoints(metadata_job):
    """Guards PY8 S2: under the default open network nothing refused the
    cloud metadata endpoints, so an agent on a cloud VM's sandbox reached
    the VM's credentials server. The agent user's connects are now refused
    by BenchFlow's own rule (the record's proof is the rule's counter), and
    the trial scores as before."""
    trial = metadata_job.trial("metadata-probe")
    probes = _probes(trial)
    assert set(probes) == set(METADATA_V4), probes
    for address in METADATA_V4:
        assert probes[address] == "refused", probes
    record = h.read_json(trial / "cloud-metadata.json")
    assert record["state"] == "on", record
    assert "the rule counted" in record["why"], record
    result = h.read_json(trial / "result.json")
    assert result["rewards"]["reward"] == 1.0
    assert "cloud_metadata_block" in result["timing"]


@needs_sandbox
def test_cloud_metadata_opt_out_is_recorded_and_installs_nothing(metadata_job):
    """Guards PY8 S2's opt-out: metadata.x-allow-cloud-metadata = true leaves
    the endpoints as an open network always had them, and the trial folder
    says so."""
    trial = metadata_job.trial("metadata-opt-out")
    record = h.read_json(trial / "cloud-metadata.json")
    assert record["state"] == "off", record
    assert "x-allow-cloud-metadata" in record["why"], record
    # No rule: every connect reaches the mock on main's own loopback.
    assert _probes(trial) == dict.fromkeys(METADATA_V4, "reached"), _probes(trial)
    result = h.read_json(trial / "result.json")
    assert result["rewards"]["reward"] == 1.0
    assert "cloud_metadata_block" not in result["timing"]


def test_the_metadata_probe_connects_nowhere_without_the_mock():
    """The probe never connects to an endpoint that is not a local address
    of its namespace: with no mock it prints not-mocked for each of them.
    (The container has no network, so a connect it did make would read
    unreachable or failed, never not-mocked.)"""
    if not h._docker_available():
        pytest.skip("docker required")
    scripts = json.loads((h.FAKE_LLM_DIR / "scripts.json").read_text())
    command = scripts["metadata-probe"][0]["input"]["command"].replace(
        " > /app/hello.txt", " > /dev/null"
    )
    done = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "ubuntu:24.04",
            "bash",
            "-c",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr
    probes = dict(re.findall(r"(?m)^probe (\S+) (\S+)$", done.stdout))
    assert probes == dict.fromkeys(METADATA_V4, "not-mocked"), done.stdout


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
def codex_uncatalogued_job(det_root, fake_url) -> h.CliRun:
    return _run(
        det_root,
        "codex-uncatalogued",
        ["eval", "run"],
        [CODEX_VARIANTS["codex-slow-model"]],
        fake_url,
        route="codex-uncatalogued",
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
            "body.messages[*].content[*].content, body.messages[*] (system)",
            "the adapter hands the SDK the session directory as an additional "
            "working directory, which adds an 'Environment update' reminder "
            "(Additional working directories added: /app) to the first tool "
            "result; on a model Claude Code sends system messages for "
            "(claude-opus-5-5), a system message of its own after that tool "
            "result, which then holds the request's cache breakpoint",
        ),
        (
            "body.messages[*].content",
            "the ACP session starts in the default permission mode, whose "
            "deferred-tools reminder (Claude Code 2.1.316) also lists "
            "EnterPlanMode and ExitPlanMode, which bypassPermissions does not "
            "offer (in the first user message, or the environment system "
            "message)",
        ),
        (
            "body.messages[*].content, body.tools[Bash].description",
            "each permission mode's own guidance, in Claude Code 2.1.316's "
            "claude-opus-5-5 prompt (the claude-haiku-4-5 one has neither): "
            "bypassPermissions adds 'While bypass permissions mode is active: "
            "You can do much of your work through the Bash tool ...' to the "
            "environment message, and the default mode keeps 'IMPORTANT: Avoid "
            "using this tool to run cat, head, tail, sed, awk, or echo ...' in "
            "the Bash tool",
        ),
        (
            "body.tools[Bash].description, "
            "body.tools[Bash].input_schema.properties.run_in_background",
            "a background command's default time limit: 1,800,000 ms in the "
            "Agent SDK session, 600,000 ms in print mode",
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
# Claude Code 2.1.280's ACP session (default permission mode) also offered
# EnterPlanMode and ExitPlanMode as tools, which bypassPermissions does not;
# 2.1.316 defers them (a ToolSearch reminder in the first user message) in
# the ACP session and leaves them out in print mode.
_ACP_ONLY_TOOLS: list[str] = []
_ACP_ONLY_DEFERRED_TOOLS = ["EnterPlanMode", "ExitPlanMode"]
_DEFERRED_TOOLS_REMINDER = "The following deferred tools are now available"
# "(default 1800000ms / 30 minutes, ...)" in the description, "(default 1800000
# ms, ...)" in the parameter's; "in the background (default 1800000, max ...)"
# in claude-opus-5-5's description.
_BACKGROUND_DEFAULT = re.compile(r"default \d+ ?ms(?: / \d+ minutes)?")
_BACKGROUND_DEFAULT_BARE = re.compile(r"in the background \(default \d+,")
_ACP_ENVIRONMENT_UPDATE = re.compile(
    r"\n\n<system-reminder>\n# Environment update\n"
    r" - Additional working directories added:\n(?:  - [^\n]*\n)+</system-reminder>"
)
_ACP_ENVIRONMENT_UPDATE_MESSAGE = re.compile(
    r"# Environment update\n - Additional working directories added:"
    r"(?:\n  - [^\n]*)+"
)
_BYPASS_MODE_GUIDANCE = re.compile(
    r"While bypass permissions mode is active:\n\n"
    r"You can do much of your work through the Bash tool [^\n]*\n\n"
)
_DEFAULT_MODE_BASH_GUIDANCE = re.compile(
    r"\n- IMPORTANT: Avoid using this tool to run `cat`, [^\n]*"
)
_CODEX_CLIENT_NAME = re.compile(
    r"^(?:benchflow|codex_exec)/|\((?:benchflow|codex_exec); [\w.]+\)$"
)


def _without_acp_only_deferred_tools(text: str) -> str:
    if _DEFERRED_TOOLS_REMINDER not in text:
        return text
    for name in _ACP_ONLY_DEFERRED_TOOLS:
        assert f"\n{name}\n" in text, name
        text = text.replace(f"\n{name}\n", "\n")
    return text


def _is_environment_update_message(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "system"
        and isinstance(content, list)
        and len(content) == 1
        and isinstance(content[0].get("text"), str)
        and _ACP_ENVIRONMENT_UPDATE_MESSAGE.fullmatch(content[0]["text"]) is not None
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
    messages = body.get("messages") or []
    if acp:
        tools = body.get("tools") or []
        assert [t["name"] for t in tools if t["name"] in _ACP_ONLY_TOOLS] == (
            _ACP_ONLY_TOOLS
        )
        body["tools"] = [t for t in tools if t["name"] not in _ACP_ONLY_TOOLS]
        for index in reversed(range(len(messages))):
            if not _is_environment_update_message(messages[index]):
                continue
            # The cache breakpoint goes back to the message before it.
            (block,) = messages.pop(index)["content"]
            previous = messages[index - 1].get("content") if index else None
            if "cache_control" in block and isinstance(previous, list) and previous:
                previous[-1]["cache_control"] = block["cache_control"]

    def edit(text: str) -> str:
        if acp:
            return _without_acp_only_deferred_tools(text)
        return _BYPASS_MODE_GUIDANCE.sub("", text)

    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = edit(content)
        for block in content if isinstance(content, list) else []:
            if (
                acp
                and block.get("type") == "tool_result"
                and isinstance(block.get("content"), str)
            ):
                block["content"] = _ACP_ENVIRONMENT_UPDATE.sub("", block["content"])
            if isinstance(block.get("text"), str):
                block["text"] = edit(block["text"])
    for tool in body.get("tools") or []:
        if tool.get("name") != "Bash":
            continue
        description = _BACKGROUND_DEFAULT.sub(
            "default <n>", tool.get("description", "")
        )
        description = _BACKGROUND_DEFAULT_BARE.sub(
            "in the background (default <n>,", description
        )
        if acp:
            description = _DEFAULT_MODE_BASH_GUIDANCE.sub("", description)
        tool["description"] = description
        background = ((tool.get("input_schema") or {}).get("properties") or {}).get(
            "run_in_background"
        ) or {}
        if isinstance(background.get("description"), str):
            background["description"] = _BACKGROUND_DEFAULT.sub(
                "default <n>", background["description"]
            )
    return request


def _message_texts(body: dict[str, Any]) -> list[str]:
    """Every text in a Claude request's messages, string contents too."""
    texts: list[str] = []
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        for block in content if isinstance(content, list) else []:
            if isinstance(block.get("text"), str):
                texts.append(block["text"])
    return texts


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


# Codex's exec tool output carries its own timing ("Wall time: 0.1639
# seconds"): 0.0000 on an idle machine, more under pytest -n auto's load.
# Claude Code's Agent tool result names the sub-agent's random id and its
# run time.
_WIRE_VOLATILE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|"
    r"[0-9a-f]{64}|Chunk ID: [0-9a-f]+|Wall time: [0-9.]+ seconds|"
    r"agentId: [0-9a-f]+|to: '[0-9a-f]+'|duration_ms: \d+"
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
    # LiteLLM 1.103's proxy puts its drop_params setting into the request data
    # on some routes; it is a setting LiteLLM consumes, never sent upstream.
    body.pop("drop_params", None)
    return request


def _check_claude_wire_parity(
    acp_job: h.CliRun, native_job: h.CliRun, marker: str
) -> None:
    acp = [_drop_volatile_keys(r) for r in _wire(acp_job, marker)]
    native = [_drop_volatile_keys(r) for r in _wire(native_job, marker)]
    # The session-title side call is a race on the ACP side; compare the loop.
    acp = [r for r in acp if r["body"].get("tools")]
    native = [r for r in native if r["body"].get("tools")]
    assert len(acp) == len(native) > 0
    # The adapter's reminder is really there, in a tool result or a system
    # message of its own.
    assert any(
        _ACP_ENVIRONMENT_UPDATE.search(text) for r in acp for text in _tool_results(r)
    ) or any(
        _is_environment_update_message(m) for r in acp for m in r["body"]["messages"]
    )
    for a, n in zip(acp, native, strict=True):
        assert (
            _json_diff(_normalize_claude(a, acp=True), _normalize_claude(n, acp=False))
            == []
        )


def _check_codex_wire_parity(
    acp_job: h.CliRun, native_job: h.CliRun, marker: str
) -> None:
    acp = [_drop_volatile_keys(r) for r in _wire(acp_job, marker)]
    native = [_drop_volatile_keys(r) for r in _wire(native_job, marker)]
    # codex-acp's session-title side call (no tools) races the end of the
    # turn, as Claude's does (it reached the proxy for codex-wrong-answer in
    # a pytest -n auto run). Compare the loop.
    acp = [r for r in acp if r["body"].get("tools")]
    native = [r for r in native if r["body"].get("tools")]
    assert len(acp) == len(native) > 0
    for a, n in zip(acp, native, strict=True):
        assert _json_diff(_normalize_codex(a), _normalize_codex(n)) == []


@needs_sandbox
@pytest.mark.parametrize("variant", ["hello-pass", "wrong-answer"])
def test_claude_code_wire_parity(eval_job, native_eval_job, variant):
    """Both harnesses send the proxy the same requests, bar the written list."""
    marker = h.marker(EVAL_VARIANTS[variant].script)
    _check_claude_wire_parity(eval_job, native_eval_job, marker)


@needs_sandbox
@pytest.mark.parametrize("variant", ["codex-hello-pass", "codex-wrong-answer"])
def test_codex_wire_parity(codex_acp_job, codex_native_job, variant):
    marker = h.marker(CODEX_VARIANTS[variant].script)
    _check_codex_wire_parity(codex_acp_job, codex_native_job, marker)


# ----- a requested model and effort, and the sub-agent tool -----------------

# Each pair again, at a model and an effort set the way a run sets them
# (--model, --reasoning-effort), on both harnesses. A benchmark's lesson:
# codex-acp took no effort on BenchFlow 0.7.5, and its xhigh Codex
# rows ran with no apply_patch and no sub-agents.
# Claude Code 2.1.316 sends an effort only for a model it knows takes one, as
# output_config.effort with adaptive thinking (beta effort-2025-11-24).
# claude-opus-5-5 (the benchmark's Opus 5.5 rows, at max) takes one.
# claude-haiku-4-5, the tier's model, does not: the CLI ignores --effort, and
# over ACP the adapter offers no effort option, so BenchFlow refuses the run.
CLAUDE_EFFORT_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT = "max"
# The Claude Code task hands the file to a general-purpose sub-agent (its
# Agent tool, in the foreground), whose own requests carry the child marker.
CLAUDE_SUBAGENT_MARKER = h.marker("subagent-child")
# Codex sends it as reasoning.effort and in its turn metadata. The native
# harness used to refuse max.
CODEX_EFFORT = "max"
# Side calls: Claude Code's session title names the run's model; codex-acp's
# (TitleGenerator.ts, 2.0.1) runs one tool-less turn on a hard-wired model,
# whichever model the run has (see codex_config.codex_home_config).
CODEX_ACP_TITLE_MODEL = "gpt-5.6-luna"
CLAUDE_EFFORT_VARIANT = h.TaskVariant("claude-effort", "subagent")
CODEX_EFFORT_VARIANT = h.TaskVariant("codex-effort", "hello-pass")


def _effort_jobs(
    det_root, fake_url, variant: h.TaskVariant, route: str, model: str, effort: str
) -> dict[str, h.CliRun]:
    return {
        harness: _run(
            det_root,
            f"{variant.name}-{harness}",
            ["eval", "run"],
            [variant],
            fake_url,
            route=route,
            model=model,
            extra=[
                "--retry-attempts",
                "0",
                "--harness",
                harness,
                "--reasoning-effort",
                effort,
            ],
        )
        for harness in ("acp", "native")
    }


@pytest.fixture(scope="session")
def claude_effort_jobs(det_root, fake_url) -> dict[str, h.CliRun]:
    return _effort_jobs(
        det_root,
        fake_url,
        CLAUDE_EFFORT_VARIANT,
        "proxy",
        CLAUDE_EFFORT_MODEL,
        CLAUDE_EFFORT,
    )


@pytest.fixture(scope="session")
def codex_effort_jobs(det_root, fake_url) -> dict[str, h.CliRun]:
    return _effort_jobs(
        det_root, fake_url, CODEX_EFFORT_VARIANT, "codex", h.CODEX_MODEL, CODEX_EFFORT
    )


def _requests(job: h.CliRun) -> list[dict[str, Any]]:
    """Every request the job's harness sent the proxy, side calls too."""
    return [
        r for p in sorted(job.wire_dir.glob("wire-*.jsonl")) for r in h.read_jsonl(p)
    ]


def _check_effort_trial(
    job: h.CliRun, variant: h.TaskVariant, model: str, effort: str
) -> None:
    """The trial passed, its config records the setting, and BenchFlow's own
    record of every agent call (model-calls.jsonl) names the effort."""
    trial = job.trial(variant.name)
    result = h.read_json(trial / "result.json")
    assert result["rewards"] == {"reward": 1.0}, (
        result.get("error"),
        job.output[-4000:],
    )
    config = h.read_json(trial / "config.json")
    assert (config["model"], config["reasoning_effort"]) == (model, effort)
    calls = h.read_jsonl(trial / "trajectory" / "model-calls.jsonl")
    # The agent's turns: a session title it asks the same model for offers
    # no tools and carries the default effort, and is a side call.
    agent_calls = [c for c in calls if c["purpose"] == "agent" and c["tools"]]
    assert agent_calls and {c["effort"] for c in agent_calls} == {effort}, calls
    # The session's word and the calls' agree with the request.
    assert config["reasoning_effort_applied"] == effort
    assert result["effort"] == {
        "requested": effort,
        "applied": effort,
        "route": None,
        "sent": {effort: len(agent_calls)},
        "sent_source": "proxy",
        "matches": True,
    }, result["effort"]


@needs_sandbox
@pytest.mark.parametrize("harness", ["acp", "native"])
def test_claude_code_sends_the_requested_model_and_effort(
    claude_effort_jobs, det_root, harness
):
    """Every request names the run's model (BenchFlow's proxy alias for it;
    the session-title call too), every agent-loop request carries the
    effort, and the sub-agent tool (Agent) is offered, through the adapter
    and through the CLI alike. The sub-agent the task delegates to inherits
    both: its own requests name the same model and effort."""
    from benchflow.providers.litellm_config import safe_model_alias

    job = claude_effort_jobs[harness]
    _check_effort_trial(job, CLAUDE_EFFORT_VARIANT, CLAUDE_EFFORT_MODEL, CLAUDE_EFFORT)
    requests = _requests(job)
    assert {r["body"]["model"] for r in requests} == {
        safe_model_alias(CLAUDE_EFFORT_MODEL)
    }
    loop = [r for r in requests if r["body"].get("tools")]
    assert any(
        CLAUDE_SUBAGENT_MARKER in json.dumps(r["body"]["messages"][0]) for r in loop
    ), "the sub-agent sent no request"
    for request in loop:
        body = request["body"]
        assert body["output_config"] == {"effort": CLAUDE_EFFORT}, body
        assert body["thinking"]["type"] == "adaptive", body["thinking"]
        assert "effort-2025-11-24" in request["headers"]["anthropic-beta"].split(",")
        assert "Agent" in {tool["name"] for tool in body["tools"]}
    # The provider got the effort too, past the proxy.
    upstream = det_root / UPSTREAM_WIRE
    if upstream.is_file():
        received = [
            r["body"]
            for r in h.read_jsonl(upstream)
            if r["body"].get("model") == CLAUDE_EFFORT_MODEL and r["body"].get("tools")
        ]
        assert received
        assert all(
            b.get("output_config") == {"effort": CLAUDE_EFFORT} for b in received
        )


@needs_sandbox
@pytest.mark.parametrize("harness", ["acp", "native"])
def test_codex_sends_the_requested_model_and_effort(
    codex_effort_jobs, det_root, harness
):
    """Every agent-loop request names the run's model and carries the
    effort, in reasoning.effort and in Codex's own turn metadata; the one
    other model a request may name is codex-acp's session title's. The tool
    surface is the catalogued one: apply_patch, and the multi-agent tools,
    which Codex 0.159.3 defers behind tool_search for this model."""
    job = codex_effort_jobs[harness]
    _check_effort_trial(job, CODEX_EFFORT_VARIANT, h.CODEX_MODEL, CODEX_EFFORT)
    requests = _requests(job)
    loop = [r for r in requests if r["body"].get("tools")]
    assert loop
    for body in (r["body"] for r in requests):
        if body["model"] != h.CODEX_MODEL:
            assert (harness, body["model"], bool(body.get("tools"))) == (
                "acp",
                CODEX_ACP_TITLE_MODEL,
                False,
            )
    for request in loop:
        body = request["body"]
        assert body["reasoning"]["effort"] == CODEX_EFFORT, body["reasoning"]
        turn = json.loads(body["client_metadata"]["x-codex-turn-metadata"])
        assert (turn["model"], turn["reasoning_effort"]) == (
            h.CODEX_MODEL,
            CODEX_EFFORT,
        )
        tools = {t.get("name") or t.get("type"): t for t in body["tools"]}
        assert tools["apply_patch"]["type"] == "custom"
        assert "Multi-agent tools" in tools["tool_search"]["description"]
    # The trial's record of that surface: Codex's own, not its fallback.
    surface = h.read_json(job.trial(CODEX_EFFORT_VARIANT.name) / "result.json")[
        "tool_surface"
    ]
    assert "apply_patch" in surface["names"] and surface["fallback"] is False
    upstream = det_root / UPSTREAM_WIRE
    if upstream.is_file():
        threads = {r["body"].get("prompt_cache_key") for r in loop}
        received = [
            r["body"]
            for r in h.read_jsonl(upstream)
            if r["body"].get("prompt_cache_key") in threads
        ]
        assert received
        assert {b["reasoning"]["effort"] for b in received} == {CODEX_EFFORT}


@needs_sandbox
def test_claude_code_wire_parity_at_the_requested_model_and_effort(
    claude_effort_jobs,
):
    acp_job, native_job = claude_effort_jobs["acp"], claude_effort_jobs["native"]
    marker = h.marker(CLAUDE_EFFORT_VARIANT.script)
    # This model's prompt really carries each permission mode's guidance, on
    # its own side only (WIRE_EXPECTED_DIFFERENCES).
    for job, ours, theirs in (
        (acp_job, _DEFAULT_MODE_BASH_GUIDANCE, _BYPASS_MODE_GUIDANCE),
        (native_job, _BYPASS_MODE_GUIDANCE, _DEFAULT_MODE_BASH_GUIDANCE),
    ):
        body = next(r["body"] for r in _wire(job, marker) if r["body"].get("tools"))
        bash = next(t["description"] for t in body["tools"] if t["name"] == "Bash")
        texts = [bash, *_message_texts(body)]
        found = [any(p.search(t) for t in texts) for p in (ours, theirs)]
        assert found == [True, False], found
    _check_claude_wire_parity(acp_job, native_job, marker)


@needs_sandbox
def test_codex_wire_parity_at_the_requested_model_and_effort(codex_effort_jobs):
    _check_codex_wire_parity(
        codex_effort_jobs["acp"],
        codex_effort_jobs["native"],
        h.marker(CODEX_EFFORT_VARIANT.script),
    )


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


@needs_sandbox
def test_codex_warning_is_a_notice_not_a_reply(codex_uncatalogued_job):
    """On a model its catalog lacks, Codex warns as soon as the turn starts.

    codex-acp sends that warning as a session notice, recorded apart from the
    agent's words (it used to arrive as agent text and read as the agent's
    work). The client stops the trial there rather than run the model on
    Codex's fallback tools, and the trial is an unscored
    agent_model_catalog integration failure: the setup's, not an ACP error of
    the agent's that a retry repeats, and never a scored 0.
    """
    trial = codex_uncatalogued_job.trial("codex-slow-model")
    facts = _trial_facts(trial)
    assert facts["outcome"] == {
        "execution": "integration_failed",
        "assessment": "unscored",
        "reward": None,
    }
    events = facts["acp_trajectory"]
    assert [e["type"] for e in events if e["type"] != "notice"] == ["user_message"]
    notices = [e for e in events if e["type"] == "notice"]
    assert any(
        n["severity"] == "warning"
        and f"Model metadata for `{h.CODEX_UNCATALOGUED_MODEL}` not found" in n["title"]
        for n in notices
    ), notices
    # The run's own records say the same as bf.load_trial: the job summary
    # counts an integration failure, not a scored 0.
    result = h.read_json(trial / "result.json")
    assert result["error_category"] == "agent_integration"
    assert result["error"].startswith(
        "agent integration failure [agent_model_catalog]: codex-acp ran "
        f"{h.CODEX_UNCATALOGUED_MODEL} on fallback model metadata: Model metadata "
        f"for `{h.CODEX_UNCATALOGUED_MODEL}` not found"
    ), result["error"]
    assert result["integration_failure_info"]["cause"] == "agent_model_catalog"
    assert result["agent_timeout_info"] is None
    summary = h.read_json(trial.parent / "summary.json")
    assert summary["integration_failures"]["total"] == 1, summary


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


# ----- the in-sandbox proxy's install ----------------------------------------


def _key_paths(value: Any, prefix: str = "") -> list[str]:
    """Every key path in a JSON value, list positions folded to ``[]``."""
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths.update(_key_paths(child, path))
    elif isinstance(value, list):
        for child in value:
            paths.update(_key_paths(child, f"{prefix}[]"))
    return sorted(paths)


def _usage_record(trial: Path) -> dict[str, Any]:
    """The first proxy call's record: its schema and its token fields.

    The request body is the agent's (Codex's prompt and tools) and the
    response output is the fake's script, so neither is part of the record's
    schema; everything the proxy and its callback write is.
    """
    record = h.read_jsonl(trial / "trajectory" / "llm_trajectory.jsonl")[0]
    shape = json.loads(json.dumps(record))
    shape["request"].pop("body", None)
    shape["response"]["body"].pop("output", None)
    agent = h.read_json(trial / "result.json")["agent_result"]
    return {
        "record_keys": _key_paths(shape),
        "usage": record["response"]["body"]["usage"],
        "agent_result": {
            key: agent.get(key)
            for key in (
                "n_input_tokens",
                "n_output_tokens",
                "n_cache_tokens",
                "cost_usd",
                "usage_source",
            )
        },
    }


# Usage-model keys that only the OpenAI SDK's version decides. The golden was
# recorded from the fresh resolve, which installed openai 2.54.0 on 2026-10-16;
# the locked set has 2.38.0, whose PromptTokensDetails lacks this key. It was
# null there, and BenchFlow never reads it (cache writes come from
# cache_creation_input_tokens, trajectories/types.py), so it is left out of
# the comparison on both sides; any other key or token change still fails.
_SDK_ONLY_USAGE_KEYS = (("prompt_tokens_details", "cache_write_tokens"),)

# LiteLLM 1.103 adds the call's cost, priced by its own map, to the usage
# block the proxy returns. BenchFlow prices calls itself and never reads it;
# agent_result's cost_usd still pins the cost, so it is left out the same way.
_LITELLM_ONLY_USAGE_KEYS = ("cost",)


def _without_sdk_only_keys(shape: dict[str, Any]) -> dict[str, Any]:
    shape = json.loads(json.dumps(shape))
    for parent, key in _SDK_ONLY_USAGE_KEYS:
        (shape["usage"].get(parent) or {}).pop(key, None)
        path = f"response.body.usage.{parent}.{key}"
        shape["record_keys"] = [k for k in shape["record_keys"] if k != path]
    for key in _LITELLM_ONLY_USAGE_KEYS:
        shape["usage"].pop(key, None)
        path = f"response.body.usage.{key}"
        shape["record_keys"] = [k for k in shape["record_keys"] if k != path]
    return shape


@needs_sandbox
def test_in_sandbox_proxy_usage_record_is_unchanged(codex_allowlist_jobs):
    """The in-sandbox proxy installs the locked, hashed set instead of a
    fresh resolve. One fixed request through the fake gives a record
    with the same schema and token fields as before: the golden was recorded
    from the fresh-resolve install on integ/py-0.8 2ea2ad00 and is not
    regenerated from a later tree."""
    trial = codex_allowlist_jobs["acp"].trial("codex-allowlist")
    actual = _usage_record(trial)
    golden = h.Golden("in-sandbox-proxy-usage")
    if os.environ.get(h.UPDATE_GOLDEN_ENV) == "1":
        return
    expected = h.read_json(golden.path)
    assert _without_sdk_only_keys(actual) == _without_sdk_only_keys(expected), (
        json.dumps(actual, indent=2, sort_keys=True)
    )


@needs_sandbox
def test_in_sandbox_proxy_records_its_locked_install(codex_allowlist_jobs):
    from benchflow.providers import proxy_requirements as reqs

    trial = codex_allowlist_jobs["acp"].trial("codex-allowlist")
    assert h.read_json(trial / "config.json")["litellm_proxy"] == {
        "location": "sandbox",
        "requirements": "litellm-proxy.txt",
        "requirements_sha256": reqs.requirements_sha256(reqs.PROXY_GROUP),
        "extra_requirements": [],
    }


# ----- a model refusal, and a trial budget (0.10.0rc2) ---------------------

# The model looks around once, then refuses: Anthropic's stop_reason refusal
# to Claude Code on the proxy route, OpenAI's content filter to Codex on its
# Responses route (an incomplete answer whose reason is content_filter), and a
# Chat Completions content_filter to Claude Code through the proxy's Messages
# bridge (the vllm route). All run under the default retry policy: a refusal
# must not be retried away.
REFUSAL_VARIANT = h.TaskVariant("refusal", "refuse-after-look")
REFUSAL_ROUTES = ("proxy", "codex", "vllm")
# Eight steps before the one that writes hello.txt: four turns stop the agent
# long before it.
TURN_BUDGET_VARIANT = h.TaskVariant("turn-budget", "many-steps")
TURN_BUDGET = 4
# The agent hands the look around to a sub-agent first; each main or sub-agent
# call is 1,050 tokens (the fake's usage), so 3,000 tokens are spent by the
# sub-agent's second call, and its third is refused before hello.txt exists.
TOKEN_BUDGET_VARIANT = h.TaskVariant("token-budget", "subagent-explore")
TOKEN_BUDGET = 3000
SUBAGENT_CHILD_MARKER = h.marker("subagent-explore-child")


@pytest.fixture(scope="session")
def refusal_jobs(det_root, fake_url) -> dict[str, h.CliRun]:
    return {
        route: _run(
            det_root,
            f"refusal-{route}",
            ["eval", "run"],
            [REFUSAL_VARIANT],
            fake_url,
            route=route,
        )
        for route in REFUSAL_ROUTES
    }


@pytest.fixture(scope="session")
def budget_jobs(det_root, fake_url) -> dict[str, h.CliRun]:
    return {
        "turns": _run(
            det_root,
            "budget-turns",
            ["eval", "run"],
            [TURN_BUDGET_VARIANT],
            fake_url,
            extra=["--turn-budget", str(TURN_BUDGET)],
        ),
        "tokens": _run(
            det_root,
            "budget-tokens",
            ["eval", "run"],
            [TOKEN_BUDGET_VARIANT],
            fake_url,
            extra=["--token-budget", str(TOKEN_BUDGET)],
        ),
    }


def _proxy_rows(trial: Path) -> list[dict[str, Any]]:
    return h.read_jsonl(trial / "trajectory" / "llm_trajectory.jsonl")


@needs_sandbox
@pytest.mark.parametrize("route", REFUSAL_ROUTES)
def test_a_model_refusal_is_its_own_ending_counted_beside_the_score(
    refusal_jobs, route
):
    """The trial's result names the refused call and its reason, it counts 0
    (a scored failure: the agent never wrote hello.txt), it ran once under a
    policy that retries, and summary.json, the run card and the end-of-run
    lines count it as a model refusal with its rate."""
    job = refusal_jobs[route]
    # job.trial asserts one trial folder: the refusal was never retried.
    trial = job.trial(REFUSAL_VARIANT.name)
    result = h.read_json(trial / "result.json")
    assert result["agent_outcome"]["status"] == "refusal", result["agent_outcome"]
    ending = result["model_refusal"]
    assert ending is not None, job.output[-4000:]
    calls = h.read_jsonl(trial / "trajectory" / "model-calls.jsonl")
    agent_calls = [c for c in calls if c["purpose"] == "agent" and c["tools"]]
    assert ending["call"]["index"] == agent_calls[-1]["index"], (ending, calls)
    assert ending["call"]["source"] == "proxy"
    # LiteLLM records Anthropic's refusal, as OpenAI's filter, content_filter.
    assert ending["call"]["finish_reason"] == "content_filter", ending
    assert ending["refused_calls"] >= 1
    # Graded on what the agent left (hello.txt never written): a failure,
    # whatever the agent reported after the refusal.
    assert result["rewards"] == {"reward": 0.0}, result.get("error")
    assert _verifier_ran(trial)
    if route == "proxy":
        # Claude Code ends its turn with ACP's refusal stop reason.
        assert (ending["by"], ending["stop_reason"]) == ("stop_reason", "refusal")
        assert result["error"] is None
    (job_dir,) = job.job_dirs()
    summary = h.read_json(job_dir / "summary.json")
    assert summary["model_refusals"] == 1, summary
    assert summary["model_refusal_rate"] == 1.0
    assert summary["scoring"]["model_refusals"] == 1
    assert summary["scoring"]["outcomes"]["failed"] == 1
    assert summary["guard_refusals"] == 0
    card = h.read_json(job_dir / "run-card.json")
    assert card["after_run"]["scoring"]["model_refusals"] == 1
    output = " ".join(job.output.split())
    assert "1 of 1 trials (100.0%) ended on a model refusal" in output, output[-4000:]
    assert "Refusals: 1 of 1 trials" in output
    assert "1 ended on a model refusal or a content filter" in output


def _check_budget_stop(job: h.CliRun, trial: Path, measure: str, limit: int) -> dict:
    result = h.read_json(trial / "result.json")
    budget = result["trial_budget"]
    assert budget is not None, job.output[-4000:]
    assert budget["budget"][measure] == limit
    assert budget["source"] == {measure: "run"}
    assert budget["stopped_by"] == measure, (budget, job.output[-4000:])
    assert budget["stopped_at"][measure] >= limit
    assert budget["refused_calls"] >= 1
    assert result["error_category"] == "trial_budget", result["error"]
    outcome = result["agent_outcome"]
    assert (outcome["ended_by"], outcome["harness_cause"]) == (
        "harness",
        "trial_budget",
    )
    # Stopped before hello.txt: graded like a timeout, and it counts 0.
    assert result["rewards"] == {"reward": 0.0}
    assert _verifier_ran(trial)
    (job_dir,) = job.job_dirs()
    summary = h.read_json(job_dir / "summary.json")
    assert (summary["budget_stops"], summary["budget_stop_rate"]) == (1, 1.0)
    checks = summary["run_checks"]["trial_budget"]
    assert checks["stopped_by"][measure] == 1
    card = h.read_json(job_dir / "run-card.json")
    assert card["trial_budget"][measure] == {"min": limit, "max": limit}
    assert card["after_run"]["trial_budget"]["trials"] == 1
    output = " ".join(job.output.split())
    unit = "token" if measure == "tokens" else "turn"
    assert f"Per trial: budget {limit:,} {unit}s (the run's)" in output, output[-4000:]
    assert "1 of 1 trials stopped at it" in output
    assert "1 stopped at their trial budget" in output
    return result


@needs_sandbox
def test_a_turn_budget_stops_a_real_agent_mid_trial(budget_jobs):
    """Four model calls allowed: the proxy forwards four and refuses the
    fifth, the agent is stopped eight steps short of hello.txt, and the
    trial's record says so."""
    job = budget_jobs["turns"]
    trial = job.trial(TURN_BUDGET_VARIANT.name)
    result = _check_budget_stop(job, trial, "turns", TURN_BUDGET)
    assert result["trial_budget"]["used"]["turns"] == TURN_BUDGET
    assert len(_proxy_rows(trial)) == TURN_BUDGET
    assert result["trial_budget"]["stopped_at"]["turns"] == TURN_BUDGET


@needs_sandbox
def test_a_token_budget_counts_the_subagents_calls_and_stops_the_agent(budget_jobs):
    """The sub-agent's calls go through the proxy and count: they spend the
    budget, its next call is refused, and the record gives the sub-agents'
    share from Claude Code's own log."""
    job = budget_jobs["tokens"]
    trial = job.trial(TOKEN_BUDGET_VARIANT.name)
    result = _check_budget_stop(job, trial, "tokens", TOKEN_BUDGET)
    rows = _proxy_rows(trial)
    child = [
        row
        for row in rows
        if SUBAGENT_CHILD_MARKER
        in json.dumps(((row.get("request") or {}).get("body") or {}).get("messages"))
    ]
    assert len(child) >= 1, rows
    used = result["trial_budget"]["used"]
    assert used["tokens"] == result["agent_result"]["total_tokens"] >= TOKEN_BUDGET
    subagents = result["trial_budget"]["subagents"]
    assert subagents is not None and subagents["source"] == "claude-code-session-log"
    assert subagents["turns"] >= 1 and subagents["tokens"] > 0, subagents
