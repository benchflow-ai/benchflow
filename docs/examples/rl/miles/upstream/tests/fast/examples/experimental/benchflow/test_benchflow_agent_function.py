"""Offline tests for the BenchFlow agent function (no BenchFlow, no sandbox, no GPU).

A scripted BenchFlow environment server (httpx.MockTransport) answers /run and
/abort. Covered: the /run request (session URL, prompt, sampling settings and
metadata forwarded unchanged), the verdict mapping onto sample metadata, the
failure contract (a discarded episode raises InfraAbort, or returns
``reward: None`` on a Miles without it; a timeout or server error scores 0;
a refused request raises), the abort hook, the reward hook and the per-step
metrics, and the launcher preflight.

    pytest tests/fast/examples/experimental/benchflow -q
"""

import asyncio
import json
from types import SimpleNamespace

import benchflow_agent_function as baf
import httpx
import launch_common
import pytest

BASE_URL = "http://10.0.0.5:30001/sessions/abc"
METADATA = {"instance_id": "sql-000004", "kind": "sql", "max_seq_len": 16384}

SCORED = {
    "reward": 1.0,
    "dropped": False,
    "exit_status": "Submitted",
    "flagged": False,
    "detail": None,
    "eval_report": {"task_id": "sql-000004", "ended": "submitted", "rewards": {"reward": 1.0}},
    "agent_metrics": {"turns": 3, "tool_calls": 2, "clipped_replies": 0, "eval_time": 13.1},
}


@pytest.fixture(autouse=True)
def server(monkeypatch):
    """A scripted environment server; ``server.reply`` is its next /run answer."""
    state = SimpleNamespace(requests=[], reply=httpx.Response(200, json=SCORED))

    def handler(request):
        state.requests.append(request)
        reply = state.reply(request) if callable(state.reply) else state.reply
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(baf, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setenv("BENCHFLOW_ENV_URL", "http://env.test:12100/")
    monkeypatch.delenv("BENCHFLOW_ENV_TOKEN_FILE", raising=False)
    return state


class FakeInfraAbort(Exception):
    def __init__(self, exit_status, message=None):
        super().__init__(message or exit_status)
        self.exit_status = exit_status


def run(**kwargs):
    return asyncio.run(
        baf.run(
            base_url=BASE_URL,
            prompt=[{"role": "user", "content": "How many orders?"}],
            request_kwargs={"temperature": 1.0, "max_tokens": 1024},
            metadata=dict(METADATA),
            **kwargs,
        )
    )


def test_run_posts_the_session_url_and_sample_unchanged(server):
    out = run()
    (request,) = server.requests
    assert str(request.url) == "http://env.test:12100/run"
    body = json.loads(request.content)
    assert body == {
        "session_url": f"{BASE_URL}/v1",
        "prompt": [{"role": "user", "content": "How many orders?"}],
        "request_kwargs": {"temperature": 1.0, "max_tokens": 1024},
        "metadata": METADATA,
    }
    assert "authorization" not in request.headers
    assert out == {
        "reward": 1.0,
        "exit_status": "Submitted",
        "eval_report": {**SCORED["eval_report"], "flagged": False},
        "agent_metrics": SCORED["agent_metrics"],
    }


def test_the_token_file_is_sent_as_a_bearer_token(server, monkeypatch, tmp_path):
    token = tmp_path / "token"
    token.write_text("s3cret\n")
    monkeypatch.setenv("BENCHFLOW_ENV_TOKEN_FILE", str(token))
    run()
    assert server.requests[0].headers["authorization"] == "Bearer s3cret"


def test_a_flagged_exploit_keeps_its_zero_and_the_flag(server):
    flagged = {
        **SCORED,
        "reward": 0.0,
        "exit_status": "IntegrityViolation",
        "flagged": True,
    }
    server.reply = httpx.Response(200, json=flagged)
    out = run()
    assert out["reward"] == 0.0 and out["exit_status"] == "IntegrityViolation"
    assert out["eval_report"]["flagged"] is True


@pytest.mark.parametrize(
    "exit_status", ["SandboxUnavailable", "ModelEndpointFailed", "VerifierCrashCleanRun", "Aborted"]
)
def test_a_dropped_episode_raises_infra_abort(server, monkeypatch, exit_status):
    monkeypatch.setattr(baf, "InfraAbort", FakeInfraAbort)
    server.reply = httpx.Response(
        200, json={"reward": None, "dropped": True, "exit_status": exit_status, "detail": "quota exceeded"}
    )
    with pytest.raises(FakeInfraAbort) as caught:
        run()
    assert caught.value.exit_status == exit_status
    assert "quota exceeded" in str(caught.value)


def test_without_infra_abort_a_dropped_episode_has_no_reward(server, monkeypatch):
    """Miles before #2801: the missing-reward filter drops the group."""
    monkeypatch.setattr(baf, "InfraAbort", None)
    server.reply = httpx.Response(
        200, json={"reward": None, "dropped": True, "exit_status": "SandboxUnavailable", "detail": "x"}
    )
    out = run()
    assert out["reward"] is None and out["exit_status"] == "SandboxUnavailable"
    sample = SimpleNamespace(metadata=out)
    assert asyncio.run(baf.reward_func(None, sample)) is None


def test_an_unreachable_server_is_discarded(server, monkeypatch):
    monkeypatch.setattr(baf, "InfraAbort", FakeInfraAbort)
    server.reply = httpx.ConnectError("connection refused")
    with pytest.raises(FakeInfraAbort) as caught:
        run()
    assert caught.value.exit_status == "ServerUnreachable"


def test_a_server_error_scores_zero(server):
    server.reply = httpx.Response(500, text="boom")
    out = run()
    assert out["reward"] == 0.0 and out["exit_status"] == "AgentError"


def test_a_refused_request_raises(server):
    server.reply = httpx.Response(400, json={"error": "unknown task 'sql-000004'"})
    with pytest.raises(RuntimeError, match="unknown task"):
        run()


def test_the_backstop_timeout_scores_zero(server, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_EPISODE_TIMEOUT", "0.05")

    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json=SCORED)

    monkeypatch.setattr(baf, "_client", httpx.AsyncClient(transport=httpx.MockTransport(slow)))
    out = run()
    assert out["reward"] == 0.0 and out["exit_status"] == "TimeLimitExceeded"


def test_abort_asks_the_server_to_cancel_its_episodes(server):
    server.reply = httpx.Response(200, json={"cancelled": 3})
    asyncio.run(baf.abort(SimpleNamespace()))
    (request,) = server.requests
    assert request.method == "POST" and str(request.url) == "http://env.test:12100/abort"


def test_abort_never_raises(server):
    server.reply = httpx.ConnectError("down")
    asyncio.run(baf.abort(SimpleNamespace()))


def test_reward_hook_reads_the_episodes_reward():
    samples = [SimpleNamespace(metadata={"reward": 1.0}), SimpleNamespace(metadata={"reward": 0.0})]
    assert asyncio.run(baf.reward_func(None, samples)) == [1.0, 0.0]


def test_step_metrics_count_exit_statuses_and_clipped_replies():
    def sample(status, turns, clipped, ended, flagged=False):
        return SimpleNamespace(
            metadata={
                "exit_status": status,
                "agent_metrics": {"turns": turns, "clipped_replies": clipped, "tool_calls": turns - 1},
                "eval_report": {"ended": ended, "flagged": flagged},
            }
        )

    metrics = baf.benchflow_metrics(
        [
            sample("Submitted", 3, 0, "submitted"),
            sample("SequenceLengthLimitExceeded", 2, 1, "response_truncated"),
            sample("SequenceLengthLimitExceeded", 4, 0, "context_exhausted"),
            sample("IntegrityViolation", 1, 0, "submitted", flagged=True),
        ]
    )
    assert metrics["benchflow/exit_status/Submitted"] == 0.25
    assert metrics["benchflow/exit_status/SequenceLengthLimitExceeded"] == 0.5
    assert metrics["benchflow/clipped_reply_ratio"] == 1 / 10
    assert metrics["benchflow/context_exhausted_ratio"] == 0.25
    assert metrics["benchflow/flagged"] == 1.0
    assert metrics["benchflow/turns_mean"] == 2.5


# --- launcher ---------------------------------------------------------------


def test_train_args_wire_the_agent_function_reward_and_filters():
    args = launch_common.agentic_train_args(tito_model="qwen3", session_server_workers=4)
    assert "--custom-agent-function-path benchflow_agent_function.run" in args
    assert "--custom-rm-path benchflow_agent_function.reward_func" in args
    assert "--rollout-function-path benchflow_rollout.RolloutFn" in args
    assert "apply_reward_nonzero_std_filter" in args and "--over-sampling-batch-size 1" in args
    assert "--tito-model qwen3" in args and "--use-session-server" in args
    plain = launch_common.agentic_train_args(
        tito_model="qwen3", session_server_workers=4, drop_constant_reward_groups=False
    )
    assert "dynamic-sampling-filter" not in plain


def test_worker_env_forwards_the_token_path_not_the_token(tmp_path):
    token = tmp_path / "token"
    token.write_text("s3cret")
    env = launch_common.benchflow_env_vars(env_url="http://127.0.0.1:12100/", token_file=str(token))
    assert env == {"BENCHFLOW_ENV_URL": "http://127.0.0.1:12100", "BENCHFLOW_ENV_TOKEN_FILE": str(token)}
    assert "s3cret" not in json.dumps(env)


def test_preflight_refuses_tasks_the_server_does_not_serve(monkeypatch, tmp_path):
    data = tmp_path / "train.jsonl"
    rows = [{"prompt": [], "metadata": {"instance_id": task}} for task in ("a", "b", "c")]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def fake_get(url, headers=None, timeout=None):
        payload = {"tasks": ["a", "b"]} if url.endswith("/tasks") else {"tasks": 2, "sandbox": "daytona"}
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(launch_common.httpx, "get", fake_get)
    with pytest.raises(RuntimeError, match=r"1 task\(s\).*\['c'\]"):
        launch_common.preflight(env_url="http://env.test:12100", prompt_data=str(data))
    data.write_text("".join(json.dumps(row) + "\n" for row in rows[:2]))
    assert launch_common.preflight(env_url="http://env.test:12100", prompt_data=str(data))["tasks"] == 2


def test_preflight_says_how_to_start_a_missing_server(monkeypatch, tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("")

    def refuse(url, headers=None, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(launch_common.httpx, "get", refuse)
    with pytest.raises(RuntimeError, match="benchflow.integrations.miles serve"):
        launch_common.preflight(env_url="http://env.test:12100", prompt_data=str(data))
