"""The Miles environment: one bash/submit episode against a TITO session.

A scripted session server (httpx MockTransport) stands in for Miles, and a
scripted runtime for the BenchFlow sandbox. The tests check what Miles relies
on: every request extends the previous one with the assistant message exactly
as the server returned it; infrastructure failures are dropped with a named
cause and never become a 0; failures the policy could have caused score 0 with
a named exit status; the sandbox is always released.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from benchflow.integrations.miles import (
    HARNESS_MESSAGE,
    EpisodeRequest,
    EpisodeSettings,
    dataset_rows,
    run_episode,
)
from benchflow.integrations.miles import episode as episode_module
from benchflow.integrations.miles.server import create_app, serve
from benchflow.integrations.trl import BashHarnessConfig, BenchFlowSpec

REPO = Path(__file__).resolve().parents[1]
SESSION = "http://sessions.test/sessions/abc/v1"


def _make_task(parent: Path, name: str) -> Path:
    task = parent / name
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        f'version = "1.0"\n\n[task]\nname = "benchflow/{name}"\n\n'
        "[verifier]\ntimeout_sec = 60\n\n[agent]\ntimeout_sec = 60\n\n[environment]\n"
    )
    (task / "instruction.md").write_text("How many rows are there?")
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests").mkdir()
    test_sh = task / "tests" / "test.sh"
    test_sh.write_text("#!/usr/bin/env bash\necho 1 >/logs/verifier/reward.txt\n")
    test_sh.chmod(0o755)
    return task


@dataclass
class _Bash:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class _Verified:
    reward: float | None
    rollout_dir: Path
    error: str | None = None
    verifier_error: str | None = None
    rewards: dict | None = None
    integrity: Any = None


class _Runtime:
    """A TaskRuntime stand-in whose start, commands and verdict are scripted."""

    start_error: ClassVar[Exception | None] = None
    verify_outcome: ClassVar[Any] = None  # an exception, or _Verified fields
    bash_gate: ClassVar[asyncio.Event | None] = None
    created: ClassVar[list[_Runtime]] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.rollout_dir = Path(config.jobs_dir) / f"rollout-{len(_Runtime.created)}"
        self.commands: list[str] = []
        self.verified = 0
        self.closed = False

    @classmethod
    async def create(cls, config: Any) -> _Runtime:
        if cls.start_error is not None:
            raise cls.start_error
        runtime = cls(config)
        cls.created.append(runtime)
        return runtime

    async def bash(self, command: str, *, timeout_sec: int = 30) -> _Bash:
        self.commands.append(command)
        if _Runtime.bash_gate is not None:
            await _Runtime.bash_gate.wait()
        return _Bash(stdout=f"ran {command}\n")

    async def verify(self) -> _Verified:
        self.verified += 1
        outcome = _Runtime.verify_outcome
        if isinstance(outcome, Exception):
            raise outcome
        fields = {"reward": 1.0, "rewards": {"reward": 1.0}}
        fields.update(outcome or {})
        return _Verified(rollout_dir=self.rollout_dir, **fields)

    async def close(self) -> None:
        self.closed = True


def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "index": 0,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _reply(
    content: str = "",
    calls: list[dict[str, Any]] | None = None,
    *,
    reasoning: str | None = None,
    finish: str = "tool_calls",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return {
        "id": "chatcmpl-1",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish
                if calls
                else ("stop" if finish == "tool_calls" else finish),
                "meta_info": {"output_token_logprobs": [[-0.1, 7, None]]},
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }


class _Sessions:
    """A session server that answers from a script and records every request."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sessions/abc/v1/chat/completions"
        self.requests.append(json.loads(request.content))
        step = self.script.pop(0) if self.script else _reply("done")
        if isinstance(step, Exception):
            raise step
        if isinstance(step, httpx.Response):
            return step
        return httpx.Response(200, json=step)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


@pytest.fixture
def tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _Runtime.start_error = None
    _Runtime.verify_outcome = None
    _Runtime.bash_gate = None
    _Runtime.created = []
    monkeypatch.setattr(episode_module.TaskRuntime, "create", _Runtime.create)
    root = tmp_path / "tasks"
    _make_task(root, "alpha")
    return root


def _settings(tasks: Path, tmp_path: Path, **overrides: Any) -> EpisodeSettings:
    settings = EpisodeSettings(
        tasks_dir=tasks,
        harness=BashHarnessConfig(jobs_dir=tmp_path / "jobs", max_output_chars=2000),
        transport_retries=0,
        **overrides,
    )
    return settings.normalized()


def _request(**metadata: Any) -> EpisodeRequest:
    return EpisodeRequest(
        session_url=SESSION,
        prompt=[{"role": "user", "content": "How many rows?" + HARNESS_MESSAGE}],
        request_kwargs={"temperature": 1.0, "max_tokens": 64},
        metadata={"instance_id": "alpha", **metadata},
    )


async def _run(
    tasks: Path,
    tmp_path: Path,
    script: list[Any],
    *,
    metadata: dict[str, Any] | None = None,
    **settings: Any,
):
    sessions = _Sessions(script)
    async with sessions.client() as client:
        outcome = await run_episode(
            _request(**(metadata or {})),
            _settings(tasks, tmp_path, **settings),
            client=client,
        )
    return outcome, sessions


# --- the shared harness ---------------------------------------------------------


def test_harness_matches_the_rl_cookbooks_shared_harness():
    """The episode's limits and message are rl-core's (docs/examples/rl/common/harness.py)."""

    path = REPO / "docs" / "examples" / "rl" / "common" / "harness.py"
    spec = importlib.util.spec_from_file_location("rl_common_harness", path)
    assert spec is not None and spec.loader is not None
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    assert episode_module.HARNESS_MESSAGE == shared.HARNESS_MESSAGE
    assert episode_module.BASH_TIMEOUT_SEC == shared.BASH_TIMEOUT_SEC
    assert episode_module.MAX_OUTPUT_CHARS == shared.MAX_OUTPUT_CHARS
    assert episode_module.MAX_TURNS == shared.MAX_TURNS
    harness = shared.harness_config()
    assert harness.submit_path == episode_module.SUBMIT_PATH
    assert harness.sandbox_user == "agent"


def test_prompt_rows_are_the_evaluators_first_message(tasks: Path):
    rows = dataset_rows(tasks, split="train")
    (spec_row,) = BenchFlowSpec(tasks_dir=tasks).train_dataset_rows
    expected = [dict(m) for m in spec_row["prompt"]]
    expected[-1]["content"] += HARNESS_MESSAGE  # what evaluate.py sends
    assert rows == [
        {"prompt": expected, "metadata": {"instance_id": "alpha", "split": "train"}}
    ]


# --- token-in/token-out: the replayed history ------------------------------------


async def test_each_request_extends_the_last_with_the_servers_own_message(
    tasks: Path, tmp_path: Path
):
    first = _reply(
        "Let me look.",
        [_tool_call("run_bash", {"command": "ls"}, "call_1")],
        reasoning="I should list files.",
    )
    second = _reply("", [_tool_call("submit", {"answer": "42"}, "call_2")])
    outcome, sessions = await _run(tasks, tmp_path, [first, second])

    one, two = sessions.requests
    assert two["messages"][: len(one["messages"])] == one["messages"]
    replayed = two["messages"][len(one["messages"])]
    returned = first["choices"][0]["message"]
    assert replayed == returned  # reasoning_content and tool_calls included
    assert two["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "ran ls\n",
    }
    # The Miles sampling settings ride along; the tools are the TRL adapter's.
    assert one["temperature"] == 1.0 and one["max_tokens"] == 64
    assert [t["function"]["name"] for t in one["tools"]] == ["run_bash", "submit"]
    assert "stream" not in one
    assert outcome.exit_status == "Submitted"
    assert outcome.decision is not None and outcome.decision.reward == 1.0
    runtime = _Runtime.created[0]
    assert runtime.commands[0] == "ls"
    assert "printf %s 42 > /workdir/answer.txt" in runtime.commands[1]
    assert runtime.verified == 1 and runtime.closed


async def test_a_truncated_reply_ends_the_episode_without_running_its_tools(
    tasks: Path, tmp_path: Path
):
    """The session server refuses to extend a turn cut at max_tokens."""

    cut = _reply(
        "", [_tool_call("run_bash", {"command": "rm -rf x"}, "c")], finish="length"
    )
    outcome, sessions = await _run(tasks, tmp_path, [cut])
    assert len(sessions.requests) == 1
    assert _Runtime.created[0].commands == []
    assert outcome.clipped_replies == 1
    assert outcome.exit_status == "SequenceLengthLimitExceeded"
    assert outcome.decision is not None and outcome.decision.reason == "scored"


async def test_turn_limit_runs_max_turns_tool_turns_and_one_more_call(
    tasks: Path, tmp_path: Path
):
    loop = [
        _reply("", [_tool_call("run_bash", {"command": f"echo {i}"}, f"c{i}")])
        for i in range(4)
    ]
    outcome, sessions = await _run(tasks, tmp_path, loop, max_turns=3)
    assert len(sessions.requests) == 4
    assert _Runtime.created[0].commands == ["echo 0", "echo 1", "echo 2"]
    assert outcome.exit_status == "TurnLimitExceeded"
    assert outcome.turns == 4 and outcome.tool_calls == 3


async def test_a_reply_without_tool_calls_is_verified_as_left(
    tasks: Path, tmp_path: Path
):
    _Runtime.verify_outcome = {"reward": 0.0, "rewards": {"reward": 0.0}}
    outcome, _ = await _run(tasks, tmp_path, [_reply("I give up.")])
    assert outcome.exit_status == "NoToolCall"
    assert outcome.decision is not None and outcome.decision.reward == 0.0
    assert not outcome.policy_acted


async def test_bad_tool_arguments_are_fed_back_like_the_trl_adapter(
    tasks: Path, tmp_path: Path
):
    bad = _reply("", [_tool_call("run_bash", {"cmd": "ls"}, "c1")])
    outcome, sessions = await _run(tasks, tmp_path, [bad, _reply("stop")])
    tool_message = sessions.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"]) == {"error": "'command'"}
    assert outcome.tool_errors == 1 and not outcome.policy_acted


async def test_the_episode_stops_before_its_context_outgrows_max_seq_len(
    tasks: Path, tmp_path: Path
):
    """Miles trains on at most --max-seq-len tokens per episode (sample metadata)."""

    acted = _reply("", [_tool_call("run_bash", {"command": "ls"}, "c1")])
    # usage 100 + 10 tokens, a 64-token reply budget, and the tool result: > 150.
    outcome, sessions = await _run(
        tasks, tmp_path, [acted], metadata={"max_seq_len": 150}
    )
    assert len(sessions.requests) == 1
    assert outcome.exit_status == "SequenceLengthLimitExceeded"
    assert outcome.response()["eval_report"]["ended"] == "context_exhausted"
    assert _Runtime.created[0].verified == 1
    # With room to spare the episode goes on.
    _Runtime.created.clear()
    outcome, sessions = await _run(
        tasks, tmp_path, [acted, _reply("done")], metadata={"max_seq_len": 4096}
    )
    assert len(sessions.requests) == 2 and outcome.exit_status == "NoToolCall"


# --- the failure contract --------------------------------------------------------


async def test_sandbox_that_never_starts_is_dropped_before_any_model_call(
    tasks: Path, tmp_path: Path
):
    """No model call: on a Miles without InfraAbort the sample is still discarded."""

    _Runtime.start_error = RuntimeError("Daytona: quota exceeded")
    outcome, sessions = await _run(tasks, tmp_path, [_reply("hi")])
    assert sessions.requests == []
    assert outcome.dropped and outcome.exit_status == "SandboxUnavailable"
    body = outcome.response()
    assert body["dropped"] is True and body["reward"] is None
    assert "quota exceeded" in body["detail"]


@pytest.mark.parametrize(
    ("failure", "exit_status"),
    [
        (
            httpx.Response(502, json={"error": "backend transport error"}),
            "ModelEndpointFailed",
        ),
        (
            httpx.Response(404, json={"error": "session not found"}),
            "ModelEndpointFailed",
        ),
        (httpx.Response(503, json={"error": "aborted"}), "GenerationAborted"),
        (httpx.ConnectError("refused"), "ModelEndpointFailed"),
    ],
)
async def test_model_server_failures_are_dropped_after_the_policy_acted(
    tasks: Path, tmp_path: Path, failure: Any, exit_status: str
):
    acted = _reply("", [_tool_call("run_bash", {"command": "ls"}, "c1")])
    outcome, _ = await _run(tasks, tmp_path, [acted, failure])
    runtime = _Runtime.created[0]
    assert runtime.commands == ["ls"]
    assert outcome.dropped and outcome.exit_status == exit_status
    assert runtime.verified == 0 and runtime.closed


@pytest.mark.parametrize(
    ("failure", "exit_status"),
    [
        (
            httpx.Response(
                400, json={"error": "maximum context length is 16384 tokens"}
            ),
            "SequenceLengthLimitExceeded",
        ),
        (
            httpx.Response(400, json={"error": "messages are not append-only"}),
            "AgentError",
        ),
        (httpx.Response(500, json={"error": "TITO prefix mismatch"}), "AgentError"),
    ],
)
async def test_refusals_the_policy_can_cause_end_the_episode_and_are_scored(
    tasks: Path, tmp_path: Path, failure: Any, exit_status: str
):
    acted = _reply("", [_tool_call("run_bash", {"command": "ls"}, "c1")])
    outcome, _ = await _run(tasks, tmp_path, [acted, failure])
    assert not outcome.dropped
    assert outcome.exit_status == exit_status
    assert outcome.decision is not None and outcome.decision.reward == 1.0
    assert _Runtime.created[0].verified == 1


async def test_verifier_crash_scores_zero_after_the_policy_acted(
    tasks: Path, tmp_path: Path
):
    _Runtime.verify_outcome = RuntimeError("verifier sandbox lost")
    acted = _reply("", [_tool_call("run_bash", {"command": "ls"}, "c1")])
    outcome, _ = await _run(tasks, tmp_path, [acted, _reply("done")])
    assert not outcome.dropped
    assert outcome.decision is not None and outcome.decision.reward == 0.0
    assert outcome.exit_status == "VerifierError"


async def test_verifier_crash_on_an_untouched_sandbox_is_dropped(
    tasks: Path, tmp_path: Path
):
    _Runtime.verify_outcome = RuntimeError("verifier image missing")
    outcome, _ = await _run(tasks, tmp_path, [_reply("no idea")])
    assert outcome.dropped and outcome.exit_status == "VerifierCrashCleanRun"


async def test_an_exploit_verdict_scores_zero_and_is_flagged(
    tasks: Path, tmp_path: Path
):
    _Runtime.verify_outcome = {
        "integrity": {
            "exploited": True,
            "reason": "read /workdir/.grader/expected.json",
        }
    }
    submit = _reply("", [_tool_call("submit", {"answer": "7"}, "c1")])
    outcome, _ = await _run(tasks, tmp_path, [submit])
    body = outcome.response()
    assert body["reward"] == 0.0 and body["flagged"] is True
    assert body["exit_status"] == "IntegrityViolation"
    assert body["eval_report"]["integrity"]["exploited"] is True


async def test_a_wall_clock_overrun_scores_zero_and_releases_the_sandbox(
    tasks: Path, tmp_path: Path
):
    _Runtime.bash_gate = asyncio.Event()  # never set: the command hangs
    acted = _reply("", [_tool_call("run_bash", {"command": "sleep 999"}, "c1")])
    outcome, _ = await _run(tasks, tmp_path, [acted], episode_timeout_sec=0.2)
    assert outcome.exit_status == "TimeLimitExceeded"
    assert outcome.decision is not None and outcome.decision.reward == 0.0
    assert _Runtime.created[0].closed and _Runtime.created[0].verified == 0


async def test_a_cancelled_episode_releases_its_sandbox(tasks: Path, tmp_path: Path):
    _Runtime.bash_gate = asyncio.Event()
    acted = _reply("", [_tool_call("run_bash", {"command": "sleep 999"}, "c1")])
    sessions = _Sessions([acted])
    async with sessions.client() as client:
        job = asyncio.create_task(
            run_episode(_request(), _settings(tasks, tmp_path), client=client)
        )
        while not (_Runtime.created and _Runtime.created[0].commands):
            await asyncio.sleep(0.01)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
    assert _Runtime.created[0].closed
    records = (tmp_path / "jobs" / "rollouts.jsonl").read_text().splitlines()
    assert json.loads(records[-1])["exit_status"] == "Aborted"


async def test_every_episode_is_kept_for_audit(tasks: Path, tmp_path: Path):
    submit = _reply("", [_tool_call("submit", {"answer": "3"}, "c1")])
    outcome, _ = await _run(tasks, tmp_path, [submit])
    record = json.loads((tmp_path / "jobs" / "rollouts.jsonl").read_text())
    assert record["exit_status"] == "Submitted" and record["reward"] == 1.0
    assistant, tool = record["messages"][-2:]
    assert assistant["tool_calls"][0]["function"]["name"] == "submit"
    assert tool == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "submission recorded",
    }
    saved = json.loads((outcome.rollout_dir / "policy" / "messages.json").read_text())
    assert saved["episode_id"] == outcome.episode_id


def test_integrity_audit_is_refused_when_the_runtime_has_none(
    tasks: Path, tmp_path: Path
):
    fields = {f for f in episode_module.TaskRuntimeConfig.__dataclass_fields__}
    if "integrity" in fields:
        pytest.skip("this BenchFlow has the integrity audit")
    with pytest.raises(ValueError, match="no integrity audit"):
        _settings(tasks, tmp_path, integrity="audit")


# --- the server ------------------------------------------------------------------


def _app(tasks: Path, tmp_path: Path, sessions: _Sessions, **kwargs: Any) -> Any:
    return create_app(
        EpisodeSettings(
            tasks_dir=tasks,
            harness=BashHarnessConfig(jobs_dir=tmp_path / "jobs"),
            transport_retries=0,
        ),
        max_sandboxes=2,
        client_factory=sessions.client,
        **kwargs,
    )


_BODY = {
    "session_url": SESSION,
    "prompt": [{"role": "user", "content": "q"}],
    "metadata": {"instance_id": "alpha"},
}


async def test_server_runs_an_episode_and_refuses_unknown_tasks(
    tasks: Path, tmp_path: Path
):
    sessions = _Sessions([_reply("", [_tool_call("submit", {"answer": "1"}, "c1")])])
    app = _app(tasks, tmp_path, sessions)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://env"
        ) as http:
            ok = (await http.post("/run", json=_BODY)).json()
            unknown = await http.post(
                "/run", json={**_BODY, "metadata": {"instance_id": "nope"}}
            )
            health = (await http.get("/health")).json()
    assert ok["exit_status"] == "Submitted" and ok["reward"] == 1.0
    assert ok["dropped"] is False and ok["agent_metrics"]["turns"] == 1
    assert unknown.status_code == 400 and "unknown task" in unknown.json()["error"]
    assert health["episodes"] == 1 and health["exit_status"] == {"Submitted": 1}
    assert _Runtime.created[0].closed


async def test_abort_cancels_episodes_in_flight_and_releases_sandboxes(
    tasks: Path, tmp_path: Path
):
    _Runtime.bash_gate = asyncio.Event()  # the command never returns
    sessions = _Sessions(
        [_reply("", [_tool_call("run_bash", {"command": "sleep 9"}, "c")])]
    )
    app = _app(tasks, tmp_path, sessions)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://env"
        ) as http:
            running = asyncio.create_task(http.post("/run", json=_BODY))
            while not (_Runtime.created and _Runtime.created[0].commands):
                await asyncio.sleep(0.01)
            aborted = (await http.post("/abort")).json()
            result = (await running).json()
    assert aborted == {"cancelled": 1}
    assert result["dropped"] is True and result["exit_status"] == "Aborted"
    assert _Runtime.created[0].closed


async def test_server_checks_the_bearer_token(tasks: Path, tmp_path: Path):
    app = _app(tasks, tmp_path, _Sessions([]), token="s3cret")
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://env"
        ) as http:
            anonymous = await http.get("/tasks")
            forged = await http.post("/abort", headers={"Authorization": "Bearer nope"})
            signed = await http.get(
                "/tasks", headers={"Authorization": "Bearer s3cret"}
            )
    assert anonymous.status_code == 401 and forged.status_code == 401
    assert signed.json() == {"tasks": ["alpha"]}


def test_binding_beyond_loopback_needs_a_token(tasks: Path):
    with pytest.raises(ValueError, match="token"):
        serve(EpisodeSettings(tasks_dir=tasks), host="0.0.0.0", port=1)
