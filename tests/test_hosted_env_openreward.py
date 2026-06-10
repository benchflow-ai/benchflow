"""Unit coverage for the OpenReward (ORS) hosted environment runner.

Everything runs against a FAKED OpenReward client and model transport — no
network, no API keys. The fakes mirror the documented client surface
(https://docs.openreward.ai/quickstart.md, https://pypi.org/project/openreward/):
``environments.get(name=..., variant=...)``, ``list_tasks(split=...)``,
``list_tools(format="openai")``, ``session(task=...)`` context manager,
``session.get_prompt()`` block list, and ``session.call_tool`` returning
``ToolOutput`` objects with ``blocks`` / ``reward`` / ``finished``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.hosted_env import (
    HostedEnvError,
    HostedEnvRef,
    HostedEnvRunConfig,
    prime_env_info,
    run_hosted_env,
)
from benchflow.hosted_env_openreward import (
    ChatTurn,
    OpenRewardRunConfig,
    OpenRewardRunResult,
    _default_client_factory,
    resolve_model_endpoint,
    run_openreward_env,
)

PROXY_ENV = {
    "BENCHFLOW_PROVIDER_BASE_URL": "http://proxy.test/v1",
    "BENCHFLOW_PROVIDER_API_KEY": "proxy-key",
    "BENCHFLOW_PROVIDER_MODEL": "served-model",
    "BENCHFLOW_PROVIDER_NAME": "litellm",
    "BENCHFLOW_PROVIDER_PROTOCOL": "openai-completions",
}

PROMPT = "Capture the flag hidden in the container."


# ── Fakes for the documented OpenReward client surface ──


class FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeToolOutput:
    def __init__(
        self,
        text: str = "",
        reward: float | None = None,
        finished: bool = False,
    ) -> None:
        self.blocks = [FakeBlock(text)]
        self.reward = reward
        self.finished = finished
        self.metadata: dict = {}


class FakeSession:
    """Context-manager session with scripted call_tool outputs."""

    def __init__(self, prompt: str, outputs: list) -> None:
        self._prompt = prompt
        self._outputs = outputs
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def get_prompt(self) -> list[FakeBlock]:
        return [FakeBlock(self._prompt)]

    def call_tool(self, name: str, arguments: dict) -> FakeToolOutput:
        self.calls.append((name, arguments))
        out = self._outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class FakeEnvironment:
    def __init__(self, tasks: list, tools: list, sessions: list[FakeSession]) -> None:
        self._tasks = tasks
        self._tools = tools
        self._sessions = sessions
        self.sessions_used: list[FakeSession] = []
        self.split_seen: str | None = None

    def list_tasks(self, split: str) -> list:
        self.split_seen = split
        return self._tasks

    def list_tools(self, format: str) -> list:
        assert format == "openai"
        return self._tools

    def session(self, task: dict) -> FakeSession:
        session = self._sessions.pop(0)
        self.sessions_used.append(session)
        return session


class FakeClient:
    def __init__(self, environment: FakeEnvironment) -> None:
        self.get_kwargs: dict = {}

        def get(**kwargs):
            self.get_kwargs = kwargs
            return environment

        self.environments = SimpleNamespace(get=get)


def tool_call(name: str, arguments: str, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def scripted_chat(turns: list[ChatTurn]):
    """Chat transport fake — returns scripted turns, records every request."""
    requests: list[dict] = []

    def chat(*, endpoint, messages, tools, max_tokens, temperature) -> ChatTurn:
        requests.append(
            {
                "endpoint": endpoint,
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return turns.pop(0)

    return chat, requests


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def run_fake(
    tmp_path: Path,
    *,
    turns: list[ChatTurn],
    outputs_per_task: list[list],
    tasks: list | None = None,
    source_env: str = "openreward:GeneralReasoning/CTF",
    version: str | None = None,
    num_examples: int = 1,
    max_turns: int = 16,
) -> tuple[OpenRewardRunResult, FakeClient, FakeEnvironment, list[dict]]:
    tasks = tasks if tasks is not None else [{"task_id": "t0"}]
    sessions = [FakeSession(PROMPT, outputs) for outputs in outputs_per_task]
    environment = FakeEnvironment(tasks, TOOLS, sessions)
    client = FakeClient(environment)
    chat, requests = scripted_chat(turns)
    result = run_openreward_env(
        OpenRewardRunConfig(
            source_env=HostedEnvRef.parse(source_env, version=version),
            model="openai/gpt-test",
            env_args={"split": "train"},
            agent="openreward-driver",
            jobs_dir=tmp_path,
            num_examples=num_examples,
            max_turns=max_turns,
            max_tokens=512,
            temperature=0.0,
        ),
        env=PROXY_ENV,
        client_factory=lambda: client,
        chat_completion=chat,
    )
    return result, client, environment, requests


def happy_turns() -> list[ChatTurn]:
    return [
        ChatTurn(
            message={
                "content": "Let me solve this.",
                "tool_calls": [tool_call("submit_answer", '{"answer": 42}')],
            },
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
    ]


def happy_outputs() -> list[list]:
    return [[FakeToolOutput("correct", reward=1.0, finished=True)]]


# ── Reference parsing / provider recognition ──


def test_hosted_env_ref_recognizes_openreward_provider():
    ref = HostedEnvRef.parse("openreward:GeneralReasoning/CTF")

    assert ref.provider == "openreward"
    assert ref.owner == "GeneralReasoning"
    assert ref.name == "CTF"
    assert ref.env_id == "GeneralReasoning/CTF"
    assert ref.env_uid == "openreward:GeneralReasoning/CTF@latest"
    assert ref.hub_url == "https://openreward.ai/environments"


def test_hosted_env_ref_openreward_variant_rides_the_version_slot():
    ref = HostedEnvRef.parse("openreward:GeneralReasoning/ArithmeticEnv@bitwise")

    assert ref.version == "bitwise"
    assert ref.env_uid == "openreward:GeneralReasoning/ArithmeticEnv@bitwise"


def test_hosted_env_ref_still_rejects_unknown_providers():
    with pytest.raises(HostedEnvError, match="primeintellect, openreward"):
        HostedEnvRef.parse("otherhub:owner/name")


def test_run_hosted_env_rejects_openreward_refs(tmp_path):
    with pytest.raises(HostedEnvError, match="run_openreward_env"):
        run_hosted_env(
            HostedEnvRunConfig(
                source_env=HostedEnvRef.parse("openreward:GeneralReasoning/CTF"),
                model="openai/gpt-test",
                jobs_dir=tmp_path,
            )
        )


def test_run_openreward_env_rejects_prime_refs(tmp_path):
    with pytest.raises(HostedEnvError, match=r"benchflow\.hosted_env"):
        run_openreward_env(
            OpenRewardRunConfig(
                source_env=HostedEnvRef.parse(
                    "primeintellect/general-agent", version="0.1.1"
                ),
                model="openai/gpt-test",
                jobs_dir=tmp_path,
            ),
            env=PROXY_ENV,
        )


def test_prime_env_info_rejects_openreward_refs():
    with pytest.raises(HostedEnvError, match="primeintellect"):
        prime_env_info(HostedEnvRef.parse("openreward:GeneralReasoning/CTF"))


# ── Model endpoint resolution (fail-closed) ──


def test_resolve_model_endpoint_honors_explicit_provider_contract():
    endpoint = resolve_model_endpoint("openai/gpt-test", PROXY_ENV)

    assert endpoint.base_url == "http://proxy.test/v1"
    assert endpoint.api_key == "proxy-key"
    assert endpoint.model_id == "served-model"
    assert endpoint.provider == "litellm"


def test_resolve_model_endpoint_rejects_non_openai_protocol_contract():
    env = {**PROXY_ENV, "BENCHFLOW_PROVIDER_PROTOCOL": "anthropic-messages"}
    with pytest.raises(HostedEnvError, match="anthropic-messages"):
        resolve_model_endpoint("openai/gpt-test", env)


def test_resolve_model_endpoint_uses_registered_provider_registry():
    env = {
        "DEEPSEEK_BASE_URL": "https://api.deepseek.test/v1",
        "DEEPSEEK_API_KEY": "ds-key",
    }
    endpoint = resolve_model_endpoint("deepseek/deepseek-chat", env)

    assert endpoint.base_url == "https://api.deepseek.test/v1"
    assert endpoint.api_key == "ds-key"
    assert endpoint.model_id == "deepseek-chat"
    assert endpoint.provider == "deepseek"


def test_resolve_model_endpoint_normalizes_bare_openai_models():
    endpoint = resolve_model_endpoint("gpt-5-mini", {"OPENAI_API_KEY": "oa-key"})

    assert endpoint.base_url == "https://api.openai.com/v1"
    assert endpoint.model_id == "gpt-5-mini"
    assert endpoint.provider == "openai"


def test_resolve_model_endpoint_fails_closed_without_api_key():
    env = {"DEEPSEEK_BASE_URL": "https://api.deepseek.test/v1"}
    with pytest.raises(HostedEnvError, match="DEEPSEEK_API_KEY"):
        resolve_model_endpoint("deepseek/deepseek-chat", env)


def test_resolve_model_endpoint_fails_closed_for_unknown_models():
    with pytest.raises(HostedEnvError, match="No OpenAI-compatible provider"):
        resolve_model_endpoint("gemini-2.5-flash", {})


def test_resolve_model_endpoint_rejects_anthropic_only_providers():
    env = {"AZURE_RESOURCE": "res", "AZURE_API_KEY": "az-key"}
    with pytest.raises(HostedEnvError, match="openai-completions"):
        resolve_model_endpoint("azure-foundry-anthropic/claude-test", env)


def test_resolve_model_endpoint_requires_base_url_for_serverless_providers():
    with pytest.raises(HostedEnvError, match="BENCHFLOW_PROVIDER_BASE_URL"):
        resolve_model_endpoint("vllm/Qwen/Qwen3", {"OPENAI_API_KEY": "k"})


# ── Default client factory (auth fail-closed, no network) ──


def test_default_client_factory_requires_api_key():
    with pytest.raises(HostedEnvError, match="OPENREWARD_API_KEY"):
        _default_client_factory({})


def test_default_client_factory_requires_openreward_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "openreward", None)
    with pytest.raises(HostedEnvError, match="pip install openreward"):
        _default_client_factory({"OPENREWARD_API_KEY": "or-key"})


# ── Episode happy path ──


def test_happy_path_drives_episode_and_scores(tmp_path):
    result, client, environment, requests = run_fake(
        tmp_path, turns=happy_turns(), outputs_per_task=happy_outputs()
    )

    assert result.error is None
    assert result.reward == 1.0
    assert result.total_tool_calls == 1
    assert result.normalized_model == "served-model"
    assert client.get_kwargs == {"name": "GeneralReasoning/CTF"}
    assert environment.split_seen == "train"
    episode = result.episodes[0]
    assert episode.finished is True
    assert episode.truncated is False
    assert episode.reward == 1.0
    assert episode.task_id == "t0"

    # The model request carried the resolved endpoint, prompt, and tools.
    assert requests[0]["endpoint"].base_url == "http://proxy.test/v1"
    assert requests[0]["messages"] == [{"role": "user", "content": PROMPT}]
    assert requests[0]["tools"] == TOOLS
    assert requests[0]["max_tokens"] == 512


def test_happy_path_parses_tool_arguments_as_json(tmp_path):
    result, _, environment, _ = run_fake(
        tmp_path, turns=happy_turns(), outputs_per_task=happy_outputs()
    )
    # The faked session recorded the exact call the runner made: JSON-parsed
    # arguments, not the raw string.
    assert environment.sessions_used[0].calls == [("submit_answer", {"answer": 42})]
    assert result.episodes[0].n_tool_calls == 1


def test_happy_path_trajectory_events_exact(tmp_path):
    result, _, _, _ = run_fake(
        tmp_path, turns=happy_turns(), outputs_per_task=happy_outputs()
    )

    traj_path = result.run_dir / "trajectory" / "acp_trajectory.jsonl"
    events = [json.loads(line) for line in traj_path.read_text().splitlines() if line]
    assert [e["type"] for e in events] == [
        "user_message",
        "agent_message",
        "tool_call",
        "reward",
    ]
    assert events[0]["text"] == PROMPT
    assert events[0]["example_index"] == 0
    assert events[1]["text"] == "Let me solve this."
    assert events[2]["title"] == "submit_answer"
    assert events[2]["status"] == "completed"
    assert events[2]["tool_call_id"] == "call_1"
    assert events[2]["content"] == [{"type": "content", "text": "correct"}]
    assert events[3]["value"] == 1.0
    assert events[3]["source"] == "openreward"


def test_happy_path_variant_is_passed_to_environments_get(tmp_path):
    _, client, _, _ = run_fake(
        tmp_path,
        turns=happy_turns(),
        outputs_per_task=happy_outputs(),
        source_env="openreward:GeneralReasoning/ArithmeticEnv",
        version="bitwise",
    )
    assert client.get_kwargs == {
        "name": "GeneralReasoning/ArithmeticEnv",
        "variant": "bitwise",
    }


# ── Rollout artifact contract ──


def test_openreward_writes_contract_result_json(tmp_path):
    result, _, _, _ = run_fake(
        tmp_path, turns=happy_turns(), outputs_per_task=happy_outputs()
    )
    payload = json.loads((result.run_dir / "result.json").read_text())

    for key in (
        "task_name",
        "rollout_name",
        "rewards",
        "agent",
        "agent_name",
        "model",
        "n_tool_calls",
        "n_prompts",
        "agent_result",
        "final_metrics",
        "trajectory_summary",
        "error",
        "verifier_error",
        "partial_trajectory",
        "trajectory_source",
        "started_at",
        "finished_at",
        "timing",
        "source",
    ):
        assert key in payload, f"missing contract key: {key}"

    assert payload["task_name"] == "openreward:GeneralReasoning/CTF@latest"
    assert payload["rewards"] == {
        "reward": 1.0,
        "rubric": [{"name": "example_0", "score": 1.0}],
    }
    assert payload["agent"] == "openreward-driver"
    assert payload["agent_name"] == "openreward"
    assert payload["model"] == "served-model"
    assert payload["n_tool_calls"] == 1
    assert payload["trajectory_source"] == "hosted_env"
    assert payload["error"] is None
    assert payload["source"]["type"] == "hosted_env"
    assert payload["source"]["provider"] == "openreward"
    assert payload["source"]["env_uid"] == "openreward:GeneralReasoning/CTF@latest"
    assert payload["source"]["runner"] == "openreward"
    # Token usage flows from the provider response into the contract.
    assert payload["agent_result"]["n_input_tokens"] == 10
    assert payload["agent_result"]["n_output_tokens"] == 5
    assert payload["agent_result"]["total_tokens"] == 15
    assert payload["agent_result"]["usage_source"] == "provider_response"
    assert payload["trajectory_summary"]["tool_call_steps"] == 1


def test_openreward_writes_rewards_jsonl(tmp_path):
    result, _, _, _ = run_fake(
        tmp_path, turns=happy_turns(), outputs_per_task=happy_outputs()
    )
    rewards_path = result.run_dir / "rewards.jsonl"
    events = [
        json.loads(line) for line in rewards_path.read_text().splitlines() if line
    ]
    terminal = [e for e in events if e["type"] == "terminal"]
    rubric = [e for e in events if e["type"] == "process"]
    assert len(terminal) == 1
    assert terminal[0]["value"] == 1.0
    assert terminal[0]["tag"] == "reward"
    assert terminal[0]["source"] == "verifier"
    assert [(e["tag"], e["value"]) for e in rubric] == [("example_0", 1.0)]


def test_openreward_writes_trainer_verifiers_jsonl(tmp_path):
    result, _, _, _ = run_fake(
        tmp_path, turns=happy_turns(), outputs_per_task=happy_outputs()
    )
    records = [
        json.loads(line)
        for line in (result.run_dir / "trainer" / "verifiers.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    assert len(records) == 1
    record = records[0]
    assert record["example_id"] == 0
    assert record["reward"] == 1.0
    assert record["is_completed"] is True
    assert record["is_truncated"] is False
    assert record["prompt"] == [{"role": "user", "content": PROMPT}]
    assert record["completion"] == [
        {"role": "assistant", "content": "Let me solve this."},
        {"role": "assistant", "content": "[tool_call: submit_answer]\ncorrect"},
    ]
    assert record["info"]["task_id"] == "openreward:GeneralReasoning/CTF@latest#t0"
    assert record["info"]["environment"] == "openreward:GeneralReasoning/CTF@latest"
    assert record["info"]["model"] == "served-model"
    assert record["info"]["reward_valid"] is True


def test_openreward_writes_config_timing_prompts_and_evidence(tmp_path):
    result, _, _, _ = run_fake(
        tmp_path, turns=happy_turns(), outputs_per_task=happy_outputs()
    )
    run_dir = result.run_dir
    assert json.loads((run_dir / "prompts.json").read_text()) == [PROMPT]
    assert "total" in json.loads((run_dir / "timing.json").read_text())

    config = json.loads((run_dir / "config.json").read_text())
    assert config["environment"] == "hosted_env"
    assert config["hosted_env"]["provider"] == "openreward"
    assert config["hosted_env"]["runner"] == "openreward"
    assert config["hosted_env"]["env_uid"] == "openreward:GeneralReasoning/CTF@latest"
    assert config["hosted_env"]["split"] == "train"
    assert config["hosted_env"]["max_turns"] == 16

    hosted = json.loads((run_dir / "hosted_env" / "hosted_run.json").read_text())
    assert hosted["env_uid"] == "openreward:GeneralReasoning/CTF@latest"
    assert hosted["runner"] == "openreward"
    assert hosted["model_base_url"] == "http://proxy.test/v1"
    assert "api_key" not in json.dumps(hosted).lower()
    episodes = [
        json.loads(line)
        for line in (run_dir / "hosted_env" / "episodes.jsonl").read_text().splitlines()
        if line
    ]
    assert episodes == [
        {
            "example_index": 0,
            "task_id": "t0",
            "task": {"task_id": "t0"},
            "reward": 1.0,
            "finished": True,
            "truncated": False,
            "n_tool_calls": 1,
            "usage": {"input": 10, "output": 5},
            "error": None,
        }
    ]


# ── Finished-flag handling ──


def test_episode_continues_until_finished_flag(tmp_path):
    turns = [
        ChatTurn(
            message={
                "content": "Trying a probe first.",
                "tool_calls": [tool_call("submit_answer", '{"answer": 1}', "call_1")],
            }
        ),
        ChatTurn(
            message={
                "content": None,
                "tool_calls": [tool_call("submit_answer", '{"answer": 42}', "call_2")],
            }
        ),
    ]
    outputs = [
        [
            FakeToolOutput("nope", reward=0.25, finished=False),
            FakeToolOutput("correct", reward=1.0, finished=True),
        ]
    ]
    result, _, _, requests = run_fake(tmp_path, turns=turns, outputs_per_task=outputs)

    assert len(requests) == 2
    episode = result.episodes[0]
    assert episode.finished is True
    assert episode.n_tool_calls == 2
    # Final reward is the last reward observed in the episode.
    assert result.reward == 1.0
    # The first tool result was threaded back to the model as a tool message.
    assert {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "nope",
    } in requests[1]["messages"]
    # Both step rewards are captured as reward events.
    traj = (result.run_dir / "trajectory" / "acp_trajectory.jsonl").read_text()
    reward_events = [
        json.loads(line) for line in traj.splitlines() if '"reward"' in line
    ]
    assert [e["value"] for e in reward_events if e["type"] == "reward"] == [0.25, 1.0]


def test_episode_stops_when_model_returns_no_tool_calls(tmp_path):
    turns = [ChatTurn(message={"content": "I cannot solve this."})]
    result, _, _, requests = run_fake(tmp_path, turns=turns, outputs_per_task=[[]])

    assert len(requests) == 1
    episode = result.episodes[0]
    assert episode.finished is False
    assert episode.truncated is False
    assert episode.reward is None
    # Unscored episodes fail closed to 0 in the aggregate.
    assert result.reward == 0.0
    assert result.error is None
    record = json.loads(
        (result.run_dir / "trainer" / "verifiers.jsonl").read_text().splitlines()[0]
    )
    assert record["reward"] == 0.0
    assert record["is_completed"] is False


def test_episode_truncates_at_max_turns(tmp_path):
    turns = [
        ChatTurn(
            message={
                "content": None,
                "tool_calls": [tool_call("submit_answer", "{}", f"call_{i}")],
            }
        )
        for i in range(2)
    ]
    outputs = [[FakeToolOutput("keep going") for _ in range(2)]]
    result, _, _, requests = run_fake(
        tmp_path, turns=turns, outputs_per_task=outputs, max_turns=2
    )

    assert len(requests) == 2  # hard stop at max_turns
    episode = result.episodes[0]
    assert episode.truncated is True
    assert episode.finished is False
    record = json.loads(
        (result.run_dir / "trainer" / "verifiers.jsonl").read_text().splitlines()[0]
    )
    assert record["is_truncated"] is True


# ── Error handling (fail-closed) ──


def test_tool_error_fails_episode_closed(tmp_path):
    turns = happy_turns()
    outputs = [[RuntimeError("boom")]]
    result, _, _, _ = run_fake(tmp_path, turns=turns, outputs_per_task=outputs)

    episode = result.episodes[0]
    assert episode.error == "RuntimeError: boom"
    assert episode.reward is None
    assert result.error == "t0: RuntimeError: boom"
    assert result.reward == 0.0

    payload = json.loads((result.run_dir / "result.json").read_text())
    assert payload["error"] == "t0: RuntimeError: boom"
    record = json.loads(
        (result.run_dir / "trainer" / "verifiers.jsonl").read_text().splitlines()[0]
    )
    assert record["reward"] == 0.0
    assert record["info"]["reward_valid"] is False


def test_malformed_tool_arguments_fail_episode_closed(tmp_path):
    turns = [
        ChatTurn(
            message={
                "content": None,
                "tool_calls": [tool_call("submit_answer", "not-json")],
            }
        )
    ]
    result, _, _, _ = run_fake(tmp_path, turns=turns, outputs_per_task=[[]])

    episode = result.episodes[0]
    assert episode.error is not None
    assert "non-JSON arguments" in episode.error
    assert episode.n_tool_calls == 0
    assert result.reward == 0.0


def test_environment_resolution_failure_raises(tmp_path):
    def broken_factory():
        def get(**kwargs):
            raise ConnectionError("hub unreachable")

        return SimpleNamespace(environments=SimpleNamespace(get=get))

    with pytest.raises(HostedEnvError, match="hub unreachable"):
        run_openreward_env(
            OpenRewardRunConfig(
                source_env=HostedEnvRef.parse("openreward:GeneralReasoning/CTF"),
                model="openai/gpt-test",
                jobs_dir=tmp_path,
            ),
            env=PROXY_ENV,
            client_factory=broken_factory,
        )


def test_empty_task_split_raises(tmp_path):
    environment = FakeEnvironment([], TOOLS, [])
    with pytest.raises(HostedEnvError, match="no tasks"):
        run_openreward_env(
            OpenRewardRunConfig(
                source_env=HostedEnvRef.parse("openreward:GeneralReasoning/CTF"),
                model="openai/gpt-test",
                jobs_dir=tmp_path,
            ),
            env=PROXY_ENV,
            client_factory=lambda: FakeClient(environment),
        )


def test_model_endpoint_failure_blocks_run_before_any_client_call(tmp_path):
    """Endpoint resolution fails closed before the hosted client is built."""
    constructed: list[bool] = []

    def factory():
        constructed.append(True)
        raise AssertionError("client must not be constructed")

    with pytest.raises(HostedEnvError, match="No OpenAI-compatible provider"):
        run_openreward_env(
            OpenRewardRunConfig(
                source_env=HostedEnvRef.parse("openreward:GeneralReasoning/CTF"),
                model="mystery-model",
                jobs_dir=tmp_path,
            ),
            env={},
            client_factory=factory,
        )
    assert constructed == []


# ── Multi-episode aggregation ──


def test_rewards_aggregate_as_mean_over_all_episodes(tmp_path):
    turns = [
        ChatTurn(
            message={
                "content": None,
                "tool_calls": [tool_call("submit_answer", "{}", "call_a")],
            }
        ),
        ChatTurn(
            message={
                "content": None,
                "tool_calls": [tool_call("submit_answer", "{}", "call_b")],
            }
        ),
    ]
    outputs = [
        [FakeToolOutput("right", reward=1.0, finished=True)],
        [FakeToolOutput("wrong", reward=0.0, finished=True)],
    ]
    result, _, _, _ = run_fake(
        tmp_path,
        turns=turns,
        outputs_per_task=outputs,
        tasks=[{"task_id": "t0"}, {"task_id": "t1"}],
        num_examples=2,
    )

    assert result.reward == 0.5
    assert result.total_tool_calls == 2
    payload = json.loads((result.run_dir / "result.json").read_text())
    assert payload["rewards"]["rubric"] == [
        {"name": "example_0", "score": 1.0},
        {"name": "example_1", "score": 0.0},
    ]
    records = (result.run_dir / "trainer" / "verifiers.jsonl").read_text().splitlines()
    assert [json.loads(r)["example_id"] for r in records] == [0, 1]


def test_num_examples_limits_episode_count(tmp_path):
    result, _, _, requests = run_fake(
        tmp_path,
        turns=happy_turns(),
        outputs_per_task=happy_outputs(),
        tasks=[{"task_id": "t0"}, {"task_id": "t1"}, {"task_id": "t2"}],
        num_examples=1,
    )
    assert len(result.episodes) == 1
    assert len(requests) == 1


# ── CLI routing ──


def test_eval_create_openreward_routes_to_openreward_runner(tmp_path, monkeypatch):
    seen: dict[str, object] = {}

    def fake_run(config: OpenRewardRunConfig, **kwargs) -> OpenRewardRunResult:
        seen["config"] = config
        return OpenRewardRunResult(
            source_env=config.source_env,
            run_dir=tmp_path / "run",
            model=config.model,
            normalized_model="gpt-test",
            reward=1.0,
            total_tool_calls=3,
            episodes=[],
            error=None,
        )

    monkeypatch.setattr("benchflow.hosted_env_openreward.run_openreward_env", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--source-env",
            "openreward:GeneralReasoning/CTF",
            "--source-env-arg",
            "split=train",
            "--source-env-num-examples",
            "2",
            "--source-env-max-turns",
            "8",
            "--model",
            "openai/gpt-test",
            "--jobs-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    config = seen["config"]
    assert isinstance(config, OpenRewardRunConfig)
    assert config.source_env.env_uid == "openreward:GeneralReasoning/CTF@latest"
    assert config.env_args == {"split": "train"}
    assert config.num_examples == 2
    assert config.max_turns == 8
    assert config.model == "openai/gpt-test"
    assert "openreward:GeneralReasoning/CTF@latest" in result.output


def test_eval_create_openreward_run_error_exits_nonzero(tmp_path, monkeypatch):
    def fake_run(config: OpenRewardRunConfig, **kwargs) -> OpenRewardRunResult:
        return OpenRewardRunResult(
            source_env=config.source_env,
            run_dir=tmp_path / "run",
            model=config.model,
            normalized_model="gpt-test",
            reward=0.0,
            total_tool_calls=0,
            episodes=[],
            error="t0: RuntimeError: boom",
        )

    monkeypatch.setattr("benchflow.hosted_env_openreward.run_openreward_env", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--source-env",
            "openreward:GeneralReasoning/CTF",
            "--model",
            "openai/gpt-test",
            "--jobs-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "boom" in result.output
