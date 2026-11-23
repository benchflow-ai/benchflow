"""An opt-in pin of reasoning effort in the model proxy (benchflow#1246).

A sub-agent can run at another effort than the run's: Claude Code's Agent tool
takes an ``effort`` argument (1,024 against 4,096 thinking tokens in our
probe) and Codex's ``spawn_agent`` a ``reasoning_effort`` (``max`` in a
``high`` run). With ``effort_pin`` (``--effort-pin``, off by default) the
proxy's request guard rewrites each request's effort to the run's, on every
wire: Anthropic ``output_config.effort``, OpenAI ``reasoning_effort`` and
Responses ``reasoning.effort``; each rewrite is counted in ``model-calls.json``
(``effort_rewrites``). An Anthropic thinking budget with no level is the
agent's own (Claude Code's ``MAX_THINKING_TOKENS``, its main loop's too) and is
left as sent. Off, nothing is rewritten and a request whose level differs
from the run's is counted (``effort_mismatches``).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from benchflow.providers import litellm_request_guard as guard
from benchflow.trajectories.model_calls import write_model_calls

PIN_ENV = "BENCHFLOW_EFFORT_PIN"

WIRES = {
    "/v1/messages": (
        {
            "model": "m",
            "max_tokens": 32000,
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "low"},
            "thinking": {"type": "adaptive"},
        },
        lambda body: (body["output_config"]["effort"], body["thinking"]),
        ("high", {"type": "adaptive"}),
    ),
    "/v1/chat/completions": (
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "max",
        },
        lambda body: body["reasoning_effort"],
        "high",
    ),
    "/v1/responses": (
        {
            "model": "m",
            "input": "hi",
            "reasoning": {"effort": "max", "summary": "auto"},
        },
        lambda body: body["reasoning"],
        {"effort": "high", "summary": "auto"},
    ),
}


def _scope(path: str) -> dict:
    return {
        "type": "http",
        "path": path,
        "method": "POST",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }


def _receive(body: dict):
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {
                "type": "http.request",
                "body": json.dumps(body).encode(),
                "more_body": False,
            }
        await asyncio.Event().wait()

    return receive


@pytest.fixture
def proxy_log(tmp_path, monkeypatch):
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    monkeypatch.delenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", raising=False)
    monkeypatch.delenv(PIN_ENV, raising=False)
    monkeypatch.setattr(guard, "_GATE", guard._Gate())
    return log


async def _through_guard(path: str, body: dict) -> dict:
    """The body LiteLLM gets once the guard has passed the request."""
    seen: list[dict] = []

    async def app(scope, receive, send):
        first = await receive()
        seen.append(json.loads(first["body"]))
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"{}"})

    async def send(message):
        pass

    await guard.SingleRouteGuard(app)(_scope(path), _receive(body), send)
    (got,) = seen
    return got


def _ends(log: Path) -> list[dict]:
    return [
        row
        for row in (json.loads(line) for line in log.read_text().splitlines())
        if row["event"] == "request_end"
    ]


# --- J6: pin on --------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(WIRES))
async def test_pin_on_rewrites_each_wire_s_effort_to_the_run_s(
    proxy_log, monkeypatch, path
):
    monkeypatch.setenv(PIN_ENV, "high")
    body, read, want = WIRES[path]
    got = await _through_guard(path, body)
    assert read(got) == want
    (end,) = _ends(proxy_log)
    assert end["effort_pin"] == "high"
    assert end["effort_sent"] != end["effort_asked"]
    assert end["effort_asked"] in {"low", "max"}


@pytest.mark.asyncio
async def test_pin_on_leaves_a_request_that_carries_no_effort_or_counts_tokens(
    proxy_log, monkeypatch
):
    monkeypatch.setenv(PIN_ENV, "high")
    plain = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert await _through_guard("/v1/chat/completions", plain) == plain
    counted = dict(WIRES["/v1/messages"][0])
    assert await _through_guard("/v1/messages/count_tokens", counted) == counted
    assert all("effort_sent" not in end for end in _ends(proxy_log))


#: A main loop's request at the run's level (high), as each wire carries it.
AT_THE_RUN_S_LEVEL = {
    "/v1/messages": {
        "model": "m",
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "hi"}],
        "output_config": {"effort": "high"},
        "thinking": {"type": "adaptive"},
    },
    "/v1/chat/completions": {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high",
    },
    "/v1/responses": {
        "model": "m",
        "input": "hi",
        "reasoning": {"effort": "high", "summary": "auto"},
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(AT_THE_RUN_S_LEVEL))
async def test_pin_on_passes_a_request_at_the_run_s_level_byte_for_byte(
    proxy_log, monkeypatch, path
):
    monkeypatch.setenv(PIN_ENV, "high")
    body = AT_THE_RUN_S_LEVEL[path]
    got = await _through_guard(path, body)
    assert json.dumps(got, separators=(",", ":")) == json.dumps(
        body, separators=(",", ":")
    )
    (end,) = _ends(proxy_log)
    assert (end["effort_asked"], end["effort_pin"]) == ("high", "high")
    assert "effort_sent" not in end


@pytest.mark.asyncio
async def test_pin_on_leaves_a_thinking_budget_with_no_level_as_sent(
    proxy_log, monkeypatch
):
    # Claude Code's own budget (MAX_THINKING_TOKENS), its main loop's as much
    # as a sub-agent's: no level to pin, so the main agent's budget stays.
    monkeypatch.setenv(PIN_ENV, "high")
    body = {
        "model": "m",
        "max_tokens": 64000,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 31999},
    }
    assert await _through_guard("/v1/messages", body) == body
    (end,) = _ends(proxy_log)
    assert end["effort_asked"] == "budget_tokens=31999"
    assert "effort_sent" not in end


def test_the_guard_s_levels_are_benchflow_s():
    from benchflow._utils.config import EFFORT_LEVELS

    assert guard._EFFORT_LEVELS == EFFORT_LEVELS


def test_a_pin_that_is_not_a_level_stops_the_proxy_from_starting(monkeypatch):
    # BENCHFLOW_EFFORT_PIN=1 is the run's switch; only the run's level may
    # reach the proxy (effort_request.with_route_effort turns one into the
    # other). Anything else fails closed, at the proxy's start: install(),
    # which the proxy's sitecustomize runs before LiteLLM builds its app.
    monkeypatch.setenv(PIN_ENV, "1")
    with pytest.raises(RuntimeError, match="BENCHFLOW_EFFORT_PIN"):
        guard.effort_pin()
    with pytest.raises(RuntimeError, match="BENCHFLOW_EFFORT_PIN"):
        guard.install()


@pytest.mark.asyncio
async def test_an_effort_the_agent_sent_with_a_control_character_is_kept_printable(
    proxy_log,
):
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "max\x1b[2J",
    }
    await _through_guard("/v1/chat/completions", body)
    (end,) = _ends(proxy_log)
    assert end["effort_asked"] == "(unprintable)"


def test_a_pin_in_benchflow_s_own_environment_is_refused(monkeypatch):
    """A host proxy starts from BenchFlow's own environment under the run's
    agent env, so a pin set in the shell (or .env) would hold a run that never
    asked for one: refused before launch."""
    from benchflow.effort_request import EffortRequestError, settle_effort

    monkeypatch.setenv(PIN_ENV, "high")
    with pytest.raises(EffortRequestError, match="own environment"):
        settle_effort("codex-acp", "deepseek/deepseek-flash", "high", {})
    # The run's own --effort-pin overlays it, as the proxy's env does.
    settled = settle_effort(
        "codex-acp", "deepseek/deepseek-flash", "high", {PIN_ENV: "1"}
    )
    assert settled.effort == "high"
    monkeypatch.setenv(PIN_ENV, "0")
    assert settle_effort("codex-acp", "deepseek/deepseek-flash", "high", {}).effort


@pytest.mark.parametrize("value", ["high", "1"])
def test_every_rollout_refuses_a_pin_from_benchflow_s_own_environment(
    monkeypatch, value
):
    """The rollout's own step (with_route_effort, which every rollout runs
    before its proxy starts, bf.run's too) refuses it as well, whatever
    built the config."""
    from benchflow.effort_request import EffortRequestError, with_route_effort

    monkeypatch.setenv(PIN_ENV, value)
    with pytest.raises(EffortRequestError, match="own environment"):
        with_route_effort(
            "codex-acp", "deepseek/deepseek-flash", "high", {"DEEPSEEK_API_KEY": "x"}
        )
    env = with_route_effort(
        "codex-acp",
        "deepseek/deepseek-flash",
        "high",
        {"DEEPSEEK_API_KEY": "x", PIN_ENV: "1"},
    )
    assert env[PIN_ENV] == "high"


# --- J7: pin off (the default) -----------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(WIRES))
async def test_pin_off_rewrites_nothing_and_names_the_effort_asked(proxy_log, path):
    body, read, _ = WIRES[path]
    got = await _through_guard(path, body)
    assert read(got) == read(body)
    (end,) = _ends(proxy_log)
    assert "effort_sent" not in end and "effort_pin" not in end
    assert end["effort_asked"] in {"low", "max"}


# --- the counts in model-calls.json ------------------------------------------


def _trial(tmp_path: Path, run_effort: str | None, ends: list[dict]) -> Path:
    trial = tmp_path / "trial"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({"reasoning_effort": run_effort}))
    row = {
        "request": {"method": "POST", "path": "/v1/responses", "body": {}},
        "response": {
            "status_code": 200,
            "body": {"model": "m", "status": "completed", "usage": {}},
        },
        "duration_ms": 1.0,
        "metadata": {"request_model": "m", "call_purpose": "agent"},
    }
    (trial / "trajectory" / "llm_trajectory.jsonl").write_text(json.dumps(row) + "\n")
    (trial / "trajectory" / "provider_requests.jsonl").write_text(
        "".join(
            json.dumps({"event": "request_end", "request_id": str(i), **end}) + "\n"
            for i, end in enumerate(ends)
        )
    )
    return trial


def test_pinned_trial_counts_its_rewrites(tmp_path):
    trial = _trial(
        tmp_path,
        "high",
        [
            {"status": 200, "effort_pin": "high", "effort_asked": "high"},
            {
                "status": 200,
                "effort_pin": "high",
                "effort_asked": "max",
                "effort_sent": "high",
            },
            {
                "status": 200,
                "effort_pin": "high",
                "effort_asked": "low",
                "effort_sent": "high",
            },
        ],
    )
    summary = write_model_calls(trial)
    assert summary is not None
    assert summary["effort_pin"] == "high"
    assert summary["effort_rewrites"] == 2
    assert summary["effort_mismatches"] == 0


def test_unpinned_trial_counts_requests_at_another_effort(tmp_path):
    trial = _trial(
        tmp_path,
        "high",
        [
            *({"status": 200, "effort_asked": "high"} for _ in range(10)),
            {"status": 200, "effort_asked": "max"},
            {"status": 200, "effort_asked": "max"},
            # A thinking budget with no level is the agent's own: no level
            # to compare with the run's.
            {"status": 200, "effort_asked": "budget_tokens=4096"},
            {"status": 200, "effort_asked": "budget_tokens=1024"},
            # A request the gateway refused ran at no effort.
            {
                "status": 400,
                "effort_asked": "low",
                "refused": {"rule": "model-not-served", "model": "x"},
            },
            # One that carried none is not counted either way.
            {"status": 200},
        ],
    )
    summary = write_model_calls(trial)
    assert summary is not None
    assert summary["effort_pin"] is None
    assert summary["effort_rewrites"] == 0
    assert summary["effort_mismatches"] == 2


def test_no_run_effort_leaves_nothing_to_compare(tmp_path):
    trial = _trial(tmp_path, None, [{"status": 200, "effort_asked": "max"}])
    summary = write_model_calls(trial)
    assert summary is not None
    assert summary["effort_mismatches"] is None


# --- the run setting ---------------------------------------------------------


def test_effort_pin_is_listed_by_help_all_only():
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    runner = CliRunner()
    short = runner.invoke(app, ["run", "--help"], terminal_width=200).output
    full = runner.invoke(app, ["run", "--help-all"], terminal_width=200).output
    assert "--effort-pin" not in short
    assert "--effort-pin" in full
    eval_short = runner.invoke(app, ["eval", "run", "--help"], terminal_width=200)
    assert "--effort-pin" not in eval_short.output


def test_a_run_config_s_effort_pin_rides_the_agent_env(tmp_path):
    from benchflow.evaluation import Evaluation

    tasks = tmp_path / "tasks" / "task-0"
    (tasks / "tests").mkdir(parents=True)
    (tasks / "task.toml").write_text('version = "1.0"\n[agent]\ntimeout_sec = 60\n')
    (tasks / "instruction.md").write_text("hi\n")
    (tasks / "tests" / "test.sh").write_text("#!/bin/sh\n")
    config = tmp_path / "run.yaml"
    config.write_text(
        f"tasks_dir: {tasks.parent}\njobs_dir: {tmp_path / 'jobs'}\n"
        "agent: codex-acp\nmodel: deepseek/deepseek-flash\n"
        "reasoning_effort: high\neffort_pin: true\n"
    )
    job = Evaluation.from_yaml(config)
    assert job._config.agent_env.get(PIN_ENV) == "1"


def test_the_pin_needs_an_effort_and_the_proxy():
    from benchflow.effort_request import (
        EffortRequestError,
        settle_effort,
        with_route_effort,
    )

    with pytest.raises(EffortRequestError, match="effort"):
        settle_effort("codex-acp", "deepseek/deepseek-flash", None, {PIN_ENV: "1"})
    env = with_route_effort(
        "codex-acp",
        "deepseek/deepseek-flash",
        "high",
        {PIN_ENV: "1", "DEEPSEEK_API_KEY": "sk-x"},
    )
    assert env[PIN_ENV] == "high"


@pytest.mark.parametrize(
    ("agent", "model", "env"),
    [
        # A Claude login, an endpoint the run names, a Codex ChatGPT login:
        # each goes around the proxy, so nothing there could pin.
        (
            "claude-agent-acp",
            "claude-opus-5-5",
            {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test"},
        ),
        (
            "claude-agent-acp",
            "xiaomi/mimo-v2.6-pro",
            {
                "ANTHROPIC_BASE_URL": "https://api.example.test/anthropic",
                "ANTHROPIC_AUTH_TOKEN": "test-token",
            },
        ),
        ("codex-acp", "openai/gpt-5.5", {"CODEX_ACCESS_TOKEN": "test-token"}),
    ],
)
def test_the_pin_is_refused_on_each_route_around_the_proxy(
    agent, model, env, monkeypatch, tmp_path
):
    from benchflow.agents import env as agent_env_module
    from benchflow.effort_request import EffortRequestError, with_route_effort

    for key in agent_env_module.inheritable_env_names():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(agent_env_module, "load_dotenv_env", lambda: {})
    monkeypatch.setattr(
        agent_env_module, "login_path", lambda name: tmp_path / Path(name).name
    )
    with pytest.raises(EffortRequestError, match="proxy"):
        with_route_effort(agent, model, "high", {PIN_ENV: "1", **env})


def test_run_card_records_the_effort_pin(tmp_path):
    from tests.test_subagent_model_pin import _card

    card, lines = _card(
        tmp_path,
        agent="codex-acp",
        model="deepseek/deepseek-flash",
        reasoning_effort="high",
        agent_env={PIN_ENV: "1"},
    )
    assert card["model"]["effort_pin"] is True
    assert any("Effort pin:" in line and "high" in line for line in lines), lines
    off, _ = _card(tmp_path / "off", agent="codex-acp", model="deepseek/deepseek-flash")
    assert off["model"]["effort_pin"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("shell", "run", "want"),
    [
        ("high", {}, None),
        ("1", {}, None),
        ("high", {PIN_ENV: "low"}, "low"),
        (None, {PIN_ENV: "max"}, "max"),
    ],
)
async def test_a_host_proxy_takes_the_pin_from_the_run_s_env_only(
    monkeypatch, shell, run, want
):
    """A host proxy starts from BenchFlow's own environment with the run's
    agent env over it; the pin is the run's alone, so every path that starts
    one (benchflow continue's among them) is covered, whatever checked the
    run before."""
    from benchflow.providers import litellm_runtime as runtime
    from benchflow.providers.litellm_config import resolve_litellm_route

    class Started(Exception):
        pass

    seen: dict[str, str] = {}

    async def litellm_for(*_args, **_kwargs):
        return "litellm"

    def daemon_proxy(*, env, **_kwargs):
        seen.update(env)
        raise Started

    monkeypatch.setattr(runtime, "_host_litellm_for", litellm_for)
    monkeypatch.setattr(runtime, "_daemon_proxy", daemon_proxy)
    monkeypatch.setattr(runtime, "_host_bind_address", lambda _environment: "127.0.0.1")
    if shell is None:
        monkeypatch.delenv(PIN_ENV, raising=False)
    else:
        monkeypatch.setenv(PIN_ENV, shell)
    agent_env = {"DEEPSEEK_API_KEY": "sk-x", **run}
    route = resolve_litellm_route("deepseek/deepseek-flash", agent_env)
    with pytest.raises(Started):
        await runtime._start_host_litellm(
            route=route,
            master_key="sk-runtime-test",
            agent_env=agent_env,
            environment="docker",
            session_id="s",
            agent_name="codex-acp",
        )
    assert seen.get(PIN_ENV) == want
