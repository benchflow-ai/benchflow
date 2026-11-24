"""Claude Code's sub-agents run on the run's model, on every route (benchflow#1246).

Claude Code's Agent tool takes a ``model`` argument (``haiku``, ``opus``)
that sends a sub-agent's calls to another model. Through BenchFlow's model
proxy those calls are refused (``model_not_served``); off the proxy (a Claude
login, an endpoint the run names) nothing stops them. Claude Code holds its
sub-agents to ``CLAUDE_CODE_SUBAGENT_MODEL`` only with
``CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1`` (probed on Claude Code 2.1.316), so
BenchFlow sets both to the run's model in Claude Code's environment unless the
run sets either itself, and the run card says which (``pinned``, ``run-set``
or ``off``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.agents import env as agent_env_module
from benchflow.agents.env import resolve_agent_env
from benchflow.agents.launch_env import agent_launch_env

PIN = "CLAUDE_CODE_SUBAGENT_MODEL"
FORCE = "CLAUDE_CODE_SUBAGENT_MODEL_FORCE"


@pytest.fixture(autouse=True)
def _no_operator_env(monkeypatch, tmp_path):
    """Nothing from the machine running the tests: no .env, no host login,
    none of the keys BenchFlow copies from the host."""
    for key in agent_env_module.inheritable_env_names():
        monkeypatch.delenv(key, raising=False)
    for key in (PIN, FORCE, "ANTHROPIC_MODEL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "no.env"))
    monkeypatch.setattr(agent_env_module, "load_dotenv_env", lambda: {})
    monkeypatch.setattr(
        agent_env_module, "login_path", lambda name: tmp_path / Path(name).name
    )


def _proxy_env(model: str, env: dict[str, str]) -> dict[str, str]:
    """The env the proxy route hands Claude Code (its wiring, then the
    launch filter), and the names that proxy's config serves."""
    from benchflow.providers.litellm_config import (
        litellm_proxy_config,
        resolve_litellm_route,
    )
    from benchflow.providers.litellm_runtime import (
        _apply_litellm_agent_env,
        _served_models_env,
    )

    route = resolve_litellm_route(model, env)
    wired = _apply_litellm_agent_env(
        agent="claude-agent-acp",
        agent_env=env,
        route=route,
        base_url="http://127.0.0.1:4000",
        master_key="sk-runtime-test",
    )
    import json

    served = json.loads(
        _served_models_env(litellm_proxy_config(route, master_key="sk-runtime-test"))[
            "BENCHFLOW_LITELLM_SERVED_MODELS"
        ]
    )
    return {**wired, "_served": json.dumps(served)}


# --- J3: the pin and FORCE on every route; a run's own setting wins ---------


def test_proxy_route_pins_subagents_to_the_main_loop_s_model_name():
    """Through the proxy the main loop asks for the proxy's name for the
    model (ANTHROPIC_MODEL, ``runtime-…``); the wiring gives the sub-agents
    the same name, so one trial asks for one model under one name, and the
    proxy's model gate serves it."""
    import json

    from benchflow.providers.litellm_config import safe_model_alias

    resolved = resolve_agent_env(
        "claude-agent-acp", "deepseek/deepseek-flash", {"DEEPSEEK_API_KEY": "sk-x"}
    )
    assert PIN not in resolved and FORCE not in resolved
    wired = _proxy_env("deepseek/deepseek-flash", resolved)
    assert wired["ANTHROPIC_MODEL"] == safe_model_alias("deepseek/deepseek-flash")
    assert (wired.get(PIN), wired.get(FORCE)) == (wired["ANTHROPIC_MODEL"], "1")
    assert wired[PIN] in json.loads(wired["_served"])


@pytest.mark.asyncio
async def test_claude_login_route_pins_subagents():
    from benchflow.providers.litellm_runtime import ensure_litellm_runtime

    model = "claude-opus-5-5"
    resolved = resolve_agent_env(
        "claude-agent-acp", model, {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test"}
    )
    launch, runtime = await ensure_litellm_runtime(
        agent="claude-agent-acp",
        agent_env=resolved,
        model=model,
        runtime=None,
        environment="docker",
        usage_tracking="off",
    )
    assert runtime is None  # a login goes around the proxy
    launch = agent_launch_env("claude-agent-acp", model, launch)
    assert (launch.get(PIN), launch.get(FORCE)) == ("claude-opus-5-5", "1")


@pytest.mark.asyncio
async def test_named_endpoint_route_pins_subagents_to_the_endpoint_s_model():
    from benchflow.providers.litellm_runtime import ensure_litellm_runtime

    model = "xiaomi/mimo-v2.6-pro"
    resolved = resolve_agent_env(
        "claude-agent-acp",
        model,
        {
            "ANTHROPIC_BASE_URL": "https://api.example.test/anthropic",
            "ANTHROPIC_AUTH_TOKEN": "test-token",
        },
    )
    launch, runtime = await ensure_litellm_runtime(
        agent="claude-agent-acp",
        agent_env=resolved,
        model=model,
        runtime=None,
        environment="docker",
        usage_tracking="off",
    )
    assert runtime is None
    launch = agent_launch_env("claude-agent-acp", model, launch)
    # The same id the main loop runs on there (ANTHROPIC_MODEL).
    assert launch["ANTHROPIC_MODEL"] == "mimo-v2.6-pro"
    assert (launch.get(PIN), launch.get(FORCE)) == ("mimo-v2.6-pro", "1")


@pytest.mark.parametrize(
    "given",
    [
        {PIN: "claude-haiku-5-5"},
        {FORCE: "0"},
        {PIN: "claude-sonnet-5-5", FORCE: "1"},
    ],
)
def test_a_run_s_own_setting_wins_and_nothing_is_added(given):
    resolved = resolve_agent_env(
        "claude-agent-acp",
        "deepseek/deepseek-flash",
        {"DEEPSEEK_API_KEY": "sk-x", **given},
    )
    wired = _proxy_env("deepseek/deepseek-flash", resolved)
    for env in (resolved, wired):
        assert {k: env.get(k) for k in (PIN, FORCE)} == {
            PIN: given.get(PIN),
            FORCE: given.get(FORCE),
        }


@pytest.mark.parametrize(
    ("model", "pinned"),
    [
        ("claude-opus-5-5", "claude-opus-5-5"),
        ("sonnet", "sonnet"),
        ("opus[1m]", "opus[1m]"),
        ("opusplan", "opusplan"),
        ("claude-opus-5-5[high]", "claude-opus-5-5"),
        ("claude-sonnet-5-5[1m][max]", "claude-sonnet-5-5[1m]"),
    ],
)
def test_a_login_s_pin_is_the_id_its_main_loop_runs(model, pinned):
    """On a Claude login the main loop runs the model ACP sets
    (acp.runtime._format_acp_model of the settled model): the pin is that id,
    a Claude Code alias as itself, a context size kept, never an effort."""
    from benchflow.acp.runtime import _format_acp_model
    from benchflow.effort_request import split_model_effort

    resolved = resolve_agent_env(
        "claude-agent-acp", model, {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test"}
    )
    settled, _ = split_model_effort(model)
    assert settled is not None
    assert resolved[PIN] == pinned == _format_acp_model(settled, "claude-agent-acp")
    assert resolved[FORCE] == "1"


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5-5",
        "anthropic/claude-opus-5-5",
        "anthropic/claude-sonnet-5-5[1m]",
        "sonnet",
        "claude-opus-5-5[high]",
    ],
)
def test_an_api_key_route_s_pin_is_served_by_its_proxy(model):
    """Through the proxy the main loop runs the proxy's alias for the model
    (ANTHROPIC_MODEL); the pin is a name the same config serves, so the
    model gate passes the sub-agent's calls, and never carries an effort."""
    import json

    from benchflow.effort_request import split_model_effort

    settled, _ = split_model_effort(model)
    assert settled is not None
    resolved = resolve_agent_env(
        "claude-agent-acp", settled, {"ANTHROPIC_API_KEY": "sk-ant-api03-test"}
    )
    wired = _proxy_env(settled, resolved)
    assert wired[PIN] == wired["ANTHROPIC_MODEL"], wired[PIN]
    assert "[high]" not in wired[PIN]
    assert wired[PIN] in json.loads(wired["_served"]), (wired[PIN], wired["_served"])
    assert wired[FORCE] == "1"


def test_the_native_harness_launches_with_the_pin():
    """The native CLI gets its launch env from the same resolution
    (native_harness.runtime.connect_native: apply_codex_launch_config, then
    check_model_route)."""
    from benchflow.agents.codex_config import apply_codex_launch_config
    from benchflow.native_harness.harnesses import native_harness_for
    from benchflow.native_harness.runtime import check_model_route

    model = "claude-opus-5-5"
    resolved = resolve_agent_env(
        "claude-agent-acp", model, {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test"}
    )
    launch, _ = apply_codex_launch_config(
        "claude-agent-acp", resolved, model=model, reasoning_effort=None, sandboxed=True
    )
    launch = check_model_route(
        native_harness_for("claude-agent-acp"), "claude-agent-acp", model, dict(launch)
    )
    assert (launch.get(PIN), launch.get(FORCE)) == ("claude-opus-5-5", "1")


def test_no_model_and_other_agents_get_no_pin():
    no_model = resolve_agent_env(
        "claude-agent-acp", None, {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test"}
    )
    assert PIN not in no_model and FORCE not in no_model
    codex = resolve_agent_env(
        "codex-acp", "deepseek/deepseek-flash", {"DEEPSEEK_API_KEY": "sk-x"}
    )
    assert PIN not in codex and FORCE not in codex


# --- J4: the run card records the pin's state --------------------------------


def _tasks(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    task = root / "task-0"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n[agent]\ntimeout_sec = 600\n[verifier]\ntimeout_sec = 120\n'
    )
    (task / "instruction.md").write_text("Write 42 to /app/answer.txt.\n")
    (task / "tests" / "test.sh").write_text("#!/bin/sh\n")
    return root


def _card(tmp_path: Path, **config) -> tuple[dict, list[str]]:
    from benchflow.evaluation import Evaluation, EvaluationConfig
    from benchflow.run_card import build_run_card, card_lines

    base = {"agent": "claude-agent-acp", "model": "claude-sonnet-5"}
    base.update(config)
    job = Evaluation(
        tasks_dir=str(_tasks(tmp_path)),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(**base),
        preflight=False,
    )
    card = build_run_card(job, job._get_task_dirs())
    return card, card_lines(card, tmp_path / "job")


def _setting(card: dict, name: str) -> dict | None:
    return next((s for s in card["settings"] if s["name"] == name), None)


def test_run_card_records_the_pin(tmp_path):
    from benchflow.providers.litellm_config import safe_model_alias

    # No login here, so the run goes through the proxy: the main loop's name
    # for the model there is the proxy's.
    alias = safe_model_alias("claude-sonnet-5")
    card, lines = _card(tmp_path)
    assert card["agent"]["subagent_model_pin"] == "pinned"
    assert card["agent"]["subagent_model"] == alias
    pin, force = _setting(card, PIN), _setting(card, FORCE)
    assert pin is not None and force is not None
    assert (pin["value"], pin["source"]) == (alias, "benchflow_default")
    assert (force["value"], force["source"]) == ("1", "benchflow_default")
    assert any(
        "Sub-agents:" in line and "claude-sonnet-5" in line and FORCE in line
        for line in lines
    ), lines


def test_run_card_names_a_login_s_pin_as_the_model_id(tmp_path):
    card, _ = _card(
        tmp_path, agent_env={"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test"}
    )
    assert card["agent"]["subagent_model"] == "claude-sonnet-5"
    pin = _setting(card, PIN)
    assert pin is not None and pin["value"] == "claude-sonnet-5"


def test_run_card_records_a_run_s_own_setting(tmp_path):
    card, lines = _card(tmp_path, agent_env={FORCE: "0"})
    assert card["agent"]["subagent_model_pin"] == "run-set"
    assert _setting(card, PIN) is None
    force = _setting(card, FORCE)
    assert force is not None and (force["value"], force["source"]) == (
        "0",
        "agent_env",
    )
    assert any("Sub-agents:" in line and "run's own" in line for line in lines), lines


def test_run_card_says_off_without_a_model_and_nothing_for_other_agents(tmp_path):
    from benchflow.agents.env import subagent_model_pin
    from benchflow.providers.litellm_config import safe_model_alias
    from benchflow.run_card import card_lines

    # With no --model the run has the agent's default, which is pinned (here
    # through the proxy, under the proxy's name for it).
    card, _ = _card(tmp_path / "a", model=None)
    assert card["agent"]["subagent_model_pin"] == "pinned"
    assert card["agent"]["subagent_model"] == safe_model_alias(card["model"]["id"])
    # Only a launch with no model at all (the SDK's) leaves it off.
    assert subagent_model_pin("claude-agent-acp", None, {}) == ("off", None)
    card["agent"].update(subagent_model_pin="off", subagent_model=None)
    lines = card_lines(card, tmp_path / "job")
    assert any("Sub-agents:" in line and "not pinned" in line for line in lines), lines
    codex, codex_lines = _card(
        tmp_path / "b", agent="codex-acp", model="deepseek/deepseek-flash"
    )
    assert codex["agent"]["subagent_model_pin"] is None
    assert not any("Sub-agents:" in line for line in codex_lines)
