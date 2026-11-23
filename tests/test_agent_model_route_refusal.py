"""The claude agent runs MiMo, and an unroutable agent/model pair is refused
before a job exists (REPORT B11).

``--agent claude --model xiaomi/mimo-v2.6-pro`` failed inside the job ("Agent
'claude-agent-acp' requires provider protocol 'anthropic-messages', but provider
'xiaomi' ... only supports openai-completions") although MiMo's token plan
serves Anthropic Messages at ``${XIAOMI_BASE_URL%/v1}/anthropic``; and a pair
no provider serves created a job whose every trial errored.

Claude Code calls the gateway, which calls each route in the route's own
protocol, so no provider refuses it (``-a claude -m
openrouter/anthropic/claude-opus-5.5`` was refused, by the run and not its dry
run). Codex is still refused, by both, where the provider serves no Responses
API.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.agents.env import resolve_agent_env, resolve_provider_env
from benchflow.agents.providers import PROVIDERS, resolve_base_url
from benchflow.cli.main import app
from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.providers.litellm_config import resolve_litellm_route
from benchflow.rollout import RolloutConfig
from benchflow.runtime import check_rollout_config

MIMO = "xiaomi/mimo-v2.6-pro"
TOKEN_PLAN = "https://token-plan.example.test/v1"
# Z.ai's coding plan answers 404 on /responses: no Responses API for Codex.
UNROUTABLE = "zai-coding/glm-5.3-flash"
OPENROUTER_CLAUDE = "openrouter/anthropic/claude-opus-5.5"


def _xiaomi_env(base: str = TOKEN_PLAN) -> dict[str, str]:
    return {"XIAOMI_API_KEY": "mimo-test-key", "XIAOMI_BASE_URL": base}


@pytest.mark.parametrize(
    ("base", "anthropic"),
    [
        (TOKEN_PLAN, "https://token-plan.example.test/anthropic"),
        (f"{TOKEN_PLAN}/", "https://token-plan.example.test/anthropic"),
        ("https://mimo.example.test/api", "https://mimo.example.test/api/anthropic"),
    ],
)
def test_xiaomi_declares_its_anthropic_messages_endpoint(base, anthropic):
    xiaomi = PROVIDERS["xiaomi"]
    assert "anthropic-messages" in xiaomi.all_endpoints
    env = _xiaomi_env(base)
    assert resolve_base_url(xiaomi, env, protocol="anthropic-messages") == anthropic
    # The OpenAI surface is unchanged.
    assert resolve_base_url(xiaomi, env, protocol="openai-completions") == base


def test_the_claude_agent_resolves_mimo_to_its_anthropic_endpoint():
    env = _xiaomi_env()
    resolve_provider_env(env, MIMO, "claude-agent-acp")

    assert env["BENCHFLOW_PROVIDER_BASE_URL"] == (
        "https://token-plan.example.test/anthropic"
    )
    assert env["ANTHROPIC_BASE_URL"] == env["BENCHFLOW_PROVIDER_BASE_URL"]
    assert env["ANTHROPIC_MODEL"] == "mimo-v2.6-pro"
    assert env["BENCHFLOW_PROVIDER_PROTOCOL"] == "anthropic-messages"


def test_the_gateway_calls_mimos_openai_endpoint_for_the_claude_agent():
    """The LiteLLM gateway speaks OpenAI chat completions to MiMo whatever the
    agent speaks (as for zai-coding, PR #1074), so the agent's Anthropic URL
    must not become the gateway's upstream."""
    env = resolve_agent_env("claude-agent-acp", MIMO, _xiaomi_env())
    route = resolve_litellm_route(MIMO, env)

    assert route.upstream_model == "openai/mimo-v2.6-pro"
    assert route.litellm_params["api_base"] == TOKEN_PLAN
    assert route.litellm_params["api_key"] == "os.environ/XIAOMI_API_KEY"
    assert route.required_env == ("XIAOMI_API_KEY",)


def test_the_mimo_agents_route_is_unchanged():
    env = resolve_agent_env("mimo", MIMO, _xiaomi_env())
    route = resolve_litellm_route(MIMO, env)

    assert env["BENCHFLOW_PROVIDER_BASE_URL"] == TOKEN_PLAN
    assert route.upstream_model == "openai/mimo-v2.6-pro"
    assert route.litellm_params["api_base"] == TOKEN_PLAN


def test_an_explicit_gateway_base_still_wins_for_xiaomi():
    env = {
        **_xiaomi_env(),
        "BENCHFLOW_PROVIDER_BASE_URL": "https://proxy.example.test/v1",
        "BENCHFLOW_PROVIDER_API_KEY": "proxy-key",
    }
    resolve_provider_env(env, MIMO, "claude-agent-acp")
    route = resolve_litellm_route(MIMO, env)

    assert route.litellm_params["api_base"] == "https://proxy.example.test/v1"
    assert route.litellm_params["api_key"] == "os.environ/BENCHFLOW_PROVIDER_API_KEY"


# ── Refused before the job ──────────────────────────────────────────────────


def _task(root: Path) -> Path:
    task = root / "task"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )
    (task / "instruction.md").write_text("Do it.\n")
    return task


@pytest.mark.parametrize("agent", ["codex-acp", "codex"])
def test_bench_eval_run_refuses_an_unroutable_pair_before_a_job_exists(
    tmp_path, monkeypatch, agent
):
    started: list[object] = []
    monkeypatch.setattr(
        "benchflow.cli.main.run_batch_eval", lambda *a, **k: started.append(a)
    )
    task = _task(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            agent,
            "--model",
            UNROUTABLE,
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--agent-env",
            "ZAI_API_KEY=test-key",
        ],
    )

    assert result.exit_code == 1, result.output
    assert started == []
    assert not (tmp_path / "jobs").exists()
    assert "requires provider protocol 'openai-responses'" in result.stderr
    assert "No job was created" in result.stderr


def test_bench_eval_run_lets_claude_run_mimo(tmp_path, monkeypatch):
    started: list[object] = []
    monkeypatch.setattr(
        "benchflow.cli.main.run_batch_eval", lambda *a, **k: started.append(a)
    )
    task = _task(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            "claude",
            "--model",
            MIMO,
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--agent-env",
            "XIAOMI_API_KEY=test-key",
            "--agent-env",
            f"XIAOMI_BASE_URL={TOKEN_PLAN}",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(started) == 1


def test_evaluation_run_refuses_an_unroutable_pair_before_a_job_exists(tmp_path):
    task = _task(tmp_path / "tasks")
    evaluation = Evaluation(
        tasks_dir=task.parent,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent="codex-acp", model=UNROUTABLE, environment="docker"
        ),
    )

    with pytest.raises(ValueError, match="requires provider protocol"):
        evaluation._check_before_run()
    assert not (tmp_path / "jobs").exists()


def test_a_single_rollout_refuses_an_unroutable_pair(tmp_path):
    task = _task(tmp_path)
    with pytest.raises(
        ValueError, match="requires provider protocol 'openai-responses'"
    ):
        check_rollout_config(
            RolloutConfig(
                task_path=task,
                agent="codex-acp",
                model=UNROUTABLE,
                environment="docker",
            )
        )


@pytest.mark.parametrize(
    ("agent", "model"),
    [
        ("oracle", UNROUTABLE),
        ("claude-agent-acp", "claude-haiku-4-5"),
        ("claude-agent-acp", "zai/glm-5"),
        ("claude-agent-acp", "vllm/Qwen/Qwen3-Coder"),
        ("claude-agent-acp", UNROUTABLE),
        ("claude", UNROUTABLE),
        ("claude-agent-acp", OPENROUTER_CLAUDE),
        ("claude", OPENROUTER_CLAUDE),
    ],
)
def test_routable_pairs_pass(tmp_path, agent, model):
    task = _task(tmp_path)
    check_rollout_config(
        RolloutConfig(task_path=task, agent=agent, model=model, environment="docker")
    )


# ── Claude Code on a provider that serves no Anthropic Messages ────


def test_the_gateway_calls_openrouter_chat_completions_for_claude_code():
    """Claude Code's Messages calls go to the gateway, which calls OpenRouter's
    chat completions: the route of ``bench run -a claude -m
    openrouter/anthropic/claude-opus-5.5 -e max``, refused before the run."""
    env = resolve_agent_env(
        "claude-agent-acp", OPENROUTER_CLAUDE, {"OPENROUTER_API_KEY": "or-test-key"}
    )
    route = resolve_litellm_route(OPENROUTER_CLAUDE, env)

    assert route.upstream_model == "openai/anthropic/claude-opus-5.5"
    assert route.litellm_params["api_base"] == "https://openrouter.ai/api/v1"


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("agent", ["claude", "claude-agent-acp"])
def test_bench_run_takes_claude_code_on_openrouter(
    tmp_path, monkeypatch, agent, dry_run
):
    started: list[object] = []
    monkeypatch.setattr(
        "benchflow.cli.main.run_batch_eval", lambda *a, **k: started.append(a)
    )
    task = _task(tmp_path)
    args = [
        "run",
        str(task),
        "-a",
        agent,
        "-m",
        OPENROUTER_CLAUDE,
        "-e",
        "max",
        "--sandbox",
        "docker",
        "--jobs-dir",
        str(tmp_path / "jobs"),
        "--agent-env",
        "OPENROUTER_API_KEY=or-test-key",
    ]

    result = CliRunner().invoke(app, [*args, *(["--dry-run"] if dry_run else [])])

    assert result.exit_code == 0, result.output
    assert len(started) == (0 if dry_run else 1)


@pytest.mark.parametrize("dry_run", [True, False])
def test_bench_run_and_its_dry_run_refuse_the_same_pair(tmp_path, monkeypatch, dry_run):
    started: list[object] = []
    monkeypatch.setattr(
        "benchflow.cli.main.run_batch_eval", lambda *a, **k: started.append(a)
    )
    task = _task(tmp_path)
    args = [
        "run",
        str(task),
        "-a",
        "codex",
        "-m",
        UNROUTABLE,
        "--sandbox",
        "docker",
        "--jobs-dir",
        str(tmp_path / "jobs"),
        "--agent-env",
        "ZAI_API_KEY=test-key",
    ]

    result = CliRunner().invoke(app, [*args, *(["--dry-run"] if dry_run else [])])

    assert result.exit_code == 1, result.output
    assert started == []
    assert not (tmp_path / "jobs").exists()
    assert "requires provider protocol 'openai-responses'" in result.stderr
    assert "No job was created" in result.stderr
