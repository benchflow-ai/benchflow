"""Rollout kernel architecture tests."""

from __future__ import annotations

import ast
import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from benchflow.agents.registry import (
    AGENT_INSTALLERS,
    AGENT_LAUNCH,
    AGENTS,
    AgentConfig,
    register_agent,
)
from benchflow.rollout import (
    Rollout,
    RolloutConfig,
    _build_rollout_result,
    _write_config,
)
from benchflow.skill_policy import SKILL_MODE_NO_SKILL, resolve_task_skill_policy

# ``rollout.py`` was split into the ``benchflow.rollout`` package; the kernel
# invariant now spans every module in it.
ROLL_OUT = Path("src/benchflow/rollout")

CONCRETE_PLANE_MODULES = {
    "benchflow.acp.client",
    "benchflow.acp.runtime",
    "benchflow.environment.manifest_env",
    "benchflow.providers.runtime",
    "benchflow.sandbox.daytona",
    "benchflow.sandbox.lockdown",
    "benchflow.sandbox.setup",
    "benchflow.sandbox.user",
}

COMPOSITION_BOUNDARY_MODULES = {
    "benchflow.acp.runtime",
    "benchflow.environment.manifest_env",
    "benchflow.providers.runtime",
    "benchflow.sandbox.lockdown",
    "benchflow.sandbox.setup",
}


def _imported_modules(path: Path) -> set[str]:
    sources = sorted(path.glob("*.py")) if path.is_dir() else [path]
    modules: set[str] = set()
    for source in sources:
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
    return modules


def test_rollout_kernel_does_not_import_concrete_planes() -> None:
    """Guards the fix from PR #515 for issue #415: rollout imports contracts."""
    imported = _imported_modules(ROLL_OUT)

    assert imported.isdisjoint(CONCRETE_PLANE_MODULES)
    assert "benchflow.contracts" in imported
    assert "benchflow.rollout_planes" not in imported


def test_concrete_plane_bindings_live_at_composition_boundary() -> None:
    """Guards the fix from PR #515 for issue #415: concrete imports stay outside."""
    imported = _imported_modules(Path("src/benchflow/rollout_planes.py"))

    assert imported >= COMPOSITION_BOUNDARY_MODULES


@pytest.fixture
def explicit_policy_agent():
    name = "explicit-policy-rollout-test"
    config = register_agent(
        name,
        install_cmd="true",
        launch_cmd="true",
        requires_env=["DEEPSEEK_API_KEY"],
        environment_policy="explicit",
    )
    yield config
    AGENTS.pop(name, None)
    AGENT_INSTALLERS.pop(name, None)
    AGENT_LAUNCH.pop(name, None)


@pytest.mark.asyncio
async def test_runtime_credentials_are_keyword_only_and_isolated_per_rollout(
    tmp_path: Path, explicit_policy_agent
) -> None:
    async def resolve(secret: str) -> dict[str, str]:
        config = RolloutConfig(
            task_path=tmp_path,
            agent=explicit_policy_agent.name,
            agent_env={
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
                "OPENAI_API_KEY": "configured-openai",
                "AWS_ACCESS_KEY_ID": "configured-aws",
                "DATABASE_URL": "postgresql://ambient.example/db",
                "PATH": "/unreviewed/bin",
                "LD_PRELOAD": "/tmp/inject.so",
            },
        )
        rollout = await Rollout.create(
            config,
            runtime_credentials={
                "DEEPSEEK_API_KEY": secret,
                "OPENAI_API_KEY": f"unrelated-{secret}",
                "AWS_ACCESS_KEY_ID": f"unrelated-aws-{secret}",
            },
        )
        return rollout._resolve_agent_environment(
            config.primary_agent,
            config.primary_model,
            config.agent_env,
        )

    first, second = await asyncio.gather(
        resolve("runtime-secret-1"),
        resolve("runtime-secret-2"),
    )

    assert first["DEEPSEEK_API_KEY"] == "runtime-secret-1"
    assert second["DEEPSEEK_API_KEY"] == "runtime-secret-2"
    for resolved_env in (first, second):
        assert "OPENAI_API_KEY" not in resolved_env
        assert "AWS_ACCESS_KEY_ID" not in resolved_env
        assert "DATABASE_URL" not in resolved_env
        assert "PATH" not in resolved_env
        assert "LD_PRELOAD" not in resolved_env
        assert resolved_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"

    config = RolloutConfig(task_path=tmp_path, agent=explicit_policy_agent.name)
    with pytest.raises(TypeError):
        await Rollout.create(config, {"DEEPSEEK_API_KEY": "positional-secret"})


@pytest.mark.asyncio
async def test_runtime_credentials_never_enter_serializable_surfaces(
    tmp_path: Path, caplog, explicit_policy_agent
) -> None:
    secret = "runtime-secret-never-serialize"
    deepseek_secret = "runtime-deepseek-secret"
    explicit_policy_agent.requires_env.append("LICENSE")
    supplied = {
        "DEEPSEEK_API_KEY": deepseek_secret,
        "LICENSE": secret,
    }
    config = RolloutConfig(
        task_path=tmp_path,
        agent=explicit_policy_agent.name,
        agent_env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"},
    )
    rollout = await Rollout.create(config, runtime_credentials=supplied)
    supplied["DEEPSEEK_API_KEY"] = "mutated-after-create"

    resolved_env = rollout._resolve_agent_environment(
        config.primary_agent,
        config.primary_model,
        config.agent_env,
    )
    assert resolved_env["DEEPSEEK_API_KEY"] == deepseek_secret
    assert resolved_env["LICENSE"] == secret

    skill_policy = resolve_task_skill_policy(
        task_path=tmp_path,
        skill_mode=SKILL_MODE_NO_SKILL,
        runtime_skills_dir=None,
        declared_sandbox_skills_dir=None,
    )
    _write_config(
        tmp_path,
        task_path=tmp_path,
        agent=config.primary_agent,
        model=config.primary_model,
        environment=config.environment,
        environment_policy=explicit_policy_agent.environment_policy,
        skill_policy=skill_policy,
        sandbox_user=config.sandbox_user,
        context_root=None,
        timeout=60,
        started_at=datetime(2026, 1, 1),
        agent_env=resolved_env,
        runtime_credential_names=set(supplied),
        runtime_credential_values={secret, deepseek_secret},
    )
    _build_rollout_result(
        tmp_path,
        task_name="explicit-policy-task",
        rollout_name="rollout",
        agent=config.primary_agent,
        agent_name=config.primary_agent,
        model=None,
        n_tool_calls=0,
        prompts=[],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards=None,
        started_at=datetime(2026, 1, 1),
        timing={},
        skill_policy=skill_policy,
    )

    recorded_config = json.loads((tmp_path / "config.json").read_text())
    assert recorded_config["environment_policy"] == "explicit"
    assert "DEEPSEEK_API_KEY" not in recorded_config["agent_env"]
    assert "LICENSE" not in recorded_config["agent_env"]

    serialized_surfaces = "\n".join(
        [
            (tmp_path / "config.json").read_text(),
            (tmp_path / "result.json").read_text(),
            repr(config),
            repr(rollout),
            caplog.text,
        ]
    )
    assert secret not in serialized_surfaces
    assert deepseek_secret not in serialized_surfaces

    missing = await Rollout.create(
        config,
        runtime_credentials={"OPENAI_API_KEY": "unrelated-runtime-secret"},
    )
    with pytest.raises(ValueError) as exc_info:
        missing._resolve_agent_environment(
            config.primary_agent,
            config.primary_model,
            config.agent_env,
        )
    assert secret not in str(exc_info.value)
    assert "unrelated-runtime-secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_plane_owned_explicit_policy_controls_resolution_and_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    agent = "plane-owned-explicit-agent"
    plane_config = AgentConfig(
        name=agent,
        install_cmd="true",
        launch_cmd="true",
        requires_env=["LICENSE"],
        environment_policy="explicit",
    )
    config = RolloutConfig(
        task_path=tmp_path,
        agent=agent,
        agent_env={
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
            "PATH": "/unreviewed/bin",
        },
    )
    rollout = await Rollout.create(
        config,
        runtime_credentials={"LICENSE": "plane-runtime-secret"},
    )
    monkeypatch.setattr(
        rollout._planes,
        "agent_config",
        lambda requested: plane_config if requested == agent else None,
    )
    monkeypatch.setattr(
        rollout._planes,
        "resolve_agent_env",
        lambda _agent, _model, env: {
            **dict(env or {}),
            "DATABASE_URL": "postgresql://plane.example/db",
        },
    )

    resolved = rollout._resolve_agent_environment(
        config.primary_agent,
        config.primary_model,
        config.agent_env,
    )

    assert resolved == {
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
        "LICENSE": "plane-runtime-secret",
    }
    assert rollout._agent_environment_policy(agent) == "explicit"
