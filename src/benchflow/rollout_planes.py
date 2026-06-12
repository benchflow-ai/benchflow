"""Concrete rollout plane adapters.

``benchflow.rollout`` is the kernel: lifecycle orchestration, reward/result
assembly, and tree growth. This module is the composition boundary that binds
the kernel to today's concrete ACP, provider, sandbox, and environment-plane
implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchflow.acp.runtime import connect_acp, execute_prompts
from benchflow.agents.credentials import (
    upload_subscription_auth,
    write_credential_files,
)
from benchflow.agents.env import resolve_agent_env
from benchflow.agents.install import (
    _link_skill_paths,
    apply_web_tool_policy,
    deploy_skills,
    install_agent,
)
from benchflow.agents.registry import AGENT_LAUNCH, AGENTS
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.providers.runtime import (
    ensure_litellm_runtime,
    extract_usage,
    stop_provider_runtime,
)
from benchflow.sandbox.lockdown import (
    _resolve_locked_paths,
    _seed_verifier_workspace,
    _snapshot_build_config,
    cleanup_verifier_python_hooks,
    clear_verifier_output_dir,
    ensure_legacy_app_dir,
    lockdown_paths,
    setup_sandbox_user,
)
from benchflow.sandbox.setup import (
    _create_environment,
    _inject_skills_into_dockerfile,
    _patch_docker_dind,
    stage_dockerfile_deps,
)


class DefaultRolloutPlanes:
    """Default bindings for the four concrete planes."""

    def agent_launch(self, agent: str, *, disallow_web_tools: bool) -> str:
        launch = AGENT_LAUNCH.get(agent, agent)
        if not disallow_web_tools:
            return launch
        agent_cfg = AGENTS.get(agent)
        if agent_cfg and agent_cfg.disallow_web_tools_launch_suffix:
            return launch + agent_cfg.disallow_web_tools_launch_suffix
        return launch

    def agent_config(self, agent: str) -> Any:
        return AGENTS.get(agent)

    def resolve_agent_env(
        self,
        agent: str,
        model: str | None,
        agent_env: dict[str, str] | None,
    ) -> dict[str, str]:
        return resolve_agent_env(agent, model, agent_env)

    def install_docker_compat(self) -> None:
        _patch_docker_dind()

    def resolve_locked_paths(
        self, sandbox_user: str | None, locked_paths: list[str] | None
    ) -> list[str]:
        return _resolve_locked_paths(sandbox_user, locked_paths)

    def stage_dockerfile_deps(self, task_path: Path, context_root: Path) -> None:
        stage_dockerfile_deps(task_path, context_root)

    def inject_skills_into_dockerfile(
        self, task_path: Path, skills_dir: Path, *, sandbox_dir: str = "/skills"
    ) -> None:
        _inject_skills_into_dockerfile(task_path, skills_dir, sandbox_dir=sandbox_dir)

    def create_environment(
        self,
        environment: str,
        task: Any,
        task_path: Path,
        rollout_name: str | None,
        rollout_paths: Any,
        *,
        preserve_agent_network: bool,
        environment_manifest: EnvironmentManifest | None,
    ) -> Any:
        return _create_environment(
            environment,
            task,
            task_path,
            rollout_name or task_path.name,
            rollout_paths,
            preserve_agent_network=preserve_agent_network,
            environment_manifest=environment_manifest,
        )

    def manifest_environment(
        self, manifest: EnvironmentManifest, *, sandbox: Any
    ) -> ManifestEnvironment:
        return ManifestEnvironment(manifest, sandbox=sandbox)

    async def setup_sandbox_user(
        self,
        env: Any,
        sandbox_user: str,
        *,
        workspace: str,
        timeout_sec: int = 120,
    ) -> str:
        return await setup_sandbox_user(
            env,
            sandbox_user,
            workspace=workspace,
            timeout_sec=timeout_sec,
        )

    async def snapshot_build_config(self, env: Any, *, workspace: str) -> None:
        await _snapshot_build_config(env, workspace=workspace)

    async def seed_verifier_workspace(
        self, env: Any, *, workspace: str, sandbox_user: str | None
    ) -> None:
        await _seed_verifier_workspace(
            env, workspace=workspace, sandbox_user=sandbox_user
        )

    async def deploy_skills(self, *args: Any, **kwargs: Any) -> None:
        await deploy_skills(*args, **kwargs)

    async def lockdown_paths(self, env: Any, locked_paths: list[str]) -> None:
        await lockdown_paths(env, locked_paths)

    async def install_agent(
        self,
        env: Any,
        agent: str,
        rollout_dir: Path,
        *,
        sandbox_setup_timeout: int = 120,
    ) -> Any:
        return await install_agent(
            env, agent, rollout_dir, sandbox_setup_timeout=sandbox_setup_timeout
        )

    async def write_credential_files(self, *args: Any, **kwargs: Any) -> None:
        await write_credential_files(*args, **kwargs)

    async def upload_subscription_auth(self, *args: Any, **kwargs: Any) -> None:
        await upload_subscription_auth(*args, **kwargs)

    async def apply_web_tool_policy(self, *args: Any, **kwargs: Any) -> None:
        await apply_web_tool_policy(*args, **kwargs)

    async def link_skill_paths(self, *args: Any, **kwargs: Any) -> None:
        await _link_skill_paths(*args, **kwargs)

    async def ensure_litellm_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return await ensure_litellm_runtime(*args, **kwargs)

    async def stop_provider_runtime(self, runtime: Any) -> None:
        await stop_provider_runtime(runtime)

    def extract_usage(self, runtime: Any) -> dict[str, Any]:
        return extract_usage(runtime)

    async def connect_acp(self, *args: Any, **kwargs: Any) -> Any:
        return await connect_acp(*args, **kwargs)

    async def execute_prompts(self, *args: Any, **kwargs: Any) -> Any:
        return await execute_prompts(*args, **kwargs)

    async def harden_before_verify(self, *args: Any, **kwargs: Any) -> None:
        from benchflow.sandbox.lockdown import harden_before_verify as _harden

        await _harden(*args, **kwargs)

    def verifier(self, *args: Any, **kwargs: Any) -> Any:
        from benchflow.task import Verifier as _Verifier

        return _Verifier(*args, **kwargs)

    async def clear_verifier_output_dir(self, *args: Any, **kwargs: Any) -> None:
        await clear_verifier_output_dir(*args, **kwargs)

    async def ensure_legacy_app_dir(self, *args: Any, **kwargs: Any) -> None:
        await ensure_legacy_app_dir(*args, **kwargs)

    async def cleanup_verifier_python_hooks(self, *args: Any, **kwargs: Any) -> None:
        await cleanup_verifier_python_hooks(*args, **kwargs)
