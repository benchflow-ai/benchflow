"""Concrete rollout plane adapters.

``benchflow.rollout`` is the kernel: lifecycle orchestration, reward/result
assembly, and tree growth. This module is the composition boundary that binds
the kernel to today's concrete ACP, provider, sandbox, and environment-plane
implementations.
"""

from __future__ import annotations

import shlex
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

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
from benchflow.agents.registry import (
    AGENT_LAUNCH,
    AGENTS,
    is_explicit_raw_agent_command,
)
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.providers.runtime import (
    ensure_litellm_runtime,
    extract_usage,
    host_proxy_claim,
    release_host_proxy,
    reserve_host_proxy,
    stop_provider_runtime,
)
from benchflow.sandbox._exec_output import exec_raw
from benchflow.sandbox.egress_denylist import (
    EgressDenylist,
    start_egress_denylist,
    stop_egress_denylist,
)
from benchflow.sandbox.lockdown import (
    VERIFIER_BASELINE_PROBE,
    KeptRelease,
    _kill_sandbox_user_procs,
    _kill_sandbox_user_procs_keeping,
    _prepare_log_dirs,
    _resolve_locked_paths,
    _seed_verifier_workspace,
    _snapshot_build_config,
    block_cloud_metadata,
    check_loader_custody,
    cleanup_verifier_python_hooks,
    clear_verifier_output_dir,
    end_kept_services,
    enforce_agent_egress_firewall,
    enforce_sandbox_uid_egress,
    ensure_legacy_app_dir,
    kill_switched_accounts,
    lockdown_paths,
    probe_kept_service_freezer,
    release_kept_services,
    setup_sandbox_user,
)
from benchflow.sandbox.setup import (
    _create_environment,
    _inject_skills_into_dockerfile,
    _patch_docker_dind,
    override_dockerfile_base_image,
    stage_dockerfile_deps,
)


class DefaultRolloutPlanes:
    """Default bindings for the four concrete planes."""

    def agent_launch(
        self,
        agent: str,
        *,
        disallow_web_tools: bool,
        disallow_hosted_search: bool = False,
    ) -> str:
        launch = AGENT_LAUNCH.get(agent)
        if launch is None:
            if not is_explicit_raw_agent_command(agent):
                raise KeyError(f"Unknown agent: {agent!r}")
            launch = agent
        agent_cfg = AGENTS.get(agent)
        if agent_cfg is None:
            return launch
        if disallow_web_tools:
            return launch + agent_cfg.disallow_web_tools_launch_suffix
        if disallow_hosted_search:
            return launch + agent_cfg.disallow_hosted_search_launch_suffix
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

    def override_dockerfile_base_image(self, task_path: Path, base_image: str) -> int:
        return override_dockerfile_base_image(task_path, base_image)

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
        block_cloud_metadata: bool = False,
        cloud_metadata_root_agent: bool = False,
    ) -> Any:
        return _create_environment(
            environment,
            task,
            task_path,
            rollout_name or task_path.name,
            rollout_paths,
            preserve_agent_network=preserve_agent_network,
            environment_manifest=environment_manifest,
            block_cloud_metadata=block_cloud_metadata,
            cloud_metadata_root_agent=cloud_metadata_root_agent,
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
        guard_daemon: bool = True,
    ) -> str:
        return await setup_sandbox_user(
            env,
            sandbox_user,
            workspace=workspace,
            timeout_sec=timeout_sec,
            guard_daemon=guard_daemon,
        )

    async def snapshot_build_config(self, env: Any, *, workspace: str) -> None:
        await _snapshot_build_config(env, workspace=workspace)

    async def prepare_verifier_runtime(
        self, env: Any, *, task: Any, sandbox_user: str | None
    ) -> None:
        from benchflow.sandbox.lockdown import prepare_verifier_runtime

        await prepare_verifier_runtime(env, task, sandbox_user)

    async def prepare_separate_verifier_runtime(
        self, env: Any, *, task: Any, workspace: str
    ) -> None:
        from benchflow.sandbox.lockdown import prepare_separate_verifier_runtime

        await prepare_separate_verifier_runtime(env, task, workspace=workspace)

    async def seed_verifier_workspace(
        self, env: Any, *, workspace: str, sandbox_user: str | None
    ) -> None:
        await _seed_verifier_workspace(
            env, workspace=workspace, sandbox_user=sandbox_user
        )

    async def probe_snapshot_start(
        self, env: Any, *, agent_binary: str | None
    ) -> dict[str, bool]:
        """What a sandbox started from a branch snapshot already holds: the
        agent binary (``agent``) and the verifier's pre-agent baseline
        (``baseline``). One exec; anything unreadable counts as absent."""
        parts = [": benchflow-snapshot-start"]
        if agent_binary:
            quoted = shlex.quote(agent_binary)
            check = (
                f"test -x {quoted}"
                if agent_binary.startswith("/")
                else f"command -v {quoted} >/dev/null"
            )
            parts.append(f"{{ {check} && echo agent; }}")
        parts.append(f"{{ {VERIFIER_BASELINE_PROBE}; }}")
        result = await exec_raw(
            env.exec, "; ".join(parts) + "; true", user="root", timeout_sec=30
        )
        found = set((getattr(result, "stdout", "") or "").split())
        return {"agent": "agent" in found, "baseline": "baseline" in found}

    async def prepare_log_dirs(self, env: Any, *, sandbox_user: str | None) -> None:
        await _prepare_log_dirs(env, sandbox_user=sandbox_user)

    async def deploy_skills(self, *args: Any, **kwargs: Any) -> None:
        await deploy_skills(*args, **kwargs)

    async def lockdown_paths(self, env: Any, locked_paths: list[str]) -> None:
        await lockdown_paths(env, locked_paths)

    async def enforce_agent_egress_firewall(
        self, env: Any, sandbox_user: str | None, agent_env: dict[str, str]
    ) -> None:
        await enforce_agent_egress_firewall(env, sandbox_user, agent_env)

    async def install_agent(
        self,
        env: Any,
        agent: str,
        rollout_dir: Path,
        *,
        sandbox_setup_timeout: int = 120,
        install_layer_bypass_reason: str | None = None,
    ) -> Any:
        return await install_agent(
            env,
            agent,
            rollout_dir,
            sandbox_setup_timeout=sandbox_setup_timeout,
            install_layer_bypass_reason=install_layer_bypass_reason,
        )

    async def write_credential_files(self, *args: Any, **kwargs: Any) -> list[str]:
        return await write_credential_files(*args, **kwargs)

    async def upload_subscription_auth(self, *args: Any, **kwargs: Any) -> list[str]:
        return await upload_subscription_auth(*args, **kwargs)

    async def apply_web_tool_policy(self, *args: Any, **kwargs: Any) -> None:
        await apply_web_tool_policy(*args, **kwargs)

    async def link_skill_paths(self, *args: Any, **kwargs: Any) -> None:
        await _link_skill_paths(*args, **kwargs)

    async def ensure_litellm_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return await ensure_litellm_runtime(*args, **kwargs)

    def reserve_host_proxy(self, **kwargs: Any) -> Any:
        return reserve_host_proxy(**kwargs)

    def host_proxy_claim(self, reservation: Any) -> AbstractContextManager[None]:
        return host_proxy_claim(reservation)

    def release_host_proxy(self, reservation: Any) -> None:
        release_host_proxy(reservation)

    async def stop_provider_runtime(self, runtime: Any) -> None:
        await stop_provider_runtime(runtime)

    async def start_egress_denylist(
        self,
        env: Any,
        sandbox_user: str | None,
        denylist: EgressDenylist,
        *,
        model_gateway_url: str | None = None,
    ) -> None:
        if denylist.native_claude_model_only and sandbox_user is None:
            raise ValueError("Native OAuth transport requires a sandbox user")
        await start_egress_denylist(
            env, sandbox_user, denylist, model_gateway_url=model_gateway_url
        )
        if denylist.native_claude_model_only:
            assert sandbox_user is not None
            await enforce_sandbox_uid_egress(env, sandbox_user)

    async def stop_egress_denylist(self, env: Any, rollout_dir: Path) -> None:
        await stop_egress_denylist(env, rollout_dir)

    async def block_cloud_metadata(
        self, env: Any, sandbox_user: str | None
    ) -> dict[str, Any]:
        return await block_cloud_metadata(env, sandbox_user)

    def extract_usage(self, runtime: Any) -> dict[str, Any]:
        return extract_usage(runtime)

    async def connect_acp(self, *args: Any, **kwargs: Any) -> Any:
        # Here, not at the top: the ACP runtime loads the ACP SDK's schema
        # (about 0.2 s), which a rollout that starts no ACP agent never needs.
        from benchflow.acp.runtime import connect_acp

        return await connect_acp(*args, **kwargs)

    async def connect_native(self, *args: Any, **kwargs: Any) -> Any:
        from benchflow.native_harness.runtime import connect_native

        return await connect_native(*args, **kwargs)

    async def execute_prompts(self, *args: Any, **kwargs: Any) -> Any:
        from benchflow.acp.runtime import execute_prompts

        return await execute_prompts(*args, **kwargs)

    async def connect_session_factory(self, *args: Any, **kwargs: Any) -> Any:
        from benchflow.rollout.session_factory_runtime import (
            connect_session_factory,
        )

        return await connect_session_factory(*args, **kwargs)

    async def execute_prompts_session_factory(self, *args: Any, **kwargs: Any) -> Any:
        from benchflow.rollout.session_factory_runtime import (
            execute_prompts_session_factory,
        )

        return await execute_prompts_session_factory(*args, **kwargs)

    async def quiesce_agent(
        self, env: Any, sandbox_user: str, *, keep_ports: tuple[int, ...] = ()
    ) -> Any:
        # The accounts the task lets the agent switch to stop first, as the
        # agent user's processes do (K1-R11-1).
        await kill_switched_accounts(env)
        if not keep_ports:
            await _kill_sandbox_user_procs(env, sandbox_user)
            return None
        # The turn-end kill of a task that keeps services: the first kill of
        # the trial, so it waits for their listeners. They stay frozen while
        # the submission and the evidence are captured (K2-R1-A-3); the kill
        # before grading finds and keeps them, and the verifier lets them go
        # on just before its first command. Returns what it killed and kept,
        # for the trial's record.
        return await _kill_sandbox_user_procs_keeping(
            env, sandbox_user, keep_ports, wait=True
        )

    async def probe_kept_service_freezer(self, env: Any) -> None:
        # Before any agent code: a sandbox that cannot hold kept services is
        # refused now, not in the kill after the agent's turn (K1-R9-1).
        await probe_kept_service_freezer(env)

    async def check_loader_custody(
        self, env: Any, workspace: str, *, timeout_sec: int = 120
    ) -> None:
        # Before the agent's install: an image the seal is sure to refuse
        # for its loader files is refused now (DFR-p3-5).
        await check_loader_custody(env, workspace, timeout_sec=timeout_sec)

    async def release_kept_services(
        self,
        env: Any,
        sandbox_user: str,
        pids: tuple[int, ...],
        close: tuple[str, ...] = (),
    ) -> KeptRelease:
        return await release_kept_services(env, sandbox_user, pids, close)

    async def end_kept_services(
        self,
        env: Any,
        sandbox_user: str,
        opened: tuple[tuple[str, str, str], ...] = (),
    ) -> int:
        return await end_kept_services(env, sandbox_user, opened)

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

    async def soft_verifier_bash_env(self, *args: Any, **kwargs: Any) -> str:
        from benchflow.sandbox.lockdown import soft_verifier_bash_env

        return await soft_verifier_bash_env(*args, **kwargs)
