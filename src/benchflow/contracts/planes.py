"""Rollout composition-plane contract.

The rollout kernel owns lifecycle orchestration. Concrete ACP, sandbox,
provider, and environment implementations are bound through this protocol at
the composition edge so the kernel does not import those concrete modules.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from benchflow.environment.manifest import EnvironmentManifest

if TYPE_CHECKING:
    from benchflow.sandbox.egress_denylist import EgressDenylist


class LiveUsageGateway(Protocol):
    """A provider gateway process that reports live cumulative token usage.

    The eval dashboard's mid-prompt token signal reads this accessor through
    the rollout kernel (``Rollout.activity_snapshot`` →
    ``_gateway_live_tokens``), which cannot import the concrete provider
    plane — so the NAME is agreed here, at the contract boundary.
    ``LiteLLMProcess`` asserts static conformance in
    ``benchflow.providers.litellm_runtime``; renaming the accessor on either
    side fails the type checker there instead of silently blanking the
    dashboard signal.
    """

    def live_usage_tokens(self) -> int | None: ...


@runtime_checkable
class RolloutPlanes(Protocol):
    """Concrete-plane operations the rollout kernel needs."""

    def agent_launch(
        self,
        agent: str,
        *,
        disallow_web_tools: bool,
        disallow_hosted_search: bool = False,
    ) -> str: ...
    def agent_config(self, agent: str) -> Any: ...

    def resolve_agent_env(
        self,
        agent: str,
        model: str | None,
        agent_env: dict[str, str] | None,
    ) -> dict[str, str]: ...

    def install_docker_compat(self) -> None: ...

    def resolve_locked_paths(
        self, sandbox_user: str | None, locked_paths: list[str] | None
    ) -> list[str]: ...

    def stage_dockerfile_deps(self, task_path: Path, context_root: Path) -> None: ...
    def override_dockerfile_base_image(
        self, task_path: Path, base_image: str
    ) -> int: ...
    def inject_skills_into_dockerfile(
        self, task_path: Path, skills_dir: Path, *, sandbox_dir: str = "/skills"
    ) -> None: ...

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
    ) -> Any: ...

    def manifest_environment(
        self, manifest: EnvironmentManifest, *, sandbox: Any
    ) -> Any: ...

    async def setup_sandbox_user(
        self,
        env: Any,
        sandbox_user: str,
        *,
        workspace: str,
        agents: list[str] | None = None,
        timeout_sec: int = 120,
        guard_daemon: bool = True,
    ) -> str: ...

    async def snapshot_build_config(self, env: Any, *, workspace: str) -> None: ...
    async def seed_verifier_workspace(
        self, env: Any, *, workspace: str, sandbox_user: str | None
    ) -> None: ...
    async def prepare_verifier_runtime(
        self, env: Any, *, task: Any, sandbox_user: str | None
    ) -> None: ...
    async def prepare_separate_verifier_runtime(
        self, env: Any, *, task: Any, workspace: str
    ) -> None: ...
    async def deploy_skills(self, *args: Any, **kwargs: Any) -> None: ...
    async def lockdown_paths(self, env: Any, locked_paths: list[str]) -> None: ...
    async def prepare_log_dirs(self, env: Any, *, sandbox_user: str | None) -> None: ...
    async def probe_snapshot_start(
        self, env: Any, *, agent_binary: str | None
    ) -> dict[str, bool]: ...
    async def install_agent(
        self,
        env: Any,
        agent: str,
        rollout_dir: Path,
        *,
        sandbox_setup_timeout: int = 120,
    ) -> Any: ...
    async def write_credential_files(
        self, *args: Any, **kwargs: Any
    ) -> list[str] | None: ...
    async def upload_subscription_auth(
        self, *args: Any, **kwargs: Any
    ) -> list[str] | None: ...
    async def apply_web_tool_policy(self, *args: Any, **kwargs: Any) -> None: ...
    async def link_skill_paths(self, *args: Any, **kwargs: Any) -> None: ...
    async def ensure_litellm_runtime(self, *args: Any, **kwargs: Any) -> Any: ...
    async def stop_provider_runtime(self, runtime: Any) -> None: ...
    async def start_egress_denylist(
        self,
        env: Any,
        sandbox_user: str | None,
        denylist: EgressDenylist,
        *,
        model_gateway_url: str | None = None,
    ) -> None:
        """Return only after the proxy is healthy and its policy is enforced.

        Native model-only policies also require sandbox-UID direct egress to be
        blocked before returning. Raise on failure; callers must not start ACP
        or publish successful admission until this method completes.
        """
        ...

    async def stop_egress_denylist(self, env: Any, rollout_dir: Path) -> None: ...
    async def enforce_agent_egress_firewall(
        self, env: Any, sandbox_user: str | None, agent_env: dict[str, str]
    ) -> None: ...
    async def enforce_offline_uid_egress(self, env: Any, sandbox_user: str) -> None:
        """The loopback-only uid firewall, DNS refused, for an agent with no model route."""
        ...

    async def block_cloud_metadata(
        self, env: Any, sandbox_user: str | None
    ) -> dict[str, Any]: ...
    def extract_usage(self, runtime: Any) -> dict[str, Any]: ...
    async def connect_acp(self, *args: Any, **kwargs: Any) -> Any: ...
    async def connect_native(self, *args: Any, **kwargs: Any) -> Any:
        """Connect the agent's own CLI (``harness="native"``); same slots as ACP."""
        ...

    async def execute_prompts(self, *args: Any, **kwargs: Any) -> Any: ...
    async def connect_session_factory(self, *args: Any, **kwargs: Any) -> Any: ...
    async def execute_prompts_session_factory(
        self, *args: Any, **kwargs: Any
    ) -> Any: ...
    async def quiesce_agent(
        self, env: Any, sandbox_user: str, *, keep_ports: tuple[int, ...] = ()
    ) -> Any: ...
    async def probe_kept_service_freezer(self, env: Any) -> None: ...
    async def check_loader_custody(
        self, env: Any, workspace: str, *, timeout_sec: int = 120
    ) -> None: ...
    async def release_kept_services(
        self,
        env: Any,
        sandbox_user: str,
        pids: tuple[int, ...],
        close: tuple[str, ...] = (),
    ) -> Any: ...
    async def end_kept_services(
        self,
        env: Any,
        sandbox_user: str,
        opened: tuple[tuple[str, str, str], ...] = (),
    ) -> int: ...
    async def harden_before_verify(self, *args: Any, **kwargs: Any) -> None: ...
    def verifier(self, *args: Any, **kwargs: Any) -> Any: ...
    async def clear_verifier_output_dir(self, *args: Any, **kwargs: Any) -> None: ...
    async def ensure_legacy_app_dir(self, *args: Any, **kwargs: Any) -> None: ...
    async def cleanup_verifier_python_hooks(
        self, *args: Any, **kwargs: Any
    ) -> None: ...


def default_rollout_planes() -> RolloutPlanes:
    """Construct the default concrete plane bundle at the composition edge."""

    from benchflow.rollout_planes import DefaultRolloutPlanes

    return DefaultRolloutPlanes()


async def install_agent_with_policy(
    planes: RolloutPlanes,
    env: Any,
    agent: str,
    rollout_dir: Path,
    *,
    sandbox_setup_timeout: int = 120,
    install_layer_bypass_reason: str | None = None,
) -> Any:
    """Pass cache policy to the default adapter at the composition edge.

    Caller-supplied install implementations retain their calling convention
    and own their installation behavior, just as before cached layers.
    """
    from benchflow.rollout_planes import DefaultRolloutPlanes

    # Instance overrides can differ from the class method. Check and invoke
    # the same selected callable so only the default adapter gets cache policy.
    install = planes.install_agent
    if (
        isinstance(install, MethodType)
        and install.__func__ is DefaultRolloutPlanes.install_agent
    ):
        return await install(
            env,
            agent,
            rollout_dir,
            sandbox_setup_timeout=sandbox_setup_timeout,
            install_layer_bypass_reason=install_layer_bypass_reason,
        )
    return await install(
        env, agent, rollout_dir, sandbox_setup_timeout=sandbox_setup_timeout
    )
