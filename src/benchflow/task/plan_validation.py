"""The sandbox-free task plan gate used by check and rollout setup.

Keep policy in its owning validators: this function only puts the run's
gates in one order, and lets their original exceptions and messages through.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchflow.sandbox.egress_denylist import EgressDenylist
    from benchflow.task.task import Task


def agent_egress_policy(
    task: Task,
    *,
    agent: str,
    sandbox_user: str | None,
    disallow_web_tools: bool = False,
) -> EgressDenylist | None:
    """Resolve the run's agent-only egress policy and its user prerequisite."""
    from benchflow.agents.registry import is_operator_script, is_scripted_agent
    from benchflow.sandbox.egress_denylist import (
        agent_network_sandbox_config,
        egress_denylist_for,
    )

    if (
        (is_scripted_agent(agent) and not is_operator_script(agent))
        or agent == "task-runtime"
        or disallow_web_tools
        or getattr(task.config.sandbox, "allow_internet", True) is False
    ):
        return None
    policy = egress_denylist_for(agent_network_sandbox_config(task.config))
    if (
        is_operator_script(agent)
        and policy is not None
        and policy.native_claude_model_only
    ):
        raise ValueError("script: cannot use a native Claude model-only network policy")
    if policy is not None and not sandbox_user:
        raise ValueError(f"network_mode={policy.mode!r} requires a sandbox_user")
    return policy


def validate_agent_transport(*, sandbox: str, agent: str, harness: str) -> None:
    """Check one selected connection without resolving credentials or a proxy."""
    from benchflow.agents.registry import is_scripted_agent, session_factory_entrypoint
    from benchflow.sandbox.providers import PROVIDERS_BY_NAME
    from benchflow.task.runtime_capabilities import (
        UnsupportedTaskFeature,
        UnsupportedTaskFeatureError,
    )

    provider = PROVIDERS_BY_NAME.get(sandbox)
    if (
        provider is not None
        and not provider.supports_live_process
        and not is_scripted_agent(agent)
        and agent != "task-runtime"
        and (harness != "acp" or session_factory_entrypoint(agent) is None)
    ):
        raise UnsupportedTaskFeatureError(
            [
                UnsupportedTaskFeature(
                    path="agent.harness",
                    reason="ACP and native harnesses require a live stdio transport; select a supported sandbox",
                    sandbox=sandbox,
                )
            ]
        )


def validate_task_plan(
    task_dir: Path,
    *,
    sandbox: str,
    agent: str,
    model: str | None = None,
    harness: str = "acp",
    sandbox_user: str | None = "agent",
    agent_env: dict[str, str] | None = None,
    task: Task | None = None,
    force_sandbox_local: bool = False,
    custom_backend: bool = False,
    sandbox_instance: object | None = None,
    disallow_web_tools: bool = False,
    execute_solution: bool | None = None,
) -> dict[str, object] | None:
    """Refuse a task this run cannot honor, without creating a sandbox.

    The caller materializes source formats first, just as RolloutConfig does.
    A rollout supplies its loaded task and resolved agent environment, so
    config overrides and the selected authentication route are validated too.
    ``execute_solution`` overrides the normal oracle execution intent for
    verifier-only children that use oracle provisioning without its solver.
    Return the proxy install record the run persists; native-login routes
    return None and do not check extras for a proxy they do not use.
    """
    from benchflow.sandbox.providers import is_known_provider
    from benchflow.sandbox.setup import _validate_task_runtime_for_launch
    from benchflow.task.task import Task
    from benchflow.taskmd.launch import check_launch

    task = task if task is not None else Task(task_dir)
    validate_agent_transport(sandbox=sandbox, agent=agent, harness=harness)
    from benchflow.agents.registry import is_scripted_agent
    from benchflow.task.config import TaskConfig
    from benchflow.task.env import check_task_env, resolve_env_vars

    # S1's task-level refusal names every field before phase-local resolution
    # can raise a context-free error for the first individual env table.
    if isinstance(task.config, TaskConfig):
        check_task_env(task.config, task=task_dir.name)

    # Sandbox env is needed at creation. A setup capture only exists after
    # its command succeeds: never admit a forward/self dependency, or fake a
    # value in os.environ. Verification and the solution run after all setup.
    resolve_env_vars(task.config.sandbox.env)
    produced_env: dict[str, str] = {}
    for command in task.config.sandbox.setup_commands:
        resolve_env_vars(command.env, extra_env=produced_env)
        if command.capture_dir and command.capture_dir_b64_env:
            produced_env[command.capture_dir_b64_env] = ""
    resolve_env_vars(task.config.verifier.env, extra_env=produced_env)
    if execute_solution is None:
        execute_solution = agent == "oracle"
    if execute_solution:
        resolve_env_vars(task.config.solution.env, extra_env=produced_env)

    from benchflow.agents.muse_config import validate_muse_roles

    validate_muse_roles([(agent, model)], sandbox_user=sandbox_user)
    disallow_web_tools = (
        not is_scripted_agent(agent)
        and agent != "task-runtime"
        and (disallow_web_tools or task.config.sandbox.allow_internet is False)
    )
    policy = agent_egress_policy(
        task,
        agent=agent,
        sandbox_user=sandbox_user,
        disallow_web_tools=disallow_web_tools,
    )
    # An unknown SDK backend needs a positive capability report from its
    # sandbox for identity-switch admission. Without an instance there is
    # no evidence of the barrier, so refuse before calling its factory.
    # Also check a supplied instance under a built-in name: the registry
    # cannot vouch for a caller's implementation (K1-R15-2 B1).
    from benchflow.task import identity_switch

    if identity_switch.declared(task.config) and (
        sandbox_instance is not None
        or (custom_backend and not is_known_provider(sandbox))
    ):
        identity_switch.require_barrier(sandbox_instance, backend=sandbox)
    # Other custom-backend capabilities remain the plane's responsibility.
    # Built-in names retain the shared named capability gate.
    if not custom_backend or is_known_provider(sandbox):
        _validate_task_runtime_for_launch(
            task, sandbox_type=sandbox, task_path=task_dir
        )
    check_launch(task_dir, primary_agent=agent, sandbox_user=sandbox_user)

    from benchflow.task.keep_services import static_refusal

    # Service retention a run's setup would refuse, where no image is needed
    # to tell; same words and next step as setup (DT2-R9-1). A sandbox that
    # cannot hold kept services is refused at setup, from the measured
    # identities (keep_services.provider_refusal).
    refusal = static_refusal(task, agent, sandbox_user, sandbox)
    if refusal is not None:
        raise ValueError(refusal)

    from benchflow.agents.env import unroutable_agent_model
    from benchflow.agents.registry import is_scripted_agent

    if not is_scripted_agent(agent):
        reason = unroutable_agent_model(agent, model)
        if reason is not None:
            raise ValueError(reason)

    # S1 has not landed at D3's fixed base. Once it does, use its job gate
    # unchanged (including Compose and forwarded env), not a second parser.
    # Do not catch a refusal: new S1 policy must fail check and run together.
    from benchflow import evaluation

    env_gate = getattr(evaluation, "_refuse_unnamed_task_env", None)
    if env_gate is not None:
        env_gate([task_dir], manifest=None, sandbox=sandbox)

    from benchflow.providers.litellm_runtime import proxy_install_record

    if agent_env is None:
        # Check has no agent-env overlay: inherit the same well-known host
        # variables as a run, without requiring a key just to inspect a plan.
        from benchflow.agents.env import inherit_agent_auth_env

        agent_env = inherit_agent_auth_env(agent, model)
    from benchflow.sandbox.native_oauth import (
        allowlist_model_transport,
        native_oauth_egress_policy,
    )

    # Use the same pure auth-dependent gates as connect(), before allocation.
    # Scripted seats have no model transport or agent network restriction.
    if not is_scripted_agent(agent) and agent != "task-runtime":
        native_oauth_egress_policy(
            agent, model, agent_env, no_web=disallow_web_tools
        ) or allowlist_model_transport(policy, agent, model, agent_env)
    return proxy_install_record(
        agent=agent,
        model=model,
        agent_env=dict(agent_env or {}),
        environment=sandbox,
        force_sandbox_local=force_sandbox_local
        or policy is not None
        or getattr(task.config.sandbox, "allow_internet", True) is False,
    )
