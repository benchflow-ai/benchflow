"""Contract tests for the default rollout-plane bindings.

``DefaultRolloutPlanes`` is the composition boundary that binds the rollout
kernel to today's concrete ACP / provider / sandbox / environment
implementations. It was previously only exercised indirectly. These pin the
contract: the default bundle satisfies the ``RolloutPlanes`` Protocol, the
factory returns a satisfying instance, and the small amount of real logic the
adapter owns (``agent_launch`` web-tool suffixing, registry delegation) behaves.
"""

from __future__ import annotations

from benchflow.agents.registry import AGENT_LAUNCH, AGENTS
from benchflow.contracts.planes import RolloutPlanes, default_rollout_planes
from benchflow.rollout_planes import DefaultRolloutPlanes

# The Protocol surface every plane binding must implement. Pinning the names
# means a new abstract method on RolloutPlanes (or a renamed binding) fails here
# instead of at first live use.
_REQUIRED_METHODS = (
    "agent_launch",
    "agent_config",
    "resolve_agent_env",
    "install_docker_compat",
    "resolve_locked_paths",
    "stage_dockerfile_deps",
    "inject_skills_into_dockerfile",
    "create_environment",
    "manifest_environment",
    "setup_sandbox_user",
    "snapshot_build_config",
    "seed_verifier_workspace",
    "deploy_skills",
    "lockdown_paths",
    "install_agent",
    "write_credential_files",
    "upload_subscription_auth",
    "apply_web_tool_policy",
    "link_skill_paths",
    "ensure_litellm_runtime",
    "stop_provider_runtime",
    "extract_usage",
    "connect_acp",
    "execute_prompts",
    "harden_before_verify",
    "verifier",
    "clear_verifier_output_dir",
    "ensure_legacy_app_dir",
    "cleanup_verifier_python_hooks",
)


def test_default_planes_satisfies_protocol() -> None:
    planes = DefaultRolloutPlanes()
    # runtime_checkable Protocol isinstance check.
    assert isinstance(planes, RolloutPlanes)
    for name in _REQUIRED_METHODS:
        assert callable(getattr(planes, name)), f"missing plane method: {name}"


def test_factory_returns_satisfying_instance() -> None:
    planes = default_rollout_planes()
    assert isinstance(planes, DefaultRolloutPlanes)
    assert isinstance(planes, RolloutPlanes)


def test_agent_launch_passthrough_without_web_policy() -> None:
    planes = DefaultRolloutPlanes()
    agent = "codex-acp"
    expected = AGENT_LAUNCH.get(agent, agent)
    assert planes.agent_launch(agent, disallow_web_tools=False) == expected


def test_agent_launch_appends_web_tool_suffix_when_disallowed() -> None:
    planes = DefaultRolloutPlanes()
    agent = "codex-acp"
    cfg = AGENTS[agent]
    suffix = cfg.disallow_web_tools_launch_suffix
    assert suffix, "fixture agent must carry a web-tool suffix to exercise the branch"
    launched = planes.agent_launch(agent, disallow_web_tools=True)
    assert launched == AGENT_LAUNCH.get(agent, agent) + suffix
    # The suffix is only appended under the disallow flag.
    assert planes.agent_launch(agent, disallow_web_tools=True) != planes.agent_launch(
        agent, disallow_web_tools=False
    )


def test_agent_launch_unknown_agent_falls_back_to_name() -> None:
    planes = DefaultRolloutPlanes()
    # An unknown agent has no launch mapping and no config: returns the name,
    # and the disallow flag is a no-op (no config -> no suffix).
    assert planes.agent_launch("not-a-real-agent", disallow_web_tools=False) == (
        "not-a-real-agent"
    )
    assert planes.agent_launch("not-a-real-agent", disallow_web_tools=True) == (
        "not-a-real-agent"
    )


def test_agent_config_delegates_to_registry() -> None:
    planes = DefaultRolloutPlanes()
    assert planes.agent_config("codex-acp") is AGENTS["codex-acp"]
    assert planes.agent_config("not-a-real-agent") is None
