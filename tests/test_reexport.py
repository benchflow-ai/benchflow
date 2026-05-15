"""Verify benchflow public API re-exports."""


def test_native_task_and_sandbox_reexports():
    """Native task and sandbox classes should be importable from benchflow."""
    from benchflow import (
        ExecResult,
        SandboxSpec,
        Task,
        TaskConfig,
    )

    assert Task.__module__.startswith("benchflow.tasks")
    assert TaskConfig.__module__.startswith("benchflow.tasks")
    assert ExecResult.__module__.startswith("benchflow.sandboxes")
    assert SandboxSpec.__module__.startswith("benchflow.sandboxes")


def test_benchflow_job_shadows_harbor():
    """benchflow.Job is benchflow's own Job, not Harbor's."""
    from benchflow import Job

    assert Job.__module__ == "benchflow.job"


def test_benchflow_additions():
    """benchflow's own additions should be importable."""
    from benchflow import (
        ACPClient,
        TrajectoryProxy,
    )

    assert ACPClient.__module__.startswith("benchflow")
    assert TrajectoryProxy.__module__.startswith("benchflow")


def test_rollout_runner_importable():
    from benchflow import RolloutConfig, run

    assert callable(run)
    assert callable(RolloutConfig)


def test_extracted_modules_importable():
    """Symbols moved to models, _trajectory, _env_setup are importable from canonical paths."""
    from benchflow._env_setup import _dep_local_name, stage_dockerfile_deps
    from benchflow._trajectory import _capture_session_trajectory
    from benchflow.models import AgentInstallError, AgentTimeoutError, RunResult

    assert RunResult.__module__ == "benchflow.models"
    assert AgentInstallError.__module__ == "benchflow.models"
    assert AgentTimeoutError.__module__ == "benchflow.models"
    assert callable(_capture_session_trajectory)
    assert callable(stage_dockerfile_deps)
    assert callable(_dep_local_name)


def test_public_api_reexports():
    """Public API symbols are still importable from benchflow top-level."""
    from benchflow import (
        AgentInstallError,
        AgentTimeoutError,
        RolloutConfig,
        RolloutResult,
        RunResult,
        SandboxSpec,
        stage_dockerfile_deps,
    )

    assert callable(RolloutConfig)
    assert callable(RolloutResult)
    assert callable(SandboxSpec)
    assert callable(RunResult)
    assert callable(AgentInstallError)
    assert callable(AgentTimeoutError)
    assert callable(stage_dockerfile_deps)


def test_register_agent():
    """Custom agents can be registered at runtime."""
    from benchflow import AGENTS, AgentCapability, get_agent, register_agent
    from benchflow.agents.registry import AGENT_INSTALLERS, AGENT_LAUNCH

    try:
        register_agent(
            name="test-custom-agent",
            install_cmd="echo installed",
            launch_cmd="test-agent --acp",
            requires_env=["TEST_KEY"],
            description="Test agent",
            capabilities=[AgentCapability("agent-as-tool")],
        )

        assert "test-custom-agent" in AGENTS
        cfg, alias_model = get_agent("test-custom-agent")
        assert cfg.launch_cmd == "test-agent --acp"
        assert cfg.requires_env == ["TEST_KEY"]
        assert cfg.capabilities == [AgentCapability("agent-as-tool")]
        assert alias_model == ""
    finally:
        # register_agent writes to all three dicts; clean up all three to keep
        # the global registries in sync for downstream tests.
        AGENTS.pop("test-custom-agent", None)
        AGENT_INSTALLERS.pop("test-custom-agent", None)
        AGENT_LAUNCH.pop("test-custom-agent", None)
