"""Verify Harbor re-exports and benchflow additions work."""


def test_harbor_reexports():
    """Harbor classes should be importable from benchflow."""
    from benchflow import (
        BaseEnvironment,
        TaskConfig,
    )

    assert TaskConfig.__module__.startswith("harbor")
    assert BaseEnvironment.__module__.startswith("harbor")


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


def test_sdk_importable():
    from benchflow.sdk import SDK

    sdk = SDK()
    assert hasattr(sdk, "run")


def test_extracted_modules_importable():
    """Symbols moved to _models, _trajectory, _env_setup are importable from canonical paths."""
    from benchflow._env_setup import _dep_local_name, stage_dockerfile_deps
    from benchflow._models import AgentInstallError, AgentTimeoutError, RunResult
    from benchflow._trajectory import _capture_session_trajectory

    assert RunResult.__module__ == "benchflow._models"
    assert AgentInstallError.__module__ == "benchflow._models"
    assert AgentTimeoutError.__module__ == "benchflow._models"
    assert callable(_capture_session_trajectory)
    assert callable(stage_dockerfile_deps)
    assert callable(_dep_local_name)


def test_public_api_reexports():
    """Public API symbols are still importable from benchflow top-level."""
    from benchflow import (
        SDK,
        AgentInstallError,
        AgentTimeoutError,
        RunResult,
        stage_dockerfile_deps,
    )

    assert callable(SDK)
    assert callable(RunResult)
    assert callable(AgentInstallError)
    assert callable(AgentTimeoutError)
    assert callable(stage_dockerfile_deps)


def test_register_agent():
    """Custom agents can be registered at runtime."""
    from benchflow import AGENTS, get_agent, register_agent
    from benchflow.agents.registry import AGENT_INSTALLERS, AGENT_LAUNCH

    try:
        register_agent(
            name="test-custom-agent",
            install_cmd="echo installed",
            launch_cmd="test-agent --acp",
            requires_env=["TEST_KEY"],
            description="Test agent",
        )

        assert "test-custom-agent" in AGENTS
        cfg, alias_model = get_agent("test-custom-agent")
        assert cfg.launch_cmd == "test-agent --acp"
        assert cfg.requires_env == ["TEST_KEY"]
        assert alias_model == ""
    finally:
        # register_agent writes to all three dicts; clean up all three to keep
        # the global registries in sync for downstream tests.
        AGENTS.pop("test-custom-agent", None)
        AGENT_INSTALLERS.pop("test-custom-agent", None)
        AGENT_LAUNCH.pop("test-custom-agent", None)
