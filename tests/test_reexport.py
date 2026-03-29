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


def test_register_agent():
    """Custom agents can be registered at runtime."""
    from benchflow import register_agent, AGENTS, get_agent

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

    # Cleanup
    del AGENTS["test-custom-agent"]
