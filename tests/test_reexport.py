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
