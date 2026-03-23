"""Verify Harbor re-exports and benchflow additions work."""


def test_harbor_reexports():
    """All Harbor classes should be importable from benchflow."""
    from benchflow import (
        BaseEnvironment,
        Job,
        TaskConfig,
    )

    # These should be Harbor's classes
    assert Job.__module__.startswith("harbor")
    assert TaskConfig.__module__.startswith("harbor")
    assert BaseEnvironment.__module__.startswith("harbor")


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
