from __future__ import annotations

from benchflow.environment.adapters import environment_adapter_report


def test_browser_environment_adapter_reports_verified_docker() -> None:
    """Guards 0.7 browser environment adapter evidence from being only a label."""
    report = environment_adapter_report(
        benchmark_adapter="browser-use-benchmark",
        sandbox="docker",
    )

    assert report.name == "browser"
    assert report.world == "browser"
    assert report.provider_support == "verified"
    assert "browser-runtime" in report.required_capabilities
    assert report.to_dict()["verified_sandboxes"] == ["docker"]


def test_browser_environment_adapter_reports_unverified_other_sandbox() -> None:
    """Guards provider-honest reporting for browser tasks before scale."""
    report = environment_adapter_report(
        benchmark_adapter="browser-use-benchmark",
        sandbox="cua",
    )

    assert report.name == "browser"
    assert report.provider_support == "unverified"
    assert report.note is not None
    assert "needs original-vs-BenchFlow parity" in report.note


def test_desktop_environment_adapter_reports_local_cua_capabilities() -> None:
    """Guards 0.7 desktop environment adapter capability reporting."""
    report = environment_adapter_report(
        benchmark_adapter="use-computer-cookbook",
        sandbox="cua",
        provider_mode="local",
    )

    assert report.name == "desktop"
    assert report.provider_support == "verified"
    assert report.provider_mode == "local"
    assert "screenshot" in report.required_capabilities
    assert "file-transfer" in report.required_capabilities


def test_desktop_environment_adapter_requires_probe_for_cloud_cua() -> None:
    """Cua cloud must not inherit local-Cua verification without runtime evidence."""
    report = environment_adapter_report(
        benchmark_adapter="use-computer-cookbook",
        sandbox="cua",
        provider_mode="cloud",
    )

    assert report.name == "desktop"
    assert report.provider_support == "runtime-probe-required"
    assert report.provider_mode == "cloud"
    assert report.note is not None
    assert "--probe-runtime" in report.note


def test_desktop_environment_adapter_accepts_probed_cloud_cua() -> None:
    """A passing Cua cloud runtime probe upgrades provider support to verified."""
    report = environment_adapter_report(
        benchmark_adapter="use-computer-cookbook",
        sandbox="cua",
        provider_mode="cloud",
        runtime_probe_ready=True,
    )

    assert report.provider_support == "verified"
    assert report.provider_mode == "cloud"
    assert report.to_dict()["verified_sandboxes"] == ["cua:local", "cua:cloud-probed"]
