"""Environment-adapter capability reports for foreign benchmark tasks.

The inbound benchmark adapter says where a task came from. The environment
adapter says what kind of world the agent acts in. Keeping this mapping small
and explicit lets CLI/resource flows report the Browser/Desktop plane without
making benchmark adapters own sandbox lifecycle or agent-loop behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchflow.sandbox.providers import SANDBOX_PROVIDERS


@dataclass(frozen=True)
class EnvironmentAdapterReport:
    """Provider-honest description of the environment plane for one task."""

    name: str
    world: str
    benchmark_adapter: str | None
    status: str
    provider_support: str
    required_capabilities: tuple[str, ...]
    verified_sandboxes: tuple[str, ...]
    provider_mode: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "world": self.world,
            "benchmark_adapter": self.benchmark_adapter,
            "status": self.status,
            "provider_support": self.provider_support,
            "required_capabilities": list(self.required_capabilities),
            "verified_sandboxes": list(self.verified_sandboxes),
        }
        if self.provider_mode:
            payload["provider_mode"] = self.provider_mode
        if self.note:
            payload["note"] = self.note
        return payload


def environment_adapter_report(
    *,
    benchmark_adapter: str | None,
    sandbox: str,
    provider_mode: str | None = None,
    runtime_probe_ready: bool = False,
) -> EnvironmentAdapterReport:
    """Return the environment-plane report for a task/provider pairing."""

    match benchmark_adapter:
        case "browser-use-benchmark":
            return _browser_report(benchmark_adapter=benchmark_adapter, sandbox=sandbox)
        case "computer-use-benchmark" | "use-computer-cookbook":
            return _desktop_report(
                benchmark_adapter=benchmark_adapter,
                sandbox=sandbox,
                provider_mode=provider_mode,
                runtime_probe_ready=runtime_probe_ready,
            )
        case _:
            return EnvironmentAdapterReport(
                name="repo",
                world="repo",
                benchmark_adapter=benchmark_adapter,
                status="ready",
                provider_support="native",
                required_capabilities=("shell", "filesystem"),
                verified_sandboxes=SANDBOX_PROVIDERS,
            )


def _browser_report(
    *, benchmark_adapter: str, sandbox: str
) -> EnvironmentAdapterReport:
    verified = ("docker",)
    provider_support = "verified" if sandbox in verified else "unverified"
    note = None
    if provider_support == "unverified":
        note = (
            "browser environment adapter is verified on docker; this sandbox "
            "needs original-vs-BenchFlow parity before scale"
        )
    return EnvironmentAdapterReport(
        name="browser",
        world="browser",
        benchmark_adapter=benchmark_adapter,
        status="ready",
        provider_support=provider_support,
        required_capabilities=(
            "browser-runtime",
            "local-http-fixture",
            "screenshot-artifacts",
            "trace-artifacts",
        ),
        verified_sandboxes=verified,
        note=note,
    )


def _desktop_report(
    *,
    benchmark_adapter: str,
    sandbox: str,
    provider_mode: str | None,
    runtime_probe_ready: bool,
) -> EnvironmentAdapterReport:
    verified = ("cua:local",)
    if sandbox == "cua" and provider_mode == "local":
        provider_support = "verified"
    elif sandbox == "cua" and runtime_probe_ready:
        provider_support = "verified"
        verified = ("cua:local", "cua:cloud-probed")
    elif sandbox == "cua":
        provider_support = "runtime-probe-required"
    else:
        provider_support = "unverified"
    note = None
    if provider_support == "runtime-probe-required":
        note = (
            "desktop environment adapter is verified on local Cua; Cua cloud "
            "must pass `bench environment check --probe-runtime --sandbox cua` "
            "before scale"
        )
    elif provider_support == "unverified":
        note = (
            "desktop environment adapter is verified on local Cua; this sandbox "
            "needs runtime and parity evidence before scale"
        )
    return EnvironmentAdapterReport(
        name="desktop",
        world="desktop",
        benchmark_adapter=benchmark_adapter,
        status="ready",
        provider_support=provider_support,
        required_capabilities=(
            "shell",
            "file-transfer",
            "screenshot",
            "display-dimensions",
            "cleanup",
        ),
        verified_sandboxes=verified,
        provider_mode=provider_mode,
        note=note,
    )
