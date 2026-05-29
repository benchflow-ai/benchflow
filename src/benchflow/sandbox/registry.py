"""Sandbox provider registry — how concrete substrates join the kernel.

Design principle 2 (``docs/architecture.md``): *four planes, each swappable,
each managed + BYO.* The kernel never imports ``DockerSandbox`` /
``DaytonaSandbox`` / a Modal class directly; it asks this registry to
``resolve_sandbox(name, ctx)`` and gets back something that satisfies the
:class:`benchflow.contracts.Sandbox` Protocol. Built-in providers register
themselves (see :mod:`benchflow.sandbox.setup`); a third party adds a backend
with :func:`register_sandbox` and no kernel edit.

This mirrors the Agent plane's existing data-driven registry
(:mod:`benchflow.agents.registry`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchflow.contracts import Sandbox


@dataclass(frozen=True)
class SandboxBuildContext:
    """The resolved inputs every provider factory needs to build a sandbox.

    The shared, provider-independent preamble (manifest image/env resolution,
    network-policy adjustments) runs once before resolution and lands here;
    each factory only does its own provider-specific construction.
    """

    environment_dir: Path
    environment_name: str
    session_id: str
    rollout_paths: Any
    task_env_config: Any
    persistent_env: dict[str, str] | None


# A factory takes the resolved context and returns a Sandbox-Protocol object.
SandboxFactory = Callable[[SandboxBuildContext], "Sandbox"]

_SANDBOX_REGISTRY: dict[str, SandboxFactory] = {}


def register_sandbox(name: str, factory: SandboxFactory) -> None:
    """Register a sandbox provider under ``name`` (e.g. ``"docker"``)."""
    _SANDBOX_REGISTRY[name] = factory


def available_sandboxes() -> list[str]:
    """Names of all registered sandbox providers, sorted."""
    return sorted(_SANDBOX_REGISTRY)


def resolve_sandbox(name: str, ctx: SandboxBuildContext) -> Sandbox:
    """Build a sandbox by provider name, raising for unknown names."""
    try:
        factory = _SANDBOX_REGISTRY[name]
    except KeyError:
        known = ", ".join(repr(n) for n in available_sandboxes()) or "(none)"
        raise ValueError(
            f"Unknown sandbox provider {name!r} (registered: {known})"
        ) from None
    return factory(ctx)
