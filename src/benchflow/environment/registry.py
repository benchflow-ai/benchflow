"""Environment provider registry — how concrete worlds join the kernel.

Design principle 2 (``docs/architecture.md``): the Environment plane is
swappable + BYO. The kernel does not import ``ManifestEnvironment`` directly;
it asks :func:`resolve_environment` for a :class:`benchflow.contracts.Environment`
built from the rollout's manifest. ``ManifestEnvironment`` is the built-in
default adapter that reads a declarative manifest; a third party registers an
alternative world (a fixed sidecar fleet, a custom simulator host) with
:func:`register_environment` and no kernel edit.

Mirrors :mod:`benchflow.sandbox.registry` and the Agent plane's
:mod:`benchflow.agents.registry`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchflow.contracts import Environment

# A factory builds an Environment from its manifest, bound to a live sandbox.
EnvironmentFactory = Callable[[Any, Any], "Environment"]

_ENVIRONMENT_REGISTRY: dict[str, EnvironmentFactory] = {}

# The manifest field that selects the provider; absent => the default below.
DEFAULT_ENVIRONMENT_KIND = "manifest"


def register_environment(kind: str, factory: EnvironmentFactory) -> None:
    """Register an environment provider under ``kind`` (e.g. ``"manifest"``)."""
    _ENVIRONMENT_REGISTRY[kind] = factory


def available_environments() -> list[str]:
    """Names of all registered environment providers, sorted."""
    return sorted(_ENVIRONMENT_REGISTRY)


def resolve_environment(kind: str, manifest: Any, *, sandbox: Any) -> Environment:
    """Build an Environment by provider kind, raising for unknown kinds."""
    try:
        factory = _ENVIRONMENT_REGISTRY[kind]
    except KeyError:
        known = ", ".join(repr(n) for n in available_environments()) or "(none)"
        raise ValueError(
            f"Unknown environment provider {kind!r} (registered: {known})"
        ) from None
    return factory(manifest, sandbox)


def _build_manifest_environment(manifest: Any, sandbox: Any) -> Environment:
    # Lazy import: keep the concrete adapter out of the registry's import path.
    from benchflow.environment.manifest_env import ManifestEnvironment

    return ManifestEnvironment(manifest, sandbox=sandbox)


register_environment(DEFAULT_ENVIRONMENT_KIND, _build_manifest_environment)
