"""Canonical registry of sandbox providers — the single source of truth.

Before this module the provider set ``{docker, daytona, modal}`` was hand-copied
across ~10 sites (typer ``--sandbox`` help strings, validation membership checks,
error messages, the optional-extra map, and two ``{daytona, modal}`` "off-box
model" subsets) with no registry and no drift test, so adding a provider meant
editing every copy. All of those now derive from the tuple below; the drift test
``tests/test_sandbox_provider_registry_drift.py`` fails if a literal set
reappears elsewhere.

Import-safe by design: stdlib only, so every caller (CLI options, eval planning,
runtime capabilities, the litellm runtime, the replay orchestrator) can import it
without a cycle. The provider→implementation dispatch deliberately stays in
``sandbox/setup.py:_create_sandbox_environment`` (its per-branch lazy imports and
daytona-only resource clamps make per-branch code the safer shape); it validates
against this registry rather than re-listing the names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelProxyLocation(StrEnum):
    """Where BenchFlow runs the mandatory LiteLLM proxy for a sandbox."""

    HOST = "host"
    SANDBOX = "sandbox"


@dataclass(frozen=True)
class SandboxProvider:
    """One sandbox backend and the facts that were previously duplicated."""

    name: str
    extra: str | None  # pip/uv optional-dependency extra; None when none is needed
    model_proxy: ModelProxyLocation
    #: Whether the backend can actually enforce ``network_mode = "no-network"``.
    #: Declared here so the runtime capability gate reads a fact off the
    #: registry instead of growing a ``sandbox == "<name>"`` special case per
    #: backend that cannot isolate the network.
    enforces_no_network: bool = True
    #: Whether the backend can run a task's docker-compose side services.
    #: ``False`` means a multi-service task must be refused, not run partially.
    supports_compose: bool = False

    @property
    def off_box_model(self) -> bool:
        """Backward-compatible alias for the old, ambiguously named flag."""

        return self.model_proxy is ModelProxyLocation.SANDBOX


# Ordered, docker-first. This tuple is the ONLY place the set is spelled out.
_PROVIDERS: tuple[SandboxProvider, ...] = (
    SandboxProvider(
        "docker",
        extra=None,
        model_proxy=ModelProxyLocation.HOST,
        supports_compose=True,
    ),
    SandboxProvider(
        "daytona",
        extra="sandbox-daytona",
        model_proxy=ModelProxyLocation.SANDBOX,
        # The DinD strategy runs compose inside the sandbox VM.
        supports_compose=True,
    ),
    SandboxProvider(
        "modal",
        extra="sandbox-modal",
        model_proxy=ModelProxyLocation.SANDBOX,
    ),
    SandboxProvider(
        "apple-container",
        extra=None,
        model_proxy=ModelProxyLocation.SANDBOX,
        enforces_no_network=False,
    ),
    SandboxProvider(
        "agentcore",
        extra="sandbox-agentcore",
        model_proxy=ModelProxyLocation.SANDBOX,
        # AgentCore's CreateAgentRuntime networkConfiguration.networkMode only
        # accepts PUBLIC or VPC — there is no isolated mode, so a no-network
        # task cannot be honored here.
        enforces_no_network=False,
    ),
)

PROVIDERS_BY_NAME: dict[str, SandboxProvider] = {p.name: p for p in _PROVIDERS}

#: Ordered provider names — use for help text / messages (stable order).
SANDBOX_PROVIDERS: tuple[str, ...] = tuple(p.name for p in _PROVIDERS)
#: O(1) membership set — use for validation.
SANDBOX_PROVIDER_SET: frozenset[str] = frozenset(SANDBOX_PROVIDERS)
#: provider → optional-dependency extra (only providers that need one).
OPTIONAL_SANDBOX_EXTRAS: dict[str, str] = {
    p.name: p.extra for p in _PROVIDERS if p.extra is not None
}
#: Providers whose LiteLLM proxy must run inside the sandbox.
SANDBOX_MODEL_PROXY_PROVIDERS: frozenset[str] = frozenset(
    p.name for p in _PROVIDERS if p.model_proxy is ModelProxyLocation.SANDBOX
)
# Backward-compatible import alias. New code uses the literal placement contract.
OFF_BOX_MODEL_PROVIDERS = SANDBOX_MODEL_PROXY_PROVIDERS


#: Providers that run exactly one container and cannot host compose services.
SINGLE_CONTAINER_PROVIDERS: frozenset[str] = frozenset(
    p.name for p in _PROVIDERS if not p.supports_compose
)
#: Providers that cannot enforce ``network_mode = "no-network"``.
NO_NETWORK_UNSUPPORTED_PROVIDERS: frozenset[str] = frozenset(
    p.name for p in _PROVIDERS if not p.enforces_no_network
)


def is_known_provider(name: str) -> bool:
    """True if ``name`` is a registered sandbox provider."""
    return name in SANDBOX_PROVIDER_SET


def provider_extra(name: str) -> str | None:
    """The optional-dependency extra for ``name`` (None if absent/unknown)."""
    p = PROVIDERS_BY_NAME.get(name)
    return p.extra if p else None


def providers_phrase(*, quote: bool = False) -> str:
    """Human list of providers, e.g. ``docker, daytona, or modal``.

    Kept byte-identical to the strings it replaces so every help/error message
    is unchanged. ``quote=True`` renders ``'docker', 'daytona', or 'modal'``.
    """
    items = [f"'{n}'" if quote else n for n in SANDBOX_PROVIDERS]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", or " + items[-1]
