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
    #: Whether the backend can enforce ``network_mode = "denylist"`` (a
    #: root-owned loopback proxy behind the uid firewall).
    enforces_denylist: bool = False
    #: Whether the backend can enforce ``network_mode = "allowlist"`` (the same
    #: proxy in allow mode, plus the DNS filter and CIDR rules on the uid
    #: firewall). Set only for backends validated by a live check.
    enforces_allowlist: bool = False
    #: Whether an agent can hold an interactive ACP stdio transport.
    supports_live_process: bool = False
    #: Whether the backend runs a task's verifier in a fresh sandbox of its own
    #: (``verifier.sandbox_mode: separate``, ``--verifier-sandbox separate``).
    runs_separate_verifier: bool = False
    #: When that sandbox starts in ``run()``'s own lifecycle: ``"prestart"``
    #: (from task files at verify() entry, while the agent's run is captured)
    #: or ``"after_stop"`` (once the agent's sandbox has stopped); None keeps
    #: the serial order.
    separate_verifier_start: str | None = None
    #: Whether the agent's container is created with no_new_privs, so no
    #: process in it can gain an identity: "always", "compose"
    #: (only a task with a compose file, run Docker-in-Docker) or "never".
    #: x-agent-identity-switch runs only where it holds (K1-R15-2).
    agent_container_no_new_privs: str = "never"

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
        supports_live_process=True,
        runs_separate_verifier=True,
        separate_verifier_start="prestart",
        supports_compose=True,
        enforces_denylist=True,
        enforces_allowlist=True,
        agent_container_no_new_privs="always",
    ),
    SandboxProvider(
        # The local provider on a Docker host the user controls (ssh:// or
        # tcp:// with TLS). The model proxy runs in the sandbox because the
        # remote container cannot reach one on the caller's machine.
        "remote-docker",
        extra=None,
        model_proxy=ModelProxyLocation.SANDBOX,
        supports_live_process=True,
        runs_separate_verifier=True,
        separate_verifier_start="prestart",
        supports_compose=True,
        enforces_denylist=True,
        enforces_allowlist=True,
        agent_container_no_new_privs="always",
    ),
    SandboxProvider(
        "daytona",
        extra="sandbox-daytona",
        model_proxy=ModelProxyLocation.SANDBOX,
        supports_live_process=True,
        # After the agent's sandbox stopped, until an agent there is shown
        # unable to reach an idle verifier sandbox.
        runs_separate_verifier=True,
        separate_verifier_start="after_stop",
        # The DinD strategy runs compose inside the sandbox VM.
        supports_compose=True,
        enforces_denylist=True,
        enforces_allowlist=True,
        # Only the DinD strategy's compose services; a direct sandbox has none.
        agent_container_no_new_privs="compose",
    ),
    SandboxProvider(
        "modal",
        extra="sandbox-modal",
        model_proxy=ModelProxyLocation.SANDBOX,
        # The agent's stdio over Sandbox.exec streams (ModalProcess).
        supports_live_process=True,
        # after the agent's sandbox stopped, so a GPU trial never
        # holds both sandboxes' GPUs.
        runs_separate_verifier=True,
        separate_verifier_start="after_stop",
    ),
    SandboxProvider(
        "apple-container",
        extra=None,
        model_proxy=ModelProxyLocation.SANDBOX,
        supports_live_process=True,
        enforces_no_network=False,
    ),
    SandboxProvider(
        "agentcore",
        extra="sandbox-agentcore",
        model_proxy=ModelProxyLocation.SANDBOX,
        supports_live_process=True,
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
#: Providers that cannot enforce ``network_mode = "denylist"``.
DENYLIST_UNSUPPORTED_PROVIDERS: frozenset[str] = frozenset(
    p.name for p in _PROVIDERS if not p.enforces_denylist
)

#: Providers that cannot enforce ``network_mode = "allowlist"``.
ALLOWLIST_UNSUPPORTED_PROVIDERS: frozenset[str] = frozenset(
    p.name for p in _PROVIDERS if not p.enforces_allowlist
)
#: Providers that run a separate verifier sandbox.
SEPARATE_VERIFIER_PROVIDERS: frozenset[str] = frozenset(
    p.name for p in _PROVIDERS if p.runs_separate_verifier
)


def is_known_provider(name: str) -> bool:
    """True if ``name`` is a registered sandbox provider."""
    return name in SANDBOX_PROVIDER_SET


def provider_extra(name: str) -> str | None:
    """The optional-dependency extra for ``name`` (None if absent/unknown)."""
    p = PROVIDERS_BY_NAME.get(name)
    return p.extra if p else None


def extra_install_hint(extra: str) -> str:
    """How to add an optional extra, for a ``uv tool`` install and a checkout.

    The checkout form keeps ``dev``: ``uv sync`` removes every extra it is not
    given, so ``uv sync --extra <extra>`` alone would uninstall the test tools.
    """
    return (
        f"uv tool install --python 3.12 --upgrade 'benchflow[{extra}]' "
        f"(source checkout: uv sync --extra {extra} --extra dev --locked)"
    )


# Top-level module each extra-backed provider's SDK installs.
_SANDBOX_SDK_MODULES: dict[str, str] = {
    "daytona": "daytona",
    "agentcore": "bedrock_agentcore",
}


def sandbox_sdk_missing(name: str) -> bool:
    """True when ``name`` needs an SDK from an extra that is not installed.

    Uses ``find_spec`` so nothing is imported; Modal keeps its own import
    check in ``eval_plan``.
    """
    module = _SANDBOX_SDK_MODULES.get(name)
    if module is None:
        return False
    import importlib.util

    try:
        return importlib.util.find_spec(module) is None
    except (ImportError, ValueError):
        return True


def providers_phrase(*, quote: bool = False) -> str:
    """Human list of providers, e.g. ``docker, daytona, or modal``.

    Kept byte-identical to the strings it replaces so every help/error message
    is unchanged. ``quote=True`` renders ``'docker', 'daytona', or 'modal'``.
    """
    items = [f"'{n}'" if quote else n for n in SANDBOX_PROVIDERS]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", or " + items[-1]
