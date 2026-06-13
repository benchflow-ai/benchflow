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


@dataclass(frozen=True)
class SandboxProvider:
    """One sandbox backend and the facts that were previously duplicated."""

    name: str
    extra: str | None  # pip/uv optional-dependency extra; None for built-in docker
    off_box_model: bool  # model traffic exits the sandbox → host proxy (not docker)


# Ordered, docker-first. This tuple is the ONLY place the set is spelled out.
_PROVIDERS: tuple[SandboxProvider, ...] = (
    SandboxProvider("docker", extra=None, off_box_model=False),
    SandboxProvider("daytona", extra="sandbox-daytona", off_box_model=True),
    SandboxProvider("modal", extra="sandbox-modal", off_box_model=True),
    SandboxProvider("cua", extra="sandbox-cua", off_box_model=True),
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
#: Providers whose model traffic must reach the host proxy off-box (≡ non-docker).
OFF_BOX_MODEL_PROVIDERS: frozenset[str] = frozenset(
    p.name for p in _PROVIDERS if p.off_box_model
)


def is_known_provider(name: str) -> bool:
    """True if ``name`` is a registered sandbox provider."""
    return name in SANDBOX_PROVIDER_SET


def provider_extra(name: str) -> str | None:
    """The optional-dependency extra for ``name`` (None for docker/unknown)."""
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
