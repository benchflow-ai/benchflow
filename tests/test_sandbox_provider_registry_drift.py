"""Drift guard for the canonical sandbox-provider registry (dev-ex #14).

Before ``benchflow.sandbox.providers`` the set ``{docker, daytona, modal}`` (and
its model-proxy placement subset) was hand-copied across ~10 sites with no
single source of truth. These tests fail if (a) a literal provider set reappears
outside the registry, (b) the derived facts (phrase, extras, proxy placement) go
stale, or (c) the registry and the dispatch table drift apart.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from benchflow.sandbox.providers import (
    OFF_BOX_MODEL_PROVIDERS,
    OPTIONAL_SANDBOX_EXTRAS,
    PROVIDERS_BY_NAME,
    SANDBOX_MODEL_PROXY_PROVIDERS,
    SANDBOX_PROVIDER_SET,
    SANDBOX_PROVIDERS,
    ModelProxyLocation,
    providers_phrase,
)

_SRC = Path(__file__).resolve().parents[1] / "src" / "benchflow"
_REGISTRY = _SRC / "sandbox" / "providers.py"


def test_registry_is_the_single_source_of_truth() -> None:
    # Locks the current set + docker-first order; adding a provider is then a
    # deliberate edit here + a test update, never a silent scatter.
    assert SANDBOX_PROVIDERS == (
        "docker",
        "daytona",
        "modal",
        "apple-container",
        "agentcore",
        "fystash",
    )
    assert frozenset(SANDBOX_PROVIDERS) == SANDBOX_PROVIDER_SET


def test_providers_phrase_is_byte_identical() -> None:
    # The refactor must be behavior-preserving for every help/error string that
    # used to hand-write this phrase.
    assert providers_phrase() == (
        "docker, daytona, modal, apple-container, agentcore, or fystash"
    )
    assert providers_phrase(quote=True) == (
        "'docker', 'daytona', 'modal', 'apple-container', 'agentcore', or 'fystash'"
    )


def test_model_proxy_placement_is_explicit_for_every_provider() -> None:
    """Guards PR #936 against routing host loopback into an Apple VM."""

    assert PROVIDERS_BY_NAME["docker"].model_proxy is ModelProxyLocation.HOST
    for provider in ("daytona", "modal", "apple-container", "agentcore", "fystash"):
        assert PROVIDERS_BY_NAME[provider].model_proxy is ModelProxyLocation.SANDBOX
    assert (
        frozenset({"daytona", "modal", "apple-container", "agentcore", "fystash"})
        == SANDBOX_MODEL_PROXY_PROVIDERS
    )
    assert OFF_BOX_MODEL_PROVIDERS is SANDBOX_MODEL_PROXY_PROVIDERS


def test_no_divergent_provider_set_literal_outside_the_registry() -> None:
    """The core drift catcher: the literal provider set / off-box subset must
    appear ONLY in providers.py. A reintroduced hand-rolled set fails here."""
    # Ordered triple in any bracket form, e.g. {"docker", "daytona", "modal"} or
    # ("docker", "daytona", "modal"); and the bare {"daytona", "modal"} subset.
    triple = re.compile(
        r'["\']docker["\']\s*,\s*["\']daytona["\']\s*,\s*["\']modal["\']'
    )
    off_box = re.compile(r'["\']daytona["\']\s*,\s*["\']modal["\']')
    offenders: list[str] = []
    for py in _SRC.rglob("*.py"):
        if py.resolve() == _REGISTRY.resolve():
            continue
        text = py.read_text()
        for rx, what in ((triple, "provider-set"), (off_box, "off-box-subset")):
            for m in rx.finditer(text):
                line = text[: m.start()].count("\n") + 1
                offenders.append(
                    f"{py.relative_to(_SRC.parent.parent)}:{line} ({what})"
                )
    assert not offenders, (
        "Hardcoded sandbox-provider literal(s) found outside "
        "benchflow.sandbox.providers — derive from the registry instead:\n  "
        + "\n  ".join(offenders)
    )


def test_optional_extras_match_pyproject() -> None:
    # Every provider extra must be a real packaging extra; fails if an extra is
    # renamed in pyproject without updating the registry.
    pyproject = tomllib.loads((_SRC.parent.parent / "pyproject.toml").read_text())
    declared = set(pyproject["project"]["optional-dependencies"])
    assert set(OPTIONAL_SANDBOX_EXTRAS.values()) <= declared, (
        f"registry extras {set(OPTIONAL_SANDBOX_EXTRAS.values())} not all declared "
        f"in pyproject optional-dependencies {declared}"
    )
    # Apple Container is a system CLI backend, so it needs no Python extra.
    assert OPTIONAL_SANDBOX_EXTRAS == {
        "daytona": "sandbox-daytona",
        "modal": "sandbox-modal",
        "agentcore": "sandbox-agentcore",
        "fystash": "sandbox-fystash",
    }


def test_every_registry_provider_has_a_dispatch_branch() -> None:
    # Registry name with no branch in _create_sandbox_environment would fall to
    # the "Unknown sandbox_type" ValueError — prove they can't drift apart.
    setup_src = (_SRC / "sandbox" / "setup.py").read_text()
    body = setup_src[setup_src.index("def _create_sandbox_environment") :]
    for name in SANDBOX_PROVIDERS:
        assert f'sandbox_type == "{name}"' in body, (
            f"provider {name!r} is in the registry but has no dispatch branch in "
            "_create_sandbox_environment"
        )
