"""Guards the v0.5 Phase 0 contracts/kernel seam.

Phase 0 (docs/architecture.md design principle 1: "the kernel depends only on
contracts") introduced:

* ``benchflow.contracts`` — the single aggregation point for the four-plane
  Protocols, so the kernel imports Protocols, never concrete providers.
* ``benchflow.sandbox.registry`` / ``benchflow.environment.registry`` — the
  data-driven provider registries the kernel resolves through, so a backend
  joins by registration rather than a kernel ``if/elif`` edit.

These tests pin that structure so a later refactor can't quietly collapse the
contract surface or the registry seam back into the kernel.
"""

from __future__ import annotations

import pytest


def test_contracts_exposes_the_four_planes():
    """The single import surface carries all four plane Protocols."""
    import benchflow.contracts as c

    # One Protocol (or pair) per plane — the kernel's whole type surface.
    for name in ("Sandbox", "Agent", "Session", "Environment", "Reward"):
        assert hasattr(c, name), f"contracts is missing {name!r}"
        assert name in c.__all__

    # The seam dataclasses that cross kernel<->plane must travel with them.
    for name in ("ExecResult", "ReadinessProbe", "StateSnapshot", "VerifyResult"):
        assert hasattr(c, name), f"contracts is missing {name!r}"


def test_contracts_reexports_are_the_plane_protocols():
    """contracts re-exports the canonical Protocols, not parallel copies."""
    import benchflow.contracts as c
    from benchflow.agents.protocol import Agent as AgentProto
    from benchflow.environment.protocol import Environment as EnvProto
    from benchflow.rewards.protocol import Reward as RewardProto
    from benchflow.sandbox.protocol import Sandbox as SandboxProto

    assert c.Sandbox is SandboxProto
    assert c.Agent is AgentProto
    assert c.Environment is EnvProto
    assert c.Reward is RewardProto


def test_sandbox_registry_has_builtin_providers():
    """The three managed sandbox backends register themselves on import."""
    from benchflow.sandbox.registry import available_sandboxes

    assert {"docker", "daytona", "modal"} <= set(available_sandboxes())


def test_sandbox_registry_resolves_and_rejects():
    """resolve_sandbox dispatches by name and raises for unknown providers."""
    from benchflow.sandbox import registry

    sentinel = object()
    registry.register_sandbox("byo-test", lambda ctx: sentinel)
    ctx = registry.SandboxBuildContext(
        environment_dir=None,  # type: ignore[arg-type]
        environment_name="t",
        session_id="s",
        rollout_paths=None,
        task_env_config=None,
        persistent_env=None,
    )
    assert registry.resolve_sandbox("byo-test", ctx) is sentinel

    with pytest.raises(ValueError, match="Unknown sandbox provider"):
        registry.resolve_sandbox("does-not-exist", ctx)


def test_environment_registry_has_manifest_default_and_rejects():
    """The manifest provider is the registered default; unknown kinds raise."""
    from benchflow.environment import registry

    assert registry.DEFAULT_ENVIRONMENT_KIND in registry.available_environments()

    sentinel = object()
    registry.register_environment("byo-env-test", lambda m, sb: sentinel)
    assert (
        registry.resolve_environment("byo-env-test", object(), sandbox=object())
        is sentinel
    )

    with pytest.raises(ValueError, match="Unknown environment provider"):
        registry.resolve_environment("nope", object(), sandbox=object())


def test_kernel_imports_contracts_not_concrete_planes():
    """rollout.py types its plane handles against contracts, not concretes.

    Guards the Phase 0 reseat: the kernel module must import the plane
    Protocols from ``benchflow.contracts`` and must not re-introduce a direct
    ``ManifestEnvironment`` import (it now resolves the environment via the
    registry).
    """
    import inspect

    import benchflow.rollout as rollout

    src = inspect.getsource(rollout)
    assert "from benchflow.contracts import" in src
    assert "resolve_environment" in src
    # The concrete environment adapter is no longer imported into the kernel.
    assert "from benchflow.environment.manifest_env import" not in src
