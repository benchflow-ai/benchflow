"""The kernel's contract surface — BenchFlow's four planes, in one place.

Design principle 1 (``docs/architecture.md``): *the kernel depends only on
contracts.* This package is that single import point. The kernel (Rollout
lifecycle, reward, trajectory) types itself against the ``Protocol`` classes
re-exported here; concrete providers (``DockerSandbox``, ``ManifestEnvironment``,
the ACP session adapter, ``RewardFunc``s) live in their plane packages and join
the kernel through the per-plane registries — never by being imported into the
kernel directly.

The four planes map onto Han's ``E = {T, H, V, S, C}``:

* ``Sandbox``                 — where it runs (the substrate).
* ``Agent`` / ``Session``     — who acts (Han's H, the harness).
* ``Environment``             — the world (Han's S, the stateful state machine).
* ``Reward`` / ``RewardFunc`` — how it's scored (Han's V, the verifier).

This module re-exports **only Protocols and the small dataclasses that cross
the kernel↔plane seam**. It imports no concrete provider, so importing
``benchflow.contracts`` can never pull a sandbox/agent/reward backend into the
kernel. Keep it that way.

Note on the ``Agent`` / ``Environment`` names: ``benchflow.contracts.Agent`` and
``benchflow.contracts.Environment`` are the *Protocols*. The top-level
``benchflow.Agent`` / ``benchflow.Environment`` are the ``runtime`` module's
convenience *config* dataclasses (a different thing). Import the contract from
here when you mean the plane Protocol.
"""

from __future__ import annotations

# ── Agent plane (Han's H) ──────────────────────────────────────────────────
from benchflow.agents.protocol import (
    Agent,
    AgentCapabilities,
    AskUserHandler,
    AskUserRequest,
    Session,
    StopReason,
)

# ── Environment plane (Han's S) ────────────────────────────────────────────
from benchflow.environment.protocol import (
    EnvHandle,
    Environment,
    EnvState,
    ReadinessProbe,
    StateSnapshot,
)

# ── Reward plane (Han's V) ─────────────────────────────────────────────────
from benchflow.rewards.protocol import (
    Reward,
    RewardFunc,
    VerifyResult,
)

# ── Sandbox plane (the substrate) ──────────────────────────────────────────
from benchflow.sandbox.protocol import (
    ExecResult,
    ImageBuilder,
    ImageConfig,
    ImageRef,
    Sandbox,
    SandboxImage,
    SandboxSnapshotNotSupported,
    SandboxStartupError,
)

__all__ = [
    # Sandbox plane
    "Sandbox",
    "ExecResult",
    "SandboxImage",
    "SandboxSnapshotNotSupported",
    "SandboxStartupError",
    "ImageBuilder",
    "ImageConfig",
    "ImageRef",
    # Agent plane
    "Agent",
    "Session",
    "AgentCapabilities",
    "AskUserHandler",
    "AskUserRequest",
    "StopReason",
    # Environment plane
    "Environment",
    "EnvHandle",
    "ReadinessProbe",
    "EnvState",
    "StateSnapshot",
    # Reward plane
    "Reward",
    "RewardFunc",
    "VerifyResult",
]
