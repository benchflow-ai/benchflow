"""The Environment contract — the third of BenchFlow's four planes.

The kernel depends only on this Protocol; ``ManifestEnvironment`` is the
default concrete implementation (architecture.md, "The four contracts").

The contract surface is **stable**, but split by altitude:

* **Core** — ``provision`` / ``readiness`` / ``query`` / ``teardown``. Enough
  to provision a stateful world, gate the agent on its readiness, inspect it
  for the verifier, and tear it down.
* **Roll-back** — ``snapshot`` / ``restore``. The substrate ``Rollout.branch()``
  runs on; ``ManifestEnvironment`` implements them for SQLite-backed state.
* **Platform layer** — ``reset``. Declared so it wires in later without a
  contract change; ``ManifestEnvironment`` raises ``NotImplementedError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class EnvHandle:
    """Live handle to a provisioned environment.

    ``endpoints`` maps each exposed port to a reachable base URL.
    """

    name: str
    endpoints: dict[int, str] = field(default_factory=dict)


@dataclass
class ReadinessProbe:
    """Outcome of a readiness check — the gate the agent never runs before."""

    ready: bool
    checked: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class EnvState:
    """A snapshot of environment state for the verifier to inspect."""

    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateSnapshot:
    """A handle to a saved environment-state snapshot — the unit of roll-back.

    ``path`` is the in-sandbox directory the state files were captured to;
    ``restore`` copies them back from there.
    """

    id: str
    path: str = ""


@runtime_checkable
class Environment(Protocol):
    """The stateful world the agent acts in (Han's "S")."""

    # --- core ---
    async def provision(self, ctx: Any) -> EnvHandle: ...
    async def readiness(self) -> ReadinessProbe: ...
    async def query(self) -> EnvState: ...
    async def teardown(self) -> None: ...

    # --- roll-back (the substrate branching runs on) ---
    async def snapshot(self) -> StateSnapshot: ...
    async def restore(self, snap: StateSnapshot) -> None: ...
    async def reset(self) -> None: ...
