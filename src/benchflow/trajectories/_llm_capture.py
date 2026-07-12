"""Crash-safe live persistence for provider LLM trajectories."""

from __future__ import annotations

import os
from pathlib import Path

from benchflow.trajectories.types import Trajectory


class LiveLLMTrajectoryWriter:
    """Atomically publish redacted snapshots of completed LLM exchanges.

    LiteLLM's callback log is append-only, but the public BenchFlow artifact is
    rewritten from a parsed snapshot. This keeps concurrent readers from ever
    observing a partial JSON line and lets end-of-run reconciliation repair a
    missed live poll without changing the trajectory schema.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self._tmp.unlink(missing_ok=True)
        self._last_payload: str | None = None

    def write(self, trajectory: Trajectory | None) -> bool:
        """Publish *trajectory* when it is non-empty and changed."""
        if trajectory is None or not trajectory.exchanges:
            return False
        payload = trajectory.to_jsonl(redact_keys=True)
        if payload == self._last_payload:
            return False
        self._tmp.write_text(payload)
        os.replace(self._tmp, self.path)
        self._last_payload = payload
        return True

    def reconcile(self, trajectory: Trajectory | None) -> bool:
        """Publish the authoritative final snapshot.

        This deliberately shares the same serialization and atomic replacement
        path as live writes; callers may invoke it even if the last poll already
        captured the final exchange.
        """
        return self.write(trajectory)
