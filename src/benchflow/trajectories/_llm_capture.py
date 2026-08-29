"""Crash-safe live persistence for provider LLM trajectories."""

from __future__ import annotations

import json
import os
from pathlib import Path

from benchflow.trajectories.types import Trajectory


class LiveLLMTrajectoryWriter:
    """Atomically publish cumulative redacted snapshots of LLM exchanges.

    LiteLLM's callback log is append-only, but the public BenchFlow artifact is
    rewritten from parsed snapshots. Rows already published by an earlier
    provider runtime are retained when a scene switches agents or models. This
    keeps concurrent readers from ever observing a partial JSON line and lets
    end-of-run reconciliation repair a missed live poll without changing the
    trajectory schema.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self._tmp.unlink(missing_ok=True)
        self._last_payload: str | None = None
        self._base_payload = self._valid_existing_payload()

    def write(self, trajectory: Trajectory | None) -> bool:
        """Publish *trajectory* when it is non-empty and changed."""
        if trajectory is None or not trajectory.exchanges:
            return False
        snapshot = trajectory.to_jsonl(redact_keys=True)
        if snapshot == self._last_payload:
            return False
        payload = _join_jsonl(self._base_payload, snapshot)
        if payload == self._valid_existing_payload():
            self._last_payload = snapshot
            return False
        self._tmp.write_text(payload)
        os.replace(self._tmp, self.path)
        self._last_payload = snapshot
        return True

    def reconcile(self, trajectory: Trajectory | None) -> bool:
        """Publish the authoritative final snapshot.

        This deliberately shares the same serialization and atomic replacement
        path as live writes; callers may invoke it even if the last poll already
        captured the final exchange.
        """
        return self.write(trajectory)

    def _valid_existing_payload(self) -> str:
        if not self.path.is_file():
            return ""
        try:
            payload = self.path.read_text()
            for line in payload.splitlines():
                if line.strip():
                    json.loads(line)
        except (OSError, json.JSONDecodeError):
            return ""
        return payload.strip()


def _join_jsonl(existing: str, snapshot: str) -> str:
    """Join a prior-runtime prefix to the current runtime's full snapshot."""

    return "\n".join(part.strip() for part in (existing, snapshot) if part.strip())
