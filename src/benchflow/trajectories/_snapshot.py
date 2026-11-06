"""Reuse redacted records while preserving complete trajectory snapshots."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from benchflow.trajectories.types import redact_trajectory_obj


class RedactedJSONLSnapshot:
    """Serialize current records, redacting only changed or new records.

    Each writer owns its cache; raw JSON is retained only in memory and only
    for the current snapshot. Comparing serialized values, rather than object
    identity, catches nested in-place changes. Reordering and truncation are
    handled by replacing the cache with the current sequence on every call.
    """

    def __init__(self) -> None:
        self._entries: list[tuple[str, str]] = []
        self._keys: list[object] = []

    def serialize(
        self,
        records: Iterable[dict[str, Any]],
        *,
        cache_keys: list[object] | None = None,
    ) -> str:
        entries: list[tuple[str, str]] = []
        for index, record in enumerate(records):
            # Only the ACP live sink supplies these trusted immutable-event or
            # tool-revision keys. General callers keep full value comparison.
            if (
                cache_keys is not None
                and cache_keys[index] is not None
                and index < len(self._keys)
                and cache_keys[index] == self._keys[index]
                and index < len(self._entries)
            ):
                entries.append(self._entries[index])
                continue
            raw = json.dumps(record, default=str)
            if index < len(self._entries) and raw == self._entries[index][0]:
                redacted = self._entries[index][1]
            else:
                # Normalize non-JSON leaves before applying the same canonical
                # value redactor as the uncached serializers. Never redact the
                # serialized text: that can corrupt backslash escapes.
                redacted = json.dumps(redact_trajectory_obj(json.loads(raw)))
            entries.append((raw, redacted))

        payload = "\n".join(redacted for _raw, redacted in entries)
        self._entries = entries
        self._keys = list(cache_keys) if cache_keys is not None else []
        return payload
